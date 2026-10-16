"""Checks of the framework adapters (qrec_verify/external.py, adapters/).

Two reference states, each written in every framework under adapters/examples/:
- a complex RBM on the 4x4 J1-J2 model (J2 = 0.2, the shipped instance J1J2/square_16_P_0.2);
- a Slater determinant with a Jastrow factor on a 3x4 Hubbard model (3 up + 3 down, U = 4,
  t' = -0.25, periodic x, antiperiodic y).
The flax version is the reference. For every adapter the test checks log psi on random
configurations (eager, jit and vmap) to 1e-10, the full-sum energy to 1e-10, and a Monte Carlo
energy with the scorer's samplers against the full sum; with --cli also op_verify.py logpsi-check,
ed-gate and score per adapter.

Needs NetKet; the torch, julia and c adapters also need torch, juliacall (with JLD2 in its Julia
project) and a C++ compiler. A missing toolchain is reported as SKIP, never as a pass.

    python tests/test_adapters.py [--cli] [--only torch,c] [--keep DIR]
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
EX = ROOT / "adapters" / "examples"
sys.path.insert(0, str(ROOT))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import flax  # noqa: E402

from qrec_verify import external as E, hamiltonians as Hm, instance as I, score as SC, stats as S  # noqa: E402

ADAPTERS = ["flax", "jax", "python", "torch", "c", "julia"]
TOL = 1e-10
RESULTS: dict = {}


def have(kind: str) -> bool:
    if kind == "torch":
        return importlib.util.find_spec("torch") is not None
    if kind == "julia":
        return importlib.util.find_spec("juliacall") is not None
    if kind == "c":
        return shutil.which("c++") is not None
    return True


def hubbard_inst() -> dict:
    return {"instance_id": "test/hubbard-3x4", "model": "Hubbard", "lattice": "rectangular-3x4", "n_sites": 12,
            "boundary": "PA", "params": {"Nf": 3, "U": 4.0, "t2": -0.25}, "rows": []}


def cplx(rng, shape, scale):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


# ------------------------------------------------------------------ export to every framework

def export(ws: pathlib.Path, name: str, P: dict, header: tuple, cpp: str):
    """Write one parameter set as flax msgpack, npz, torch state_dict, raw binary (C) and JLD2,
    and build the C++ library."""
    (ws / f"{name}.msgpack").write_bytes(flax.serialization.msgpack_serialize({"params": P}))
    np.savez(ws / f"{name}.npz", **P)
    with open(ws / f"{name}.bin", "wb") as fh:
        fh.write(struct.pack("<ii", *header))
        for k in P:                       # dict order = the order the C++ reader expects
            fh.write(np.ascontiguousarray(P[k]).tobytes())
    if have("torch"):
        import torch
        torch.save({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in P.items()}, ws / f"{name}.pt")
    if have("julia"):
        r = subprocess.run([sys.executable, str(ROOT / "tests" / "jld2_export.py"), str(ws / f"{name}.npz"),
                            str(ws / f"{name}.jld2")], capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0, r.stderr[-3000:]
    if have("c"):
        r = subprocess.run(["c++", "-O2", "-shared", "-fPIC", "-I", str(ROOT / "adapters"), str(EX / cpp), "-o",
                            str(ws / f"lib{name}.so")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def make_states(ws: pathlib.Path) -> dict:
    rng = np.random.default_rng(20261015)
    states = {}
    # RBM, 4x4 J1-J2
    inst = I.load("J1J2/square_16_P_0.2")
    n = 16
    P = {"W": cplx(rng, (n, n), 0.25), "b": cplx(rng, (n,), 0.1), "a": cplx(rng, (n,), 0.05)}
    export(ws, "rbm", P, (n, n), "rbm.cpp")
    states["rbm"] = dict(
        inst=inst, system=Hm.build(inst), small="j1j2:4x4:0.2", inst_file=str(ROOT / "instances" / "J1J2" / "square_16_P_0.2.json"),
        specs={"flax": (f"{EX}/rbm_flax.py:build_model", ws / "rbm.msgpack"),
               "jax": (f"jax:{EX}/rbm_jax.py:build", ws / "rbm.npz"),
               "python": (f"python:{EX}/rbm_numpy.py:build", ws / "rbm.npz"),
               "torch": (f"torch:{EX}/rbm_torch.py:RBM", ws / "rbm.pt"),
               "c": (f"c:{ws}/librbm.so", ws / "rbm.bin"),
               "julia": (f"julia:{EX}/rbm.jl", ws / "rbm.jld2")})
    # Slater-Jastrow, 3x4 Hubbard
    inst = hubbard_inst()
    no, nf = 24, 6
    A = rng.normal(size=(no, no))
    P = {"M": rng.normal(size=(no, nf)), "V": 0.1 * (A + A.T)}
    export(ws, "sj", P, (no, nf), "slater_jastrow.cpp")
    f = ws / "hubbard-3x4.json"
    f.write_text(json.dumps(inst))
    states["sj"] = dict(
        inst=inst, system=Hm.build(inst), small="hubbard:3x4:PA:3,3:4:-0.25", inst_file=str(f),
        specs={"flax": (f"{EX}/slater_jastrow_flax.py:build_model", ws / "sj.msgpack"),
               "jax": (f"jax:{EX}/slater_jastrow_jax.py:build", ws / "sj.npz"),
               "python": (f"python:{EX}/slater_jastrow_numpy.py:build", ws / "sj.npz"),
               "torch": (f"torch:{EX}/slater_jastrow_torch.py:build", ws / "sj.pt"),
               "c": (f"c:{ws}/libsj.so", ws / "sj.bin"),
               "julia": (f"julia:{EX}/slater_jastrow.jl", ws / "sj.jld2")})
    return states


def load(st: dict, kind: str):
    spec, params = st["specs"][kind]
    return E.load_state(spec, str(params), st["inst"])


def phase_diff(a, b):
    return np.abs(np.angle(np.exp(1j * (np.imag(a) - np.imag(b)))))


def per_adapter(kinds, one) -> dict:
    """one(kind) -> dict for every adapter; failures are collected, so one broken adapter does not
    hide the numbers of the others."""
    out, bad = {}, []
    for k in kinds:
        try:
            out[k] = one(k)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out[k] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            bad.append(k)
    if bad:
        raise AssertionError(f"failed for {bad}: " + json.dumps({k: out[k] for k in kinds}, default=float))
    return out


# ------------------------------------------------------------------ checks

def check_logpsi(st: dict, ref, kinds):
    """log psi of every adapter against flax on 512 random configurations: eager, jit, vmap."""
    x = np.asarray(st["system"].hilbert.random_state(jax.random.key(7), 512))
    lr = np.asarray(ref.model.apply(ref.variables, jnp.asarray(x)))

    def one(k):
        s = load(st, k)
        try:
            t0 = time.perf_counter()
            le = np.asarray(s.model.apply(s.variables, jnp.asarray(x)))
            lj = np.asarray(jax.jit(s.model.apply)(s.variables, jnp.asarray(x)))
            lv = np.asarray(jax.vmap(lambda xx: s.model.apply(s.variables, xx))(jnp.asarray(x.reshape(4, 128, -1))))
            dt = time.perf_counter() - t0
        finally:
            s.close()
        dre = max(float(np.max(np.abs(np.real(l) - np.real(lr)))) for l in (le, lj, lv.reshape(-1)))
        dph = max(float(np.max(phase_diff(l, lr))) for l in (le, lj, lv.reshape(-1)))
        assert dre < TOL and dph < TOL, (k, dre, dph)
        return {"max_dlog_abs": dre, "max_dphase": dph, "s": dt}
    return per_adapter(kinds, one)


def check_fullsum(st: dict, e_ref: float, kinds):
    def one(k):
        s = load(st, k)
        try:
            e = SC.exact_energy_fullsum(st["system"], s.model, s.variables)
        finally:
            s.close()
        assert abs(e - e_ref) < TOL * max(1.0, abs(e_ref)), (k, e, e_ref)
        return {"fullsum_total": e, "delta": e - e_ref}
    return per_adapter(kinds, one)


def check_mc(st: dict, e_ref: float, kinds, spin: bool):
    """The scorer's own sampler (smaller than the frozen settings) through every adapter: the energy
    agrees with the full sum within 4 sigma. With the same seed, the chains of all adapters are the
    same as flax's (log psi agrees to 1e-10), so the energies agree to rounding."""
    system = st["system"]

    def one(k):
        s = load(st, k)
        try:
            if spin:
                smp = SC.spin_sampler(system, n_chains=64, n_replicas=8)
                kw = dict(n_burn=256, block=16, min_samples=64 * 16 * 16)
            else:
                smp = SC.fermion_sampler(system, n_chains=32)
                kw = dict(n_burn=128, block=32, min_samples=32 * 32 * 16)
            t0 = time.perf_counter()
            ser = SC.collect_adaptive(system, s.model, s.variables, smp, seed=11, sigma_target_total=None,
                                      max_seconds=1800, log=lambda *_: None, **kw)
            wall = time.perf_counter() - t0
        finally:
            s.close()
        d = S.diagnose(ser["e_loc"], ser["logabs"], ser["acc"])
        z = (d.mean - e_ref) / d.sigma
        assert abs(z) < 4, (k, z)
        return {"mc_total": float(d.mean), "sigma_total": float(d.sigma), "z_vs_fullsum": float(z),
                "n_samples": int(ser["e_loc"].size), "wall_s": wall}
    out = per_adapter(kinds, one)
    m0 = out[kinds[0]]["mc_total"]
    for k in kinds:
        out[k]["delta_vs_" + kinds[0]] = out[k]["mc_total"] - m0
    return out


def run_cli(st: dict, kinds, ws: pathlib.Path, e_ref: float, name: str):
    """op_verify.py logpsi-check, ed-gate and score for every adapter."""
    ref = load(st, "flax")
    x = np.asarray(st["system"].hilbert.random_state(jax.random.key(4), 64))
    np.savez(ws / f"cfg_{name}.npz", x=x, logpsi=np.asarray(ref.model.apply(ref.variables, jnp.asarray(x))))
    cli = str(ROOT / "op_verify.py")

    def one(k):
        spec, params = st["specs"][k]
        base = ["--instance-file", st["inst_file"], "--state", spec, "--params", str(params)]
        res = {}
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, cli, "logpsi-check", *base, "--configs", str(ws / f"cfg_{name}.npz")],
                           cwd=ws, capture_output=True, text=True, timeout=1800)
        assert "{" in r.stdout, (k, r.returncode, r.stderr[-3000:])
        j = json.loads(r.stdout[r.stdout.index("{"):])
        res["logpsi_check"] = {"rc": r.returncode, "max_dlog_abs": j["max_abs_dlog_abs"], "max_dphase": j["max_abs_dphase"]}
        assert r.returncode == 0 and j["framework"] == k, (k, r.stdout[-2000:], r.stderr[-3000:])
        r = subprocess.run([sys.executable, cli, "ed-gate", *base, "--small", st["small"], "--seed", "1", "--rtol", "10",
                            "--out", f"ed_{name}_{k}.json"], cwd=ws, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, (k, r.stderr[-3000:])
        j = json.loads((ws / f"ed_{name}_{k}.json").read_text())
        res["ed_gate"] = {"rc": r.returncode, "ed_total": j["ed_total"], "state_total": j["state_total"], "how": j["how"]}
        assert abs(j["state_total"] - e_ref) < 1e-8 * max(1.0, abs(e_ref)), (k, j)
        r = subprocess.run([sys.executable, cli, "score", *base, "--seed", "5", "--min-samples", "16384",
                            "--max-hours", "0.5", "--out", f"score_{name}_{k}.json"], cwd=ws, capture_output=True,
                           text=True, timeout=3600)
        assert r.returncode in (0, 3) and (ws / f"score_{name}_{k}.json").exists(), (k, r.stderr[-3000:])
        c = json.loads((ws / f"score_{name}_{k}.json").read_text())
        z = (c["energy_total"] - e_ref) / c["sigma_total"] if c["sigma_total"] else float("nan")
        res["score"] = {"rc": r.returncode, "energy_total": c["energy_total"], "sigma_total": c["sigma_total"],
                        "z_vs_fullsum": z, "status": c["status"], "qualified": c["qualified"], "wall_s": c["wall_s"],
                        "framework": c.get("state_framework"), "state_hash": c["state_hash"],
                        "host_calls": c.get("host_calls")}
        assert c.get("state_framework") == k and abs(z) < 5, (k, res["score"])
        res["wall_s"] = time.perf_counter() - t0
        print(f"    cli {name} {k}: {json.dumps(res, default=float)}", flush=True)
        return res
    return per_adapter(kinds, one)


def main(argv):
    kinds = [k for k in ADAPTERS if have(k)]
    if "--only" in argv:
        want = argv[argv.index("--only") + 1].split(",")
        kinds = [k for k in kinds if k in want or k == "flax"]
    skipped = [k for k in ADAPTERS if not have(k)]
    for k in skipped:
        print(f"SKIP {k}: toolchain not installed", flush=True)
    keep = argv[argv.index("--keep") + 1] if "--keep" in argv else None
    ws = pathlib.Path(keep) if keep else pathlib.Path(tempfile.mkdtemp(prefix="op-adapters-",
                                                                        dir=os.environ.get("TMPDIR") or None))
    ws.mkdir(parents=True, exist_ok=True)
    ws = ws.resolve()
    failed = 0
    try:
        states = make_states(ws)
        print(f"  states exported to {', '.join(kinds)} in {ws}", flush=True)
        for name, st in states.items():
            spin = name == "rbm"
            ref = load(st, "flax")
            e_ref = SC.exact_energy_fullsum(st["system"], ref.model, ref.variables)
            RESULTS[name] = {"fullsum_flax_total": e_ref}
            steps = [("logpsi", lambda: check_logpsi(st, ref, kinds)),
                     ("fullsum", lambda: check_fullsum(st, e_ref, kinds)),
                     ("mc", lambda: check_mc(st, e_ref, kinds, spin))]
            if "--cli" in argv:
                steps.append(("cli", lambda: run_cli(st, kinds, ws, e_ref, name)))
            for step, fn in steps:
                try:
                    r = fn()
                    RESULTS[name][step] = r
                    print(f"PASS {name} {step}", flush=True)
                    for k, v in r.items():
                        print(f"    {name} {step} {k}: {json.dumps(v, default=float)}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    failed += 1
                    print(f"FAIL {name} {step}: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
        (ws / "summary.json").write_text(json.dumps(RESULTS, indent=1, default=float))
        print(f"  summary: {ws / 'summary.json'}")
    finally:
        if not keep:
            shutil.rmtree(ws, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
