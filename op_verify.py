"""QMBL: Open Problems re-measurement: the command line around qrec_verify.

qrec_verify is QMBL's scorer (v1.7). This file reads the instance from a QMBL instance file
(`data/<model>/<instance>.json` of a QMBL checkout, so the energy to beat is QMBL's live one) and
adds `combine`, which turns two measurements of one state into one entry.

A state is a function returning log psi in the canonical basis (any sign rule belongs to the
state) plus its parameter file, written in any framework: `--state` names the adapter
(qrec_verify/external.py, adapters/README.md). The default is `build_model()` returning a flax
module plus a flax msgpack of its variables. For big fermion systems a state may add a bounded
local-energy kernel, used only after it agrees with NetKet's generic estimator on the verifier's
own samples.

    --state file.py:build_model --params state.msgpack     flax (default; also flax:file.py:build_model)
    --state torch:file.py:NAME  --params state.pt          PyTorch module + state_dict
    --state jax:file.py:NAME    --params state.npz         JAX function apply_fn(params, x) + pytree
    --state julia:state.jl     [--params FILE]             Julia logpsi(x::Matrix{Int8})
    --state c:libstate.so      [--params FILE]             C ABI logpsi (adapters/qmbl_state.h)
    --state python:file.py:NAME [--params FILE]            numpy host function

    python op_verify.py info        --instance-file F
    python op_verify.py neel-check  --instance-file F          (spins: Neel state at full size)
    python op_verify.py fermi-sea   --instance-file F          (fermions: Fermi sea at full size)
    python op_verify.py logpsi-check --instance-file F --state model.py:build_model --params state.msgpack --configs x.npz
    python op_verify.py ed-gate     --instance-file F --state ... --params ... --small SPEC --seed S
    python op_verify.py score       --instance-file F --state ... --params ... --seed S --out score_S.json
    python op_verify.py combine     --instance-file F score_A.json score_B.json [--out entry.json]
    python op_verify.py mps-score   --instance-file F --mps state.h5 --seed S --out score_S.json

An MPS can instead be submitted as an HDF5 file (MPS_FORMAT.md); `mps-score` evaluates it exactly by
contraction (sigma 0) and writes a scorecard of the same shape, so `combine` and the entry rule are the
same; its two measurements differ only in the seed of the sampling cross-check.

Every energy printed per site is in QMBL's per-site units (S.S per site for spins, i.e. the Pauli
total / 4N; total / N for fermions).

The entry rule (2026-10-08 and 2026-10-12): every entry is listed wherever its energy
lands. A state is measured twice with independent sampler seeds, each with a fixed sample budget
(no precision target: sigma is reported as reached). The entry verifies when both measurements
qualify (every gate passes or only warns) and agree within 2.5 sigma combined; otherwise it is
listed with its problems as flags and takes no record mark. The entry's energy is their
inverse-variance mean. Its gap is (E - E_target) / |E_target|, where the target is QMBL's best
variational energy; it is a new record when it lies below the target by more than 2 sigma
combined (QMBL's tie rule, RULES.md 6). Where the instance has no variational energy yet, the
first verified entry sets the target and takes the mark; combine leaves that to the page.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qrec_verify import instance as I          # noqa: E402
from qrec_verify import score as SC            # noqa: E402
from qrec_verify import external as EX         # noqa: E402

VERSION = "op-verify 1 (qrec_verify v1.7)"
AGREE_SIGMAS = 2.5
RECORD_SIGMAS = 2.0
DETERMINISTIC_RTOL = 1e-11      # two exact evaluations of one state (sigma 0) must agree to this


def load_instance(path: str) -> dict:
    inst = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for k in ("instance_id", "model", "lattice", "n_sites", "params", "rows"):
        if k not in inst:
            raise SystemExit(f"{path}: not a QMBL instance file (no {k!r})")
    return inst


def combine(inst: dict, a: dict, b: dict) -> dict:
    """Two measurements of one state -> one entry, against the live target."""
    div = I.per_site_divisor(inst) or 1.0
    out = {"instance_id": inst["instance_id"], "verifier": VERSION,
           "measurements": [{k: c.get(k) for k in ("seed", "energy_per_site", "sigma_per_site", "status",
                                                    "qualified", "n_samples", "wall_s", "state_hash")}
                            for c in (a, b)]}
    problems = []
    if a.get("state_hash") and b.get("state_hash") and a["state_hash"] != b["state_hash"]:
        problems.append("the two measurements are of different states")
    if a.get("seed") is not None and a.get("seed") == b.get("seed"):
        problems.append("the two measurements used the same seed")
    for c in (a, b):
        if not c.get("qualified"):
            problems.append(f"seed {c.get('seed')}: not qualified ({c.get('status')})")
    ea, eb = a["energy_per_site"], b["energy_per_site"]
    sa, sb = a["sigma_per_site"], b["sigma_per_site"]
    s_ab = math.hypot(sa, sb)
    if s_ab > 0:
        out["agreement_sigmas"] = abs(ea - eb) / s_ab
    else:   # deterministic (MPS contraction): equal up to rounding of the contraction order
        out["agreement_sigmas"] = 0.0 if abs(ea - eb) <= DETERMINISTIC_RTOL * max(1.0, abs(ea)) else math.inf
    if out["agreement_sigmas"] > AGREE_SIGMAS:
        problems.append(f"the two measurements differ by {out['agreement_sigmas']:.1f} sigma (limit {AGREE_SIGMAS})")
    if sa > 0 and sb > 0:
        w = (1 / sa ** 2, 1 / sb ** 2)
        e = (w[0] * ea + w[1] * eb) / (w[0] + w[1])
        s = 1 / math.sqrt(w[0] + w[1])
    else:   # a deterministic or zero-variance state: the two must agree exactly anyway
        e, s = (ea + eb) / 2, max(sa, sb)
    out["energy_per_site"], out["sigma_per_site"] = e, s
    out["energy_total"], out["sigma_total"] = e * div, s * div

    # On an open instance the target is the best variational energy; on a solved one (calibration)
    # it is the exact energy, which no state can beat.
    t = I.target(inst)
    if t is None:
        # No variational energy on the instance yet: the first verified entry sets the target and
        # takes the record mark, which only the list of entries can decide (the page orders them).
        out["target"] = None
        out["gap"] = None
        out["new_record"] = None
        out["first_state_candidate"] = not problems
    else:
        te, ts = t["energy"] / div, (t.get("sigma") or 0.0) / div
        out["target"] = {"energy_per_site": te, "sigma_per_site": t.get("sigma") and ts,
                         "method": t.get("method"), "reference": t.get("reference")}
        out["gap"] = (e - te) / abs(te)
        out["new_record"] = t.get("bound_type") != "exact" and (te - e) > RECORD_SIGMAS * math.hypot(s, ts)
    # Projected or extrapolated energies this entry lies below: allowed, but a second measurement
    # with another sampler comes before any claim (DESIGN.md, plausibility).
    out["below_estimates"] = [{"energy_per_site": r["energy"] / div, "bound_type": r["bound_type"],
                                "method": r.get("method")}
                               for r in inst["rows"] if r.get("bound_type") in ("projected", "extrapolated")
                               and not r.get("defect") and out["energy_total"] < r["energy"]]
    ex = I.exact_record(inst)
    if ex is not None and out["energy_total"] < ex["energy"] - 3 * math.hypot(out["sigma_total"], ex.get("sigma") or 0.0):
        problems.append("below the exact energy: a bug, not a result")
    out["problems"] = problems
    out["verified"] = not problems
    if problems and out["new_record"]:
        out["new_record"] = False          # a flagged entry is listed, never marked
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("info", "neel-check", "fermi-sea"):
        sub.add_parser(name).add_argument("--instance-file", required=True)
    for name in ("score", "ed-gate", "logpsi-check"):
        p = sub.add_parser(name)
        p.add_argument("--instance-file", required=True)
        p.add_argument("--state", required=True,
                       help="[flax:|torch:|jax:|python:]file.py:NAME, julia:file.jl or c:lib.so (adapters/README.md)")
        p.add_argument("--params", default=None,
                       help="parameter file: flax msgpack, torch state_dict, npz/msgpack pytree, or a file the "
                            "julia/c/python state reads itself")
        p.add_argument("--host-chunk", type=int, default=None,
                       help=f"most configurations per call of a host state (default {EX.DEFAULT_CHUNK})")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--kernel", default=None, help="file.py:factory(model, instance) -> fn(variables, x) -> e_loc")
        p.add_argument("--max-hours", type=float, default=SC.MAX_HOURS_DEFAULT)
        p.add_argument("--chunk-size", type=int, default=None)
        p.add_argument("--min-samples", type=int, default=None)
        p.add_argument("--sigma-target", type=float, default=None)
        if name == "ed-gate":
            p.add_argument("--small", required=True)
            p.add_argument("--rtol", type=float, default=1e-3)
        if name == "logpsi-check":
            p.add_argument("--configs", required=True)
    p = sub.add_parser("mps-score", help="exact energy of an MPS file (MPS_FORMAT.md)")
    p.add_argument("--instance-file", required=True)
    p.add_argument("--mps", required=True, help="HDF5 file in the format of MPS_FORMAT.md")
    p.add_argument("--seed", type=int, required=True, help="seed of the sampling cross-check")
    p.add_argument("--out", default=None)
    p.add_argument("--n-samples", type=int, default=512, help="perfect samples for the cross-check (0: skip)")
    p.add_argument("--max-gb", type=float, default=None, help="memory limit of the contraction")
    p.add_argument("--max-tflop", type=float, default=None, help="operation limit of the contraction")
    p.add_argument("--variance", choices=("auto", "yes", "no"), default="auto")
    p = sub.add_parser("combine")
    p.add_argument("--instance-file", required=True)
    p.add_argument("scores", nargs=2)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    inst = load_instance(a.instance_file)
    div = I.per_site_divisor(inst)

    if a.cmd == "combine":
        cards = [json.loads(pathlib.Path(f).read_text()) for f in a.scores]
        r = combine(inst, *cards)
        txt = SC.to_json(r)
        if a.out:
            pathlib.Path(a.out).write_text(txt)
        print(txt)
        return 0 if r["verified"] else 1

    if a.cmd == "mps-score":
        from qrec_verify import mps as MPS
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
        except ImportError:
            pass        # the sampling cross-check is skipped with a warning
        lim = {k: v for k, v in (("max_gb", a.max_gb), ("max_tflop", a.max_tflop)) if v is not None}
        card = MPS.evaluate(inst, a.mps, seed=a.seed, n_samples=a.n_samples, limits=lim, variance=a.variance)
        card["verifier"] = VERSION
        card["__METRIC__"] = card["energy_per_site"]
        out = SC.to_json(card)
        if a.out:
            pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(a.out).write_text(out)
        e = card["energy_per_site"]
        print(f"[mps-score] E/site {'none' if e is None else f'{e:.12f}'} (exact contraction), status "
              f"{card['status']}, qualified {card['qualified']}; "
              + "; ".join(f"{g['name']} {g['status']}" for g in card["gates"]), flush=True)
        return 0 if card["qualified"] else 3

    import jax
    jax.config.update("jax_enable_x64", True)

    if a.cmd == "info":
        t, q = I.target(inst), I.record(inst)
        info = {"instance_id": inst["instance_id"], "model": inst["model"], "params": inst["params"],
                "boundary": inst.get("boundary"), "n_sites": inst["n_sites"], "per_site_divisor": div,
                "dof": I.expected_dof(inst), "einf": I.expected_einf(inst),
                "target": None if t is None else {"energy_per_site": t["energy"] / div, "sigma_per_site":
                                                  None if t.get("sigma") is None else t["sigma"] / div,
                                                  "method": t.get("method"), "bound_type": t.get("bound_type")},
                "qmbl_record": None if q is None else {"energy_per_site": q["energy"] / div,
                                                       "method": q.get("method"), "bound_type": q.get("bound_type")},
                "budget_samples": (SC.SPIN if SC.is_spin(inst) else SC.FERMION)["min_samples"],
                "sampler": SC.SPIN if SC.is_spin(inst) else SC.FERMION}
        info["known_state"] = SC.neel_check(inst) if SC.is_spin(inst) else SC.fermi_sea_reference(inst)
        print(SC.to_json(info))
        return 0
    if a.cmd == "neel-check":
        r = SC.neel_check(inst)
        print(SC.to_json(r))
        return 0 if r["pass"] else 1
    if a.cmd == "fermi-sea":
        print(SC.to_json(SC.fermi_sea_reference(inst)))
        return 0

    def load(i):
        try:
            return EX.load_state(a.state, a.params, i, chunk=a.host_chunk)
        except ValueError as exc:
            ap.error(str(exc))

    if a.cmd == "logpsi-check":
        import numpy as np
        import jax.numpy as jnp
        st = load(inst)
        model, variables = st.model, st.variables
        z = np.load(a.configs)
        lp = np.asarray(model.apply(variables, jnp.asarray(z["x"])))
        ref = np.asarray(z["logpsi"])
        dre = float(np.max(np.abs(np.real(lp) - np.real(ref))))
        dph = float(np.max(np.abs(np.angle(np.exp(1j * (np.imag(lp) - np.imag(ref)))))))
        r = {"n": int(len(ref)), "max_abs_dlog_abs": dre, "max_abs_dphase": dph, "pass": bool(dre < 1e-6 and dph < 1e-6),
             "framework": st.kind}
        st.close()
        print(SC.to_json(r))
        return 0 if r["pass"] else 1

    if a.seed is None:
        ap.error("--seed is required")
    if a.cmd == "ed-gate":
        small_inst, system = SC.small_system(a.small)
        st = load(small_inst)
        model, variables = st.model, st.variables
        e0 = SC.ground_state_energy(system)
        sdiv = I.per_site_divisor(small_inst)
        try:
            e = SC.exact_energy_fullsum(system, model, variables)
            sig, how = 0.0, "full sum over the sector"
        except ValueError:
            kernel = SC.make_kernel(a.kernel, model, small_inst)
            card = SC.evaluate(small_inst, model, variables, seed=a.seed, max_hours=a.max_hours, kernel=kernel,
                               chunk_size=a.chunk_size, min_samples=16384, sigma_target=2e-4, system=system)
            e, sig, how = card["energy_total"], card["sigma_total"], f"Monte Carlo ({card['status']})"
        d = e - e0
        ok = (d >= -3 * sig - 1e-9) and (d <= max(3 * sig, a.rtol * abs(e0)))
        r = {"small": a.small, "ed_total": e0, "state_total": e, "sigma_total": sig, "how": how,
             "delta_total": d, "delta_per_site": d / sdiv, "rel": d / abs(e0), "rtol": a.rtol, "pass": bool(ok),
             "framework": st.kind}
        st.close()
        print(SC.to_json(r))
        if a.out:
            pathlib.Path(a.out).write_text(SC.to_json(r))
        return 0 if ok else 1

    # score
    st = load(inst)
    model, variables = st.model, st.variables
    kernel = SC.make_kernel(a.kernel, model, inst)
    card = SC.evaluate(inst, model, variables, seed=a.seed, max_hours=a.max_hours, kernel=kernel,
                       chunk_size=a.chunk_size, state_id=str(pathlib.Path(a.params or a.state)),
                       min_samples=a.min_samples, sigma_target=a.sigma_target)
    card["seed"] = a.seed
    card["verifier"] = VERSION
    card["state_spec"] = a.state
    card["state_framework"] = st.kind
    if st.host is not None:
        card["host_calls"] = {"calls": st.host.n_calls, "configurations": st.host.n_configs}
    st.close()
    out = SC.to_json(card)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(out)
    print(f"[score] E/site {card['energy_per_site']:.8f} +- {card['sigma_per_site']:.2g}, status {card['status']}, "
          f"qualified {card['qualified']}, {card['n_samples']} samples, {card['wall_s'] / 3600:.2f} h", flush=True)
    return 0 if card["qualified"] else 3


if __name__ == "__main__":
    sys.exit(main())
