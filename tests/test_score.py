"""Checks of the frozen scorer (qrec_verify.score) and of op_verify.py's command line.
Needs NetKet:  python tests/test_score.py [--cli]
"""

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
import netket as nk  # noqa: E402

from qrec_verify import hamiltonians as Hm, instance as I, score as SC, stats as S  # noqa: E402


def small_hubbard(lx=3, ly=4, nu=3, nd=3, **kw):
    kw = dict(dict(U=4.0, t1=1.0, t2=-0.25, axes=("P", "A")), **kw)
    hi, H, g = Hm.hubbard(lx, ly, nu, nd, **kw)
    _, Hplain, _ = Hm.hubbard(lx, ly, nu, nd, jax_operator=False, **kw)
    return Hm.System(f"test/{lx}x{ly}", hi, H, g, lx * ly, lx, ly, {"axes": kw["axes"]}), Hplain


def fake_hubbard_inst(lx, ly, nu, U=4.0):
    return {"instance_id": f"test/{lx}x{ly}", "model": "Hubbard", "lattice": f"rectangular-{lx}x{ly}",
            "n_sites": lx * ly, "boundary": "PA", "params": {"Nf": nu, "U": U, "t2": -0.25}, "rows": []}


def table_model(system, psi):
    """Exact amplitude table over the sector (small systems)."""
    hi = system.hilbert
    logpsi = jnp.asarray(np.log(np.maximum(np.abs(psi), 1e-300)) + 1j * np.pi * (psi.real < 0))

    class Table(nn.Module):
        @nn.compact
        def __call__(self, x):
            c = self.param("c", nn.initializers.zeros, (), jnp.float64)
            return logpsi[hi.states_to_numbers(x)] + c
    return Table()


def test_site_exchange_conserves():
    system, _ = small_hubbard(4, 4, 5, 3)
    rule = SC.site_exchange_rule(system)
    sampler = nk.sampler.MetropolisSampler(system.hilbert, rule, n_chains=16)
    x = system.hilbert.random_state(jax.random.key(1), 16)
    up, dn = SC._site_orbitals(system)
    for k in range(20):
        xp, _ = rule.transition(sampler, None, None, None, jax.random.key(10 + k), x)
        xp = np.asarray(xp)
        assert np.all(xp[:, up].sum(1) == 5) and np.all(xp[:, dn].sum(1) == 3)
        diff = (xp != np.asarray(x)).sum(1)
        assert np.all(diff <= 4)
        x = jnp.asarray(xp)
    print("  site exchange keeps (N_up, N_dn) = (5, 3) over 20 x 16 proposals")


def test_fermion_sampler_vs_fullsum():
    system, Hplain = small_hubbard()
    model = nk.models.Slater2nd(system.hilbert, param_dtype=float)
    variables = model.init(jax.random.key(5), system.hilbert.all_states()[:1])
    exact = float(np.real(nk.vqs.FullSumState(system.hilbert, model, variables=variables).expect(Hplain).mean))
    ser = SC.collect_adaptive(system, model, variables, SC.fermion_sampler(system), seed=13, n_burn=128,
                              block=32, min_samples=64 * 32 * 16, sigma_target_total=None, max_seconds=600,
                              log=lambda *_: None)
    d = S.diagnose(ser["e_loc"], ser["logabs"], ser["acc"])
    z = (d.mean - exact) / d.sigma
    print(f"  fermion hop+exchange {d.mean:.6f} +- {d.sigma:.6f} vs full sum {exact:.6f} (z {z:+.2f}), "
          f"stop {ser['stop']}, gates {[(g.name, g.status) for g in d.gates]}")
    assert abs(z) < 4, z


def test_spin_pt_vs_fullsum():
    inst = I.load("J1J2/square_16_P_0.5")
    system = Hm.build(inst)
    model = nk.models.RBM(alpha=1, param_dtype=complex)
    variables = model.init(jax.random.key(3), np.ones((1, 16)))
    variables = jax.tree_util.tree_map(lambda a: a * 3.0, variables)
    exact = float(np.real(nk.vqs.FullSumState(system.hilbert, model, variables=variables).expect(system.hamiltonian).mean))
    ser = SC.collect_adaptive(system, model, variables, SC.spin_sampler(system, n_chains=64, n_replicas=8),
                              seed=11, n_burn=256, block=16, min_samples=64 * 16 * 32, sigma_target_total=None,
                              max_seconds=600, log=lambda *_: None)
    d = S.diagnose(ser["e_loc"], ser["logabs"], ser["acc"])
    z = (d.mean - exact) / d.sigma
    print(f"  spin PT {d.mean:.6f} +- {d.sigma:.6f} vs full sum {exact:.6f} (z {z:+.2f}), gates "
          f"{[(g.name, g.status) for g in d.gates]}")
    assert abs(z) < 4, z


def test_zero_variance_and_fullsum_gate():
    inst = I.load("J1J2/square_16_P_0.2")
    system = Hm.build(inst)
    w, v = nk.exact.lanczos_ed(system.hamiltonian, k=1, compute_eigenvectors=True)
    psi = np.asarray(v[:, 0]); psi = psi / psi[np.argmax(np.abs(psi))]
    model = table_model(system, psi)
    variables = model.init(jax.random.key(0), jnp.asarray(system.hilbert.all_states()[:2]))
    e = SC.exact_energy_fullsum(system, model, variables)
    assert abs(e - float(w[0])) < 1e-9, (e, w[0])
    card = SC.evaluate(inst, model, variables, seed=3, max_hours=0.2, min_samples=4096)
    assert abs(card["energy_total"] - float(w[0])) < 1e-8 and card["sigma_total"] == 0.0, card["energy_total"]
    print(f"  ED {float(w[0]):.10f}: full sum {e:.10f}, scorer (PT, zero variance) {card['energy_total']:.10f}, "
          f"status {card['status']}, qualified {card['qualified']}")


def test_kernel_check():
    system, _ = small_hubbard()
    model = nk.models.Slater2nd(system.hilbert, param_dtype=float)
    variables = model.init(jax.random.key(7), system.hilbert.all_states()[:1])
    x = np.asarray(system.hilbert.random_state(jax.random.key(8), 16))
    good = lambda v, xs: SC.generic_local_energy(system, model, v, xs)
    bad = lambda v, xs: 1.0 + 1e-5 * np.arange(len(xs)) + SC.generic_local_energy(system, model, v, xs)
    cg, cb = SC.kernel_check(system, model, variables, good, x), SC.kernel_check(system, model, variables, bad, x)
    assert cg["pass"] and not cb["pass"], (cg, cb)
    # the written-out estimator agrees with NetKet's own local estimator on the same samples
    vs = nk.vqs.MCState(nk.sampler.MetropolisFermionHop(system.hilbert, graph=system.graph), model,
                        variables=variables, n_samples=64, seed=1, sampler_seed=2)
    xs = np.asarray(vs.samples).reshape(-1, system.hilbert.size)
    le = vs.local_estimators(system.hamiltonian)
    ref = np.asarray(getattr(le, "data", le)).reshape(-1)
    mine = SC.generic_local_energy(system, model, variables, xs)
    assert np.max(np.abs(ref - mine)) < 1e-10, np.max(np.abs(ref - mine))
    print(f"  kernel check: exact kernel pass (max {cg['max_abs_diff']:.1e}), shifted kernel rejected")


def test_evaluate_with_kernel():
    system, _ = small_hubbard()
    inst = fake_hubbard_inst(3, 4, 3)
    model = nk.models.Slater2nd(system.hilbert, param_dtype=float)
    variables = model.init(jax.random.key(9), system.hilbert.all_states()[:1])
    kern = lambda v, xs: SC.generic_local_energy(system, model, v, xs)
    card = SC.evaluate(inst, model, variables, seed=1, max_hours=0.05, kernel=kern, min_samples=8192,
                       sigma_target=1e-3, system=system)
    assert all(c["pass"] for c in card["kernel_checks"]) and len(card["kernel_checks"]) == 2
    print(f"  evaluate with kernel: E/site {card['energy_per_site']:.5f} +- {card['sigma_per_site']:.1e}, "
          f"stop {card['stop']}, qualified {card['qualified']}")


def kspace_fermi_sea(lx, ly, nu, nd, t1, t2, axes):
    """Plane-wave Fermi sea, independent of the verifier's bond lists (see tests/test_physics.py)."""
    kx = 2 * np.pi * (np.arange(lx) + (0.5 if axes[0] == "A" else 0.0)) / lx
    ky = 2 * np.pi * (np.arange(ly) + (0.5 if axes[1] == "A" else 0.0)) / ly
    cx, cy = np.cos(kx)[:, None], np.cos(ky)[None, :]
    ev = np.sort((-2 * t1 * (cx + cy) - 4 * t2 * cx * cy).ravel())
    return float(ev[:nu].sum() + ev[:nd].sum())


def test_neel_and_fermi_sea():
    r = SC.neel_check(I.load("J1J2/square_100_P_0.2"))
    assert r["pass"] and abs(r["energy_total"] + 160.0) < 1e-9, r
    for inst in (I.load("Hubbard/rectangular-4x16_64_P_28_8"), I.load("Hubbard/rectangular-8x16_128_P_56_8"),
                 fake_hubbard_inst(8, 8, 25, U=8.0)):                    # t' = -0.25, antiperiodic y
        lx, ly = I.lattice_dims(inst)
        hp = I.hubbard_params(inst)
        ref = (kspace_fermi_sea(lx, ly, hp["n_up"], hp["n_dn"], hp["t1"], hp["t2"], I.boundary_axes(inst))
               + hp["U"] * hp["n_up"] * hp["n_dn"] / (lx * ly))
        f = SC.fermi_sea_reference(inst)
        assert abs(f["expected_total"] - ref) < 1e-9, (inst["instance_id"], f, ref)
    uv = dict(fake_hubbard_inst(4, 4, 5, U=8.0), params={"Nf": 5, "U": 8, "t2": -0.25, "V1": 1, "V2": 0.5})
    assert SC.fermi_sea_reference(uv)["expected_total"] is None
    print(f"  Neel 10x10 J2=0.2: {r['energy_total']} (= -160); Fermi seas match the plane-wave sums")


def test_rules():
    mk = lambda e, s, h, seed, q=True: {"energy_per_site": e, "sigma_per_site": s, "energy_total": e * 400,
                                        "sigma_total": s * 400, "state_hash": h, "seed": seed, "qualified": q}
    k = SC.keep_rule([mk(-0.5010, 1e-5, "a", 1), mk(-0.5012, 1e-5, "b", 2)], -0.5)
    assert k["keep"] and k["best_per_site"] == -0.5012, k
    assert not SC.keep_rule([mk(-0.5010, 1e-5, "a", 1)], -0.502)["keep"]
    k3 = SC.keep_rule([mk(-0.6, 1e-5, "x", 1, q=False), mk(-0.5005, 1e-5, "a", 2)], -0.5)
    assert k3["keep"] and k3["best_per_site"] == -0.5005, k3          # an unqualified score never counts
    assert SC.keep_rule([mk(-0.5001, 1e-5, "a", 1)], -0.5)["keep"]     # no margin: lowest energy wins
    inst = I.load("J1J2/square_100_P_0.2")           # record -237.1928 Pauli total, variational rows only
    rec = I.record(inst)["energy"] / 400
    c = SC.claim_rule(inst, mk(rec - 1e-3, 5e-6, "s", 1))
    assert c["claim"] and not c["audit"], c
    c2 = SC.claim_rule(inst, mk(rec - 1e-6, 5e-5, "s", 1))          # sigma disregarded: below is a claim
    assert c2["claim"], c2
    assert not SC.claim_rule(inst, mk(rec + 1e-6, 5e-6, "s", 1))["claim"]
    c3 = SC.claim_rule(inst, mk(rec - 1e-3, 5e-6, "s", 1, q=False))
    assert not c3["claim"]
    # audit direction (v1.7): only a result BELOW a projected or extrapolated row is flagged
    # (synthetic instance: target -0.4975 per site, extrapolation -0.4980, projection -0.4990)
    row = lambda e, b: {"energy": e * 400, "sigma": None, "bound_type": b, "method": b}
    inst_a = {"instance_id": "J1J2/square_100_P_0.5", "model": "J1J2", "lattice": "square", "n_sites": 100,
              "boundary": "P", "params": {"J2": 0.5},
              "rows": [row(-0.4975, "variational"), row(-0.4980, "extrapolated"), row(-0.4990, "projected")]}
    for e_site, n_below in ((-0.4976, 0), (-0.4985, 1), (-0.4995, 2)):
        ca = SC.claim_rule(inst_a, mk(e_site, 5e-5, "s", 1))
        assert ca["claim"] and len(ca["below_references"]) == n_below and ca["audit"] == (n_below > 0), (e_site, ca)
    assert len(I.references_below(inst_a, I.target(inst_a)["energy"])) == 2   # op_verify info: refs under the target
    print("  keep rule (lowest qualified energy wins), claim rule (any qualified score below the record) and "
          "audit direction behave")


def test_cli():
    """op_verify.py in a throw-away workspace: info, neel-check, logpsi-check, ed-gate (full sum),
    two scores with independent seeds, combine."""
    ws = pathlib.Path(tempfile.mkdtemp(prefix="op-cli-", dir=os.environ.get("TMPDIR") or None))
    try:
        inst_file = ROOT / "instances" / "J1J2" / "square_16_P_0.2.json"
        (ws / "model.py").write_text(
            "import netket as nk\n"
            "def build_model():\n"
            "    return nk.models.RBM(alpha=1, param_dtype=complex)\n")
        import flax
        model = nk.models.RBM(alpha=1, param_dtype=complex)
        variables = model.init(jax.random.key(3), np.ones((1, 16)))
        (ws / "state.msgpack").write_bytes(flax.serialization.msgpack_serialize(flax.core.unfreeze(variables)))
        x = np.asarray(Hm.build(I.load("J1J2/square_16_P_0.2")).hilbert.random_state(jax.random.key(4), 8))
        np.savez(ws / "cfg.npz", x=x, logpsi=np.asarray(model.apply(variables, x)))
        py = sys.executable
        cli = str(ROOT / "op_verify.py")
        run = lambda cmd, *args: subprocess.run([py, cli, cmd, "--instance-file", str(inst_file), *args], cwd=ws,
                                                capture_output=True, text=True, timeout=900)
        for args in (["info"], ["neel-check"],
                     ["logpsi-check", "--state", "model.py:build_model", "--params", "state.msgpack", "--configs", "cfg.npz"],
                     ["ed-gate", "--state", "model.py:build_model", "--params", "state.msgpack", "--small",
                      "J1J2/square_16_P_0.2", "--seed", "1", "--rtol", "10"],
                     ["score", "--state", "model.py:build_model", "--params", "state.msgpack", "--seed", "5",
                      "--out", "s5.json", "--max-hours", "0.1"],
                     ["score", "--state", "model.py:build_model", "--params", "state.msgpack", "--seed", "6",
                      "--out", "s6.json", "--max-hours", "0.1"],
                     ["combine", "s5.json", "s6.json", "--out", "entry.json"]):
            r = run(*args)
            tail = (r.stdout.strip().splitlines() or [""])[-1][:160]
            print(f"  op_verify.py {args[0]}: rc {r.returncode}; {tail}")
            if args[0] in ("info", "neel-check", "logpsi-check", "ed-gate"):
                assert r.returncode == 0, r.stderr[-2000:]
            if args[0] == "score":
                assert r.returncode in (0, 3) and (ws / args[args.index("--out") + 1]).exists(), r.stderr[-2000:]
            if args[0] == "combine":
                assert r.returncode in (0, 1) and (ws / "entry.json").exists(), r.stderr[-2000:]
        card = json.loads((ws / "s5.json").read_text())
        assert card["__METRIC__"] == card["energy_per_site"] and card["seed"] == 5
        entry = json.loads((ws / "entry.json").read_text())
        # square_16_P_0.2 is solved: an untrained RBM sits far above the exact energy, so the entry is
        # no record, and its gap is measured against the best variational row of the file.
        assert entry["new_record"] is False and len(entry["measurements"]) == 2
        lo = min(m["energy_per_site"] for m in entry["measurements"])
        hi = max(m["energy_per_site"] for m in entry["measurements"])
        assert lo - 1e-12 <= entry["energy_per_site"] <= hi + 1e-12
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_combine_rules():
    """op_verify.combine on synthetic measurements: the entry energy, the gap and the record mark
    against the best variational energy (never an extrapolation), the agreement and qualification
    checks, the exact-energy bug check, and no mark where the instance has no variational energy."""
    import op_verify as OV
    base = {"instance_id": "J1J2/square_100_P_0.5", "model": "J1J2", "lattice": "square", "n_sites": 100,
            "boundary": "P", "params": {"J2": 0.5}}
    row = lambda e, s, b: {"energy": e * 400, "sigma": None if s is None else s * 400, "bound_type": b, "method": b}
    open_inst = dict(base, rows=[row(-0.4975, 2e-6, "variational"), row(-0.4980, None, "extrapolated")])
    card = lambda e, s, seed, q=True, h="h": {"seed": seed, "energy_per_site": e, "sigma_per_site": s,
                                              "qualified": q, "status": "pass" if q else "fail", "state_hash": h}
    r = OV.combine(open_inst, card(-0.497600, 1e-6, 1), card(-0.497601, 1e-6, 2))
    assert r["verified"] and r["new_record"] and r["gap"] < 0, r
    assert abs(r["energy_per_site"] - (-0.4976005)) < 1e-9 and len(r["below_estimates"]) == 0
    r = OV.combine(open_inst, card(-0.4981, 1e-6, 1), card(-0.4981, 1e-6, 2))
    assert r["verified"] and r["new_record"] and len(r["below_estimates"]) == 1      # below the extrapolation: listed
    r = OV.combine(open_inst, card(-0.497502, 1e-6, 1), card(-0.497502, 1e-6, 2))
    assert r["verified"] and not r["new_record"] and r["gap"] < 0                     # within 2 sigma of the target
    r = OV.combine(open_inst, card(-0.4970, 1e-6, 1), card(-0.4970, 1e-6, 2))
    assert r["verified"] and not r["new_record"] and r["gap"] > 0                     # above: listed anyway
    r = OV.combine(open_inst, card(-0.4986, 1e-6, 1), card(-0.4996, 1e-6, 2))        # disagree, far below
    assert not r["verified"] and r["new_record"] is False and r["problems"], r          # flagged: listed, never marked
    assert not OV.combine(open_inst, card(-0.4976, 1e-6, 1), card(-0.4976, 1e-6, 2, q=False))["verified"]
    assert not OV.combine(open_inst, card(-0.4976, 1e-6, 1), card(-0.4976, 1e-6, 1))["verified"]        # same seed
    assert not OV.combine(open_inst, card(-0.4976, 1e-6, 1), card(-0.4976, 1e-6, 2, h="g"))["verified"]  # two states
    solved = dict(base, rows=[row(-0.50, None, "exact"), row(-0.4975, 2e-6, "variational")])
    r = OV.combine(solved, card(-0.4990, 1e-6, 1), card(-0.4990, 1e-6, 2))
    assert r["verified"] and r["new_record"] is False and r["gap"] > 0
    assert not OV.combine(solved, card(-0.5010, 1e-6, 1), card(-0.5010, 1e-6, 2))["verified"]          # below exact
    bare = dict(base, rows=[row(-0.4980, None, "extrapolated")])
    r = OV.combine(bare, card(-0.4970, 1e-6, 1), card(-0.4970, 1e-6, 2))
    assert r["verified"] and r["new_record"] is None and r["gap"] is None and r["first_state_candidate"]
    print("  combine: entry energy, gap, record mark, agreement, qualification, exact check, first state behave")


TESTS = [test_site_exchange_conserves, test_rules, test_neel_and_fermi_sea, test_kernel_check,
         test_zero_variance_and_fullsum_gate, test_fermion_sampler_vs_fullsum, test_spin_pt_vs_fullsum,
         test_evaluate_with_kernel, test_combine_rules]

if __name__ == "__main__":
    failed = 0
    tests = TESTS + ([test_cli] if "--cli" in sys.argv else [])
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
