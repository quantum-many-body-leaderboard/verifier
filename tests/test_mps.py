"""Checks of the MPS route (qrec_verify/mps.py, MPS_FORMAT.md, adapters/).

Part A (numpy, h5py): the file format, its validation, the sector gate, the cost estimate.
Part B (--netket): the verifier's MPO against NetKet's Hamiltonian on random MPS in the instance's
  sector (dense state vector over the sector, NetKet sparse matrix), several site orders, an
  antiperiodic axis, t', V1, V2, open axes; <H^2>; the sampling cross-check.
Part C (--tenpy): DMRG ground states with TeNPy on solved 4x4 instances, written with
  adapters/mps_tenpy.py, against QMBL's exact energy, TeNPy's own energy and the dense NetKet
  evaluation; a snake order; an antiperiodic axis with t' and V1 (against NetKet's ED); a wrong site
  order and wrong fermion conventions; op_verify.py mps-score twice and combine.
Part D (--itensor, needs QMBL_JULIA=path/to/julia): the same with ITensors.jl DMRG and
  adapters/mps_itensor.jl.
Part E (--bench): contraction time and memory on 8x8 cylinders at growing bond dimension.

    python tests/test_mps.py [--netket] [--tenpy] [--itensor] [--bench]
Results are also written to $MPS_TEST_DIR/summary.json (default: a temporary directory).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qrec_verify import instance as I, mps as MPS  # noqa: E402
from adapters.mps_numpy import write_mps, snake_order, column_order  # noqa: E402

OUT = pathlib.Path(os.environ.get("MPS_TEST_DIR") or tempfile.mkdtemp(prefix="mps-test-"))
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY: dict = {}
CHARGES = {"spin-half": np.array([[1], [0]]), "electron": np.array([[0, 0], [1, 0], [0, 1], [1, 1]])}


def note(key, value):
    SUMMARY[key] = value
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=str))


def fake_inst(model, lx, ly, boundary, **params):
    lat = "square" if lx == ly else f"rectangular-{lx}x{ly}"
    return {"instance_id": f"test/{model}-{lx}x{ly}-{boundary}", "model": model, "lattice": lat,
            "n_sites": lx * ly, "boundary": boundary, "params": params, "rows": []}


def row_order(lx, ly):
    """y-major (x fastest): the transpose of QMBL's numbering."""
    return [x * ly + y for y in range(ly) for x in range(lx)]


def random_sector_mps(n, basis, target, chi_q, rng, dtype=float):
    """Random MPS whose every configuration has the charges `target` (number of up spins; or
    (N_up, N_dn)): bond labels carry the charge of the left part, chi_q copies per charge."""
    ch = CHARGES[basis]
    target = np.atleast_1d(np.asarray(target))
    def labels(k):
        out = []
        for q in np.ndindex(*(target + 1)):
            q = np.array(q)
            if np.all(q <= np.minimum(target, k)) and np.all(target - q <= n - k):
                out += [q] * (1 if k in (0, n) else chi_q)
        return out
    bonds = [labels(k) for k in range(n + 1)]
    ts = []
    for k in range(n):
        L, R = bonds[k], bonds[k + 1]
        A = rng.normal(size=(len(L), len(ch), len(R)))
        if dtype is complex:
            A = A + 1j * rng.normal(size=A.shape)
        mask = np.array([[[np.array_equal(l + ch[s], r) for r in R] for s in range(len(ch))] for l in L])
        ts.append(A * mask)
    return ts


def random_mps(n, d, chi, rng):
    dims = [min(chi, d ** k, d ** (n - k)) for k in range(n + 1)]
    return [rng.normal(size=(dims[k], d, dims[k + 1])) / math.sqrt(dims[k] * d) for k in range(n)]


def sector_target(inst):
    t = MPS.sector_targets(inst)
    return (t["n_up"], t["n_dn"]) if "n_up" in t else (inst["n_sites"] // 2,)


# ------------------------------------------------------------------ part A

def test_format_and_validation():
    rng = np.random.default_rng(0)
    inst = I.load("J1J2/square_16_P_0.5")
    ts = random_sector_mps(16, "spin-half", 8, 3, rng)
    p = OUT / "fmt.h5"
    write_mps(str(p), ts, snake_order(4, 4), "spin-half", energy=-1.0, instance_id=inst["instance_id"])
    m = MPS.load(str(p), inst)
    assert m.n_sites == 16 and m.order.tolist() == snake_order(4, 4) and m.bond_dims[0] == m.bond_dims[-1] == 1
    assert len(m.sha256) == 64
    import h5py
    bad = {"order": lambda f: (f.__delitem__("site_order"), f.create_dataset("site_order", data=np.zeros(16, int))),
           "basis": lambda f: f.attrs.__setitem__("local_basis", "electron"),
           "format": lambda f: f.attrs.__setitem__("format", "other"),
           "bond": lambda f: (f["tensors"].__delitem__("3"), f["tensors"].create_dataset("3", data=np.zeros((5, 2, 5)))),
           "instance": lambda f: f.attrs.__setitem__("instance_id", "J1J2/square_16_P_0.2")}
    for name, spoil in bad.items():
        q = OUT / f"bad-{name}.h5"
        q.write_bytes(p.read_bytes())
        with h5py.File(q, "r+") as f:
            spoil(f)
        try:
            MPS.load(str(q), inst)
        except ValueError as exc:
            print(f"  rejected ({name}): {exc}")
            continue
        raise AssertionError(f"{name}: a broken file was accepted")
    # legs given as (physical, left, right) and boundary tensors without their size-1 leg
    ts2 = [np.transpose(t, (1, 0, 2)) for t in ts]
    ts2[0], ts2[-1] = ts2[0][:, 0, :], ts2[-1][:, :, 0]
    write_mps(str(OUT / "fmt2.h5"), ts2, snake_order(4, 4), "spin-half", leg_order="plr")
    for a, b in zip(MPS.load(str(OUT / "fmt2.h5")).tensors(), ts):
        assert np.array_equal(a, b)


def test_sector_gate_numpy():
    rng = np.random.default_rng(1)
    inst = I.load("Hubbard/square_16_P_4_8")
    good = random_sector_mps(16, "electron", (4, 4), 2, rng)
    q = MPS.contract(good, MPS.charge_mpo("electron", 16, MPS.sector_targets(inst)))
    bad = random_mps(16, 4, 8, rng)
    qb = MPS.contract(bad, MPS.charge_mpo("electron", 16, MPS.sector_targets(inst)))
    print(f"  sector deviation: in-sector state {q['value']:.1e}, random state {qb['value']:.2f}")
    assert abs(q["value"]) < 1e-10 and qb["value"] > 1.0


def test_cost_estimate():
    inst = fake_inst("Hubbard", 8, 8, "OP", Nf=28, U=8, t2=-0.25)
    mpo = MPS.build_mpo(inst, column_order(8, 8))
    for chi in (1000, 10000):
        dims = [min(chi, 4 ** k, 4 ** (64 - k)) for k in range(65)]
        c = MPS.cost_estimate(dims, mpo.bond_dims, 4)
        print(f"  Hubbard 8x8 cylinder, chi {chi}: D {c['mpo_D_max']}, <H> {c['tflop_energy']:.3g} Tflop, "
              f"{c['gb_energy']:.3g} GB; <H^2> {c['tflop_variance']:.3g} Tflop, {c['gb_variance']:.3g} GB; "
              f"dense file {c['gb_mps']:.3g} GB")


def test_combine_deterministic():
    """Two exact evaluations (sigma 0) of one file: equal energies verify, a difference above
    rounding does not."""
    import op_verify as OV
    inst = I.load("J1J2/square_16_P_0.5")
    e = I.exact_record(inst)["energy"] / 64 + 1e-4
    def card(energy, seed):
        return {"seed": seed, "energy_per_site": energy, "sigma_per_site": 0.0, "status": "pass",
                "qualified": True, "state_hash": "h"}
    r = OV.combine(inst, card(e, 1), card(e * (1 + 1e-13), 2))
    assert r["verified"] and r["sigma_per_site"] == 0.0 and r["new_record"] is False, r
    r = OV.combine(inst, card(e, 1), card(e * (1 + 1e-9), 2))
    assert not r["verified"], r
    print(f"  deterministic combine: equal -> verified, 1e-9 apart -> {r['problems']}")


# ------------------------------------------------------------------ dense reference with NetKet

def netket_system(inst, plain=True):
    """The verifier's Hamiltonian (hamiltonians.py); for fermions the plain FermionOperator2nd,
    whose sparse matrix NetKet builds."""
    from qrec_verify import hamiltonians as Hm
    if inst["model"] != "Hubbard" or not plain:
        return Hm.build(inst)
    lx, ly = I.lattice_dims(inst)
    hp = I.hubbard_params(inst)
    hi, H, g = Hm.hubbard(lx, ly, hp["n_up"], hp["n_dn"], hp["U"], hp["t1"], hp["t2"], I.boundary_axes(inst),
                          hp["V1"], hp["V2"], jax_operator=False)
    return Hm.System(inst["instance_id"], hi, H, g, lx * ly, lx, ly, {})


_SPARSE: dict = {}


def dense_reference(inst, tensors, order, basis, system=None, chunk=100_000):
    """<H>, <H^2> - <H>^2 and the sector norm from the dense state vector over NetKet's sector."""
    key = json.dumps([inst["instance_id"], inst["boundary"], inst["params"]], sort_keys=True)
    if key not in _SPARSE:          # the 4x4 Hubbard matrix takes minutes; build it once
        system = system or netket_system(inst)
        _SPARSE[key] = (system, system.hamiltonian.to_sparse())
    system, M = _SPARSE[key]
    xs = np.asarray(system.hilbert.all_states())
    nmap = MPS.NetKetMap(system, np.asarray(order), basis)
    psi = np.empty(len(xs), dtype=np.result_type(*tensors))
    for a in range(0, len(xs), chunk):
        idx, sgn = nmap.to_chain(xs[a:a + chunk])
        psi[a:a + chunk] = MPS.amplitudes(tensors, idx) * sgn
    nrm = np.vdot(psi, psi).real
    Hpsi = M @ psi
    E = np.vdot(psi, Hpsi).real / nrm
    var = np.vdot(Hpsi, Hpsi).real / nrm - E * E
    return float(E), float(var), float(nrm), psi


def ground_state_energy(system):
    import netket as nk
    return float(nk.exact.lanczos_ed(system.hamiltonian, k=1, compute_eigenvectors=False)[0])


# ------------------------------------------------------------------ part B

def nk_mpo_vs_netket_random():
    cases = [(I.load("J1J2/square_16_P_0.5"), snake_order(4, 4), 3, float),
             (fake_inst("J1J2", 3, 4, "O", J2=0.3), row_order(3, 4), 4, complex),
             (fake_inst("Heisenberg", 4, 3, "OP"), column_order(4, 3), 4, float),
             (I.load("Hubbard/square_16_P_4_8"), column_order(4, 4), 1, float),
             (fake_inst("Hubbard", 4, 4, "PA", Nf=3, U=4, t2=-0.25, V1=0.5, V2=0.25), None, 2, complex),
             (fake_inst("Hubbard", 3, 4, "OA", Nf=3, U=6, t2=0.3, V1=1.0), snake_order(3, 4), 3, float)]
    rows = []
    for k, (inst, order, chi_q, dt) in enumerate(cases):
        rng = np.random.default_rng(10 + k)
        n = inst["n_sites"]
        if order is None:
            order = rng.permutation(n).tolist()
        basis = MPS.basis_for(inst)
        ts = random_sector_mps(n, basis, sector_target(inst), chi_q, rng, dt)
        mpo = MPS.build_mpo(inst, order)
        e = MPS.contract(ts, mpo)
        h2 = MPS.contract(ts, mpo, layers=2)
        Ed, vard, _, _ = dense_reference(inst, ts, order, basis)
        dv = (h2["value"] - e["value"] ** 2) - vard
        rel = abs(e["value"] - Ed) / max(1, abs(Ed))
        print(f"  {inst['instance_id']} ({dt.__name__}, chi {max(t.shape[2] for t in ts)}, {mpo.description.split(': ')[1]}): "
              f"MPO {e['value']:.12f}, NetKet {Ed:.12f}, rel {rel:.1e}; variance {vard:.6f}, diff {dv:.1e}")
        rows.append({"instance": inst["instance_id"], "mpo": e["value"], "netket": Ed, "rel": rel, "var_diff": dv})
        assert rel < 1e-10 and abs(e["imag"]) < 1e-10, (rel, e)
        assert abs(dv) < 1e-8 * max(1, vard), dv
    note("random_mps_vs_netket", rows)


def nk_sampling_check_random():
    inst = fake_inst("Hubbard", 4, 4, "PA", Nf=3, U=4, t2=-0.25, V1=0.5, V2=0.25)
    rng = np.random.default_rng(3)
    order = snake_order(4, 4)
    ts = random_sector_mps(16, "electron", (3, 3), 2, rng)
    p = OUT / "random-pa.h5"
    write_mps(str(p), ts, order, "electron")
    E = MPS.contract(ts, MPS.build_mpo(inst, order))["value"]
    m = MPS.load(str(p), inst)
    s = MPS.sampling_check(inst, m, E, 2048, seed=5)
    print(f"  sampling check, random state: {s['mean_total']:.6f} +- {s['stderr_total']:.4f} vs {E:.6f} (z {s['z']:+.2f})")
    assert abs(s["z"]) < 4
    # the same samples with the wrong fermion sign must disagree: drop the reordering sign
    orig = MPS.NetKetMap.to_chain
    MPS.NetKetMap.to_chain = lambda self, x: (orig(self, x)[0], np.ones(len(x)))
    try:
        s2 = MPS.sampling_check(inst, m, E, 16384, seed=5)
    finally:
        MPS.NetKetMap.to_chain = orig
    print(f"  without the ordering sign: {s2['mean_total']:.6f} +- {s2['stderr_total']:.4f} (z {s2['z']:+.1f})")
    assert abs(s2["z"]) > 4
    note("sampling_random", {"z": s["z"], "z_without_sign": s2["z"]})


# ------------------------------------------------------------------ part C (TeNPy)

def chain_terms(inst, order):
    basis, onsite, pairs = MPS.hamiltonian_terms(inst)
    pos = np.empty(len(order), dtype=int)
    pos[np.asarray(order)] = np.arange(len(order))
    return (basis, [(int(pos[i]), name, c) for i, name, c in onsite],
            [(int(min(pos[i], pos[j])), int(max(pos[i], pos[j])), kind, c) for i, j, kind, c in pairs])


def tenpy_dmrg(inst, order, init, chi, sweeps=30):
    """DMRG with TeNPy's own coupling-model machinery (its own Jordan-Wigner handling)."""
    from tenpy.algorithms import dmrg
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingModel, MPOModel
    from tenpy.networks.mps import MPS as TMPS
    from tenpy.networks.site import SpinHalfFermionSite, SpinHalfSite
    basis, onsite, pairs = chain_terms(inst, order)
    n = inst["n_sites"]
    site = SpinHalfSite(conserve="Sz") if basis == "spin-half" else SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    lat = Chain(n, site, bc="open", bc_MPS="finite")

    class Model(CouplingModel, MPOModel):
        def __init__(self):
            CouplingModel.__init__(self, lat)
            for k, _, c in onsite:
                self.add_onsite_term(c, k, "NuNd")
            for p, q, kind, c in pairs:
                if kind == "heis":
                    self.add_coupling_term(4 * c, p, q, "Sz", "Sz")
                    self.add_coupling_term(2 * c, p, q, "Sp", "Sm")
                    self.add_coupling_term(2 * c, p, q, "Sm", "Sp")
                elif kind == "hop":       # c+_p c_q and c+_q c_p = - c_p c+_q
                    # TeNPy 1.1's add_coupling_term inserts no Jordan-Wigner string itself: the string
                    # runs between p and q ("JW") and acts on p too ("op JW", the product op @ JW)
                    for cd, cc in (("Cdu", "Cu"), ("Cdd", "Cd")):
                        self.add_coupling_term(c, p, q, cd + " JW", cc, op_string="JW")
                        self.add_coupling_term(-c, p, q, cc + " JW", cd, op_string="JW")
                elif kind == "nn":
                    self.add_coupling_term(c, p, q, "Ntot", "Ntot")
            MPOModel.__init__(self, lat, self.calc_H_MPO())

    model = Model()
    psi = TMPS.from_product_state(lat.mps_sites(), init, bc="finite")
    t0 = time.perf_counter()
    info = dmrg.run(psi, model, {"mixer": True, "max_E_err": 1e-11, "max_sweeps": sweeps, "min_sweeps": 6,
                                 "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
                                 "chi_list": {0: 32, 2: 128, 4: chi}})
    E = float(model.H_MPO.expectation_value(psi))
    return psi, E, time.perf_counter() - t0, max(psi.chi)


def check_state(label, inst, path, order, *, exact=None, dmrg_tol=None, dense=True, n_samples=512):
    """Score the file and compare: stated energy (the producing code's), exact, dense NetKet."""
    card = MPS.evaluate(inst, str(path), seed=1, n_samples=n_samples, log=lambda *a: None)
    E = card["energy_total"]
    g = {x["name"]: x["status"] for x in card["gates"]}
    row = {"label": label, "verifier": E, "stated": card.get("stated_energy_total"), "status": card["status"],
           "gates": g, "chi_max": card["chi_max"], "variance": card.get("variance_total"),
           "contraction_s": card.get("contraction_wall_s"), "sampling_z": card.get("sampling", {}).get("z")}
    if exact is not None:
        row["exact"] = exact
        row["minus_exact"] = E - exact
    if dense:
        m = MPS.load(str(path), inst)
        t0 = time.perf_counter()
        Ed, vard, nrm, _ = dense_reference(inst, m.tensors(), order, m.basis)
        row.update(dense=Ed, minus_dense=E - Ed, dense_variance=vard, dense_s=time.perf_counter() - t0)
    print(f"  {label}: verifier {E:.12f}, stated {row['stated']}, "
          + (f"exact {exact:.12f} (diff {E - exact:.2e}), " if exact is not None else "")
          + (f"dense NetKet diff {row['minus_dense']:.1e}, " if dense else "")
          + f"variance {row['variance']}, sampling z {row['sampling_z']}, gates {g}")
    note(label, row)
    if dense:
        assert abs(row["minus_dense"]) < 1e-10 * max(1, abs(E)), row
    if exact is not None:
        assert E >= exact - 1e-9 and E - exact < dmrg_tol, row
    return card, row


def tenpy_case(label, inst, order, init, chi, exact, dmrg_tol):
    from adapters.mps_tenpy import write_tenpy
    psi, E, wall, chimax = tenpy_dmrg(inst, order, init, chi)
    print(f"  TeNPy DMRG {label}: E {E:.12f}, chi {chimax}, {wall:.0f} s")
    path = OUT / f"tenpy-{label}.h5"
    write_tenpy(str(path), psi, order, energy=E, instance_id=inst["instance_id"])
    card, row = check_state(f"tenpy-{label}", inst, path, order, exact=exact, dmrg_tol=dmrg_tol)
    assert card["status"] == "pass", card["gates"]
    assert abs(row["verifier"] - E) < 1e-9 * max(1, abs(E))
    return path, E


def wrong_conventions(label, inst, path, wrong_order):
    """Rewrite one file with a wrong site order and wrong fermion conventions; the verifier's energy
    must move (it is still a valid upper bound) and the stated-energy gate must fail."""
    m = MPS.load(str(path), inst)
    ts, claim = m.tensors(), float(m.attrs["energy"])
    variants = {"site order": dict(site_order=wrong_order)}
    if m.basis == "electron":
        variants["|up+down> = c+_dn c+_up |0> (double-occupancy sign)"] = dict(double_sign=-1)
        variants["up and down swapped"] = dict(basis_map=[0, 2, 1, 3])
    else:
        variants["up and down swapped"] = dict(basis_map=[1, 0])
    rows = {}
    for name, kw in variants.items():
        q = OUT / f"wrong-{label}-{len(rows)}.h5"
        args = dict(site_order=m.order, energy=claim, instance_id=inst["instance_id"])
        args.update(kw)
        write_mps(str(q), ts, args.pop("site_order"), m.basis, **args)
        card = MPS.evaluate(inst, str(q), seed=1, n_samples=0, variance="no", log=lambda *a: None)
        g = {x["name"]: x["status"] for x in card["gates"]}
        rows[name] = {"verifier": card["energy_total"], "stated": claim, "diff": card["energy_total"] - claim,
                      "stated_gate": g["stated energy"]}
        print(f"  wrong {name}: verifier {card['energy_total']:.10f} vs stated {claim:.10f} "
              f"(diff {card['energy_total'] - claim:+.3e}), stated-energy gate {g['stated energy']}")
        if m.basis == "spin-half" and name == "up and down swapped":
            # a global spin flip: SU(2)-symmetric Hamiltonian, same energy; harmless and undetectable
            assert abs(card["energy_total"] - claim) < 1e-9 * abs(claim)
        else:
            assert g["stated energy"] == "fail" and card["energy_total"] > claim + 1e-6, rows[name]
    note(f"wrong-{label}", rows)


def tp_j1j2_snake():
    inst = I.load("J1J2/square_16_P_0.5")
    order = snake_order(4, 4)
    exact = I.exact_record(inst)["energy"]
    path, _ = tenpy_case("J1J2-4x4-snake", inst, order, ["up", "down"] * 8, 256, exact, 1e-6)
    wrong_conventions("tenpy-J1J2", inst, path, column_order(4, 4))
    return path


def tp_hubbard():
    inst = I.load("Hubbard/square_16_P_4_8")
    order = column_order(4, 4)
    exact = I.exact_record(inst)["energy"]
    chi = int(os.environ.get("MPS_TEST_CHI", "800"))
    path, _ = tenpy_case("Hubbard-4x4", inst, order, ["up", "down", "empty", "empty"] * 4, chi, exact, 5e-3)
    wrong_conventions("tenpy-Hubbard", inst, path, snake_order(4, 4))
    return path


def tp_hubbard_antiperiodic():
    inst = fake_inst("Hubbard", 4, 4, "PA", Nf=3, U=4, t2=-0.25, V1=0.5)
    order = row_order(4, 4)
    e0 = ground_state_energy(netket_system(inst, plain=False))
    note("ed-Hubbard-4x4-PA", e0)
    init = ["up", "empty", "down", "empty", "empty", "up", "empty", "down"] + ["empty"] * 4 + ["up", "down", "empty", "empty"]
    tenpy_case("Hubbard-4x4-PA", inst, order, init, 400, e0, 5e-3)


def tp_cli_combine(path):
    """op_verify.py mps-score with two seeds, then combine: the entry rule of sampled states."""
    inst_file = ROOT / "instances" / "J1J2" / "square_16_P_0.5.json"
    cards = []
    for seed in (11, 12):
        out = OUT / f"cli-{seed}.json"
        r = subprocess.run([sys.executable, str(ROOT / "op_verify.py"), "mps-score", "--instance-file", str(inst_file),
                            "--mps", str(path), "--seed", str(seed), "--out", str(out)], capture_output=True, text=True)
        print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
        assert r.returncode == 0, r.stderr[-2000:]
        cards.append(str(out))
    r = subprocess.run([sys.executable, str(ROOT / "op_verify.py"), "combine", "--instance-file", str(inst_file), *cards],
                       capture_output=True, text=True)
    entry = json.loads(r.stdout)
    print(f"  combine: verified {entry['verified']}, E/site {entry['energy_per_site']:.12f}, gap {entry['gap']:.2e}, "
          f"new record {entry['new_record']}, problems {entry['problems']}")
    assert entry["verified"] and entry["sigma_per_site"] == 0.0 and entry["new_record"] is False
    note("cli_combine", {k: entry[k] for k in ("verified", "energy_per_site", "gap", "new_record", "problems")})


# ------------------------------------------------------------------ part D (ITensors.jl)

def itensor_case(label, inst, order, init, chi, exact, dmrg_tol, sweeps=14):
    import h5py
    julia = os.environ["QMBL_JULIA"]
    basis, onsite, pairs = chain_terms(inst, order)
    job, out = OUT / f"itensor-{label}-job.h5", OUT / f"itensor-{label}.h5"
    kinds = {"heis": 0, "hop": 1, "nn": 2}
    with h5py.File(job, "w") as f:
        for k, v in dict(basis=basis, n_sites=inst["n_sites"], chi=chi, sweeps=sweeps, out=str(out),
                         instance_id=inst["instance_id"]).items():
            f.attrs[k] = v
        f["order"] = np.asarray(order, dtype=np.int64)
        f["init"] = np.asarray(init, dtype=np.int64)
        f["on_k"] = np.asarray([k for k, *_ in onsite], dtype=np.int64)
        f["on_c"] = np.asarray([c for *_, c in onsite], dtype=np.float64)
        f["p"] = np.asarray([p for p, *_ in pairs], dtype=np.int64)
        f["q"] = np.asarray([q for _, q, *_ in pairs], dtype=np.int64)
        f["kind"] = np.asarray([kinds[k] for _, _, k, _ in pairs], dtype=np.int64)
        f["c"] = np.asarray([c for *_, c in pairs], dtype=np.float64)
    t0 = time.perf_counter()
    r = subprocess.run([julia, str(ROOT / "tests" / "itensor_dmrg.jl"), str(job)], capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("ITENSOR_ENERGY")]
    if r.returncode != 0 or not line:
        print("  julia stdout:\n" + r.stdout[-3000:] + "\n  julia stderr:\n" + r.stderr[-5000:])
    assert r.returncode == 0 and line, (r.stdout[-3000:], r.stderr[-3000:])
    print(f"  {line[0]} (julia wall incl. compile {time.perf_counter() - t0:.0f} s)")
    card, row = check_state(f"itensor-{label}", inst, out, order, exact=exact, dmrg_tol=dmrg_tol)
    assert card["status"] == "pass", card["gates"]
    return out


def it_j1j2():
    inst = I.load("J1J2/square_16_P_0.5")
    order = snake_order(4, 4)
    path = itensor_case("J1J2-4x4-snake", inst, order, [1, 2] * 8, 256, I.exact_record(inst)["energy"], 1e-6)
    wrong_conventions("itensor-J1J2", inst, path, column_order(4, 4))


def it_hubbard():
    inst = I.load("Hubbard/square_16_P_4_8")
    order = column_order(4, 4)
    chi = int(os.environ.get("MPS_TEST_CHI", "800"))
    path = itensor_case("Hubbard-4x4", inst, order, [2, 3, 1, 1] * 4, chi, I.exact_record(inst)["energy"], 5e-3)
    wrong_conventions("itensor-Hubbard", inst, path, snake_order(4, 4))


def it_hubbard_antiperiodic():
    inst = fake_inst("Hubbard", 4, 4, "PA", Nf=3, U=4, t2=-0.25, V1=0.5)
    e0 = SUMMARY.get("ed-Hubbard-4x4-PA") or ground_state_energy(netket_system(inst, plain=False))
    init = [2, 1, 3, 1, 1, 2, 1, 3] + [1] * 4 + [2, 3, 1, 1]
    itensor_case("Hubbard-4x4-PA", inst, row_order(4, 4), init, 400, e0, 5e-3, sweeps=30)


# ------------------------------------------------------------------ part E (timing)

def bench():
    chis = [int(c) for c in os.environ.get("MPS_BENCH_CHI", "256,512,1024").split(",")]
    rows = []
    for inst in (fake_inst("J1J2", 8, 8, "OP", J2=0.5), fake_inst("Hubbard", 8, 8, "OP", Nf=28, U=8, t2=-0.25)):
        basis = MPS.basis_for(inst)
        d = MPS.BASES[basis]
        order = column_order(8, 8)          # y fastest: the chain winds around the cylinder
        mpo = MPS.build_mpo(inst, order)
        for chi in chis:
            ts = random_mps(64, d, chi, np.random.default_rng(chi))
            c = MPS.cost_estimate([t.shape[0] for t in ts] + [1], mpo.bond_dims, d)
            r = MPS.contract(ts, mpo)
            row = {"instance": inst["instance_id"], "chi": chi, "D": c["mpo_D_max"], "energy_s": r["wall_s"],
                   "tflop": c["tflop_energy"], "gflop_per_s": c["tflop_energy"] * 1e3 / r["wall_s"],
                   "est_gb": c["gb_energy"]}
            if chi <= 256:
                t0 = time.perf_counter()
                MPS.contract(ts, mpo, layers=2)
                row["variance_s"] = time.perf_counter() - t0
                row["variance_tflop"] = c["tflop_variance"]
            print(f"  {row}", flush=True)
            rows.append(row)
        big = max(chis)
        last = [r for r in rows if r["instance"] == inst["instance_id"] and r["chi"] == big][0]
        dims = [min(10000, d ** k, d ** (64 - k)) for k in range(65)]
        c = MPS.cost_estimate(dims, mpo.bond_dims, d)
        est_s = last["energy_s"] * c["tflop_energy"] / last["tflop"]
        ext = {"instance": inst["instance_id"], "chi": 10000, "tflop": c["tflop_energy"], "gb": c["gb_energy"],
               "gb_mps": c["gb_mps"], "seconds_at_measured_rate": est_s, "hours": est_s / 3600,
               "variance_tflop": c["tflop_variance"], "variance_gb": c["gb_variance"]}
        print(f"  extrapolated: {ext}", flush=True)
        rows.append(ext)
    note("bench", rows)


NETKET = [nk_mpo_vs_netket_random, nk_sampling_check_random]
TENPY = [tp_j1j2_snake, tp_hubbard, tp_hubbard_antiperiodic]
ITENSOR = [it_j1j2, it_hubbard, it_hubbard_antiperiodic]

if __name__ == "__main__":
    failed = 0
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    if any(a in sys.argv for a in ("--netket", "--tenpy", "--itensor")):
        import jax
        jax.config.update("jax_enable_x64", True)
    if "--netket" in sys.argv:
        tests += NETKET
    if "--tenpy" in sys.argv:
        tests += TENPY
    if "--itensor" in sys.argv:
        tests += ITENSOR
    if "--bench" in sys.argv:
        tests += [bench]
    j1j2_file = None
    for fn in tests:
        t0 = time.perf_counter()
        try:
            r = fn()
            if fn is tp_j1j2_snake:
                j1j2_file = r
            print(f"PASS {fn.__name__} ({time.perf_counter() - t0:.0f} s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    if j1j2_file is not None:
        try:
            tp_cli_combine(j1j2_file)
            print("PASS tp_cli_combine", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL tp_cli_combine: {type(exc).__name__}: {exc}", flush=True)
    print(f"summary: {OUT / 'summary.json'}")
    sys.exit(1 if failed else 0)
