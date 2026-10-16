"""Physics checks of the verifier: conventions, Hamiltonians, local energies, the measurement path.

Part A (numpy only): bond counts, Fermi-sea energies of the real-space hopping matrix against a
momentum-space sum, dof and einf against every stored row, records. Part B (NetKet): exact
diagonalisation against QMBL's exact rows, a free-fermion check with t' and an antiperiodic axis,
the trace check of einf with V terms, a zero-variance re-measurement of an exact eigenstate, and
Monte Carlo against full summation for a random spin state and a random Slater determinant.

    python tests/test_physics.py          # part A
    python tests/test_physics.py --netket # parts A and B
"""

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qrec_verify import instance as I  # noqa: E402
from qrec_verify.hamiltonians import square_bonds  # noqa: E402

ALL = sorted(str(p.relative_to(ROOT / "instances"))[:-5] for p in (ROOT / "instances").rglob("*.json"))


def single_particle_matrix(lx, ly, t1, t2, axes):
    n = lx * ly
    h = np.zeros((n, n))
    nn, nnn = square_bonds(lx, ly, axes)
    for bonds, t in ((nn, t1), (nnn, t2)):
        for i, j, s in bonds:
            h[i, j] += -t * s
            h[j, i] += -t * s
    return h


def fermi_sea_kinetic(lx, ly, n_up, n_dn, t1=1.0, t2=0.0, axes=("P", "P")):
    ev = np.linalg.eigvalsh(single_particle_matrix(lx, ly, t1, t2, axes))
    return float(ev[:n_up].sum() + ev[:n_dn].sum())


# ------------------------------------------------------------------ part A

def test_bond_counts():
    for lx, ly, axes in ((16, 16, ("P", "A")), (6, 16, ("P", "P")), (10, 10, ("P", "P"))):
        nn, nnn = square_bonds(lx, ly, axes)
        assert len(nn) == 2 * lx * ly and len(nnn) == 2 * lx * ly, (lx, ly, len(nn), len(nnn))
    nn, nnn = square_bonds(16, 16, ("P", "A"))
    # antiperiodic along y: the 16 wrapping NN bonds and the 32 wrapping NNN bonds carry the sign
    assert sum(s < 0 for *_, s in nn) == 16 and sum(s < 0 for *_, s in nnn) == 32


def fermi_sea_kspace(lx, ly, n_up, n_dn, t1=1.0, t2=0.0, axes=("P", "P")):
    """The same Fermi sea from plane waves: eps(k) = -2 t1 (cos kx + cos ky) - 4 t2 cos kx cos ky,
    k = 2 pi (m + 1/2) / L along an antiperiodic axis. Independent of square_bonds (L >= 3)."""
    kx = 2 * np.pi * (np.arange(lx) + (0.5 if axes[0] == "A" else 0.0)) / lx
    ky = 2 * np.pi * (np.arange(ly) + (0.5 if axes[1] == "A" else 0.0)) / ly
    cx, cy = np.cos(kx)[:, None], np.cos(ky)[None, :]
    ev = np.sort((-2 * t1 * (cx + cy) - 4 * t2 * cx * cy).ravel())
    return float(ev[:n_up].sum() + ev[:n_dn].sum())


def test_fermi_sea_real_vs_momentum_space():
    for lx, ly, nu, nd, t2, axes in ((16, 16, 112, 112, -0.25, ("P", "A")), (8, 8, 25, 25, -0.25, ("A", "P")),
                                     (4, 16, 28, 28, 0.0, ("P", "P")), (8, 16, 56, 56, 0.0, ("P", "P")),
                                     (6, 16, 42, 40, -0.25, ("P", "A"))):
        ek, ref = fermi_sea_kinetic(lx, ly, nu, nd, 1.0, t2, axes), fermi_sea_kspace(lx, ly, nu, nd, 1.0, t2, axes)
        assert abs(ek - ref) < 1e-9, (lx, ly, nu, nd, t2, axes, ek, ref)


def test_dof_einf_against_rows():
    for iid in ALL:
        inst = I.load(iid)
        dof, einf = I.expected_dof(inst), I.expected_einf(inst)
        for r in inst["rows"]:
            if r.get("dof") is not None:
                assert r["dof"] == dof, (iid, r["dof"], dof)
            if r.get("einf") is not None and einf is not None:
                assert abs(r["einf"] - einf) <= 1e-6 * max(1.0, abs(einf)), (iid, r["einf"], einf)


def test_records():
    # QMBL v0.4.0 (2026-10-08): any unflagged row can hold a record, an extrapolation included
    expect = {"Hubbard/rectangular-4x16_64_P_28_8": -49.11872,          # NNBF, extrapolated
              "J1J2/square_100_P_0.5": -199.52,                          # DMRG, extrapolated
              "J1J2/square_16_P_0.5": -33.83169340557936}                # exact
    for iid, e in expect.items():
        rec = I.record(I.load(iid))
        assert rec is not None and rec["energy"] == e, (iid, rec and rec["energy"])
    # v1.6 (2026-10-08): a variational method chases the state-of-the-art variational
    # energy, with or without sigma, never an extrapolation or projection; exact where solved.
    targets = {"Hubbard/rectangular-4x16_64_P_28_8": -49.03872,         # not fixed-node GFMC's -49.05408
               "Hubbard/rectangular-8x16_128_P_56_8": -96.8704,         # tensor backflow, no sigma
               "J1J2/square_100_P_0.5": -199.07756,                     # not the three extrapolations
               "J1J2/square_16_P_0.5": -33.83169340557936}              # exact
    for iid, e in targets.items():
        t = I.target(I.load(iid))
        assert t is not None and t["energy"] == e, (iid, t and t["energy"])


# ------------------------------------------------------------------ part B (NetKet)

def nk_grid_order():
    import netket as nk
    for lx, ly in ((4, 4), (3, 5), (6, 8)):
        pos = np.asarray(nk.graph.Grid(extent=[lx, ly], pbc=True).positions)
        for i, (x, y) in enumerate(pos):
            assert i == int(round(x)) * ly + int(round(y)), (lx, ly, i, x, y)


def ground_energy(H):
    import netket as nk
    return float(nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0])


def nk_j1j2_ed():
    from qrec_verify import hamiltonians as Hm
    for j2 in ("0.2", "0.5", "0.7", "1.0"):
        inst = I.load(f"J1J2/square_16_P_{j2}")
        sysm = Hm.build(inst)
        e0 = ground_energy(sysm.hamiltonian)
        ref = I.exact_record(inst)["energy"]
        assert abs(e0 - ref) < 1e-6, (j2, e0, ref)
        print(f"  J1J2 4x4 J2={j2}: ED {e0:.10f} vs QMBL {ref}")


def nk_hubbard_ed():
    from qrec_verify import hamiltonians as Hm
    inst = I.load("Hubbard/square_16_P_4_8")
    sysm = Hm.build(inst)
    e0 = ground_energy(sysm.hamiltonian)
    ref = I.exact_record(inst)["energy"]
    assert abs(e0 - ref) < 1e-8, (e0, ref)
    print(f"  Hubbard 4x4 (4,4) U=8: ED {e0:.12f} vs QMBL {ref}")


def nk_free_fermions_tprime_pa():
    from qrec_verify import hamiltonians as Hm
    for axes in (("P", "A"), ("A", "P"), ("P", "P")):
        hi, H, _ = Hm.hubbard(4, 4, 3, 3, U=0.0, t1=1.0, t2=-0.25, axes=axes)
        e0 = ground_energy(H)
        ref = fermi_sea_kinetic(4, 4, 3, 3, 1.0, -0.25, axes)
        assert abs(e0 - ref) < 1e-9, (axes, e0, ref)
        print(f"  free fermions 4x4 (3,3) t'=-0.25 axes={axes}: ED {e0:.12f} = {ref:.12f}")


def nk_einf_trace():
    import scipy.sparse as sp
    from qrec_verify import hamiltonians as Hm
    hi, H, _ = Hm.hubbard(4, 4, 3, 3, U=8.0, t1=1.0, t2=-0.25, axes=("P", "A"), V1=1.0, V2=0.5,
                          jax_operator=False)
    M = H.to_sparse()
    tr = float(np.real(M.diagonal().sum())) / M.shape[0]
    fake = {"model": "Hubbard", "lattice": "square", "n_sites": 16, "boundary": "PA",
            "params": {"Nf": 3, "U": 8, "t1": 1, "t2": -0.25, "V1": 1, "V2": 0.5}}
    ref = I.expected_einf(fake)
    assert abs(tr - ref) < 1e-9, (tr, ref)
    print(f"  einf 4x4 (3,3) U=8 V1=1 V2=0.5: trace {tr:.12f} = formula {ref:.12f}")


def nk_zero_variance():
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import netket as nk
    from qrec_verify import hamiltonians as Hm, measure as Me, stats as S
    inst = I.load("J1J2/square_16_P_0.5")
    sysm = Hm.build(inst)
    w, v = nk.exact.lanczos_ed(sysm.hamiltonian, k=1, compute_eigenvectors=True)
    psi = np.asarray(v[:, 0])
    psi = psi / psi[np.argmax(np.abs(psi))]
    logpsi = jnp.asarray(np.log(np.maximum(np.abs(psi), 1e-300)) + 1j * np.pi * (psi.real < 0))
    hi = sysm.hilbert

    class Table(nn.Module):
        @nn.compact
        def __call__(self, x):
            c = self.param("c", nn.initializers.zeros, (), jnp.float64)
            return logpsi[hi.states_to_numbers(x)] + c

    model = Table()
    variables = model.init(jax.random.key(0), jnp.asarray(hi.all_states()[:2]))
    sampler = Me.default_sampler(sysm, n_chains=64)
    ser = Me.collect(sysm, model, variables, sampler, n_steps=128, n_burn=64, seed=7)
    card = Me.scorecard(inst, ser, state_id="ed-ground-state")
    assert abs(card["energy_total"] - float(w[0])) < 1e-8, (card["energy_total"], w[0])
    assert card["sigma_total"] == 0.0 and card["status"] == "pass", card["gates"]
    print(f"  zero-variance: E {card['energy_total']:.10f}, ED {float(w[0]):.10f}, gates {card['status']}")


def nk_mc_vs_fullsum_spin():
    import jax
    import netket as nk
    from qrec_verify import hamiltonians as Hm, measure as Me
    inst = I.load("J1J2/square_16_P_0.5")
    sysm = Hm.build(inst)
    model = nk.models.RBM(alpha=1, param_dtype=complex)
    variables = model.init(jax.random.key(3), np.ones((1, 16)))
    variables = jax.tree_util.tree_map(lambda a: a * 3.0, variables)   # a non-trivial state
    fs = nk.vqs.FullSumState(sysm.hilbert, model, variables=variables)
    exact = float(np.real(fs.expect(sysm.hamiltonian).mean))
    sampler = Me.default_sampler(sysm, n_chains=64)
    ser = Me.collect(sysm, model, variables, sampler, n_steps=1024, n_burn=256, seed=11)
    card = Me.scorecard(inst, ser, state_id="random-rbm")
    z = (card["energy_total"] - exact) / card["sigma_total"]
    print(f"  spin MC {card['energy_total']:.6f} +- {card['sigma_total']:.6f} vs full sum {exact:.6f} "
          f"(z {z:+.2f}); gates {[(g['name'], g['status']) for g in card['gates']]}")
    assert abs(z) < 4, z
    assert card["status"] in ("pass", "warn"), card["gates"]


def nk_mc_vs_fullsum_fermion():
    import jax
    import netket as nk
    from qrec_verify import hamiltonians as Hm, measure as Me, stats as S
    kw = dict(U=4.0, t1=1.0, t2=-0.25, axes=("P", "A"))
    hi, H, g = Hm.hubbard(3, 4, 3, 3, **kw)
    _, Hplain, _ = Hm.hubbard(3, 4, 3, 3, jax_operator=False, **kw)   # FullSumState cannot hash H
    sysm = Hm.System("test/3x4", hi, H, g, 12, 3, 4, {})
    model = nk.models.Slater2nd(hi, param_dtype=float)
    variables = model.init(jax.random.key(5), hi.all_states()[:1])
    fs = nk.vqs.FullSumState(hi, model, variables=variables)
    exact = float(np.real(fs.expect(Hplain).mean))
    sampler = Me.default_sampler(sysm, n_chains=64)
    ser = Me.collect(sysm, model, variables, sampler, n_steps=1024, n_burn=256, seed=13)
    d = S.diagnose(ser["e_loc"], ser["logabs"], ser["acc"])
    z = (d.mean - exact) / d.sigma
    print(f"  fermion MC {d.mean:.6f} +- {d.sigma:.6f} vs full sum {exact:.6f} (z {z:+.2f}); "
          f"gates {[(x.name, x.status) for x in d.gates]}")
    assert abs(z) < 4, z


def nk_conserving_operator_matches():
    """The conserving operator used for measurements gives the same local energies as the plain
    FermionOperator2nd on random configurations of a 4x4 PA cluster with t' and V terms."""
    import jax
    import netket as nk
    from qrec_verify import hamiltonians as Hm
    kw = dict(U=8.0, t1=1.0, t2=-0.25, axes=("P", "A"), V1=1.0, V2=0.5)
    hi, Hc, _ = Hm.hubbard(4, 4, 5, 5, **kw)
    _, Hp, _ = Hm.hubbard(4, 4, 5, 5, jax_operator=False, **kw)
    model = nk.models.Slater2nd(hi, param_dtype=float)
    variables = model.init(jax.random.key(1), hi.all_states()[:1]) if hi.n_states < 2e5 else \
        model.init(jax.random.key(1), hi.random_state(jax.random.key(0), 1))
    x = hi.random_state(jax.random.key(2), 64)
    def eloc(H):
        xp, mels = H.get_conn_padded(x)
        lp = model.apply(variables, xp.reshape(-1, xp.shape[-1])).reshape(xp.shape[:-1])
        l0 = model.apply(variables, x)
        return np.asarray((mels * np.exp(lp - l0[:, None])).sum(axis=-1))
    d = np.max(np.abs(eloc(Hc) - eloc(Hp)))
    print(f"  conserving vs plain operator, 64 random configs: max |dE_loc| {d:.2e} ({type(Hc).__name__})")
    assert d < 1e-10, d


NETKET_TESTS = [nk_grid_order, nk_j1j2_ed, nk_free_fermions_tprime_pa, nk_einf_trace,
                nk_zero_variance, nk_mc_vs_fullsum_spin, nk_mc_vs_fullsum_fermion, nk_conserving_operator_matches, nk_hubbard_ed]


if __name__ == "__main__":
    failed = 0
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    if "--netket" in sys.argv:
        import jax
        jax.config.update("jax_enable_x64", True)
        tests += NETKET_TESTS
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1 if failed else 0)
