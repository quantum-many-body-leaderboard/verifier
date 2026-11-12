"""The electron gas (qrec_verify/continuum.py): Ewald sum, kinetic energy, gates, known state.

References: QMBL's checks/heg-hf/hf.mjs (qmbl 74827097), run on 2026-11-10 for the cells below.
"""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from qrec_verify import continuum as CT  # noqa: E402

# (dim, lattice, N_up, N_dn, rs, twist) -> hf.mjs output (E_HF/N, kinetic/N, exchange/N, xi)
HFMJS = [
    ((3, "sc", 7, 7, 1, None), (0.6065343288241165, 1.1209128677543463, -0.14923020092833214, -0.7302966760037951)),
    ((3, "bcc", 27, 0, 5, None), (-0.04659037571243336, 0.07562097778273431, -0.062482736449716546, -0.11945723409090227)),
    ((3, "fcc", 65, 65, 30, None), (-0.01474758424137146, 0.0012048430867346418, -0.010057510155696942, -0.011789834344818324)),
    ((3, "rect-3x3x2", 9, 9, 2, None), (0.016905630602720337, 0.28134567113244435, -0.10613673560683244, -0.31660660984578315)),
    ((2, "square", 45, 45, 30, None), (-0.019651777611229437, 0.000551610232729072, -0.016337523008869537, -0.007731729670177944)),
    ((2, "triangular", 13, 13, 10, None), (-0.055584770720247444, 0.00515161949604985, -0.03904393377900418, -0.04338491287458624)),
    ((3, "sc", 7, 7, 2, (0.25, 0.25, 0.25)), (0.004040527395137637, 0.2714710851592557, -0.08485638876316927, -0.3651483380018976)),
    ((2, "square", 13, 13, 5, (0.25, 0.25)), (-0.10372040354057864, 0.019797610509308467, -0.08036287735986578, -0.08631027338004268)),
]


def cell_of(spec):
    dim, lat, nu, nd, rs, tw = spec
    return CT.make_cell(dim, lat, nu, nd, rs, tw)


@pytest.mark.parametrize("spec,ref", HFMJS)
def test_hf_energy_matches_hfmjs(spec, ref):
    r = CT.hf_energy(cell_of(spec))
    got = (r["E_HF_per_electron"], r["kinetic_per_electron"], r["exchange_per_electron"], r["xi"])
    if r["open_shell"]:
        # a partly filled shell: which degenerate k are occupied depends on how a sort breaks ties
        # at rounding level, so the exchange energy is not unique (hf.mjs says the same)
        got, ref = got[1::2], ref[1::2]
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("spec,ref", HFMJS)
def test_ewald_xi_independent_of_kappa(spec, ref):
    cell = cell_of(spec)
    ew = CT.Ewald.build(cell.A)
    assert ew.kappa != pytest.approx(2.8 / cell.volume ** (1 / cell.dim))
    assert abs(ew.xi - ref[3]) <= 1e-10 * max(1.0, abs(ref[3]))
    assert CT.hamiltonian_gate(cell, ew)["pass"]


@pytest.mark.parametrize("spec", [s for s, _ in HFMJS[:6]])
def test_one_electron_is_half_xi(spec):
    cell = cell_of(spec)
    ew = CT.Ewald.build(cell.A)
    e = float(ew.energy_fn(1)(jnp.asarray(0.3 * cell.A[0])))
    assert abs(e - ew.xi / 2) < 1e-12


@pytest.mark.parametrize("spec", [s for s, _ in HFMJS[:6]])
def test_pair_energy_independent_of_splitting(spec):
    cell = cell_of(spec)
    a = CT.Ewald.build(cell.A)
    b = CT.Ewald.build(cell.A, n_img=3, tail=7.0)
    assert abs(a.kappa - b.kappa) > 1e-3
    x = np.random.default_rng(1).uniform(size=(5, 4, cell.dim)) @ cell.A
    fa, fb = jax.jit(jax.vmap(a.energy_fn(4))), jax.jit(jax.vmap(b.energy_fn(4)))
    np.testing.assert_allclose(np.asarray(fa(x.reshape(5, -1))), np.asarray(fb(x.reshape(5, -1))),
                               rtol=1e-11, atol=1e-12)


def test_energy_invariant_under_translation_and_wrapping():
    cell = cell_of(HFMJS[2][0])
    ew = CT.Ewald.build(cell.A)
    f = ew.energy_fn(6)
    x = np.random.default_rng(2).uniform(size=(6, 3)) @ cell.A
    e0 = float(f(jnp.asarray(x.ravel())))
    shifted = x + 0.37 * cell.A[1] - 0.11 * cell.A[2]
    wrapped = x.copy()
    wrapped[3] += cell.A[0] - 2 * cell.A[2]
    assert abs(float(f(jnp.asarray(shifted.ravel()))) - e0) < 1e-11
    assert abs(float(f(jnp.asarray(wrapped.ravel()))) - e0) < 1e-11


@pytest.mark.parametrize("spec,ref", [HFMJS[0], HFMJS[4], HFMJS[6], HFMJS[7]])
def test_slater_kinetic_is_constant(spec, ref):
    cell = cell_of(spec)
    model = CT.slater_model(cell)
    v = model.init(jax.random.PRNGKey(0), np.zeros((1, cell.n * cell.dim)))
    lp = CT.logpsi_fn(model, v)
    x = (np.random.default_rng(3).uniform(size=(4, cell.n, cell.dim)) @ cell.A).reshape(4, -1)
    want = ref[1] * cell.n
    for method in ("fwdlap", "hessian"):
        ke = jax.jit(jax.vmap(CT.kinetic_fn(lambda y: lp(y[None])[0], cell.n * cell.dim, method)))
        k = np.asarray(ke(x))
        np.testing.assert_allclose(np.real(k), want, rtol=1e-9)
        assert np.max(np.abs(np.imag(k))) < 1e-9 * max(1.0, abs(want))


def _jastrow_slater(cell):
    """Slater determinant times a periodic two-body Jastrow: a state with a non-constant kinetic energy."""
    import flax.linen as nn
    B = jnp.asarray(cell.B)
    n, dim = cell.n, cell.dim

    class JS(nn.Module):
        @nn.compact
        def __call__(self, x):
            r = x.reshape(x.shape[0], n, dim)
            d = r[:, :, None, :] - r[:, None, :, :]
            u = jnp.sum(jnp.cos(jnp.einsum("bijd,kd->bijk", d, B)), -1)
            return CT.slater_model(cell)(x) + 0.05 * jnp.sum(u, (1, 2))   # bound inside compact
    return JS()


def test_kinetic_fwdlap_matches_hessian():
    cell = cell_of(HFMJS[0][0])
    model = _jastrow_slater(cell)
    v = model.init(jax.random.PRNGKey(0), np.zeros((1, cell.n * cell.dim)))
    lp = CT.logpsi_fn(model, v)
    x = (np.random.default_rng(4).uniform(size=(3, cell.n, cell.dim)) @ cell.A).reshape(3, -1)
    one = lambda y: lp(y[None])[0]  # noqa: E731
    a = np.asarray(jax.vmap(CT.kinetic_fn(one, cell.n * cell.dim, "fwdlap"))(x))
    b = np.asarray(jax.vmap(CT.kinetic_fn(one, cell.n * cell.dim, "hessian"))(x))
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)


def test_gates_pass_slater_and_catch_broken_states():
    import flax.linen as nn
    for spec, _ in (HFMJS[0], HFMJS[6], HFMJS[7]):
        cell = cell_of(spec)
        model = CT.slater_model(cell)
        v = model.init(jax.random.PRNGKey(0), np.zeros((1, cell.n * cell.dim)))
        x = (np.random.default_rng(5).uniform(size=(4, cell.n, cell.dim)) @ cell.A).reshape(4, -1)
        gates = CT.state_gates(jax.jit(CT.logpsi_fn(model, v)), cell, x)
        assert all(g["pass"] for g in gates), gates
    cell = cell_of(HFMJS[0][0])
    n, dim = cell.n, cell.dim

    class Bosonic(nn.Module):      # symmetric: fails antisymmetry, passes the twist check at Gamma
        @nn.compact
        def __call__(self, x):
            r = x.reshape(x.shape[0], n, dim)
            return jnp.sum(jnp.cos(r @ jnp.asarray(cell.B).T), (1, 2)).astype(jnp.complex128)

    class Gaussian(nn.Module):     # not periodic: fails the twist check
        @nn.compact
        def __call__(self, x):
            return (-0.1 * jnp.sum(x ** 2, -1)).astype(jnp.complex128)

    x = (np.random.default_rng(6).uniform(size=(4, n, dim)) @ cell.A).reshape(4, -1)
    for mod, failing in ((Bosonic(), "antisymmetry"), (Gaussian(), "twist")):
        v = mod.init(jax.random.PRNGKey(0), x[:1])
        g = {r["name"]: r["pass"] for r in CT.state_gates(jax.jit(CT.logpsi_fn(mod, v)), cell, x)}
        assert g[failing] is False


def test_sign_logabs_output():
    raw = (jnp.array([1.0, -1.0]), jnp.array([0.5, 0.25]))
    lp = np.asarray(CT.as_logpsi(raw))
    np.testing.assert_allclose(np.real(lp), [0.5, 0.25])
    np.testing.assert_allclose(np.abs(np.angle(np.exp(1j * np.imag(lp)))), [0.0, math.pi])


@pytest.mark.slow
def test_hf_state_scores_hf_energy():
    """The verifier's VMC energy of the plane-wave determinant equals the HF energy (4 sigma)."""
    import op_verify
    inst = {"instance_id": "HEG3D/sc_14_P_7_7_1", "model": "HEG3D", "lattice": "sc", "n_sites": 14,
            "boundary": "P", "params": {"N_up": 7, "N_dn": 7, "rs": 1, "twist": "Gamma"}, "rows": []}
    r = op_verify.hf_check(inst, seed=3)
    assert r["pass"], r
    assert abs(r["hf_per_electron"] - HFMJS[0][1][0]) < 1e-12


def test_jax_adapter_sign_logabs_state(tmp_path):
    """A JAX function returning (sign, log|psi|), as fvmc's wave functions do, through --state jax:."""
    from qrec_verify import external as EX
    inst = {"instance_id": "HEG3D/sc_14_P_7_7_1", "model": "HEG3D", "lattice": "sc", "n_sites": 14,
            "boundary": "P", "params": {"N_up": 7, "N_dn": 7, "rs": 1, "twist": "Gamma"}, "rows": []}
    cell = CT.cell_from_instance(inst)
    ku, kd, _ = CT.occupied_k(cell)
    np.savez(tmp_path / "p.npz", ku=ku, kd=kd)
    (tmp_path / "state.py").write_text(
        "import jax.numpy as jnp\n"
        "def build(inst):\n"
        "    def apply_fn(p, x):\n"
        "        r = x.reshape(x.shape[0], 14, 3)\n"
        "        def sd(rs, k):\n"
        "            return jnp.linalg.slogdet(jnp.exp(1j * jnp.einsum('bnd,kd->bnk', rs, k)))\n"
        "        s1, l1 = sd(r[:, :7], p['ku'])\n"
        "        s2, l2 = sd(r[:, 7:], p['kd'])\n"
        "        return s1 * s2, l1 + l2\n"
        "    return apply_fn\n")
    st = EX.load_state(f"jax:{tmp_path / 'state.py'}:build", str(tmp_path / "p.npz"), inst)
    ref = CT.slater_model(cell)
    vref = ref.init(jax.random.PRNGKey(0), np.zeros((1, 42)))
    x = (np.random.default_rng(8).uniform(size=(3, 14, 3)) @ cell.A).reshape(3, -1)
    a = np.asarray(CT.logpsi_fn(st.model, st.variables)(x))
    b = np.asarray(CT.logpsi_fn(ref, vref)(x))
    np.testing.assert_allclose(np.real(a), np.real(b), atol=1e-10)
    np.testing.assert_allclose(np.angle(np.exp(1j * (np.imag(a) - np.imag(b)))), 0, atol=1e-10)
    with pytest.raises(ValueError, match="flax module or a JAX function"):
        EX.load_state(f"python:{tmp_path / 'state.py'}:build", None, inst)


# ------------------------------------------------------------------ hydrogen in continuous space (v1.10)
# qrec_verify/molecule.py: QMBL's Hchain / Hsheet instances with basis none, the electron gas's state interface.

from qrec_verify import molecule as MOL  # noqa: E402


def _quiet(*a, **k):
    pass


def _pyscf():
    return pytest.importorskip("pyscf")


def _load(iid):
    from qrec_verify import instance as I
    return I.load(iid)


def test_molecule_geometry_matches_chem_and_data_md():
    """The protons of the continuous-space instances are DATA.md's (chain along z, sheet atom I = i L + j), the same
    as the Gaussian-basis path (chem.atoms); N_up, N_dn from params."""
    from qrec_verify import chem
    for iid in ("Hchain/chain_10_O_R1.8_none", "Hsheet/square_4_O_R2.4_none"):
        inst = _load(iid)
        mol = MOL.molecule_of(inst)
        np.testing.assert_array_equal(mol.atoms, chem.atoms(inst))
        assert (mol.n_up, mol.n_dn) == (inst["params"]["N_up"], inst["params"]["N_dn"])
    sheet = MOL.molecule_of(_load("Hsheet/square_4_O_R2.4_none"))
    np.testing.assert_allclose(sheet.atoms, [[0, 0, 0], [0, 2.4, 0], [2.4, 0, 0], [2.4, 2.4, 0]])


def test_molecule_refusals():
    """Gaussian-basis instances are not continuous-space states; periodic cells are refused with the reason."""
    with pytest.raises(ValueError, match="basis none"):
        MOL.molecule_of(_load("Hchain/chain_10_O_R1.8_sto6g"))
    inst = dict(_load("Hchain/chain_10_O_R1.8_none"), boundary="P")
    with pytest.raises(ValueError, match="periodic"):
        MOL.molecule_of(inst)


def test_molecule_coulomb_against_pyscf():
    """The verifier's potential (JAX pair sums) against pyscf's point-charge Coulomb sum and Mole.energy_nuc
    (independent code), on random configurations of H10 and the H4 sheet, and the numpy term-by-term sum."""
    _pyscf()
    rng = np.random.default_rng(3)
    for iid in ("Hchain/chain_10_O_R1.8_none", "Hsheet/square_4_O_R2.4_none"):
        mol = MOL.molecule_of(_load(iid))
        x = MOL.start_positions(rng, mol, 12, 0.8)
        g = MOL.coulomb_gate(mol, x)
        assert g["pass"] and "pyscf" in g["detail"], g
        v = np.asarray(jax.vmap(MOL.potential_fn(mol))(jnp.asarray(x)))
        ref = [MOL.potential_np(xx.reshape(mol.n, 3), mol) for xx in x]
        np.testing.assert_allclose(v, ref, rtol=1e-12)
    # H2 at 1.4 bohr: the nuclear repulsion is 1 / 1.4
    assert abs(MOL.nuclear_repulsion(MOL.molecule_of(MOL.small_instance("h2:1.4"))) - 1 / 1.4) < 1e-15


def test_molecule_hydrogen_atom_exact():
    """The hydrogen atom's 1s state exp(-r): E_loc = -1/2 at every sample, sigma 0, every gate passes."""
    model, var = MOL.hydrogen_1s_model()
    card = MOL.evaluate(MOL.small_instance("h1"), model, var, seed=1, min_samples=512 * 32, log=_quiet)
    assert abs(card["energy_total"] - MOL.H_ATOM_EXACT) < 1e-12 and card["sigma_total"] < 1e-12, card
    assert all(g["pass"] for g in card["physics_gates"]), card["physics_gates"]


def test_molecule_kinetic_paths_on_rhf_h4():
    """Forward Laplacian = jax.hessian, and the Laplacian estimator = the gradient estimator per sample in the mean
    (checked in evaluate), for the RHF determinant of H4 (cc-pVDZ, s and p orbitals)."""
    _pyscf()
    model, var, _, _ = MOL.rhf_model(MOL.small_instance("h4:1.8"))
    mol = MOL.molecule_of(MOL.small_instance("h4:1.8"))
    x = MOL.start_positions(np.random.default_rng(5), mol, 8, 0.8)
    g = MOL.kinetic_paths_gate(model, var, mol, x)
    assert g["pass"] and g["value"] < 1e-10, g


def test_molecule_state_gates_catch_broken_states():
    """A symmetric (bosonic) product of 1s orbitals fails the antisymmetry gate; a state that grows far out fails
    the decay gate; the RHF determinant passes both."""
    import flax.linen as nn
    _pyscf()
    inst = MOL.small_instance("h4:1.8")
    mol = MOL.molecule_of(inst)
    atoms = jnp.asarray(mol.atoms)

    class Bosonic(nn.Module):
        @nn.compact
        def __call__(self, x):
            r = x.reshape(x.shape[0], -1, 3)
            return (-jnp.sum(jnp.linalg.norm(r - atoms[None], axis=-1), axis=-1)).astype(jnp.complex128)

    class Growing(nn.Module):
        @nn.compact
        def __call__(self, x):
            r = x.reshape(x.shape[0], -1, 3)
            m, v, _, _ = MOL.rhf_model(inst)
            return m.apply({}, x) + 0.1 * jnp.sum(r ** 2, axis=(1, 2))

    x = MOL.start_positions(np.random.default_rng(2), mol, 8, 0.8)
    lp = lambda m: CT.logpsi_fn(m, {})                     # noqa: E731
    gb = {g["name"]: g for g in MOL.state_gates(lp(Bosonic()), mol, x)}
    assert not gb["antisymmetry"]["pass"] and gb["decay"]["pass"], gb
    gg = {g["name"]: g for g in MOL.state_gates(lp(Growing()), mol, x)}
    assert gg["antisymmetry"]["pass"] and not gg["decay"]["pass"], gg
    m, v, _, _ = MOL.rhf_model(inst)
    gr = {g["name"]: g for g in MOL.state_gates(lp(m), mol, x)}
    assert gr["antisymmetry"]["pass"] and gr["decay"]["pass"], gr


def test_molecule_h2_fci_scores_pyscf_fci():
    """The two-electron FCI wave function of H2 at R = 1.4 bohr in cc-pVDZ (pyscf), evaluated by the verifier's own
    Gaussian-orbital code, scores pyscf's FCI energy within 4 sigma; both lie above the exact H2 energy
    (Pachucki 2010: -1.1744757142204434), and the score is not flagged below it."""
    _pyscf()
    model, var, e_fci = MOL.h2_fci_model(1.4)
    assert e_fci > MOL.H2_EXACT[1.4]
    card = MOL.evaluate(MOL.small_instance("h2:1.4"), model, var, seed=2, min_samples=512 * 1024, log=_quiet)
    z = (card["energy_total"] - e_fci) / card["sigma_total"]
    assert abs(z) < 4 and card["status"] != "fail", (card["energy_total"], card["sigma_total"], e_fci)
    assert card["energy_total"] > MOL.H2_EXACT[1.4] and "plausibility" not in card
    assert all(g["pass"] for g in card["physics_gates"]), card["physics_gates"]


def test_molecule_rhf_known_state_h10():
    """The known state on QMBL's H10 (R = 1.8 bohr, basis none): the RHF determinant (cc-pVDZ) scores pyscf's RHF
    energy within 3 sigma, and lies above the sources' near-exact rows of that instance (MRCI+Q, LR-DMC)."""
    _pyscf()
    inst = _load("Hchain/chain_10_O_R1.8_none")
    r = MOL.rhf_check(inst, min_samples=512 * 128, n_chains=512, log=_quiet)   # 128 samples per chain on any device
    assert r["pass"], r
    lowest = min(row["energy"] for row in inst["rows"])
    assert r["energy_total"] > lowest + 0.1, (r["energy_total"], lowest)


def test_molecule_jax_adapter_and_cli(tmp_path):
    """A JAX function of the electron positions through --state jax: (the electron gas's interface): the hydrogen
    atom's exact state as a (sign, log|psi|) pair; host states are refused for basis none."""
    from qrec_verify import external as EX
    inst = _load("Hchain/chain_10_O_R1.8_none")
    assert EX.n_dof(inst) == 30
    np.savez(tmp_path / "p.npz", a=np.array(1.0))
    (tmp_path / "state.py").write_text(
        "import jax.numpy as jnp\n"
        "def build():\n"
        "    def apply_fn(p, x):\n"
        "        r = x.reshape(x.shape[0], -1, 3)\n"
        "        return jnp.ones(x.shape[0]), -p['a'] * jnp.sum(jnp.linalg.norm(r, axis=-1), axis=-1)\n"
        "    return apply_fn\n")
    h1 = MOL.small_instance("h1")
    st = EX.load_state(f"jax:{tmp_path / 'state.py'}:build", str(tmp_path / "p.npz"), h1)
    card = MOL.evaluate(h1, st.model, st.variables, seed=4, min_samples=512 * 16, log=_quiet)
    assert abs(card["energy_total"] + 0.5) < 1e-12, card["energy_total"]
    with pytest.raises(ValueError, match="flax module or a JAX function"):
        EX.load_state(f"python:{tmp_path / 'state.py'}:build", None, inst)
