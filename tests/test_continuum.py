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
