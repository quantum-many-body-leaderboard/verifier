"""Hydrogen chains and sheets in large Gaussian bases (v1.10): more than chem.MAX_ORBITALS (128) orbitals.

The Hamiltonian is chem.py's (QMBL RULES.md 13.1-13.2 projected onto the basis, Loewdin orbitals of pyscf's atomic
orbitals, NetKet's configuration layout: the n down orbitals first, then the n up orbitals). Above 128 orbitals the
verifier does not store the n^4 two-electron integrals; it stores them factorised,

    (pq|rs) = sum_g L^g_pq L^g_rs          (pivoted Cholesky decomposition of the AO integrals, `cholesky_ao`,
                                             transformed to the Loewdin orbitals; error below CD_TOL per integral)

with n_chol about 5 to 10 n vectors, and computes local energies with them:

- `det_local_energy`: a JAX kernel for Slater-determinant states (one determinant per spin, orbital matrices
  Phi_up (n, N_up) and Phi_dn (n, N_dn) in the Loewdin orbitals). With Theta = Phi (Phi[occ])^-1 (the mixed Green's
  function G_(o_k, p) = Theta_pk = <x| c+_o_k c_p |psi> / <x|psi>) and Wick's theorem for the mixed estimator,

      E_loc = E_nuc + sum_s sum_k (h Theta_s)_(o_k, k)
              + 1/2 sum_g [ (sum_s tr Y^g_s)^2 - sum_s tr(Y^g_s Y^g_s) ],      Y^g_s = L^g[occ_s, :] Theta_s,

  exact for a determinant; O(n_chol N^2 n) per sample.
- `det_energy`: the variational energy of a determinant from its density matrices (no sampling, sigma 0).
- `connected`: every configuration connected to x with its matrix element, from h and the Cholesky vectors (the
  generic estimator for any state, used by the gates; O(N^2 n^2) configurations per sample).

States: above 128 orbitals the verifier scores determinant states (`--state det:FILE.npz`, arrays phi_up and phi_dn
in the Loewdin orbitals); a general state would need its amplitude on every connected configuration (about 10^6 per
sample for H10 in cc-pVTZ), which the verifier does not attempt.

Gates (tests/test_physics.py nk_chem_large): the Cholesky integrals reproduce pyscf's dense integrals up to 128
orbitals; the determinant kernel equals NetKet's generic estimator with the dense operator (chem.py) on the
verifier's own samples up to 128 orbitals, and `connected` equals NetKet's get_conn; above 128, pyscf's RHF energy is
reproduced by `det_energy` (sigma 0) and by Monte Carlo of the kernel, and <H> of a random CI vector in an active
space equals pyscf's `fci.direct_spin1` contraction with pyscf's own integrals.

Size: the Cholesky vectors take 8 n^2 n_chol bytes (MAX_GB bounds it); bases whose overlap matrix is nearly
singular (smallest eigenvalue below 1e-10, as chem.loewdin) have no well-defined Loewdin orbitals and are refused.
"""

from __future__ import annotations

import math

import numpy as np

CD_TOL = 1e-10               # largest remaining diagonal (mu nu|mu nu) of the residual, hartree
MAX_GB = 64.0                # peak memory of the Cholesky vectors (AO and Loewdin copies, packed)


def estimate_gb(n: int, ratio: float = 13.0) -> float:
    """Peak memory of the Cholesky vectors: about ratio x n vectors (12.4 n for H10 cc-pVTZ, 13.2 n for cc-pVQZ) of
    n (n + 1) / 2 packed pairs, held twice (AO and Loewdin, or Loewdin and the kernel's copy)."""
    return 2 * 8.0 * ratio * n * n * (n + 1) / 2 / 1e9


def _diag_ao(mol, pair, ao_loc) -> np.ndarray:
    """(mu nu|mu nu) for mu >= nu, shell pair by shell pair."""
    D = np.empty(int(pair.max()) + 1)
    for i in range(mol.nbas):
        for j in range(i + 1):
            blk = mol.intor("int2e_sph", shls_slice=(i, i + 1, j, j + 1, i, i + 1, j, j + 1))
            for a in range(blk.shape[0]):
                for b in range(blk.shape[1]):
                    mu, nu = ao_loc[i] + a, ao_loc[j] + b
                    if mu >= nu:
                        D[pair[mu, nu]] = blk[a, b, a, b]
    return D


def cholesky_ao(mol, tol: float = CD_TOL, span: float = 1e-2, max_batch: int = 256, log=None) -> np.ndarray:
    """Pivoted Cholesky decomposition of pyscf's AO two-electron integrals as the matrix V over pairs mu >= nu:
    returns L (n_chol, n_pair) with |V - L^T L| below tol on the diagonal (hence everywhere, by Cauchy-Schwarz).

    Blocked (the two-step scheme of Aquilante et al.): each round takes the candidate pivots whose residual diagonal
    exceeds span x its maximum (at most max_batch), computes their columns from pyscf's integrals (shell pair by
    shell pair, mol.intor('int2e_sph', shls_slice=...)), removes the existing vectors' part with one matrix product
    and runs the pivoted Cholesky within the candidates."""
    n = int(mol.nao)
    ao_loc = mol.ao_loc_nr()
    shell_of = np.repeat(np.arange(mol.nbas), np.diff(ao_loc))
    iu = np.tril_indices(n)
    npair = len(iu[0])
    pair = np.full((n, n), -1, dtype=np.int64)
    pair[iu] = np.arange(npair)
    pair = np.maximum(pair, pair.T)
    D = _diag_ao(mol, pair, ao_loc)
    cap = max(64, 4 * n)
    L = np.zeros((cap, npair))
    k = 0
    while True:
        dmax = float(D.max())
        if dmax < tol:
            break
        thr = max(span * dmax, tol)
        cand = np.argsort(-D)[:max_batch]
        cand = cand[D[cand] > thr]
        # columns of the candidates, by shell pair
        V = np.empty((npair, len(cand)))
        groups: dict = {}
        for c, pp in enumerate(cand):
            mu, nu = iu[0][pp], iu[1][pp]
            si, sj = shell_of[mu], shell_of[nu]
            if si < sj:
                si, sj, mu, nu = sj, si, nu, mu
            groups.setdefault((si, sj), []).append((c, mu - ao_loc[si], nu - ao_loc[sj]))
        for (si, sj), items in groups.items():
            blk = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, si, si + 1, sj, sj + 1))
            for c, a, b in items:
                V[:, c] = blk[:, :, a, b][iu]
        if k:
            V -= L[:k].T @ L[:k, cand]
        # pivoted Cholesky within the candidates
        new = []
        Dc = D[cand].copy()
        while True:
            q = int(np.argmax(Dc))
            if Dc[q] <= thr:
                break
            col = V[:, q]
            if new:
                W = np.asarray(new)
                col = col - W.T @ W[:, cand[q]]
            v = col / math.sqrt(col[cand[q]])
            new.append(v)
            D -= v * v
            D[cand[q]] = 0.0
            Dc = D[cand].copy()
            if len(new) == len(cand):
                break
        if k + len(new) > cap:
            cap = max(2 * cap, k + len(new))
            L2 = np.zeros((cap, npair))
            L2[:k] = L[:k]
            L = L2
        L[k:k + len(new)] = np.asarray(new)
        k += len(new)
        if log:
            log(f"[cholesky] {k} vectors, residual {D.max():.2e}")
    return L[:k].copy()


def to_orbitals(Lp: np.ndarray, X: np.ndarray, chunk: int = 256) -> np.ndarray:
    """AO Cholesky vectors (n_chol, n_pair over mu >= nu) -> the same in the orbitals X (columns), packed again over
    p >= q: L'_pq = sum X_mu,p L_mu,nu X_nu,q."""
    n = X.shape[0]
    m = X.shape[1]
    iu = np.tril_indices(m)
    out = np.empty((Lp.shape[0], len(iu[0])))
    for i in range(0, Lp.shape[0], chunk):
        Lf = unpack(Lp[i:i + chunk], n)
        Lo = np.einsum("mp,kmn,nq->kpq", X, Lf, X, optimize=True)
        out[i:i + chunk] = Lo[:, iu[0], iu[1]]
    return out


def unpack(Lp: np.ndarray, n: int) -> np.ndarray:
    """(n_chol, n_pair) over mu >= nu -> (n_chol, n, n) symmetric."""
    iu = np.tril_indices(n)
    L = np.zeros((Lp.shape[0], n, n))
    L[:, iu[0], iu[1]] = Lp
    L[:, iu[1], iu[0]] = Lp
    return L


# ------------------------------------------------------------------ the factorised Hamiltonian

def pair_index(n: int) -> np.ndarray:
    """(n, n) index of the packed pair (p, q) (p >= q or q >= p)."""
    iu = np.tril_indices(n)
    pair = np.full((n, n), -1, dtype=np.int64)
    pair[iu] = np.arange(len(iu[0]))
    return np.maximum(pair, pair.T)


_CACHE: dict = {}


def factorised(inst: dict, log=None, tol: float = CD_TOL) -> dict:
    """E_nuc, h (n, n) and the Cholesky vectors Lp (n_chol, n_pair) in the Loewdin orbitals of chem.py (kept for
    the last instance, so a score builds them once)."""
    from . import chem
    key = (inst.get("model"), inst.get("lattice"), int(inst["n_sites"]), str(inst["params"]["basis"]).lower(),
           float(inst["params"]["R"]), tol)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    _CACHE[key] = _factorised(inst, log, tol)
    return _CACHE[key]


def preflight(inst: dict):
    """The refusals that need no integrals: the memory bound (estimate_gb > MAX_GB) and a nearly singular overlap
    (chem.loewdin). Returns (pyscf Mole, Loewdin X)."""
    from . import chem
    mol = chem.molecule(inst)
    n = int(mol.nao)
    if estimate_gb(n) > MAX_GB:
        raise ValueError(f"{inst['instance_id']}: {n} orbitals; the Cholesky vectors would take about "
                         f"{estimate_gb(n):.0f} GB, above the verifier's bound of {MAX_GB:.0f} GB (chem_large.MAX_GB)")
    return mol, chem.loewdin(mol.intor("int1e_ovlp"))


def _factorised(inst: dict, log, tol: float) -> dict:
    mol, X = preflight(inst)
    n = int(mol.nao)
    h = X.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ X
    h = 0.5 * (h + h.T)
    Lao = cholesky_ao(mol, tol=tol, log=log)
    Lp = to_orbitals(Lao, X)
    del Lao
    return {"e_nuc": float(mol.energy_nuc()), "h": h, "Lp": Lp, "n_orb": n, "n_chol": int(Lp.shape[0]), "X": X,
            "mol": mol, "ao_labels": mol.ao_labels(), "cd_tol": tol}


def dense_eri(Lp: np.ndarray, n: int) -> np.ndarray:
    """(pq|rs) from the Cholesky vectors (gates only: n^4 numbers)."""
    L = unpack(Lp, n)
    return np.einsum("kpq,krs->pqrs", L, L)


def det_energy(ints: dict, phi_dn: np.ndarray, phi_up: np.ndarray) -> float:
    """<psi|H|psi> / <psi|psi> of the determinant with orbital matrices phi_dn (n, N_dn), phi_up (n, N_up) (real),
    from the density matrices P = Phi (Phi^T Phi)^-1 Phi^T: E_nuc + sum_s tr h P_s
    + 1/2 sum_g [(sum_s tr L^g P_s)^2 - sum_s tr L^g P_s L^g P_s]."""
    n = ints["n_orb"]
    Ps = [P @ np.linalg.solve(P.T @ P, P.T) for P in (phi_dn, phi_up) if P.shape[1]]
    e = ints["e_nuc"] + sum(float(np.sum(ints["h"] * P)) for P in Ps)
    J = 0.0
    K = 0.0
    for i in range(0, ints["Lp"].shape[0], 256):
        L = unpack(ints["Lp"][i:i + 256], n)
        tr = sum(np.einsum("kpq,pq->k", L, P) for P in Ps)
        J += float(np.sum(tr ** 2))
        for P in Ps:
            LP = np.einsum("kpq,qr->kpr", L, P)
            K += float(np.einsum("kpr,krp->", LP, LP))
    return float(e + 0.5 * (J - K))


def det_local_energy_fn(ints: dict, phi_dn, phi_up, n_dn: int, n_up: int):
    """JAX: configurations x (batch, 2n) (NetKet's layout, down orbitals first) -> E_loc of the determinant
    psi(x) = det Phi_dn[occ_dn] det Phi_up[occ_up] (module docstring)."""
    import jax
    import jax.numpy as jnp
    n = ints["n_orb"]
    h = jnp.asarray(ints["h"])
    Lp = jnp.asarray(ints["Lp"])
    pidx = jnp.asarray(pair_index(n))
    e_nuc = ints["e_nuc"]
    phis = (jnp.asarray(phi_dn), jnp.asarray(phi_up))
    counts = (n_dn, n_up)

    def one(x):
        trs, exch, e1 = [], 0.0, 0.0
        for s in range(2):
            if counts[s] == 0:
                continue
            occ = jnp.nonzero(x[s * n:(s + 1) * n] > 0.5, size=counts[s])[0]
            A = phis[s][occ]                                         # (N, N)
            theta = jnp.linalg.solve(A.T, phis[s].T).T               # Phi A^-1 (n, N)
            e1 = e1 + jnp.trace(h[occ] @ theta)
            Locc = Lp[:, pidx[occ]]                                   # (K, N, n): L^g[occ, :]
            Y = jnp.einsum("gkp,pl->gkl", Locc, theta)
            trs.append(jnp.trace(Y, axis1=1, axis2=2))
            exch = exch + jnp.einsum("gkl,glk->", Y, Y)
        tr = sum(trs)
        return e_nuc + e1 + 0.5 * (jnp.sum(tr ** 2) - exch)

    return jax.vmap(one)


def connected(x: np.ndarray, h: np.ndarray, Lp: np.ndarray, e_nuc: float = 0.0):
    """Every configuration connected to x (2n occupations, NetKet's layout: down orbitals 0..n-1, up n..2n-1) by
    H = e_nuc + sum h_pq c+_p c_q + 1/2 sum (pq|rs) c+_p c+_r c_s c_q (spin summed), with its matrix element: the
    verifier's own Slater-Condon rules from h and the Cholesky vectors (independent of NetKet). Fermion sign of
    c+_a c_i: (-1)^(occupied modes strictly between i and a) in the 2n-mode order. Returns (xs, mels)."""
    n = h.shape[0]
    L = unpack(Lp, n)
    x = np.asarray(x).astype(np.int8)
    occ = [np.nonzero(x[s * n:(s + 1) * n])[0] for s in range(2)]
    vir = [np.nonzero(x[s * n:(s + 1) * n] == 0)[0] for s in range(2)]
    allocc = [(s, i) for s in range(2) for i in occ[s]]

    def eri(p, q, r, s_):
        return float(np.dot(L[:, p, q], L[:, r, s_]))

    def apply(y, a, i):
        """c+_a c_i on y (mode indices): (y', sign) or None."""
        if y[i] == 0 or (y[a] == 1 and a != i):
            return None
        lo, hi_ = (i, a) if i < a else (a, i)
        sign = -1 if int(np.sum(y[lo + 1:hi_])) % 2 else 1
        z = y.copy()
        z[i] = 0
        z[a] = 1
        return z, sign

    xs, mels = [], []
    # diagonal
    e = e_nuc + sum(h[i, i] for s, i in allocc)
    for s, i in allocc:
        for t, j in allocc:
            e += 0.5 * eri(i, i, j, j) - (0.5 * eri(i, j, j, i) if s == t else 0.0)
    xs.append(x.copy())
    mels.append(e)
    # singles
    for s in range(2):
        for i in occ[s]:
            for a in vir[s]:
                m = h[a, i] + sum(eri(a, i, j, j) for t, j in allocc) - sum(eri(a, j, j, i) for j in occ[s])
                y, sg = apply(x, s * n + a, s * n + i)
                xs.append(y)
                mels.append(sg * m)
    # doubles
    for s in range(2):
        for t in range(s, 2):
            for ii, i in enumerate(occ[s]):
                for jj, j in enumerate(occ[t]):
                    if s == t and jj <= ii:
                        continue
                    for a in vir[s]:
                        for b in vir[t]:
                            if s == t and b <= a:
                                continue
                            m = eri(a, i, b, j) - (eri(a, j, b, i) if s == t else 0.0)
                            r1 = apply(x, t * n + b, t * n + j)
                            r2 = apply(r1[0], s * n + a, s * n + i)
                            xs.append(r2[0])
                            mels.append(r1[1] * r2[1] * m)
    return np.asarray(xs), np.asarray(mels)


class FactorisedHamiltonian:
    """The Hamiltonian of a large-basis instance for the verifier's own estimators (not a NetKet operator):
    hilbert, the factorised integrals, `local_energy_det` (the determinant kernel) and `diagonal` (H_xx)."""
    factorised = True

    def __init__(self, hilbert, ints: dict):
        self.hilbert = hilbert
        self.ints = ints

    def __repr__(self):
        return (f"FactorisedHamiltonian(n_orb={self.ints['n_orb']}, n_chol={self.ints['n_chol']}, "
                f"cd_tol={self.ints['cd_tol']})")

    def diagonal(self, x: np.ndarray) -> float:
        """H_xx by the Slater-Condon rule (det_energy of the configuration's unit-vector determinant)."""
        n = self.ints["n_orb"]
        x = np.asarray(x)
        eye = np.eye(n)
        return det_energy(self.ints, eye[:, np.nonzero(x[:n])[0]], eye[:, np.nonzero(x[n:])[0]])

    def get_conn_padded(self, x):
        raise TypeError("a factorised Hamiltonian (more than chem.MAX_ORBITALS orbitals) has no NetKet get_conn; "
                        "it scores determinant states with the verifier's kernel (chem_large.det_local_energy_fn)")


def det_model(phi_dn: np.ndarray, phi_up: np.ndarray):
    """A determinant state as a flax module without parameters: log psi(x) = log det Phi_dn[occ_dn] + log det
    Phi_up[occ_up], NetKet's layout (down orbitals first), occupied orbitals in increasing order."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    n = phi_dn.shape[0]
    pd, pu = jnp.asarray(phi_dn), jnp.asarray(phi_up)
    nd, nu = phi_dn.shape[1], phi_up.shape[1]

    def one(x):
        out = jnp.zeros((), jnp.complex128)
        for P, k, lo in ((pd, nd, 0), (pu, nu, n)):
            if k:
                occ = jnp.nonzero(x[lo:lo + n] > 0.5, size=k)[0]
                s, la = jnp.linalg.slogdet(P[occ])
                out = out + la + jnp.log(s.astype(jnp.complex128))
        return out

    class Determinant(nn.Module):
        @nn.compact
        def __call__(self, x):
            lead = x.shape[:-1]
            return jax.vmap(one)(x.reshape(-1, x.shape[-1])).reshape(lead)

    return Determinant()


def load_det_state(path: str, inst: dict) -> dict:
    """A determinant state file (.npz: phi_dn (n, N_dn), phi_up (n, N_up), real, in the instance's Loewdin
    orbitals)."""
    with np.load(path, allow_pickle=False) as z:
        pd, pu = np.asarray(z["phi_dn"], float), np.asarray(z["phi_up"], float)
    return {"phi_dn": pd, "phi_up": pu}


MAX_CHECK_CONN = 1_000_000   # connected configurations per sample for the kernel check above chem.MAX_ORBITALS


def n_connected(n: int, nd: int, nu: int) -> int:
    v = [n - nd, n - nu]
    o = [nd, nu]
    singles = o[0] * v[0] + o[1] * v[1]
    same = sum(math.comb(o[s], 2) * math.comb(v[s], 2) for s in range(2))
    return 1 + singles + same + o[0] * v[0] * o[1] * v[1]


def kernel_check(system, model, variables, kernel, x, rtol: float) -> dict:
    """The kernel against the generic estimator sum_x' H_xx' psi(x') / psi(x) with `connected` (the verifier's own
    Slater-Condon rules) on the first sample, when it has at most MAX_CHECK_CONN connected configurations;
    otherwise reported as skipped (the kernel is gated against NetKet's operator up to 128 orbitals in the tests)."""
    import jax.numpy as jnp
    ints = system.hamiltonian.ints
    n = ints["n_orb"]
    nd, nu = system.meta["n_dn"], system.meta["n_up"]
    nc = n_connected(n, nd, nu)
    if nc > MAX_CHECK_CONN:
        return {"n": 0, "max_abs_diff": float("nan"), "max_rel": float("nan"), "rtol": rtol, "pass": True,
                "skipped": f"{nc} connected configurations per sample (> {MAX_CHECK_CONN})"}
    x = np.asarray(x)[:1]
    ek = np.asarray(kernel(variables, x))
    eg = []
    for xx in x:
        ys, ms = connected(xx, ints["h"], ints["Lp"], ints["e_nuc"])
        lp = np.asarray(model.apply(variables, jnp.asarray(np.concatenate([xx[None], ys]))))
        eg.append(np.sum(ms * np.exp(lp[1:] - lp[0])))
    eg = np.real(np.asarray(eg))
    d = np.abs(ek - eg)
    return {"n": int(len(x)), "max_abs_diff": float(d.max()), "max_rel": float((d / np.maximum(1.0, np.abs(eg))).max()),
            "rtol": rtol, "pass": bool(np.all(d <= rtol * np.maximum(1.0, np.abs(eg)))),
            "estimator": "chem_large.connected"}
