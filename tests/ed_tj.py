"""Matrix-free Lanczos for the t-J model (no double occupancy) at fixed N_up, N_dn, from the verifier's own cluster
and bond lists (qrec_verify/tj.py). Independent of NetKet's fermion operators: the fermion sign of a hop is computed
here in the site-ordered basis (creation operators by increasing site index, one electron per site at most), where
moving an electron from j to i gives (-1)^(electrons strictly between i and j) and exchanging two antiparallel
spins gives +1. The spectrum does not depend on that choice of mode order.

Basis rank: rank = rank_occ(occupied sites) * C(N_e, N_up) + rank_spin(up pattern of the electrons in site order),
both in the combinatorial number system. (H v)[r] is gathered row by row, parallel over r.

    python tests/ed_tj.py INSTANCE_ID
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np
from numba import njit, prange

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _binom(n):
    B = np.zeros((n + 2, n + 2), dtype=np.int64)
    for a in range(n + 2):
        B[a, 0] = 1
        for b in range(1, a + 1):
            B[a, b] = B[a - 1, b - 1] + B[a - 1, b]
    return B


@njit(inline="always")
def _rank_bits(mask, nbits, B):
    """Rank of a bit pattern among those with the same popcount (colex order)."""
    r = 0
    k = 0
    for p in range(nbits):
        if (mask >> p) & 1:
            k += 1
            r += B[p, k]
    return r


@njit(inline="always")
def _unrank_bits(r, nbits, k, B):
    mask = 0
    for p in range(nbits - 1, -1, -1):
        if k > 0 and B[p, k] <= r:
            r -= B[p, k]
            mask |= 1 << p
            k -= 1
    return mask


@njit(inline="always")
def _compress(up, occ, n):
    """Up pattern of the electrons, in site order (bit e = electron e is up)."""
    out = 0
    e = 0
    for p in range(n):
        if (occ >> p) & 1:
            if (up >> p) & 1:
                out |= 1 << e
            e += 1
    return out


@njit(inline="always")
def _expand(spin, occ, n):
    up = 0
    e = 0
    for p in range(n):
        if (occ >> p) & 1:
            if (spin >> e) & 1:
                up |= 1 << p
            e += 1
    return up


@njit(inline="always")
def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(parallel=True)
def _matvec(n, ne, nup, n_spin, B, hi, hj, ha, ji, jj, J, v, out, diag_only):
    dim = out.shape[0]
    for r in prange(dim):
        ro = r // n_spin
        rs = r - ro * n_spin
        occ = _unrank_bits(ro, n, ne, B)
        up = _expand(_unrank_bits(rs, ne, nup, B), occ, n)
        d = 0.0
        acc = 0.0
        for b in range(len(ji)):
            i, j = ji[b], jj[b]
            if ((occ >> i) & 1) and ((occ >> j) & 1):
                if ((up >> i) & 1) != ((up >> j) & 1):
                    d -= 0.5 * J
                    if not diag_only:
                        up2 = up ^ ((1 << i) | (1 << j))
                        s2 = _rank_bits(_compress(up2, occ, n), ne, B)
                        acc += 0.5 * J * v[ro * n_spin + s2]
        if diag_only:
            out[r] = d
            continue
        acc += d * v[r]
        for b in range(len(hi)):
            for dirn in range(2):
                if dirn == 0:
                    i, j = hi[b], hj[b]
                else:
                    i, j = hj[b], hi[b]
                # c+_i c_j: an electron moves from j to i
                if ((occ >> j) & 1) and not ((occ >> i) & 1):
                    lo, hi_ = (i, j) if i < j else (j, i)
                    between = occ & (((1 << hi_) - 1) ^ ((1 << (lo + 1)) - 1))
                    sgn = -1.0 if (_popcount(between) & 1) else 1.0
                    occ2 = occ ^ ((1 << i) | (1 << j))
                    up2 = up
                    if (up >> j) & 1:
                        up2 = (up ^ (1 << j)) | (1 << i)
                    r2 = _rank_bits(occ2, n, B) * n_spin + _rank_bits(_compress(up2, occ2, n), ne, B)
                    acc += -ha[b] * sgn * v[r2]
        out[r] = acc


def ground_energy(n: int, n_up: int, n_dn: int, hops, jbonds, J: float, tol: float = 1e-12, max_iter: int = 300,
                  seed: int = 0, log=print) -> dict:
    """Lowest eigenvalue of the t-J Hamiltonian (module docstring); hops: (i, j, amplitude) of
    -amplitude (c+_i c_j + h.c.), jbonds: (i, j, _) carrying J (S.S - n n / 4)."""
    t0 = time.perf_counter()
    ne = n_up + n_dn
    B = _binom(n)
    n_spin = math.comb(ne, n_up)
    dim = math.comb(n, ne) * n_spin
    hi = np.array([h[0] for h in hops], dtype=np.int64)
    hj = np.array([h[1] for h in hops], dtype=np.int64)
    ha = np.array([h[2] for h in hops], dtype=np.float64)
    ji = np.array([b[0] for b in jbonds], dtype=np.int64)
    jj = np.array([b[1] for b in jbonds], dtype=np.int64)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    v_old = np.zeros(dim)
    w = np.empty(dim)
    alphas, betas, hist = [], [], []
    beta = 0.0
    for it in range(max_iter):
        _matvec(n, ne, n_up, n_spin, B, hi, hj, ha, ji, jj, float(J), v, w, False)
        a = float(np.dot(w, v))
        w -= a * v
        w -= beta * v_old
        beta = float(np.linalg.norm(w))
        alphas.append(a)
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        e = float(np.linalg.eigvalsh(T)[0])
        hist.append(e)
        if log and it % 20 == 0:
            log(f"[ed_tj] n={n} dim={dim:.3e} it={it} E={e:.14f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        if len(hist) > 6 and abs(hist[-1] - hist[-6]) <= tol * abs(hist[-1]):
            break
        if beta < 1e-14:
            break
        betas.append(beta)
        v_old, v = v, v_old
        np.multiply(w, 1.0 / beta, out=v)
    return {"energy": hist[-1], "dim": dim, "iterations": len(hist), "wall_s": time.perf_counter() - t0}


def instance_problem(inst: dict) -> tuple:
    """(n, n_up, n_dn, hops, jbonds, J) of a tJ instance from qrec_verify.tj (the verifier's own cluster)."""
    from qrec_verify import tj
    p = tj.params(inst)
    cl = tj.cluster(inst)
    return len(cl["pos"]), p["n_up"], p["n_dn"], tj.hop_bonds(p, cl), cl["nn"], p["J"]


if __name__ == "__main__":
    import json
    path = sys.argv[1]
    inst = json.load(open(path))
    r = ground_energy(*instance_problem(inst))
    ex = [x["energy"] for x in inst["rows"] if x.get("bound_type") == "exact"]
    print(json.dumps({"instance": inst["instance_id"], **r, "qmbl_exact": ex}))
