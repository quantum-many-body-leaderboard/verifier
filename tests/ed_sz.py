"""Matrix-free Lanczos for spin-1/2 Hamiltonians in a fixed S^z sector, from the verifier's own bond lists.

NetKet's `lanczos_ed` stores the sparse matrix, which stops at about 26 sites. This code stores only three Lanczos
vectors (C(N, N/2) doubles each: 4.8 GB at N = 32) and applies H on the fly, so the exact rows of 28- to 32-site
clusters can be checked against the same bonds and couplings the verifier samples with (`hamiltonians.spin_lattice`,
`hamiltonians.build(...).meta["bonds"]`). Used by tests/test_physics.py --big; numba and numpy only.

Hamiltonian (QMBL's Pauli form): H = sum_b J_b sigma_i . sigma_j  =  sum_b J_b [sz_i sz_j + 2 (s+_i s-_j + h.c.)].

Basis: bit i of a configuration is 1 for spin up on site i. States of popcount n_up are ranked by splitting the bits
into a high and a low half: rank(s) = start[hi] + idx_low[lo], with the states ordered by hi, then lo, both
increasing; idx_low[lo] is lo's index among the low halves of its popcount. Matrix elements are applied as a
gather, (H v)[r] = d[r] v[r] + sum_b 2 J_b v[rank(s_r with the antiparallel pair b flipped)], so the loop over r is
parallel without races.

    python tests/ed_sz.py SPEC|INSTANCE_ID        (prints the ground-state energy, Pauli total)
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np
import numba as nb
from numba import njit, prange

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _tables(n: int, n_up: int):
    n_lo = n // 2
    n_hi = n - n_lo
    lo_vals = np.arange(1 << n_lo, dtype=np.int64)
    pc_lo = np.array([bin(v).count("1") for v in range(1 << n_lo)], dtype=np.int64)
    idx_low = np.zeros(1 << n_lo, dtype=np.int64)
    lows_by_pc = []
    for k in range(n_lo + 1):
        vals = lo_vals[pc_lo == k]
        idx_low[vals] = np.arange(len(vals))
        lows_by_pc.append(vals)
    # flat array of low halves grouped by popcount, with offsets
    lo_off = np.zeros(n_lo + 2, dtype=np.int64)
    for k in range(n_lo + 1):
        lo_off[k + 1] = lo_off[k] + len(lows_by_pc[k])
    lo_flat = np.concatenate(lows_by_pc).astype(np.int64)
    start = np.full(1 << n_hi, -1, dtype=np.int64)
    his = []
    total = 0
    for hi in range(1 << n_hi):
        k = n_up - bin(hi).count("1")
        if 0 <= k <= n_lo:
            start[hi] = total
            his.append(hi)
            total += len(lows_by_pc[k])
    assert total == math.comb(n, n_up), (total, math.comb(n, n_up))
    return n_lo, np.array(his, dtype=np.int64), start, idx_low, lo_flat, lo_off, total


@njit(parallel=True, cache=False)
def _diag(his, start, lo_flat, lo_off, n_lo, n_up, bi, bj, bJ, out):
    for h in prange(len(his)):
        hi = his[h]
        k = n_up - _popcount(hi)
        base = start[hi]
        for t in range(lo_off[k + 1] - lo_off[k]):
            s = (hi << n_lo) | lo_flat[lo_off[k] + t]
            d = 0.0
            for b in range(len(bi)):
                if ((s >> bi[b]) & 1) == ((s >> bj[b]) & 1):
                    d += bJ[b]
                else:
                    d -= bJ[b]
            out[base + t] = d


@njit(inline="always")
def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(parallel=True, cache=False)
def _matvec(his, start, idx_low, lo_flat, lo_off, n_lo, n_up, bi, bj, bJ, d, v, out):
    mask_lo = (1 << n_lo) - 1
    for h in prange(len(his)):
        hi = his[h]
        k = n_up - _popcount(hi)
        base = start[hi]
        for t in range(lo_off[k + 1] - lo_off[k]):
            r = base + t
            s = (hi << n_lo) | lo_flat[lo_off[k] + t]
            acc = d[r] * v[r]
            for b in range(len(bi)):
                if ((s >> bi[b]) & 1) != ((s >> bj[b]) & 1):
                    s2 = s ^ ((1 << bi[b]) | (1 << bj[b]))
                    r2 = start[s2 >> n_lo] + idx_low[s2 & mask_lo]
                    acc += 2.0 * bJ[b] * v[r2]
            out[r] = acc


def ground_energy(n: int, bonds, n_up: int | None = None, tol: float = 1e-12, max_iter: int = 300,
                  seed: int = 0, log=print) -> dict:
    """Lowest eigenvalue of sum_b J_b sigma_i . sigma_j in the sector with n_up up spins (default n / 2).
    bonds: iterable of (i, j, J). Plain Lanczos (no reorthogonalisation; the lowest Ritz value converges
    monotonically), stopped when it changes by less than tol relative over 5 steps."""
    n_up = n // 2 if n_up is None else n_up
    t0 = time.perf_counter()
    n_lo, his, start, idx_low, lo_flat, lo_off, dim = _tables(n, n_up)
    bonds = list(bonds)
    bi = np.array([b[0] for b in bonds], dtype=np.int64)
    bj = np.array([b[1] for b in bonds], dtype=np.int64)
    bJ = np.array([b[2] for b in bonds], dtype=np.float64)
    d = np.empty(dim)
    _diag(his, start, lo_flat, lo_off, n_lo, n_up, bi, bj, bJ, d)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    v_old = np.zeros(dim)
    w = np.empty(dim)
    alphas, betas = [], []
    beta = 0.0
    hist = []
    for it in range(max_iter):
        _matvec(his, start, idx_low, lo_flat, lo_off, n_lo, n_up, bi, bj, bJ, d, v, w)
        a = float(np.dot(w, v))
        w -= a * v
        w -= beta * v_old
        beta = float(np.linalg.norm(w))
        alphas.append(a)
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        e = float(np.linalg.eigvalsh(T)[0])
        hist.append(e)
        if log and (it % 20 == 0):
            log(f"[ed_sz] n={n} dim={dim:.3e} it={it} E={e:.14f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        if len(hist) > 6 and abs(hist[-1] - hist[-6]) <= tol * abs(hist[-1]):
            break
        if beta < 1e-14:
            break
        betas.append(beta)
        v_old, v = v, v_old
        np.multiply(w, 1.0 / beta, out=v)
    return {"energy": hist[-1], "dim": dim, "iterations": len(hist), "wall_s": time.perf_counter() - t0}


def instance_bonds(inst: dict) -> tuple[int, list]:
    """(n, [(i, j, J)]) of a Heisenberg-type instance from hamiltonians.spin_lattice, the lists the verifier's
    sampled Hamiltonian and its MPO are built from: J1 on the `nn` bonds, J2 on the `nnn` bonds."""
    from qrec_verify import hamiltonians as Hm
    sl = Hm.spin_lattice(inst)
    bonds = [(i, j, sl["J1"]) for i, j, _ in sl["nn"]]
    if sl["J2"]:
        bonds += [(i, j, sl["J2"]) for i, j, _ in sl["nnn"]]
    return sl["n"], bonds


if __name__ == "__main__":
    from qrec_verify import instance as I, score as SC
    spec = sys.argv[1]
    inst, _ = SC.small_system(spec)
    r = ground_energy(*instance_bonds(inst))
    ex = I.exact_record(inst) if inst.get("rows") else None
    print(r, "QMBL exact:", ex and ex["energy"])
