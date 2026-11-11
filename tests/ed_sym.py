"""Symmetry-resolved Lanczos (v1.10) for spin-1/2 Hamiltonians in the fully symmetric sector of a cluster's symmetry.

For the exact rows of clusters beyond tests/ed_sz.py's plain S^z = 0 Lanczos (36 sites: C(36, 18) = 9.1e9 states).
The Hamiltonian is the verifier's own (QMBL's Pauli form, from `hamiltonians.build(...).meta["bonds"]`, as ed_sz.py):

    H = sum_b J_b sigma_i . sigma_j = sum_b J_b [sz_i sz_j + 2 (s+_i s-_j + h.c.)].

Group: every permutation of the sites that maps the bond list onto itself with the same couplings (found here by a
backtracking search over the weighted bond graph; translations, point group and any accidental automorphism of the
torus), times the global spin flip (S^z = 0). The sector is the trivial representation of that group (zero momentum,
fully symmetric under the point group, spin-inversion even), which is where the cited sources place the ground states
of the rows checked with it (QMBL's verified note on Heisenberg/triangular_36_P: "S^z=0, Gamma, C6v A1, spin inversion
+1"). The lowest energy in this sector is an upper bound of the ground-state energy in general; equality with a row to
1e-8 reproduces that row.

Basis: the representatives (smallest integer of each orbit of S^z = 0 configurations) and their orbit sizes O_r. With
|r~> = O_r^-1/2 sum over the orbit, <r~|H|r'~> = sqrt(O_r/O_r') sum_{s' in orbit(r') connected to r} H_(r s'), applied
as a gather (parallel without races); the matrix is built once (about 7 GB at 36 sites) and applied in CSR
form. Permutations of the 36-bit words by table lookups on four 9-bit chunks.

TFIsing (no U(1) charge): `ground_energy_tfi`, all 2^n configurations, the global flip prod sx in the group; the
Hamiltonian is stoquastic, so the trivial sector holds the ground state.

Other sectors: `--dimer-sector` (the product of singlets on the Shastry-Sutherland dimers transforms by
dimer_characters) and `--character K FLIP` (the K-th real 1D character of the group, real_characters, spin flip +-1).
Some Shastry-Sutherland 36-site rows lie outside the trivial sector (J = 0.5, 0.66697792: dimer; J = 0.67441 to 0.68:
character 3, flip +1).

    python tests/ed_sym.py INSTANCE_ID [--tol 1e-12] [--dimer-sector | --character K FLIP]
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
CHUNK = 9


def automorphisms(n: int, bonds) -> np.ndarray:
    """All permutations p (p[i] = image of site i) with {(p[i], p[j], J)} = {(i, j, J)} (bonds undirected)."""
    W = {}
    adj = [dict() for _ in range(n)]
    for i, j, J in bonds:
        key = (min(i, j), max(i, j))
        W[key] = W.get(key, 0.0) + J
    for (i, j), J in W.items():
        adj[i][j] = J
        adj[j][i] = J
    order, seen = [], set()
    for s0 in range(n):                                   # BFS order (component by component)
        if s0 in seen:
            continue
        q = [s0]
        seen.add(s0)
        while q:
            u = q.pop(0)
            order.append(u)
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    parent = {}
    pos = {v: k for k, v in enumerate(order)}
    for v in order:
        earlier = [u for u in adj[v] if pos[u] < pos[v]]
        parent[v] = min(earlier, key=lambda u: pos[u]) if earlier else None
    out = []
    perm = [-1] * n
    used = [False] * n

    def ok(v, w):
        for u, J in adj[v].items():
            if perm[u] >= 0 and abs(adj[w].get(perm[u], 0.0) - J) > 1e-12:
                return False
        return len(adj[v]) == len(adj[w])

    def rec(k):
        if k == n:
            out.append(perm.copy())
            return
        v = order[k]
        u = parent[v]
        cands = range(n) if u is None else adj[perm[u]].keys()
        for w in cands:
            if not used[w] and ok(v, w):
                perm[v] = w
                used[w] = True
                rec(k + 1)
                perm[v] = -1
                used[w] = False

    sys.setrecursionlimit(10000)
    rec(0)
    return np.array(out, dtype=np.int64)


def perm_tables(perms: np.ndarray, n: int) -> np.ndarray:
    """T[g, c, v]: the image under g of the bits v placed in chunk c."""
    nc = (n + CHUNK - 1) // CHUNK
    T = np.zeros((len(perms), nc, 1 << CHUNK), dtype=np.int64)
    for g, p in enumerate(perms):
        for c in range(nc):
            for v in range(1 << CHUNK):
                img = 0
                for b in range(CHUNK):
                    i = c * CHUNK + b
                    if i < n and (v >> b) & 1:
                        img |= 1 << int(p[i])
                T[g, c, v] = img
    return T


@njit(cache=True, inline="always")
def _image(T, g, s, nc):
    r = 0
    for c in range(nc):
        r |= T[g, c, (s >> (c * 9)) & 511]
    return r


@njit(cache=True)
def _rep_chi(T, s, nc, full, chi_g, chi_f):
    """(representative, sum of the characters over the stabiliser, character of an element mapping s to it)."""
    best = s
    cb = 1.0
    stab = 0.0
    for g in range(T.shape[0]):
        t = _image(T, g, s, nc)
        if t < best:
            best = t
            cb = chi_g[g]
        if t == s:
            stab += chi_g[g]
        u = t ^ full
        if u < best:
            best = u
            cb = chi_f[g]
        if u == s:
            stab += chi_f[g]
    return best, stab, cb


@njit(cache=True)
def _rep(T, s, nc, full):
    """(representative, stabiliser size) of s under the group and the spin flip (s ^ full)."""
    best = s
    stab = 0
    for g in range(T.shape[0]):
        t = _image(T, g, s, nc)
        if t < best:
            best = t
        if t == s:
            stab += 1
        u = t ^ full
        if u < best:
            best = u
        if u == s:
            stab += 1
    return best, stab


@njit(cache=True)
def _is_rep(T, s, nc, full):
    for g in range(T.shape[0]):
        t = _image(T, g, s, nc)
        if t < s or (t ^ full) < s:
            return False
    return True


@njit(cache=True, parallel=True)
def _row_counts(reps, bi, bj, out):
    for r in prange(len(reps)):
        s = reps[r]
        k = 0
        for b in range(len(bi)):
            if ((s >> bi[b]) & 1) != ((s >> bj[b]) & 1):
                k += 1
        out[r] = k


@njit(cache=True, parallel=True)
def _build(T, nc, full, reps, orbit, bi, bj, bJ, indptr, cols, vals, diag):
    """The sector's matrix once: diagonal, and per row the off-diagonal elements (duplicates kept, summed by the
    product)."""
    for r in prange(len(reps)):
        s = reps[r]
        d = 0.0
        k = indptr[r]
        for b in range(len(bi)):
            if ((s >> bi[b]) & 1) == ((s >> bj[b]) & 1):
                d += bJ[b]
            else:
                d -= bJ[b]
                s2 = s ^ ((1 << bi[b]) | (1 << bj[b]))
                r2, _ = _rep(T, s2, nc, full)
                c = np.searchsorted(reps, r2)
                cols[k] = c
                vals[k] = 2.0 * bJ[b] * math.sqrt(orbit[r] / orbit[c])
                k += 1
        diag[r] = d


@njit(cache=True, parallel=True)
def _matvec(indptr, cols, vals, diag, v, out):
    for r in prange(len(diag)):
        acc = diag[r] * v[r]
        for k in range(indptr[r], indptr[r + 1]):
            acc += vals[k] * v[cols[k]]
        out[r] = acc


def dimer_characters(perms: np.ndarray, dimers) -> tuple[np.ndarray, np.ndarray]:
    """The 1D representation of the product of singlets on the dimers (i, j): chi(g) = (-1)^(dimers whose orientation
    g reverses); the spin flip multiplies by (-1)^(number of dimers)."""
    D = {(min(i, j), max(i, j)) for i, j in dimers}
    chi = np.ones(len(perms))
    for g, p in enumerate(perms):
        for i, j in D:
            a, b = int(p[i]), int(p[j])
            if (min(a, b), max(a, b)) not in D:
                raise ValueError("the dimers are not mapped onto dimers")
            if a > b:
                chi[g] = -chi[g]
    return chi, chi * (-1.0) ** len(D)


def real_characters(perms: np.ndarray) -> list[np.ndarray]:
    """Every homomorphism chi: G -> {+1, -1} of the automorphism group (the real 1D representations: momenta 0 and
    pi along the torus' directions times the point group's real characters), as arrays over perms' order. Found from a
    greedy generating set: each sign assignment on the generators is propagated over G and kept if consistent."""
    idx = {tuple(int(v) for v in p): k for k, p in enumerate(perms)}
    G = len(perms)
    comp = lambda a, b: idx[tuple(int(v) for v in perms[a][perms[b]])]      # (a o b)(i) = a[b[i]]
    e = idx[tuple(range(perms.shape[1]))]
    gens, span = [], {e}
    for k in range(G):
        if k in span:
            continue
        gens.append(k)
        span, frontier = {e}, [e]                          # closure of the generators
        while frontier:
            nxt = []
            for h in frontier:
                for g in gens:
                    t = comp(g, h)
                    if t not in span:
                        span.add(t)
                        nxt.append(t)
            frontier = nxt
        if len(span) == G:
            break
    out = []
    for signs in range(1 << len(gens)):
        val = {g: (-1.0 if (signs >> a) & 1 else 1.0) for a, g in enumerate(gens)}
        chi = {e: 1.0}
        frontier, ok = [e], True
        while frontier and ok:
            nxt = []
            for h in frontier:
                for g in gens:
                    t, v = comp(g, h), val[g] * chi[h]
                    if t in chi:
                        if chi[t] != v:
                            ok = False
                            break
                    else:
                        chi[t] = v
                        nxt.append(t)
                if not ok:
                    break
            frontier = nxt
        if ok:
            out.append(np.array([chi[k] for k in range(G)]))
    return out


def ground_energy_sym(n: int, bonds, tol: float = 1e-12, max_iter: int = 400, seed: int = 0, log=print,
                      dimers=None, character=None) -> dict:
    """Lowest eigenvalue of sum_b J_b sigma_i . sigma_j in the trivial sector (S^z = 0, automorphisms, spin flip), or
    (dimers given) in the sector of the product of singlets on those bonds (dimer_characters)."""
    t0 = time.perf_counter()
    perms = automorphisms(n, bonds)
    T = perm_tables(perms, n)
    nc = T.shape[1]
    full = (1 << n) - 1
    n_group = 2 * len(perms)
    n_lo = n // 2
    n_hi = n - n_lo
    pcs = np.array([bin(v).count("1") for v in range(1 << n_lo)])
    lo_lists = np.concatenate([np.nonzero(pcs == p)[0] for p in range(n_lo + 1)]).astype(np.int64)
    lo_off = np.zeros(n_lo + 2, dtype=np.int64)
    lo_off[1:] = np.cumsum([np.sum(pcs == p) for p in range(n_lo + 1)])
    hp = np.array([bin(v).count("1") for v in range(1 << n_hi)])
    his = np.nonzero((n // 2 - hp >= 0) & (n // 2 - hp <= n_lo))[0].astype(np.int64)
    lo_off_hi = np.zeros(n_hi + 2, dtype=np.int64)       # lo_off_hi[pc(hi)] .. lo_off_hi[pc(hi)+1]
    lo_off_hi2 = np.zeros(n_hi + 2, dtype=np.int64)
    for p in range(n_hi + 1):
        q = n // 2 - p
        if 0 <= q <= n_lo:
            lo_off_hi[p], lo_off_hi2[p] = lo_off[q], lo_off[q + 1]
    offs = np.stack([lo_off_hi, lo_off_hi2], 1)
    chi_g = chi_f = None
    if dimers is not None:
        chi_g, chi_f = dimer_characters(perms, dimers)
    if character is not None:                              # (k, flip): real_characters(perms)[k], spin flip +-1
        chs = real_characters(perms)
        log(f"[ed_sym] {len(chs)} real 1D characters; sector {character[0]}, spin flip {character[1]:+d}")
        chi_g = chs[character[0]]
        chi_f = chi_g * float(character[1])
    cnt = np.zeros(len(his), dtype=np.int64)
    if chi_g is None:
        _count_reps2(T, nc, full, his, lo_lists, offs, n_lo, cnt)
    else:
        _count_reps_chi(T, nc, full, his, lo_lists, offs, n_lo, chi_g, chi_f, cnt)
    start = np.zeros(len(his), dtype=np.int64)
    start[1:] = np.cumsum(cnt)[:-1]
    dim = int(cnt.sum())
    reps = np.empty(dim, dtype=np.int64)
    orbit = np.empty(dim)
    if chi_g is None:
        _fill_reps2(T, nc, full, his, lo_lists, offs, n_lo, start, reps, orbit, float(n_group))
    else:
        _fill_reps_chi(T, nc, full, his, lo_lists, offs, n_lo, chi_g, chi_f, start, reps, orbit, float(n_group))
    order = np.argsort(reps)
    reps, orbit = reps[order], orbit[order]
    log(f"[ed_sym] {len(perms)} automorphisms (x2 spin flip), sector dimension {dim} "
        f"(C(n, n/2) / |G| = {math.comb(n, n // 2) / n_group:.4g}), {time.perf_counter() - t0:.0f} s")
    bi = np.array([b[0] for b in bonds], dtype=np.int64)
    bj = np.array([b[1] for b in bonds], dtype=np.int64)
    bJ = np.array([b[2] for b in bonds], dtype=np.float64)
    cnt = np.empty(dim, dtype=np.int64)
    _row_counts(reps, bi, bj, cnt)
    indptr = np.zeros(dim + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(cnt)
    cols = np.empty(int(indptr[-1]), dtype=np.int32)
    vals = np.empty(int(indptr[-1]))
    diag = np.empty(dim)
    if chi_g is None:
        _build(T, nc, full, reps, orbit, bi, bj, bJ, indptr, cols, vals, diag)
    else:
        _build_chi(T, nc, full, chi_g, chi_f, reps, orbit, bi, bj, bJ, indptr, cols, vals, diag)
    log(f"[ed_sym] matrix built: {indptr[-1]} off-diagonal elements, {time.perf_counter() - t0:.0f} s")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    v_old = np.zeros(dim)
    w = np.empty(dim)
    alphas, betas, hist = [], [], []
    beta = 0.0
    e = float("nan")
    for it in range(max_iter):
        _matvec(indptr, cols, vals, diag, v, w)
        a = float(np.dot(w, v))
        w -= a * v
        w -= beta * v_old
        beta = float(np.linalg.norm(w))
        alphas.append(a)
        Tm = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        e = float(np.linalg.eigvalsh(Tm)[0])
        hist.append(e)
        if log and it % 10 == 0:
            log(f"[ed_sym] iteration {it}: {e:.13f} ({time.perf_counter() - t0:.0f} s)")
        if len(hist) > 6 and abs(hist[-6] - e) < tol * abs(e):
            break
        if beta < 1e-14:
            break
        betas.append(beta)
        v_old, v = v, w / beta
        w = np.empty(dim)
    return {"energy": e, "dim": dim, "group": n_group, "iterations": len(alphas), "seconds": time.perf_counter() - t0,
            "sector": ("trivial" if dimers is None else "dimer singlets") if character is None
            else f"real character {character[0]}, spin flip {character[1]:+d}"}


@njit(cache=True, parallel=True)
def _count_reps_chi(T, nc, full, his, lo_lists, offs, n_lo, chi_g, chi_f, out):
    for h in prange(len(his)):
        hi = his[h]
        pc = 0
        x = hi
        while x:
            x &= x - 1
            pc += 1
        k = 0
        for t in range(offs[pc, 0], offs[pc, 1]):
            s = (hi << n_lo) | lo_lists[t]
            if _is_rep(T, s, nc, full):
                _, st, _ = _rep_chi(T, s, nc, full, chi_g, chi_f)
                if abs(st) > 0.5:
                    k += 1
        out[h] = k


@njit(cache=True, parallel=True)
def _fill_reps_chi(T, nc, full, his, lo_lists, offs, n_lo, chi_g, chi_f, start, reps, orbit, n_group):
    for h in prange(len(his)):
        hi = his[h]
        pc = 0
        x = hi
        while x:
            x &= x - 1
            pc += 1
        k = start[h]
        for t in range(offs[pc, 0], offs[pc, 1]):
            s = (hi << n_lo) | lo_lists[t]
            if _is_rep(T, s, nc, full):
                _, st, _ = _rep_chi(T, s, nc, full, chi_g, chi_f)
                if abs(st) > 0.5:
                    reps[k] = s
                    orbit[k] = n_group / abs(st)
                    k += 1


@njit(cache=True, parallel=True)
def _build_chi(T, nc, full, chi_g, chi_f, reps, orbit, bi, bj, bJ, indptr, cols, vals, diag):
    """As _build, in the 1D representation chi; an off-diagonal term whose representative is not in the basis
    (its stabiliser's characters cancel) is dropped (cols = -1)."""
    for r in prange(len(reps)):
        s = reps[r]
        d = 0.0
        k = indptr[r]
        for b in range(len(bi)):
            if ((s >> bi[b]) & 1) == ((s >> bj[b]) & 1):
                d += bJ[b]
            else:
                d -= bJ[b]
                s2 = s ^ ((1 << bi[b]) | (1 << bj[b]))
                r2, _, ch = _rep_chi(T, s2, nc, full, chi_g, chi_f)
                c = np.searchsorted(reps, r2)
                if c < len(reps) and reps[c] == r2:
                    cols[k] = c
                    vals[k] = 2.0 * bJ[b] * ch * math.sqrt(orbit[r] / orbit[c])
                else:
                    cols[k] = 0
                    vals[k] = 0.0
                k += 1
        diag[r] = d


@njit(cache=True, parallel=True)
def _count_reps2(T, nc, full, his, lo_lists, offs, n_lo, out):
    for h in prange(len(his)):
        hi = his[h]
        pc = 0
        x = hi
        while x:
            x &= x - 1
            pc += 1
        k = 0
        for t in range(offs[pc, 0], offs[pc, 1]):
            s = (hi << n_lo) | lo_lists[t]
            if _is_rep(T, s, nc, full):
                k += 1
        out[h] = k


@njit(cache=True, parallel=True)
def _fill_reps2(T, nc, full, his, lo_lists, offs, n_lo, start, reps, orbit, n_group):
    for h in prange(len(his)):
        hi = his[h]
        pc = 0
        x = hi
        while x:
            x &= x - 1
            pc += 1
        k = start[h]
        for t in range(offs[pc, 0], offs[pc, 1]):
            s = (hi << n_lo) | lo_lists[t]
            if _is_rep(T, s, nc, full):
                reps[k] = s
                _, st = _rep(T, s, nc, full)
                orbit[k] = n_group / st
                k += 1


# ------------------------------------------------------------------ transverse-field Ising (no U(1) charge)

@njit(cache=True, parallel=True)
def _count_reps_all(T, nc, full, n_lo, n_hi_vals, out):
    for hi in prange(n_hi_vals):
        k = 0
        for lo in range(1 << n_lo):
            if _is_rep(T, (hi << n_lo) | lo, nc, full):
                k += 1
        out[hi] = k


@njit(cache=True, parallel=True)
def _fill_reps_all(T, nc, full, n_lo, n_hi_vals, start, reps, orbit, n_group):
    for hi in prange(n_hi_vals):
        k = start[hi]
        for lo in range(1 << n_lo):
            s = (hi << n_lo) | lo
            if _is_rep(T, s, nc, full):
                reps[k] = s
                _, st = _rep(T, s, nc, full)
                orbit[k] = n_group / st
                k += 1


@njit(cache=True, parallel=True)
def _build_tfi(T, nc, full, reps, bi, bj, bJ, n, cols, diag):
    for r in prange(len(reps)):
        s = reps[r]
        d = 0.0
        for b in range(len(bi)):
            d += -bJ[b] if ((s >> bi[b]) & 1) == ((s >> bj[b]) & 1) else bJ[b]
        diag[r] = d
        for i in range(n):
            r2, _ = _rep(T, s ^ (1 << i), nc, full)
            cols[r, i] = np.searchsorted(reps, r2)


@njit(cache=True, parallel=True)
def _matvec_tfi(cols, orbit, diag, h, v, out):
    for r in prange(len(diag)):
        acc = diag[r] * v[r]
        for i in range(cols.shape[1]):
            c = cols[r, i]
            acc -= h * math.sqrt(orbit[r] / orbit[c]) * v[c]
        out[r] = acc


def ground_energy_tfi(n: int, bonds, h: float, tol: float = 1e-12, max_iter: int = 400, seed: int = 0,
                      log=print) -> dict:
    """Lowest eigenvalue of -sum_b J_b sz_i sz_j - h sum_i sx_i (Pauli; the verifier's TFIsing) in the trivial sector
    of the bond graph's automorphisms times the global flip prod sx. The Hamiltonian is stoquastic, so its ground
    state is positive (Perron-Frobenius) and lies in this sector: the result is the ground-state energy."""
    t0 = time.perf_counter()
    perms = automorphisms(n, bonds)
    T = perm_tables(perms, n)
    nc = T.shape[1]
    full = (1 << n) - 1
    n_group = 2 * len(perms)
    n_lo = n // 2
    n_hi_vals = 1 << (n - n_lo)
    cnt = np.zeros(n_hi_vals, dtype=np.int64)
    _count_reps_all(T, nc, full, n_lo, n_hi_vals, cnt)
    start = np.zeros(n_hi_vals, dtype=np.int64)
    start[1:] = np.cumsum(cnt)[:-1]
    dim = int(cnt.sum())
    reps = np.empty(dim, dtype=np.int64)
    orbit = np.empty(dim)
    _fill_reps_all(T, nc, full, n_lo, n_hi_vals, start, reps, orbit, float(n_group))
    order = np.argsort(reps)
    reps, orbit = reps[order], orbit[order]
    log(f"[ed_sym tfi] {len(perms)} automorphisms (x2 flip), sector dimension {dim} "
        f"(2^n / |G| = {2.0 ** n / n_group:.4g}), {time.perf_counter() - t0:.0f} s")
    bi = np.array([b[0] for b in bonds], dtype=np.int64)
    bj = np.array([b[1] for b in bonds], dtype=np.int64)
    bJ = np.array([b[2] for b in bonds], dtype=np.float64)
    cols = np.empty((dim, n), dtype=np.int32)
    diag = np.empty(dim)
    _build_tfi(T, nc, full, reps, bi, bj, bJ, n, cols, diag)
    del reps
    log(f"[ed_sym tfi] matrix built, {time.perf_counter() - t0:.0f} s")
    rng = np.random.default_rng(seed)
    v = np.abs(rng.standard_normal(dim))
    v /= np.linalg.norm(v)
    v_old = np.zeros(dim)
    w = np.empty(dim)
    alphas, betas, hist = [], [], []
    beta = 0.0
    e = float("nan")
    for it in range(max_iter):
        _matvec_tfi(cols, orbit, diag, h, v, w)
        a = float(np.dot(w, v))
        w -= a * v
        w -= beta * v_old
        beta = float(np.linalg.norm(w))
        alphas.append(a)
        Tm = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        e = float(np.linalg.eigvalsh(Tm)[0])
        hist.append(e)
        if log and it % 10 == 0:
            log(f"[ed_sym tfi] iteration {it}: {e:.13f} ({time.perf_counter() - t0:.0f} s)")
        if len(hist) > 6 and abs(hist[-6] - e) < tol * abs(e):
            break
        if beta < 1e-14:
            break
        betas.append(beta)
        v_old, v = v, w / beta
        w = np.empty(dim)
    return {"energy": e, "dim": dim, "group": n_group, "iterations": len(alphas), "seconds": time.perf_counter() - t0}


def instance_bonds(inst: dict):
    """The verifier's bonds (tests/ed_sz.py instance_bonds: the same lists the sampled Hamiltonian uses)."""
    sys.path.insert(0, str(ROOT / "tests"))
    import ed_sz
    return ed_sz.instance_bonds(inst)


if __name__ == "__main__":
    import argparse
    from qrec_verify import instance as I
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--dimer-sector", action="store_true",
                    help="Shastry-Sutherland: the sector of the product of singlets on the Jd bonds")
    ap.add_argument("--character", type=int, nargs=2, metavar=("K", "FLIP"),
                    help="the K-th real 1D character of the automorphism group (real_characters), spin flip +1/-1")
    a = ap.parse_args()
    inst = I.load(a.instance)
    if inst["model"] == "TFIsing":
        from qrec_verify import hamiltonians as Hm
        n, nn, h, _, _ = Hm.tfising_bonds(inst)
        r = ground_energy_tfi(n, [(i, j, 1.0) for i, j, _ in nn], h, tol=a.tol)
    else:
        n, bonds = instance_bonds(inst)
        dimers = None
        if a.dimer_sector:
            from qrec_verify import hamiltonians as Hm
            sl = Hm.spin_lattice(inst)
            dimers = [(i, j) for i, j, _ in sl["nnn"]]          # Shastry-Sutherland: J2 = Jd on the dimer bonds
        r = ground_energy_sym(n, bonds, tol=a.tol, dimers=dimers,
                              character=None if a.character is None else tuple(a.character))
    row = I.exact_record(inst)
    out = dict(r, instance=a.instance)
    if row:
        out.update(row=float(row["energy"]), rel=abs(r["energy"] - float(row["energy"])) / abs(float(row["energy"])))
    print(out, flush=True)
