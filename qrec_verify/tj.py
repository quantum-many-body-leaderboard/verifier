"""The t-J model of QMBL's tJ instances (v1.9): clusters, Hamiltonian and Hilbert space.

Hamiltonian (QMBL RULES.md 2.1, DATA.md "Units and conventions"; units of t, t = 1):

    H = - sum_b t_b s_b sum_sigma P (c+_i,sigma c_j,sigma + h.c.) P + J sum_<ij> (S_i . S_j - n_i n_j / 4)

with t_b = t (= 1) on nearest-neighbour bonds, t2 (t') on next-nearest and t3 (t'') on third-neighbour bonds where
the instance has them, s_b = -1 on a bond that wraps an antiperiodic axis (the hopping only; the exchange is
even), S = sigma / 2, and P the projector that removes doubly occupied sites. QMBL's instances all carry
`nn_term: true` (the - n_i n_j / 4); an instance without it is refused. Within the space without double
occupancy, P c+_i,s c_j,s P = c+_i,s (1 - n_i,-s) c_j,s, so every term is at most two-body and NetKet's
particle-number-conserving operator applies.

Hilbert space: NetKet's SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=(N_dn, N_up)) with the extra
constraint NoDoubleOccupancy (no site holds two electrons). A configuration is the same 2N occupation numbers as
for the Hubbard model (NetKet's orbital order: the N down orbitals, then the N up orbitals); configurations with a
doubly occupied site are not in the space. The samplers never leave it: they exchange the complete occupations of
two sites (a nearest-neighbour pair, or a random pair).

Clusters (`cluster(inst)`), sites as (x, y) positions, site index as listed:
- `square`, n = L^2, boundary P or O: the L x L square, i = x L + y (as hamiltonians.square_bonds).
- `square`, n = a^2 + b^2 not a square, boundary P: the square-shaped torus with edge vectors (a, b) and (-b, a)
  (RULES.md 2.1: 18 is (3, 3), 26 is (5, 1), 32 is (4, 4)); its sites are the integer points of the half-open
  parallelogram 0 <= f1, f2 < 1 (f the coordinates in the edge vectors), ordered by x, then y.
- `rectangular-WxL` (boundary P, O, PO, ...): as hamiltonians.square_bonds with lx = W, ly = L, i = x L + y.
- `ladder-WxL`, boundary P or A: W legs (x = 0 .. W-1, open across), L rungs (y = 0 .. L-1, periodic or
  antiperiodic along the legs), i = x L + y: the rectangular W x L layout with axes (O, P) or (O, A).
- `honeycomb-flake24`, boundary O: the 24-site flake of seven hexagons (a central hexagon and its six
  neighbours; arXiv:2406.16865 Fig. 2), nearest-neighbour distance 1, the central hexagon's centre at the
  origin and two of its sites on the x axis' perpendicular (sites at angles 30 + 60 k); sites ordered by y, then
  x (rounded to 1e-6).
Second neighbours (t2) on the square lattice are the diagonals (1, +-1), third neighbours (t3) the straight
next-but-one sites (2, 0) and (0, 2), each by minimum image on a torus.
"""

from __future__ import annotations

import math
import re

import numpy as np


# ------------------------------------------------------------------ instance parameters

def params(inst: dict) -> dict:
    """t, J, t2, t3, N_up, N_dn of a tJ instance, normalised to t = 1 (QMBL params)."""
    p = inst.get("params") or {}
    if inst.get("model") != "tJ":
        raise ValueError(f"not a tJ instance: {inst.get('model')}")
    if not p.get("nn_term", False):
        raise ValueError("tJ: only the exchange J (S.S - n n / 4) is supported (params.nn_term must be true)")
    t = float(p.get("t", 1.0))
    if t != 1.0:
        raise ValueError(f"tJ: QMBL normalises t = 1, found {t}")
    nu, nd = int(p["N_up"]), int(p["N_dn"])
    if nu + nd > int(inst["n_sites"]):
        raise ValueError("tJ: more electrons than sites")
    return {"t": 1.0, "J": float(p["J"]), "t2": float(p.get("t2", 0.0)), "t3": float(p.get("t3", 0.0)),
            "n_up": nu, "n_dn": nd}


# ------------------------------------------------------------------ clusters

def _torus_bonds(pos, reduce_fn, vectors, index):
    out, seen = [], set()
    for i, (x, y) in enumerate(pos):
        for dx, dy in vectors:
            j = index.get(reduce_fn(x + dx, y + dy))
            if j is None or j == i or frozenset((i, j)) in seen:
                continue
            seen.add(frozenset((i, j)))
            out.append((i, j, 1))
    return out


NN_VECS = ((1, 0), (0, 1))
NNN_VECS = ((1, 1), (1, -1))
NNNN_VECS = ((2, 0), (0, 2))


def tilted_square(a: int, b: int) -> dict:
    """The periodic square cluster with edge vectors (a, b) and (-b, a), N = a^2 + b^2 (module docstring)."""
    N = a * a + b * b
    T1, T2 = (a, b), (-b, a)
    det = T1[0] * T2[1] - T1[1] * T2[0]

    def frac(x, y):     # coordinates in (T1, T2), times det (integers)
        return (x * T2[1] - y * T2[0], -x * T1[1] + y * T1[0])

    pts = []
    R = abs(a) + abs(b) + 1
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            f1, f2 = frac(x, y)
            if 0 <= f1 < det and 0 <= f2 < det:
                pts.append((x, y))
    pts.sort()
    if len(pts) != N:
        raise RuntimeError(f"tilted cluster ({a}, {b}): {len(pts)} points, expected {N}")
    index = {p: i for i, p in enumerate(pts)}

    def red(x, y):
        f1, f2 = frac(x, y)
        m, n = f1 // det, f2 // det
        return (x - m * T1[0] - n * T2[0], y - m * T1[1] - n * T2[1])

    for p in pts:
        assert red(*p) == p
    return {"pos": [(float(x), float(y)) for x, y in pts], "nn": _torus_bonds(pts, red, NN_VECS, index),
            "nnn": _torus_bonds(pts, red, NNN_VECS, index), "nnnn": _torus_bonds(pts, red, NNNN_VECS, index),
            "shape": f"square-shaped torus, edge vectors ({a}, {b}) and ({-b}, {a})"}


def rect_cluster(lx: int, ly: int, axes: tuple[str, str]) -> dict:
    """lx x ly rectangle, i = x ly + y; axes 'P', 'A' (antiperiodic: sign -1 on the wrapping hops) or 'O'."""
    from .hamiltonians import square_bonds
    nn, nnn = square_bonds(lx, ly, axes)

    def straight2():
        out, seen = [], set()
        for x in range(lx):
            for y in range(ly):
                for dx, dy in NNNN_VECS:
                    x2, y2, s = x + dx, y + dy, 1
                    ok = True
                    for k, (c, L) in enumerate(((x2, lx), (y2, ly))):
                        if 0 <= c < L:
                            continue
                        if axes[k] == "O" or L < 5:
                            ok = False
                        elif axes[k] == "A":
                            s = -s
                    if not ok:
                        continue
                    i, j = x * ly + y, (x2 % lx) * ly + (y2 % ly)
                    if i != j and frozenset((i, j)) not in seen:
                        seen.add(frozenset((i, j)))
                        out.append((i, j, s))
        return out

    return {"pos": [(float(x), float(y)) for x in range(lx) for y in range(ly)], "nn": nn, "nnn": nnn,
            "nnnn": straight2(), "shape": f"{lx} x {ly}, axes {axes[0]}{axes[1]}"}


def honeycomb_flake24() -> dict:
    """The 24-site flake of seven hexagons (module docstring)."""
    s3 = math.sqrt(3.0)
    centres = [(0.0, 0.0)] + [(s3 * math.cos(math.radians(60 * k)), s3 * math.sin(math.radians(60 * k)))
                              for k in range(6)]
    pts = []
    for cx, cy in centres:
        for k in range(6):
            p = (round(cx + math.cos(math.radians(60 * k + 30)), 9), round(cy + math.sin(math.radians(60 * k + 30)), 9))
            if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in pts):
                pts.append(p)
    pts.sort(key=lambda p: (round(p[1], 6), round(p[0], 6)))
    if len(pts) != 24:
        raise RuntimeError(f"flake has {len(pts)} sites")
    nn = [(i, j, 1) for i in range(24) for j in range(i + 1, 24)
          if abs(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) - 1.0) < 1e-6]
    if len(nn) != 30:
        raise RuntimeError(f"flake has {len(nn)} bonds, expected 30")
    return {"pos": pts, "nn": nn, "nnn": [], "nnnn": [],
            "shape": "24-site honeycomb flake of seven hexagons, open (arXiv:2406.16865 Fig. 2)"}


def cluster(inst: dict) -> dict:
    """The cluster of a tJ instance: positions, bond lists 'nn', 'nnn', 'nnnn' as (i, j, sign), and a description."""
    from . import instance as I
    lat, n, b = inst["lattice"], int(inst["n_sites"]), inst.get("boundary", "P")
    if lat == "square":
        L = int(round(n ** 0.5))
        if L * L == n:
            if b not in ("P", "O"):
                raise ValueError(f"tJ square: boundary {b!r}")
            return rect_cluster(L, L, (b, b))
        if b != "P":
            raise ValueError("tJ: a square-shaped cluster that is not L x L must be periodic")
        shapes = [(a, c) for a in range(1, n) for c in range(0, a + 1) if a * a + c * c == n]
        if len(shapes) != 1:
            raise ValueError(f"tJ square {n}: {len(shapes)} square shapes (RULES.md 2.1 needs exactly one)")
        a, c = shapes[0]
        return tilted_square(a, c)
    m = re.fullmatch(r"ladder-(\d+)x(\d+)", lat)
    if m:
        W, L = int(m.group(1)), int(m.group(2))
        if W * L != n or b not in ("P", "A"):
            raise ValueError(f"tJ ladder {lat} {n} {b}")
        return rect_cluster(W, L, ("O", b))
    if lat.startswith("rectangular-"):
        lx, ly = I.lattice_dims(inst)
        return rect_cluster(lx, ly, I.boundary_axes(inst))
    if lat == "honeycomb-flake24":
        if n != 24 or b != "O":
            raise ValueError("honeycomb-flake24 is the open 24-site flake")
        return honeycomb_flake24()
    raise ValueError(f"tJ: lattice {lat} is not supported")


def hop_bonds(p: dict, cl: dict) -> list:
    """[(i, j, amplitude)] of -amplitude (c+_i c_j + h.c.): t s_b on nn, t2 s_b on nnn, t3 s_b on third neighbours
    (p: params(inst), cl: cluster(inst))."""
    out = [(i, j, p["t"] * s) for i, j, s in cl["nn"]]
    if p["t2"]:
        if not cl["nnn"]:
            raise ValueError("tJ: t2 on a cluster without second neighbours")
        out += [(i, j, p["t2"] * s) for i, j, s in cl["nnn"]]
    if p["t3"]:
        if not cl["nnnn"]:
            raise ValueError("tJ: t3 on a cluster without third neighbours")
        out += [(i, j, p["t3"] * s) for i, j, s in cl["nnnn"]]
    return out


# ------------------------------------------------------------------ NetKet: Hilbert space and operator

def no_double_constraint(n: int):
    import jax.numpy as jnp
    from netket.hilbert.constraint import DiscreteHilbertConstraint
    from netket.utils import struct

    class NoDoubleOccupancy(DiscreteHilbertConstraint):
        """No site holds two electrons: x[i] + x[N + i] <= 1 (NetKet's orbital order, down block first)."""
        n_sites: int = struct.field(pytree_node=False)

        def __init__(self, n_sites):
            self.n_sites = n_sites

        def __call__(self, x):
            return jnp.all(x[..., :self.n_sites] + x[..., self.n_sites:2 * self.n_sites] <= 1, axis=-1)

        def __hash__(self):
            return hash(("NoDoubleOccupancy", self.n_sites))

        def __eq__(self, other):
            return type(other).__name__ == "NoDoubleOccupancy" and other.n_sites == self.n_sites

    return NoDoubleOccupancy(n)


def hilbert(n: int, n_up: int, n_dn: int):
    import netket as nk
    from .hamiltonians import orbital_table
    hi = nk.hilbert.SpinOrbitalFermions(n_orbitals=n, s=0.5, n_fermions_per_spin=(n_dn, n_up),
                                        constraint=no_double_constraint(n))
    orb = orbital_table(hi, n)
    if any(orb[(i, -0.5)] != i or orb[(i, 0.5)] != n + i for i in range(n)):
        raise RuntimeError("NetKet's orbital order is not (down block, up block); the constraint assumes it")
    return hi


def operator_terms(n: int, hops, jbonds, J: float):
    """FermionOperator2nd terms and weights of the same Hamiltonian (orbital (i, down) = i, (i, up) = n + i), on the
    UNCONSTRAINED SpinOrbitalFermions space, where it leaves the space without double occupancy invariant. Used only
    by the tests, as an independent construction through NetKet's own fermion operators."""
    terms, weights = [], []
    up = lambda i: n + i
    dn = lambda i: i
    for i, j, a in hops:
        for o, ob in ((up, dn), (dn, up)):           # spin sigma (orbital o) and its opposite (ob)
            for p, q in ((i, j), (j, i)):            # c+_p,s (1 - n_p,-s) c_q,s
                terms.append(((o(p), 1), (o(q), 0)))
                weights.append(-a)
                terms.append(((o(p), 1), (ob(p), 1), (ob(p), 0), (o(q), 0)))
                weights.append(a)
    if J:
        for i, j, _ in jbonds:
            terms.append(((up(i), 1), (dn(i), 0), (dn(j), 1), (up(j), 0)))      # J/2 S+_i S-_j
            weights.append(J / 2)
            terms.append(((up(j), 1), (dn(j), 0), (dn(i), 1), (up(i), 0)))      # J/2 S-_i S+_j
            weights.append(J / 2)
            terms.append(((up(i), 1), (up(i), 0), (dn(j), 1), (dn(j), 0)))      # -J/2 n_i,up n_j,dn
            weights.append(-J / 2)
            terms.append(((dn(i), 1), (dn(i), 0), (up(j), 1), (up(j), 0)))      # -J/2 n_i,dn n_j,up
            weights.append(-J / 2)
    return terms, weights


# Moves of the jax operator: up to four elementary steps, applied right to left. Kinds: 0 annihilate, 1 create,
# 2 require the orbital empty (a projector), 3 nothing (padding).
ANNIHILATE, CREATE, EMPTY, NOP = 0, 1, 2, 3


def moves(n: int, hops, jbonds, J: float):
    """(orbitals (M, 4), kinds (M, 4), coefficients (M,)) of every off-diagonal move."""
    orbs, kinds, coef = [], [], []
    up = lambda i: n + i
    dn = lambda i: i
    for i, j, a in hops:
        for o, ob in ((up, dn), (dn, up)):
            for p, q in ((i, j), (j, i)):            # -a c+_p,s (1 - n_p,-s) c_q,s
                orbs.append((o(q), ob(p), o(p), 0))
                kinds.append((ANNIHILATE, EMPTY, CREATE, NOP))
                coef.append(-a)
    if J:
        for i, j, _ in jbonds:
            for p, q in ((i, j), (j, i)):            # J/2 c+_p,up c_p,dn c+_q,dn c_q,up
                orbs.append((up(q), dn(q), dn(p), up(p)))
                kinds.append((ANNIHILATE, CREATE, ANNIHILATE, CREATE))
                coef.append(J / 2)
    return np.array(orbs, dtype=np.int32), np.array(kinds, dtype=np.int32), np.array(coef, dtype=np.float64)


def _register_operator():
    import jax
    import jax.numpy as jnp
    from jax.tree_util import register_pytree_node_class
    from netket.operator import DiscreteJaxOperator

    @register_pytree_node_class
    class TJOperator(DiscreteJaxOperator):
        """The t-J Hamiltonian (module docstring) as a NetKet jax operator on the constrained SpinOrbitalFermions
        space: every connected configuration it returns is in the space (a move that is not allowed returns the
        configuration itself with weight 0). Fermion signs in NetKet's convention: an operator on orbital o picks up
        (-1)^(occupied orbitals below o), as NetKet's FermionOperator2nd (tests/test_physics.py compares the two)."""

        def __init__(self, hilbert, n_sites, orbs, kinds, coef, jbonds_i, jbonds_j, J):
            super().__init__(hilbert)
            self._n = int(n_sites)
            self._orbs = jnp.asarray(orbs)
            self._kinds = jnp.asarray(kinds)
            self._coef = jnp.asarray(coef)
            self._ji = jnp.asarray(jbonds_i, dtype=jnp.int32)
            self._jj = jnp.asarray(jbonds_j, dtype=jnp.int32)
            self._J = float(J)

        @property
        def dtype(self):
            return jnp.float64

        @property
        def is_hermitian(self):
            return True

        @property
        def max_conn_size(self):
            return 1 + int(self._coef.shape[0])

        def tree_flatten(self):
            return (self._orbs, self._kinds, self._coef, self._ji, self._jj), \
                {"hilbert": self.hilbert, "n": self._n, "J": self._J}

        @classmethod
        def tree_unflatten(cls, meta, data):
            return cls(meta["hilbert"], meta["n"], *data, meta["J"])

        @jax.jit
        def get_conn_padded(self, x):
            n = self._n
            shape = x.shape[:-1]
            xb = x.reshape(-1, 2 * n)
            xi = xb.astype(jnp.int32)
            nup, ndn = xi[:, n:], xi[:, :n]
            diag = -0.5 * self._J * jnp.sum(nup[:, self._ji] * ndn[:, self._jj] + ndn[:, self._ji] * nup[:, self._jj],
                                            axis=-1)
            M = self._coef.shape[0]
            y = jnp.broadcast_to(xi[:, None, :], (xi.shape[0], M, 2 * n))
            ok = jnp.ones(y.shape[:2], dtype=bool)
            sign = jnp.ones(y.shape[:2], dtype=jnp.int32)
            below = jnp.arange(2 * n)
            for s in range(4):
                o = self._orbs[:, s]                                   # (M,)
                k = self._kinds[:, s]
                occ = jnp.take_along_axis(y, jnp.broadcast_to(o[None, :, None], (y.shape[0], M, 1)), axis=-1)[..., 0]
                par = jnp.sum(jnp.where(below[None, None, :] < o[None, :, None], y, 0), axis=-1) % 2
                flip = (k == ANNIHILATE) | (k == CREATE)
                ok = ok & jnp.where(k == ANNIHILATE, occ == 1, True) & jnp.where((k == CREATE) | (k == EMPTY),
                                                                                  occ == 0, True)
                sign = sign * jnp.where(flip, 1 - 2 * par, 1)
                onehot = (below[None, None, :] == o[None, :, None]) & flip[None, :, None]
                y = jnp.where(onehot, 1 - y, y)
            mel = jnp.where(ok, sign * self._coef[None, :], 0.0)
            y = jnp.where(ok[..., None], y, xi[:, None, :])
            xp = jnp.concatenate([xi[:, None, :], y], axis=1).astype(x.dtype)
            mels = jnp.concatenate([diag[:, None], mel], axis=1)
            return xp.reshape(*shape, M + 1, 2 * n), mels.reshape(*shape, M + 1)

    return TJOperator


_TJ = []


def tj_operator(hi, n: int, hops, jbonds, J: float):
    if not _TJ:
        _TJ.append(_register_operator())
    orbs, kinds, coef = moves(n, hops, jbonds, J)
    ji = [b[0] for b in jbonds] if J else []
    jj = [b[1] for b in jbonds] if J else []
    return _TJ[0](hi, n, orbs, kinds, coef, ji, jj, J)


def build(inst: dict, overrides: dict | None = None):
    """(hilbert, H, nn graph, cluster, params) of a tJ instance."""
    import netket as nk
    p = params(inst)
    p.update(overrides or {})
    cl = cluster(inst)
    n = len(cl["pos"])
    if n != int(inst["n_sites"]):
        raise RuntimeError("cluster size")
    hi = hilbert(n, p["n_up"], p["n_dn"])
    H = tj_operator(hi, n, hop_bonds(p, cl), cl["nn"], p["J"])
    g = nk.graph.Graph(edges=[[i, j] for i, j, _ in cl["nn"]], n_nodes=n)
    return hi, H, g, cl, p


# ------------------------------------------------------------------ sampler

def random_configurations(key, n: int, n_up: int, n_dn: int, n_conf: int):
    """Uniformly random configurations of the space without double occupancy: a random permutation of
    n_up up electrons, n_dn down electrons and the holes over the sites, as NetKet occupations (down block first)."""
    import jax
    import jax.numpy as jnp
    base = jnp.array([1] * n_up + [2] * n_dn + [0] * (n - n_up - n_dn), dtype=jnp.int32)
    keys = jax.random.split(key, n_conf)
    perm = jax.vmap(lambda k: jax.random.permutation(k, base))(keys)          # (n_conf, n): 0 hole, 1 up, 2 down
    return jnp.concatenate([(perm == 2), (perm == 1)], axis=-1)


def sampler(system, n_chains: int, p_random: float = 0.25):
    """Metropolis over the space without double occupancy: with probability 1 - p_random exchange the complete
    occupations of the two sites of a random hopping bond (moves a hole, or flips an antiparallel pair), else of two
    random distinct sites. Both conserve N_up, N_dn and the constraint, and are symmetric proposals. Chains start
    from uniformly random configurations of the space (random_configurations)."""
    import jax
    import jax.numpy as jnp
    import netket as nk
    from netket.sampler.rules import MetropolisRule
    n = system.n_sites
    bonds = np.array([[i, j] for i, j, _ in system.meta["hop_bonds"]], dtype=np.int32)
    nu, nd = system.meta["n_up"], system.meta["n_dn"]

    class TJExchangeRule(MetropolisRule):
        def random_state(rule, sampler, machine, parameters, state, key):
            x = random_configurations(key, n, nu, nd, sampler.n_batches)
            return x.astype(sampler.dtype)

        def transition(rule, sampler, machine, parameters, state, key, sigma):
            k1, k2, k3, k4 = jax.random.split(key, 4)
            nc = sigma.shape[0]
            bsel = jax.random.randint(k1, (nc,), 0, len(bonds))
            bi, bj = jnp.asarray(bonds)[bsel, 0], jnp.asarray(bonds)[bsel, 1]
            ri = jax.random.randint(k2, (nc,), 0, n)
            rj = (ri + jax.random.randint(k3, (nc,), 1, n)) % n
            use_bond = jax.random.uniform(k4, (nc,)) >= p_random
            i = jnp.where(use_bond, bi, ri)
            j = jnp.where(use_bond, bj, rj)

            def one(s, a, b):
                s2 = s.at[a].set(s[b]).at[b].set(s[a])
                return s2.at[n + a].set(s[n + b]).at[n + b].set(s[n + a])
            return jax.vmap(one)(sigma, i, j), None

    return nk.sampler.MetropolisSampler(system.hilbert, TJExchangeRule(), n_chains=n_chains, sweep_size=n)
