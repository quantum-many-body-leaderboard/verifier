"""Physics checks of the verifier: conventions, Hamiltonians, local energies, the measurement path.

Part A (numpy only): bond counts, Fermi-sea energies of the real-space hopping matrix against a
momentum-space sum, dof and einf against every stored row, records. Part B (NetKet): exact
diagonalisation against QMBL's exact rows, a free-fermion check with t' and an antiperiodic axis,
the trace check of einf with V terms, a zero-variance re-measurement of an exact eigenstate, and
Monte Carlo against full summation for a random spin state and a random Slater determinant.
Since v1.9 (the lattices of qrec-verifier v1.8-v1.11): the clusters' site orders and bonds against NetKet's own
lattices and QMBL's cluster files, ED against QMBL's exact rows to 1e-8 relative where a plain Lanczos fits in memory,
the known state of every lattice at every shipped size, and Monte Carlo against full summation.

    python tests/test_physics.py          # part A
    python tests/test_physics.py --netket # parts A and B
    python tests/test_physics.py --big    # the 24- to 32-site exact rows (tests/ed_sz.py), slow
"""

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qrec_verify import instance as I  # noqa: E402
from qrec_verify.hamiltonians import frustrated_cluster, shastry_dimers, square_bonds, triangular_cluster  # noqa: E402

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


def test_triangular_clusters():
    """Cluster kinds from the instance files, bond counts, coordination."""
    kinds = {"Heisenberg/triangular_16_P": ("LxL", 4), "Heisenberg/triangular_36_P": ("LxL", 6),
             "Heisenberg/triangular_48_P": ("3m2", 4), "Heisenberg/triangular_108_P": ("3m2", 6),
             "Heisenberg/triangular_144_P": ("LxL", 12), "Heisenberg/triangular_100_O": ("LxL", 10),
             "Heisenberg/triangular_196_O": ("LxL", 14), "J1J2/triangular_108_P_0.125": ("3m2", 6),
             "J1J2/triangular_144_P_0.125": ("LxL", 12), "Heisenberg/triangular_1764_P": ("LxL", 42)}
    for iid, k in kinds.items():
        assert I.triangular_dims(I.load(iid)) == k, (iid, I.triangular_dims(I.load(iid)))
    for kind, size in (("LxL", 4), ("LxL", 6), ("LxL", 12), ("3m2", 4), ("3m2", 6)):
        cl = triangular_cluster(kind, size, "P")
        n = len(cl["coords"])
        assert len(cl["nn"]) == 3 * n and len(cl["nnn"]) == 3 * n, (kind, size, len(cl["nn"]), len(cl["nnn"]))
        deg = np.bincount(np.array([[i, j] for i, j, _ in cl["nn"]]).ravel(), minlength=n)
        assert np.all(deg == 6), (kind, size)
    for L in (4, 10, 14):
        cl = triangular_cluster("LxL", L, "O")
        assert len(cl["nn"]) == 3 * L * L - 4 * L + 1 and len(cl["nnn"]) == (L - 1) ** 2 + 2 * (L - 1) * (L - 2), L
    for iid in ("Heisenberg/triangular_144_P", "Heisenberg/triangular_100_O", "J1J2/triangular_108_P_0.125"):
        assert I.target(I.load(iid))["bound_type"] == "variational"


FRUSTRATED_KINDS = {
    "Heisenberg/kagome-2x2_12_P": ("kagome", (2, 2)), "Heisenberg/kagome-2x3_18_P": ("kagome", (2, 3)),
    "Heisenberg/kagome-4x4_48_P": ("kagome", (4, 4)), "Heisenberg/kagome-6x6_108_P": ("kagome", (6, 6)),
    "Heisenberg/kagome-8x8_192_P": ("kagome", (8, 8)),
    "Heisenberg/shuriken_24_P": ("shuriken", (2, 2)), "Heisenberg/shuriken_96_P": ("shuriken", (4, 4)),
    "Heisenberg/shuriken_216_P": ("shuriken", (6, 6)), "Heisenberg/shuriken_384_P": ("shuriken", (8, 8)),
    "Heisenberg/pyrochlore-2x2x2_32_P": ("pyrochlore", (2, 2, 2)),
    "Heisenberg/pyrochlore-2x2x3_48_P": ("pyrochlore", (2, 2, 3)),
    "Heisenberg/pyrochlore-2x2x4_64_P": ("pyrochlore", (2, 2, 4)),
    "Heisenberg/pyrochlore-2x2x6_96_P": ("pyrochlore", (2, 2, 6)),
    "Heisenberg/pyrochlore-3x3x3_108_P": ("pyrochlore", (3, 3, 3)),
    "Heisenberg/pyrochlore-2x4x6_192_P": ("pyrochlore", (2, 4, 6)),
    "Heisenberg/pyrochlore-4x4x4_256_P": ("pyrochlore", (4, 4, 4)),
    "Heisenberg/pyrochlore-2x2x2_128_P": ("pyrochlore-cubic", (2, 2, 2)),
    "Heisenberg/pyrochlore-3x3x3_432_P": ("pyrochlore-cubic", (3, 3, 3)),
    "Heisenberg/pyrochlore-4x4x4_1024_P": ("pyrochlore-cubic", (4, 4, 4)),
}


def _same_bonds_by_position(cl, pos_ref, bonds_ref, sup, shift=(0.0, 0.0)):
    """Bond sets equal after matching sites by position (the reference shifted by `shift`) modulo the supercell."""
    pos = np.array(cl["coords"])
    pos_ref = np.asarray(pos_ref, dtype=float) + np.asarray(shift)
    inv = np.linalg.inv(np.array(sup, dtype=float))

    def frac(r):
        f = np.asarray(r) @ inv
        return tuple(np.round((f - np.floor(f + 1e-9)) * 1e6).astype(int))

    mine = {frac(r): i for i, r in enumerate(pos)}
    assert len(mine) == len(pos)
    perm = [mine[frac(r)] for r in pos_ref]
    ref = {frozenset((perm[i], perm[j])) for i, j in bonds_ref}
    return ref == {frozenset((i, j)) for i, j, _ in cl["nn"]}


def _stdface_pyrochlore(W, L, H):
    """mVMC's StdFace Pyrochlore.c bonds (varbench/methods programs/mVMC_RBM), ported in QMBL's
    checks/pyrochlore-32-ed/ed_full.py (pyrochlore_stdface); site = cell (iW, iL, iH lexicographic) * 4 + isiteUC."""
    calls = [(0, 0, 0, 0, 1), (0, 0, 0, 0, 2), (0, 0, 0, 0, 3), (0, 0, 0, 2, 3), (0, 0, 0, 3, 1), (0, 0, 0, 1, 2),
             (1, 0, 0, 1, 0), (0, 1, 0, 2, 0), (0, 0, 1, 3, 0), (0, -1, 1, 3, 2), (1, 0, -1, 1, 3), (-1, 1, 0, 2, 1)]
    cells = [(a, b, c) for a in range(W) for b in range(L) for c in range(H)]
    cid = {x: k for k, x in enumerate(cells)}
    out = set()
    for (iw, il, ih) in cells:
        for (dw, dl, dh, a, b) in calls:
            j = cid[((iw + dw) % W, (il + dl) % L, (ih + dh) % H)] * 4 + b
            out.add(frozenset((cid[(iw, il, ih)] * 4 + a, j)))
    return out


def test_frustrated_clusters():
    """Cluster kinds from the instance files; bond counts and coordination; QMBL's kagome_12 and shuriken_24
    cluster files (checks/exact-recompute-2026-09/clusters) and mVMC's StdFace pyrochlore bonds, site by site."""
    import json
    for iid, k in FRUSTRATED_KINDS.items():
        assert I.frustrated_dims(I.load(iid)) == k, (iid, I.frustrated_dims(I.load(iid)))
    for iid, (kind, ext) in FRUSTRATED_KINDS.items():
        if I.load(iid)["n_sites"] > 432:
            continue
        cl = frustrated_cluster(kind, ext)
        n = len(cl["coords"])
        assert n == I.load(iid)["n_sites"]
        assert len(cl["nn"]) == (3 if kind.startswith("pyrochlore") else 2) * n, (iid, len(cl["nn"]))
    for W, L, H in ((2, 2, 2), (2, 2, 3), (3, 3, 3)):
        cl = frustrated_cluster("pyrochlore", (W, L, H))
        assert {frozenset((i, j)) for i, j, _ in cl["nn"]} == _stdface_pyrochlore(W, L, H), (W, L, H)
    sh = json.load(open(ROOT / "tests" / "data" / "shuriken_24.json"))
    cl = frustrated_cluster("shuriken", (2, 2))
    assert _same_bonds_by_position(cl, np.array(sh["positions_x4"]) / 4.0, sh["bonds"], [[2, 0], [0, 2]])
    kg = json.load(open(ROOT / "tests" / "data" / "kagome_12.json"))
    s3 = math.sqrt(3)
    ref_pos = [np.array([x / 2 + y / 4, s3 / 4 * y]) for x, y in kg["positions"]]   # doubled (a1, a2) coordinates
    cl = frustrated_cluster("kagome", (2, 2))
    assert _same_bonds_by_position(cl, ref_pos, kg["bonds"], [[2, 0], [1, s3]], shift=(0.75, s3 / 4))
    for kind, ext in (("kagome", (1, 2)), ("pyrochlore", (2, 2, 1)), ("pyrochlore", (1, 1, 1))):
        try:
            frustrated_cluster(kind, ext)
        except ValueError:
            continue
        raise AssertionError(f"{kind} {ext}: a too small cluster was accepted")


def test_shastry_dimers():
    """The Shastry-Sutherland dimers cover every site once, are diagonals of the square (minimum image),
    neighbouring dimers are orthogonal (the dimers of the two plaquettes sharing an edge with a dimer's plaquette run
    the other way), and every square bond joins two different dimers; the shipped instances parse."""
    for L in range(4, 20, 2):
        dim = shastry_dimers(L)
        assert len(dim) == L * L // 2
        assert sorted(i for b in dim for i in b[:2]) == list(range(L * L))
        partner = {}
        for i, j, _ in dim:
            partner[i], partner[j] = j, i
            dx = (j // L - i // L) % L
            dy = (j % L - i % L) % L
            assert {dx, dy} <= {1, L - 1}, (L, i, j)
        orient = {}
        for i, j, _ in dim:
            xi, yi, xj, yj = i // L, i % L, j // L, j % L
            same = ((xj - xi) % L == 1) == ((yj - yi) % L == 1)
            orient[(min(xi, xj) if abs(xi - xj) == 1 else max(xi, xj), min(yi, yj) if abs(yi - yj) == 1 else max(yi, yj))] = same
        assert len(set(orient.values())) == 2, L
        nn, _ = square_bonds(L, L, ("P", "P"))
        assert all(partner[i] != j for i, j, _ in nn), L
    for iid in ALL:
        if iid.startswith("ShastrySutherland/"):
            L, J, Jd, _ = I.shastry_params(I.load(iid))
            assert L * L == I.load(iid)["n_sites"] and Jd == 1.0, iid


def test_shastry_open():
    """Setup B (arXiv:2309.10955 App. S-2) of the open L x L square: no dimer touches a corner, no site carries two,
    and the cluster is a window of the infinite Shastry-Sutherland lattice: the torus pattern of size L + 2, shifted by
    one site along y, restricted to the L x L square, gives the same dimers."""
    for L in (4, 6, 10, 20):
        op = {frozenset(b[:2]) for b in shastry_dimers(L, "O")}
        assert len(op) == ((L - 1) ** 2 - 1) // 2, (L, len(op))
        M = L + 2
        win = set()
        for i, j, _ in shastry_dimers(M, "P"):
            (xi, yi), (xj, yj) = divmod(i, M), divmod(j, M)
            if abs(xi - xj) == 1 and abs(yi - yj) == 1:          # no wrapped dimer inside the window
                pts = [(xi, yi - 1), (xj, yj - 1)]
                if all(0 <= x < L and 0 <= y < L for x, y in pts):
                    win.add(frozenset(x * L + y for x, y in pts))
        assert op == win, L
        corners = {0, L - 1, L * (L - 1), L * L - 1}
        assert not corners & {i for b in op for i in b}


def test_tfising_dims():
    """TFIsing dimensions, axes and field from the shipped instance files; the closed-form bond count of the
    known state against the verifier's bond list."""
    from qrec_verify.hamiltonians import square_bonds
    iids = sorted(i for i in ALL if i.startswith("TFIsing/"))
    assert len(iids) == 11, iids
    want = {"TFIsing/chain_10_O_1": (10, 1, ("O", "O"), 1.0), "TFIsing/chain_32_P_0.5": (32, 1, ("P", "O"), 0.5),
            "TFIsing/square_144_O_3": (12, 12, ("O", "O"), 3.0), "TFIsing/square_36_P_3": (6, 6, ("P", "P"), 3.0)}
    for iid, w in want.items():
        assert I.tfising_dims(I.load(iid)) == w, (iid, I.tfising_dims(I.load(iid)))
    for iid in iids:
        lx, ly, axes, h = I.tfising_dims(I.load(iid))
        nn, _ = square_bonds(lx, ly, axes)
        bx = (lx if axes[0] == "P" else lx - 1) * ly if lx > 1 else 0
        by = (ly if axes[1] == "P" else ly - 1) * lx if ly > 1 else 0
        assert len(nn) == bx + by, (iid, len(nn), bx + by)


def honeycomb_reference(L1, L2):
    """An independent honeycomb torus (the rule of QMBL's checks/exact-recompute-2026-10-families/geom.py): cells
    (m, n) modulo (L1, L2), A(m, n) bonded to B(m, n), B(m - 1, n) and B(m, n - 1); site 2 (m L2 + n) for A, + 1 for B.
    Second neighbours: the six same-sublattice sites at cell offsets +-(1, 0), +-(0, 1), +-(1, -1). Returns the A/B
    positions (NetKet's units) and the two bond sets."""
    s3 = math.sqrt(3)
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, s3 / 2])
    idx = lambda m, n, s: 2 * ((m % L1) * L2 + (n % L2)) + s
    pos = np.zeros((2 * L1 * L2, 2))
    nn, nnn = set(), set()
    for m in range(L1):
        for n in range(L2):
            for s in (0, 1):
                pos[idx(m, n, s)] = m * a1 + n * a2 + (s + 1) * (a1 + a2) / 3
            for dm, dn in ((0, 0), (-1, 0), (0, -1)):
                nn.add(frozenset((idx(m, n, 0), idx(m + dm, n + dn, 1))))
            for s in (0, 1):
                for dm, dn in ((1, 0), (0, 1), (1, -1)):
                    nnn.add(frozenset((idx(m, n, s), idx(m + dm, n + dn, s))))
    return pos, nn, nnn


def test_honeycomb_clusters():
    """honeycomb-LxM: kind and extent from the instance files, 3N/2 and 3N bonds, and the bonds of an independent
    construction (honeycomb_reference) matched site by site through the positions."""
    for iid in ("Heisenberg/honeycomb-7x7_98_P", "J1J2/honeycomb-7x7_98_P_0.2"):
        assert I.frustrated_dims(I.load(iid)) == ("honeycomb", (7, 7)), iid
    for L1, L2 in ((3, 3), (4, 4), (3, 4), (7, 7)):
        cl = frustrated_cluster("honeycomb", (L1, L2))
        n = len(cl["coords"])
        assert len(cl["nn"]) == 3 * n // 2 and len(cl["nnn"]) == 3 * n, (L1, L2)
        pos, nn, nnn = honeycomb_reference(L1, L2)
        assert np.allclose(pos, np.array(cl["coords"]), atol=1e-12), (L1, L2)
        assert nn == {frozenset(b[:2]) for b in cl["nn"]} and nnn == {frozenset(b[:2]) for b in cl["nnn"]}, (L1, L2)
    try:
        frustrated_cluster("honeycomb", (2, 2))
    except ValueError:
        return
    raise AssertionError("honeycomb 2x2 (bonds through two images) was accepted")


def _tj_reference_bonds(kind, *args):
    """Independent constructions of the t-J clusters, after QMBL's checks/tj-ed-2026-10/tj_ed.py: the tilted square
    torus by fractional coordinates (`tilted`), and the seven-hexagon flake (`coronene`). Returns positions and the
    nearest-neighbour bond set (as frozensets of indices into the positions)."""
    if kind == "tilted":
        L1, L2 = args
        det = L1[0] * L2[1] - L1[1] * L2[0]
        N = abs(det)

        def red(x, y):
            a = (x * L2[1] - y * L2[0]) / det
            b = (-x * L1[1] + y * L1[0]) / det
            a -= math.floor(a + 1e-9)
            b -= math.floor(b + 1e-9)
            return (round(a * N) % N, round(b * N) % N)
        sites = {}
        R = max(map(abs, L1 + L2)) * 2 + 2
        for x in range(-R, R):
            for y in range(-R, R):
                sites.setdefault(red(x, y), (x, y))
        keys = list(sites)
        ix = {k: n for n, k in enumerate(keys)}
        bonds = {frozenset((ix[k], ix[red(x + dx, y + dy)])) for k, (x, y) in sites.items() for dx, dy in ((1, 0), (0, 1))}
        return [sites[k] for k in keys], bonds, red
    cs = [(0.0, 0.0)] + [(math.sqrt(3) * math.cos(math.radians(60 * k)), math.sqrt(3) * math.sin(math.radians(60 * k)))
                         for k in range(6)]
    pts = []
    for cx, cy in cs:
        for k in range(6):
            x, y = cx + math.cos(math.radians(60 * k + 30)), cy + math.sin(math.radians(60 * k + 30))
            if not any(abs(x - a) < 1e-6 and abs(y - b) < 1e-6 for a, b in pts):
                pts.append((x, y))
    bonds = {frozenset((i, j)) for i in range(len(pts)) for j in range(i + 1, len(pts))
             if abs(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) - 1) < 1e-6}
    return pts, bonds, None


def test_tj_clusters():
    """t-J clusters of every shipped tJ instance: bond counts in closed form; the tilted tori (18, 26, 32 sites) and
    the 24-site flake against independent constructions (_tj_reference_bonds), site by site through the
    positions; the electron numbers and dof."""
    from qrec_verify import tj
    iids = sorted(i for i in ALL if i.startswith("tJ/"))
    assert len(iids) == 27, len(iids)
    for iid in iids:
        inst = I.load(iid)
        cl = tj.cluster(inst)
        p = tj.params(inst)
        n = inst["n_sites"]
        assert len(cl["pos"]) == n and I.expected_dof(inst) == p["n_up"] + p["n_dn"], iid
        lat, b = inst["lattice"], inst["boundary"]
        if lat == "square" and b == "P":
            want = 2 * n
        elif lat == "square":
            L = int(round(n ** 0.5))
            want = 2 * L * (L - 1)
        elif lat.startswith("ladder-2x"):
            want = 3 * n // 2
        elif lat == "ladder-3x6":
            want = 30
        elif lat == "rectangular-6x16":
            want = 6 * 15 + 16 * 6
        else:
            want = 30
        assert len(cl["nn"]) == want, (iid, len(cl["nn"]), want)
        if p["t2"]:
            assert len(cl["nnn"]) == 2 * n and len(cl["nnnn"]) == 2 * n, iid
    for (a, b), n in (((3, 3), 18), ((5, 1), 26), ((4, 4), 32)):
        cl = tj.tilted_square(a, b)
        pts, bonds, red = _tj_reference_bonds("tilted", [a, b], [-b, a])
        mine = {red(int(x), int(y)): i for i, (x, y) in enumerate(cl["pos"])}
        perm = [mine[red(*q)] for q in pts]
        assert {frozenset(perm[i] for i in bd) for bd in bonds} == {frozenset(bd[:2]) for bd in cl["nn"]}, n
    cl = tj.honeycomb_flake24()
    pts, bonds, _ = _tj_reference_bonds("coronene")
    perm = [min(range(24), key=lambda i: math.hypot(cl["pos"][i][0] - x, cl["pos"][i][1] - y)) for x, y in pts]
    assert sorted(perm) == list(range(24))
    assert {frozenset(perm[i] for i in bd) for bd in bonds} == {frozenset(bd[:2]) for bd in cl["nn"]}


def _dense_fermion(n_modes, terms, weights):
    """Dense matrix of sum_k w_k (product of the ladder operators of term k) on all 2^n_modes Fock states, with the
    sign (-1)^(occupied modes below) of an operator on a mode: an independent evaluation of FermionOperator2nd terms."""
    dim = 1 << n_modes
    H = np.zeros((dim, dim), dtype=complex)
    for ops, w in zip(terms, weights):
        for st in range(dim):
            s, sg = st, 1
            for mode, dag in reversed(ops):
                occ = (s >> mode) & 1
                if occ == dag:
                    sg = 0
                    break
                sg *= -1 if bin(s & ((1 << mode) - 1)).count("1") % 2 else 1
                s ^= 1 << mode
            if sg:
                H[s, st] += w * sg
    return H


def test_kanamori_atomic_limit():
    """The impurity interaction (qrec_verify/impurity.py) against forkTPS's own atomic-limit test (test/c++/
    FTPO_impl.cpp, AIM.atomicLimit_SFPH: three orbitals, eps = -0.1, U = 1, Up = 0.8, J = 0.1): the lowest energy
    reachable from each of its ten initial occupations. The opposite sign of the spin-flip term fails one of them."""
    from qrec_verify import impurity as IMP
    U, Up, J, eps = 1.0, 0.8, 0.1, -0.1
    par = {"blocks": ["dn", "up"], "M": 3, "Nb": 0, "hloc": {"dn": eps * np.eye(3), "up": eps * np.eye(3)},
           "bath": {"dn": [[], [], []], "up": [[], [], []]}, "U": U, "J": J, "Up": Up, "soc": False}
    n, tm, w = IMP.terms(par)
    H = _dense_fermion(2 * n, tm, w)
    assert np.allclose(H, H.conj().T)
    # forkTPS sites A_up, A_dn, B_up, B_dn, C_up, C_dn = 1 .. 6; here (orbital m, up) = n + m, (m, down) = m
    mode = {1: n + 0, 2: 0, 3: n + 1, 4: 1, 5: n + 2, 6: 2}
    cases = [((1, 2), U - J + 2 * eps), ((1, 4), U - 3 * J + 2 * eps), ((1, 6), U - 3 * J + 2 * eps),
             ((3, 6), U - 3 * J + 2 * eps), ((1, 3), U - 3 * J + 2 * eps), ((2, 4), U - 3 * J + 2 * eps),
             ((1, 3, 5), 3 * U - 9 * J + 3 * eps), ((2, 4, 6), 3 * U - 9 * J + 3 * eps),
             ((1, 2, 3), 3 * U - 6 * J + 3 * eps), ((1, 4, 6), 3 * U - 9 * J + 3 * eps)]
    for occ, want in cases:
        v = np.zeros(H.shape[0], dtype=complex)
        v[sum(1 << mode[k] for k in occ)] = 1.0
        K = [v]
        for _ in range(64):
            x = H @ K[-1]
            for u in K:
                x -= np.vdot(u, x) * u
            if np.linalg.norm(x) < 1e-12:
                break
            K.append(x / np.linalg.norm(x))
        Kb = np.array(K)
        e = np.linalg.eigvalsh(Kb.conj() @ H @ Kb.T)[0]
        assert abs(e - want) < 1e-10, (occ, e, want)


def test_impurity_files():
    """Every Impurity instance reads its VarBench parameter file; orbitals per block M (1 + Nb) and the sector."""
    from qrec_verify import impurity as IMP
    iids = sorted(i for i in ALL if i.startswith("Impurity/"))
    assert len(iids) == 12, iids
    for iid in iids:
        inst = I.load(iid)
        par = IMP.read(IMP.file_for(inst))
        nb = 10 if inst["lattice"] == "SB-DMFT-MI-HF" else int(inst["n_sites"])
        assert par["Nb"] == nb and par["M"] == (3 if inst["lattice"].startswith("TB") else 1), iid
        nd, nu = IMP.sector(inst)
        assert nd + nu == IMP.n_electrons(inst) <= 2 * par["M"] * (1 + nb), iid


# ------------------------------------------------------------------ part B (NetKet)

EXACT_RTOL = 1e-8          # ED against QMBL's exact rows, relative (rule of the v1.9 merge)


def check_exact_row(iid, e0):
    """e0 (the verifier's ED) against the instance's exact row: 1e-8 relative, or the row's own sigma for an
    exact (stochastic) row, or half a unit of its printed precision where it states one."""
    row = I.exact_record(I.load(iid))
    ref, sig, prec = row["energy"], row.get("sigma") or 0.0, row.get("precision") or 0.0
    tol = max(EXACT_RTOL * abs(ref), sig, prec / 2)
    assert abs(e0 - ref) <= tol, (iid, e0, ref, tol)
    return ref, abs(e0 - ref) / abs(ref)


def _tri_inst(n, b, J2=0.0):
    return {"instance_id": f"test/tri{n}{b}", "model": "J1J2" if J2 else "Heisenberg", "lattice": "triangular",
            "n_sites": n, "boundary": b, "params": {"J2": J2} if J2 else {}, "rows": []}


def nk_triangular_netket_order():
    """The verifier's sites and bonds are NetKet's: Triangular(extent=[L, L]) for L x L, Lattice with basis
    m (a1 + a2), m (2 a2 - a1) and offsets 0, a1, a2 for 3 m^2 (positions equal, NN and NNN edge sets equal)."""
    import netket as nk
    s3 = math.sqrt(3)
    cases = [("LxL", L, b, nk.graph.Triangular(extent=[L, L], pbc=(b == "P"), max_neighbor_order=2))
             for L, b in ((4, "P"), (6, "P"), (12, "P"), (14, "P"), (4, "O"), (10, "O"), (14, "O"))]
    cases += [("3m2", m, "P", nk.graph.Lattice(basis_vectors=[[1.5, s3 / 2], [0.0, s3]],
                                               site_offsets=[[0, 0], [1.0, 0], [0.5, s3 / 2]], extent=[m, m], pbc=True,
                                               max_neighbor_order=2)) for m in (4, 6)]
    for kind, size, b, g in cases:
        cl = triangular_cluster(kind, size, b)
        pos = np.array([[x + 0.5 * y, s3 / 2 * y] for x, y in cl["coords"]])
        assert np.allclose(np.asarray(g.positions), pos, atol=1e-9), (kind, size, b)
        for color, key in ((0, "nn"), (1, "nnn")):
            mine = {frozenset((i, j)) for i, j, _ in cl[key]}
            theirs = {frozenset((i, j)) for i, j in g.edges(filter_color=color)}
            assert mine == theirs, (kind, size, b, key, len(mine), len(theirs))
    print(f"  triangular: {len(cases)} clusters, site order and NN/NNN bonds equal NetKet's lattices")


def nk_triangular_ed():
    """4x4 periodic: QMBL's exact row (VarBench ED) to 1e-8; the 12-site C6v cluster: -0.6103 per site (Bernu et al.
    1994); J1-J2 at J2 = 0.125 and the open 4x4 against an operator built on NetKet's own Triangular graph. The 36- and
    48-site exact rows need a symmetry-resolved diagonalisation (COVERAGE.md)."""
    import netket as nk
    from qrec_verify import hamiltonians as Hm
    iid = "Heisenberg/triangular_16_P"
    e0 = ground_energy(Hm.build(I.load(iid)).hamiltonian)
    ref, rel = check_exact_row(iid, e0)
    print(f"  {iid}: ED {e0:.12f} vs QMBL {ref} (rel {rel:.1e})")
    e12 = ground_energy(Hm.build(_tri_inst(12, "P")).hamiltonian) / 48
    assert abs(e12 - (-0.6103)) < 1e-4, e12
    print(f"  triangular 12-site C6v cluster: ED {e12:.7f} per site (Bernu et al.: -0.6103)")
    for b, J2 in (("P", 0.125), ("O", 0.0), ("O", 0.125)):
        g = nk.graph.Triangular(extent=[4, 4], pbc=(b == "P"), max_neighbor_order=2)
        hi = nk.hilbert.Spin(0.5, 16, total_sz=0)
        Hn = nk.operator.Heisenberg(hi, g, J=[1.0, J2] if J2 else [1.0, 0.0], sign_rule=[False, False])
        en = ground_energy(Hn)
        ev = ground_energy(Hm.build(_tri_inst(16, b, J2)).hamiltonian)
        assert abs(en - ev) < 1e-9, (b, J2, en, ev)
        print(f"  triangular 4x4 {b} J2={J2}: verifier ED {ev:.10f} = NetKet-graph ED {en:.10f}")


def nk_triangular_stripe():
    from qrec_verify import score as SC
    iids = sorted(i for i in ALL if "triangular" in i)
    for iid in iids:
        inst = I.load(iid)
        r = SC.neel_check(inst)
        assert r["pass"] and r["state"].startswith("stripe"), (iid, r)
    print(f"  stripe known state passes on {len(iids)} triangular instances, e.g. 144_P "
          f"{SC.neel_check(I.load('Heisenberg/triangular_144_P'))['energy_per_site']} per site")


def _mc_vs_fullsum(label, sysm, seed=3):
    """A random complex RBM on a small system: the verifier's parallel-tempering chains against the full sum."""
    import jax
    import netket as nk
    from qrec_verify import score as SC, stats as S
    n = sysm.n_sites
    model = nk.models.RBM(alpha=1, param_dtype=complex)
    variables = model.init(jax.random.key(seed), np.ones((1, n)))
    variables = jax.tree_util.tree_map(lambda a: a * 3.0, variables)
    exact = float(np.real(nk.vqs.FullSumState(sysm.hilbert, model, variables=variables).expect(sysm.hamiltonian).mean))
    ser = SC.collect_adaptive(sysm, model, variables, SC.spin_sampler(sysm, n_chains=64, n_replicas=8), seed=11,
                              n_burn=256, block=16, min_samples=64 * 16 * 32, sigma_target_total=None,
                              max_seconds=600, log=lambda *_: None)
    d = S.diagnose(ser["e_loc"], ser["logabs"], ser["acc"])
    z = (d.mean - exact) / d.sigma
    print(f"  {label}: PT {d.mean:.6f} +- {d.sigma:.6f} vs full sum {exact:.6f} (z {z:+.2f})")
    assert abs(z) < 4, z


def _netket_frustrated(kind, ext):
    import netket as nk
    from qrec_verify.hamiltonians import FRUSTRATED
    vecs, offs, dnn = FRUSTRATED[kind]
    return nk.graph.Lattice(basis_vectors=np.array(vecs), site_offsets=np.array(offs), extent=list(ext), pbc=True)


def nk_frustrated_netket_order():
    """The verifier's sites are NetKet's Lattice sites for the same vectors and offsets, and its bonds are NetKet's
    nearest-neighbour edges (the shuriken: NetKet's edges are the triangle legs; the squares are the extra bonds);
    kagome also against NetKet's own Kagome graph."""
    import netket as nk
    cases = [("kagome", (2, 3)), ("kagome", (6, 6)), ("kagome", (8, 8)), ("shuriken", (4, 4)),
             ("pyrochlore", (2, 2, 3)), ("pyrochlore", (2, 4, 6)), ("pyrochlore-cubic", (2, 2, 2))]
    for kind, ext in cases:
        cl = frustrated_cluster(kind, ext)
        g = _netket_frustrated(kind, ext)
        assert np.allclose(np.asarray(g.positions), np.array(cl["coords"]), atol=1e-9), (kind, ext)
        theirs = {frozenset((i, j)) for i, j in g.edges(filter_color=0)}
        mine = {frozenset((i, j)) for i, j, _ in cl["nn"]}
        if kind == "shuriken":
            nb = cl["n_basis"]
            mine = {b for b in mine if not (len({k // nb for k in b}) == 1 and {k % nb for k in b} <= {0, 1, 2, 3})}
        assert mine == theirs, (kind, ext, len(mine), len(theirs))
    for kind, ext, g in [("kagome", e, nk.graph.Kagome(extent=list(e), pbc=True)) for e in ((2, 2), (6, 6), (8, 8))] + \
            [("pyrochlore", e, nk.graph.Pyrochlore(extent=list(e), pbc=True)) for e in ((2, 2, 3), (3, 3, 3))]:
        cl = frustrated_cluster(kind, ext)
        assert np.allclose(np.asarray(g.positions), np.array(cl["coords"]), atol=1e-9), (kind, ext)
        assert {frozenset((i, j)) for i, j, _ in cl["nn"]} == {frozenset(e) for e in g.edges(filter_color=0)}, ext
    print(f"  frustrated: {len(cases)} clusters, site order and bonds equal NetKet's lattices; kagome and pyrochlore "
          f"also equal NetKet's own Kagome and Pyrochlore")


def nk_frustrated_ed():
    """ED against QMBL's exact rows: kagome 2x2 (Lauchli et al.), kagome 2x3 (VarBench ED), shuriken 24
    (arXiv:2110.08198); the 16-site cubic pyrochlore against an operator on NetKet's own lattice graph."""
    import netket as nk
    from qrec_verify import hamiltonians as Hm, score as SC
    for iid in ("Heisenberg/kagome-2x2_12_P", "Heisenberg/kagome-2x3_18_P", "Heisenberg/shuriken_24_P"):
        e0 = ground_energy(Hm.build(I.load(iid)).hamiltonian)
        ref, rel = check_exact_row(iid, e0)
        print(f"  {iid}: ED {e0:.12f} vs QMBL {ref} (rel {rel:.1e})")
    small, sysm = SC.small_system("pyrochlore-cubic:1x1x1")
    g = _netket_frustrated("pyrochlore-cubic", (1, 1, 1))
    hi = nk.hilbert.Spin(0.5, 16, total_sz=0)
    en = ground_energy(nk.operator.Heisenberg(hi, nk.graph.Graph(edges=[list(e) for e in g.edges(filter_color=0)],
                                                                 n_nodes=16), J=1.0, sign_rule=False))
    ev = ground_energy(sysm.hamiltonian)
    assert abs(en - ev) < 1e-9, (en, ev)
    print(f"  pyrochlore-cubic 1x1x1 (16 sites): verifier ED {ev:.10f} = NetKet-lattice ED {en:.10f}")


def nk_frustrated_known_state():
    from qrec_verify import score as SC
    iids = [i for i in FRUSTRATED_KINDS if I.load(i)["n_sites"] <= 432]
    for iid in iids:
        r = SC.neel_check(I.load(iid))
        assert r["pass"] and "configuration pair" in r["state"], (iid, r)
    for spec in ("kagome:2x2", "kagome:2x3", "shuriken:2", "pyrochlore-cubic:1x1x1"):
        r = SC.neel_check(SC.small_system(spec)[0])
        assert r["pass"], (spec, r)
    print(f"  configuration-pair known state passes on {len(iids)} instances and 4 small specs, e.g. pyrochlore 128 "
          f"{SC.neel_check(I.load('Heisenberg/pyrochlore-2x2x2_128_P'))['energy_per_site']} per site")


def nk_frustrated_mc_vs_fullsum():
    from qrec_verify import score as SC
    for spec in ("kagome:2x3", "pyrochlore-cubic:1x1x1"):
        _mc_vs_fullsum(spec, SC.small_system(spec)[1], seed=5)


def nk_shastry_ed():
    """ED of the 4 x 4 Shastry-Sutherland cluster against QMBL's exact row (J = 0.7), and the product of dimer
    singlets is an exact eigenstate with -3 Jd per dimer (Pauli units) at J = 0.3 and 0.7."""
    import netket as nk
    from qrec_verify import hamiltonians as Hm
    inst = I.load("ShastrySutherland/square_16_P_J0.7_Jd1")
    sysm = Hm.build(inst)
    e0 = ground_energy(sysm.hamiltonian)
    ref, rel = check_exact_row(inst["instance_id"], e0)
    print(f"  Shastry-Sutherland 4x4 J=0.7: ED {e0:.12f} vs QMBL {ref} (rel {rel:.1e})")
    for J in (0.3, 0.7):
        _, sysm = __import__("qrec_verify.score", fromlist=["x"]).small_system(f"shastry:4:{J}")
        hi = sysm.hilbert
        xs = np.asarray(hi.all_states())
        up = hi.local_states[1]
        amp = np.ones(len(xs))
        for i, j, _ in shastry_dimers(4):
            a, b = xs[:, i] == up, xs[:, j] == up
            amp *= np.where(a == b, 0.0, np.where(a, 1.0, -1.0))
        psi = amp / np.linalg.norm(amp)
        Hpsi = sysm.hamiltonian.to_sparse() @ psi
        e = float(psi @ Hpsi)
        assert abs(e + 24.0) < 1e-10 and np.linalg.norm(Hpsi - e * psi) < 1e-10, (J, e)
        print(f"  Shastry-Sutherland 4x4 J={J}: dimer product is an eigenstate, E = {e:.12f} (-3 per dimer)")


def nk_shastry_known_state():
    """The Neel pair at every shipped Shastry-Sutherland size, periodic and open."""
    from qrec_verify import score as SC
    for iid in ALL:
        if iid.startswith("ShastrySutherland/"):
            r = SC.neel_check(I.load(iid))
            assert r["pass"], (iid, r["energy_total"], r["expected_total"])
    print("  Shastry-Sutherland Neel pair: pass on every shipped instance")


def nk_shastry_open_ed():
    """The open 4 x 4 (setup B) against an operator on NetKet's own open Grid graph plus the window dimers of
    test_shastry_open, and the dimer product is no eigenstate there (edge sites without a dimer)."""
    import netket as nk
    from qrec_verify import score as SC
    _, sysm = SC.small_system("shastry:4:0.8:1:O")
    g = nk.graph.Grid(extent=[4, 4], pbc=False)
    dim = [(1, 6), (5, 8), (7, 10), (9, 14)]       # (x, y) -> 4 x + y: plaquettes (0,1), (1,0), (1,2), (2,1)
    assert {frozenset(d) for d in dim} == {frozenset(b[:2]) for b in shastry_dimers(4, "O")}
    edges = [[i, j, 0] for i, j in g.edges()] + [[i, j, 1] for i, j in dim]
    hi = nk.hilbert.Spin(0.5, 16, total_sz=0)
    Hn = nk.operator.Heisenberg(hi, nk.graph.Graph(edges=edges, n_nodes=16), J=[0.8, 1.0], sign_rule=[False, False])
    ev, en = ground_energy(sysm.hamiltonian), ground_energy(Hn)
    assert abs(ev - en) < 1e-10, (ev, en)
    print(f"  Shastry-Sutherland open 4x4 J=0.8: verifier ED {ev:.12f} = NetKet-grid ED {en:.12f}")


def _tfi_independent(lx, ly, axes, h):
    """The Ising Hamiltonian built term by term from Pauli operators on NetKet's own Grid graph (an operator path
    independent of nk.operator.Ising and of the verifier's bond list)."""
    import netket as nk
    from netket.operator.spin import sigmax, sigmaz
    pbc = [a == "P" for a in axes]
    g = nk.graph.Grid(extent=[lx, ly], pbc=pbc) if ly > 1 else nk.graph.Chain(lx, pbc=pbc[0])
    hi = nk.hilbert.Spin(0.5, lx * ly)
    H = sum(-1.0 * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges()) + sum(-h * sigmax(hi, i) for i in range(lx * ly))
    return H


def nk_tfising_ed():
    """ED against QMBL's exact rows (chain_10_O_1, chain_10_P_1) and against an independently built operator on small
    open, periodic and mixed squares."""
    from qrec_verify import hamiltonians as Hm, score as SC
    for iid in ("TFIsing/chain_10_O_1", "TFIsing/chain_10_P_1"):
        inst = I.load(iid)
        e0 = ground_energy(Hm.build(inst).hamiltonian)
        ref, rel = check_exact_row(iid, e0)
        print(f"  {iid}: ED {e0:.12f} vs QMBL {ref} (rel {rel:.1e})")
    for lx, ly, b, h in ((4, 4, "O", 3.0), (4, 4, "P", 3.0), (3, 4, "PO", 2.0), (4, 3, "OP", 4.0)):
        _, sysm = SC.small_system(f"tfising:{lx}x{ly}:{b}:{h}")
        axes = tuple(b) if len(b) == 2 else (b, b)
        ev, en = ground_energy(sysm.hamiltonian), ground_energy(_tfi_independent(lx, ly, axes, h))
        assert abs(ev - en) < 1e-9, (lx, ly, b, h, ev, en)
        print(f"  tfising {lx}x{ly} {b} h={h}: verifier ED {ev:.10f} = Pauli-term ED {en:.10f}")


def nk_tfising_known_state():
    from qrec_verify import score as SC
    iids = sorted(i for i in ALL if i.startswith("TFIsing/"))
    for iid in iids:
        r = SC.neel_check(I.load(iid))
        assert r["pass"] and r["state"].startswith("ferromagnetic"), (iid, r)
    for spec in ("tfising:4x4:O:3", "tfising:3x4:PO:2"):
        assert SC.neel_check(SC.small_system(spec)[0])["pass"], spec
    print(f"  ferromagnetic-pair known state passes on {len(iids)} TFIsing instances, e.g. 144_O "
          f"{SC.neel_check(I.load('TFIsing/square_144_O_3'))['energy_per_site']} per site")


def nk_tfising_mc_vs_fullsum():
    from qrec_verify import score as SC
    for spec in ("tfising:4x4:O:3", "tfising:4x4:P:2"):
        _mc_vs_fullsum(spec + " (single flips)", SC.small_system(spec)[1], seed=7)


def nk_honeycomb():
    """Sites and bonds equal NetKet's Honeycomb (7x7, 4x4, 3x3); ED of the 18-site 3x3 torus, Heisenberg and
    J2 = 0.2, against an operator on NetKet's own Honeycomb graph; the Neel pair at both shipped instances; the
    variational rows of the 98-site instances lie above the 32-site energies (finite-size sanity, not a gate);
    Monte Carlo against full summation on the 3x3 torus."""
    import netket as nk
    from qrec_verify import score as SC
    for ext in ((7, 7), (4, 4), (3, 3)):
        cl = frustrated_cluster("honeycomb", ext)
        g = nk.graph.Honeycomb(extent=list(ext), pbc=True, max_neighbor_order=2)
        assert np.allclose(np.asarray(g.positions), np.array(cl["coords"]), atol=1e-9), ext
        for color, key in ((0, "nn"), (1, "nnn")):
            assert {frozenset(e) for e in g.edges(filter_color=color)} == {frozenset(b[:2]) for b in cl[key]}, (ext, key)
    for J2 in (0.0, 0.2):
        _, sysm = SC.small_system(f"honeycomb:3x3:{J2}")
        g = nk.graph.Honeycomb(extent=[3, 3], pbc=True, max_neighbor_order=2)
        hi = nk.hilbert.Spin(0.5, 18, total_sz=0)
        Hn = nk.operator.Heisenberg(hi, g, J=[1.0, J2], sign_rule=[False, False])
        ev, en = ground_energy(sysm.hamiltonian), ground_energy(Hn)
        assert abs(ev - en) < 1e-10 * abs(en), (J2, ev, en)
        print(f"  honeycomb 3x3 J2={J2}: verifier ED {ev:.12f} = NetKet-Honeycomb ED {en:.12f}")
    for iid in ("Heisenberg/honeycomb-7x7_98_P", "J1J2/honeycomb-7x7_98_P_0.2"):
        r = SC.neel_check(I.load(iid))
        assert r["pass"], (iid, r)
    print("  honeycomb Neel pair passes on both 7x7 instances")
    _mc_vs_fullsum("honeycomb 3x3 J2=0.2", SC.small_system("honeycomb:3x3:0.2")[1], seed=9)


TJ_SMALL = [
    {"instance_id": "test/tj-l25", "model": "tJ", "lattice": "ladder-2x5", "n_sites": 10, "boundary": "P",
     "params": {"t": 1, "J": 0.5, "N_up": 4, "N_dn": 4, "nn_term": True}},
    {"instance_id": "test/tj-l25A", "model": "tJ", "lattice": "ladder-2x5", "n_sites": 10, "boundary": "A",
     "params": {"t": 1, "J": 0.4, "N_up": 5, "N_dn": 4, "nn_term": True}},
    {"instance_id": "test/tj-sq10", "model": "tJ", "lattice": "square", "n_sites": 10, "boundary": "P",
     "params": {"t": 1, "J": 0.3, "t2": 0.3, "t3": -0.2, "N_up": 4, "N_dn": 4, "nn_term": True}},
    {"instance_id": "test/tj-r34", "model": "tJ", "lattice": "rectangular-3x4", "n_sites": 12, "boundary": "PO",
     "params": {"t": 1, "J": 0.35, "N_up": 5, "N_dn": 5, "nn_term": True}},
]
TJ_EXACT = ["tJ/square_16_O_h2_J0.4", "tJ/ladder-3x6_18_A_h1_J0.5", "tJ/ladder-3x6_18_A_h2_J0.5",
            "tJ/square_18_P_h2_J0.4"]


def nk_tj_operator():
    """The verifier's t-J operator (qrec_verify/tj.py, TJOperator) against NetKet's FermionOperator2nd of the same
    Hamiltonian on the unconstrained space (tj.operator_terms): every matrix element between configurations without
    double occupancy, on 200 random configurations of four small clusters (periodic and antiperiodic ladders, a
    tilted torus with t2 and t3, a cylinder); and ED of the operator against the independent Lanczos of
    tests/ed_tj.py (site-ordered fermion signs)."""
    from collections import defaultdict
    import netket as nk
    sys.path.insert(0, str(ROOT / "tests"))
    import ed_tj
    from qrec_verify import tj
    from qrec_verify import hamiltonians as Hm
    for inst in TJ_SMALL:
        sysm = Hm.build(inst)
        n, p = sysm.n_sites, tj.params(inst)
        cl = tj.cluster(inst)
        hu = nk.hilbert.SpinOrbitalFermions(n, s=0.5, n_fermions_per_spin=(p["n_dn"], p["n_up"]))
        terms, w = tj.operator_terms(n, tj.hop_bonds(p, cl), cl["nn"], p["J"])
        Hn = nk.operator.FermionOperator2nd(hu, terms=terms, weights=w)
        hi = sysm.hilbert
        x = np.asarray(hi.all_states())[np.random.default_rng(0).choice(hi.n_states, 200, replace=False)]
        a, ma = map(np.asarray, sysm.hamiltonian.get_conn_padded(x))
        b, mb = map(np.asarray, Hn.get_conn_padded(x))
        dmax = 0.0
        for k in range(len(x)):
            da, db = defaultdict(float), defaultdict(float)
            for xx, m in zip(a[k], ma[k]):
                assert not np.any(xx[:n] + xx[n:] > 1), "the operator left the space"
                da[tuple(xx.astype(int))] += m
            for xx, m in zip(b[k], mb[k]):
                if not np.any(xx[:n] + xx[n:] > 1):
                    db[tuple(xx.astype(int))] += m
            dmax = max([dmax] + [abs(da[key] - db[key]) for key in set(da) | set(db)])
        assert dmax < 1e-12, (inst["instance_id"], dmax)
        e_nk = ground_energy(sysm.hamiltonian)
        r = ed_tj.ground_energy(*ed_tj.instance_problem(inst), log=None)
        assert abs(e_nk - r["energy"]) < 1e-10 * abs(e_nk), (inst["instance_id"], e_nk, r)
        print(f"  {inst['instance_id']}: elements = NetKet FermionOperator2nd (max diff {dmax:.1e}); "
              f"ED {e_nk:.12f} = ed_tj {r['energy']:.12f}")


def nk_tj_exact_rows():
    """QMBL's exact t-J rows (ED of four clusters: open 4 x 4, the antiperiodic three-leg ladder with one and two
    holes, the 18-site tilted torus) with tests/ed_tj.py on the verifier's own clusters and couplings."""
    sys.path.insert(0, str(ROOT / "tests"))
    import ed_tj
    for iid in TJ_EXACT:
        r = ed_tj.ground_energy(*ed_tj.instance_problem(I.load(iid)), log=None)
        ref, rel = check_exact_row(iid, r["energy"])
        print(f"  {iid}: Lanczos {r['energy']:.12f} (dim {r['dim']}) vs QMBL {ref} (rel {rel:.1e})")


def nk_tj_known_state():
    from qrec_verify import score as SC
    iids = sorted(i for i in ALL if i.startswith("tJ/"))
    for iid in iids:
        r = SC.known_state(I.load(iid))
        assert r["pass"], (iid, r)
    print(f"  t-J one-configuration known state passes on {len(iids)} instances")


def nk_tj_mc_vs_fullsum():
    """A random Slater determinant (NetKet's Slater2nd) on the t-J space of a 2 x 5 ladder: the verifier's sampler
    (bond and site exchanges, never a double occupancy) against the full sum over the space."""
    import jax
    import netket as nk
    from qrec_verify import hamiltonians as Hm, score as SC, stats as S
    sysm = Hm.build(TJ_SMALL[0])
    model = nk.models.Slater2nd(sysm.hilbert, param_dtype=float)
    variables = model.init(jax.random.key(4), np.asarray(sysm.hilbert.all_states()[:1]))
    exact = float(np.real(nk.vqs.FullSumState(sysm.hilbert, model, variables=variables).expect(sysm.hamiltonian).mean))
    sampler = SC.default_sampler(sysm, TJ_SMALL[0])
    ser = SC.collect_adaptive(sysm, model, variables, sampler, seed=11, n_burn=256, block=32,
                              min_samples=64 * 32 * 16, sigma_target_total=None, max_seconds=900, log=lambda *_: None)
    d = S.diagnose(ser["e_loc"], ser["logabs"], ser["acc"])
    z = (d.mean - exact) / d.sigma
    print(f"  t-J ladder 2x5: sampler {d.mean:.6f} +- {d.sigma:.6f} vs full sum {exact:.6f} (z {z:+.2f})")
    assert abs(z) < 4, z


CHEM_FAST = ["Hchain/chain_4_O_R1.8_sto6g", "Hchain/chain_6_O_R1.8_sto6g", "Hchain/chain_8_O_R1.8_sto6g",
             "Hchain/chain_10_O_R1.8_sto6g"]
# QMBL's ids carry R rounded to six figures; these exact rows were computed at the Angstrom spacing the source states
CHEM_ANGSTROM = {1.41729: 0.75, 1.51178: 0.8, 1.88973: 1.0, 2.83459: 1.5, 3.77945: 2.0}


BOHR_ANGSTROM = 0.52917721092


def check_hydrogen_row(inst, e):
    """e against the instance's exact (FCI) row. Many hydrogen rows are printed to four to six decimals, so the
    tolerance is the largest of 1e-8 relative, half a unit of the last printed decimal and half the row's stated
    precision. Returns (row energy, relative difference, tolerance)."""
    row = I.exact_record(inst)
    ref = float(row["energy"])
    text = repr(ref)
    decimals = len(text.split(".")[-1].rstrip("0")) if "." in text and "e" not in text else 12
    tol = max(EXACT_RTOL * abs(ref), 0.5 * 10.0 ** -decimals * (1 + 1e-9), (row.get("precision") or 0.0) / 2)
    assert abs(e - ref) <= tol, (inst["instance_id"], e, ref, tol)
    return ref, abs(e - ref) / abs(ref), tol


def _have(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def nk_chem():
    """Hydrogen in a Gaussian basis (qrec_verify/chem.py; needs pyscf, requirements-chem.txt): on four STO-6G chains
    the ED of the verifier's NetKet operator equals pyscf's FCI with the integrals pyscf transforms to the Loewdin
    orbitals and pyscf's FCI in canonical RHF orbitals (1e-9), and QMBL's exact row (1e-8 relative); every exact
    (FCI) row in --big; the one-determinant known state on chains of 10 to 102 atoms."""
    if not _have("pyscf"):
        print("  SKIP: pyscf is not installed (requirements-chem.txt)")
        return
    from qrec_verify import chem, score as SC
    from qrec_verify import hamiltonians as Hm
    for iid in CHEM_FAST:
        inst = I.load(iid)
        sysm = Hm.build(inst)
        e0 = ground_energy(sysm.hamiltonian)
        f = chem.fci_energy(inst, sysm.meta["integrals"])
        assert abs(e0 - f["fci_loewdin"]) < 1e-9 and abs(f["fci_loewdin"] - f["fci_canonical"]) < 1e-9, (iid, e0, f)
        ref, rel, tol = check_hydrogen_row(inst, e0)
        print(f"  {iid}: ED {e0:.12f} = FCI (Loewdin {f['fci_loewdin']:.12f}, canonical {f['fci_canonical']:.12f}) "
              f"vs QMBL {ref} (rel {rel:.1e}, tolerance {tol:.0e})")
    for iid in ("Hchain/chain_10_O_R1.8_ccpvdz", "Hchain/chain_22_O_R1.8_sto6g", "Hchain/chain_102_O_R1.8_sto6g"):
        r = SC.known_state(I.load(iid))
        assert r["pass"], (iid, r)
    print("  one-determinant known state passes on chains of 10 (cc-pVDZ), 22 and 102 (STO-6G) atoms")
    for iid in ("Hchain/chain_30_O_R1.8_ccpvdz", "Hchain/chain_10_O_R1.8_none"):
        try:
            Hm.build(I.load(iid))
        except ValueError as exc:
            print(f"  {iid} refused: {str(exc)[:100]}")
            continue
        raise AssertionError(f"{iid} was built")


def big_chem_rows():
    """Every exact (FCI) row of QMBL's hydrogen instances: pyscf's FCI with the verifier's Loewdin integrals. Rows
    whose id rounds R (CHEM_ANGSTROM) are compared at the stated Angstrom spacing and reported."""
    if not _have("pyscf"):
        print("  SKIP: pyscf is not installed")
        return
    from qrec_verify import chem
    for iid in sorted(i for i in ALL if i.startswith(("Hchain/", "Hsheet/"))):
        inst = I.load(iid)
        if not I.exact_record(inst) or not chem.is_chem(inst):
            continue
        f = chem.fci_energy(inst)
        assert abs(f["fci_loewdin"] - f["fci_canonical"]) < 1e-9, (iid, f)
        R = float(inst["params"]["R"])
        e, note = f["fci_loewdin"], ""
        if R in CHEM_ANGSTROM:
            e = chem.fci_energy(dict(inst, params=dict(inst["params"], R=CHEM_ANGSTROM[R] / BOHR_ANGSTROM)))["fci_loewdin"]
            note = f"; at {CHEM_ANGSTROM[R]} A exactly: {e:.12f}"
        ref, rel, tol = check_hydrogen_row(inst, e)
        print(f"  {iid}: FCI {f['fci_loewdin']:.12f} (canonical {f['fci_canonical']:.12f}) vs QMBL {ref} "
              f"(rel {rel:.1e}, tolerance {tol:.0e}){note}", flush=True)


def nk_impurity():
    """QMBL's exact rows of the four single-band impurity models (ED of the verifier's operator; VarBench's parameter
    files), the one-configuration known state of the eight models without spin-orbit coupling (the four spin-orbit models are
    refused: qrec_verify/impurity.py SOC_UNCONFIRMED), and the DMRG cross-check of the three-band
    models recorded in tests/xcheck_impurity.json (tests/xcheck_impurity_tenpy.py) against VarBench's forkTPS rows."""
    import json
    from qrec_verify import hamiltonians as Hm, score as SC
    for iid in sorted(i for i in ALL if i.startswith("Impurity/SB")):
        e0 = ground_energy(Hm.build(I.load(iid)).hamiltonian)
        ref, rel = check_exact_row(iid, e0)
        print(f"  {iid}: ED {e0:.12f} vs QMBL {ref} (rel {rel:.1e})")
    iids = sorted(i for i in ALL if i.startswith("Impurity/") and "-SOC_" not in i)
    for iid in iids:
        r = SC.known_state(I.load(iid))
        assert r["pass"], (iid, r)
    print(f"  one-configuration known state passes on {len(iids)} impurity instances")
    for iid in sorted(i for i in ALL if i.startswith("Impurity/TB-DMFT-SOC_")):
        try:
            Hm.build(I.load(iid))
        except ValueError as exc:
            assert "TB-DMFT-SOC" in str(exc), exc
            continue
        raise AssertionError(f"{iid} was built (the spin-orbit models are refused until confirmed)")
    print("  the spin-orbit models (TB-DMFT-SOC) are refused with the reason")
    p = ROOT / "tests" / "xcheck_impurity.json"
    if p.exists():
        for r in json.loads(p.read_text())["runs"]:
            print(f"  recorded DMRG cross-check {r['instance']} ({r['hyb']}): {r['energy']:.10f} vs VarBench "
                  f"{r['varbench'][0]:.10f} (diff {r['energy'] - r['varbench'][0]:+.1e})")


def nk_impurity_small_tenpy():
    """The three-band impurity models (TB-DMFT, TB-DMFT-SOC; no exact row in QMBL) on a truncated bath of one site per
    orbital (6 orbitals, 4096 Fock states): the lowest ED energy of the verifier's operator over the S_z (block)
    sectors equals DMRG (exact at this size) of the same parameters built independently in TeNPy
    (tests/xcheck_impurity_tenpy.py: its own fermion operators and Kanamori terms), for N = 5, 6, 7, to 1e-10."""
    if not _have("tenpy"):
        print("  SKIP: TeNPy is not installed (requirements-mps.txt)")
        return
    import netket as nk
    from qrec_verify import impurity as IMP
    sys.path.insert(0, str(ROOT / "tests"))
    import xcheck_impurity_tenpy as XT
    for iid in ("Impurity/TB-DMFT_9", "Impurity/TB-DMFT-SOC_9"):
        inst = I.load(iid)
        par = IMP.read(IMP.file_for(inst))
        par = dict(par, Nb=1, bath={b: [sites[:1] for sites in per] for b, per in par["bath"].items()})
        n, tm, w = IMP.terms(par)
        w = np.array(w, dtype=complex)
        for N in (5, 6, 7):
            e_ed = min(np.linalg.eigvalsh(nk.operator.FermionOperator2nd(
                nk.hilbert.SpinOrbitalFermions(n_orbitals=n, s=0.5, n_fermions_per_spin=(nd, N - nd)),
                terms=tm, weights=w, dtype=complex).to_dense())[0] for nd in range(max(0, N - n), min(n, N) + 1))
            e_t = XT.run(inst, chi=128, sweeps=30, par=par, n_el=N, log=lambda *a: None)["energy"]
            assert abs(e_t - e_ed) < 1e-10, (iid, N, e_ed, e_t)
            print(f"  {iid} truncated to Nb = 1, N = {N}: ED {e_ed:.12f}, TeNPy DMRG {e_t:.12f} ({e_t - e_ed:+.1e})")


def nk_triangular_mc_vs_fullsum():
    from qrec_verify import hamiltonians as Hm
    for b, J2 in (("P", 0.125), ("O", 0.0)):
        _mc_vs_fullsum(f"triangular 4x4 {b} J2={J2}", Hm.build(_tri_inst(16, b, J2)))



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


NETKET_TESTS = [nk_impurity, nk_impurity_small_tenpy,
                nk_chem,
                nk_tj_operator, nk_tj_exact_rows, nk_tj_known_state, nk_tj_mc_vs_fullsum,
                nk_honeycomb,
                nk_tfising_ed, nk_tfising_known_state, nk_tfising_mc_vs_fullsum,
                nk_shastry_ed, nk_shastry_known_state, nk_shastry_open_ed,
                nk_frustrated_netket_order, nk_frustrated_ed, nk_frustrated_known_state, nk_frustrated_mc_vs_fullsum,
                nk_triangular_netket_order, nk_triangular_ed, nk_triangular_stripe, nk_triangular_mc_vs_fullsum,
                nk_grid_order, nk_j1j2_ed, nk_free_fermions_tprime_pa, nk_einf_trace,
                nk_zero_variance, nk_mc_vs_fullsum_spin, nk_mc_vs_fullsum_fermion, nk_conserving_operator_matches, nk_hubbard_ed]


BIG_EXACT = ["Heisenberg/pyrochlore-2x2x2_32_P"]     # exact rows checked with tests/ed_sz.py (--big)
HONEYCOMB_4X4 = {0.0: None, 0.2: None}              # 32-site honeycomb torus: verifier vs independent bonds (--big)
TJ_BIG = ["tJ/ladder-2x11_22_P_h2_J0.5", "tJ/honeycomb-flake24_24_O_h1_J0.333333",   # t-J exact rows with
          "tJ/honeycomb-flake24_24_O_h2_J0.333333"]   # tests/ed_tj.py (--big): 4e7 to 2e8 states, 10 to 60 min each


def big_exact_rows():
    """Exact rows of 24- to 32-site clusters with the matrix-free S^z-sector Lanczos of tests/ed_sz.py on the
    verifier's own bond lists (hamiltonians.spin_lattice); minutes to an hour each on 32 cores."""
    import ed_sz
    for iid in BIG_EXACT:
        r = ed_sz.ground_energy(*ed_sz.instance_bonds(I.load(iid)), log=None)
        ref, rel = check_exact_row(iid, r["energy"])
        print(f"  {iid}: Lanczos {r['energy']:.12f} (dim {r['dim']}, {r['iterations']} steps, {r['wall_s']:.0f} s) "
              f"vs QMBL {ref} (rel {rel:.1e})", flush=True)
    import ed_tj
    for iid in TJ_BIG:
        r = ed_tj.ground_energy(*ed_tj.instance_problem(I.load(iid)), log=None)
        ref, rel = check_exact_row(iid, r["energy"])
        print(f"  {iid}: Lanczos {r['energy']:.12f} (dim {r['dim']}) vs QMBL {ref} (rel {rel:.1e})", flush=True)
    from qrec_verify import score as SC
    for J2 in HONEYCOMB_4X4:            # the 32-site honeycomb torus: no QMBL row, so an independent bond list
        inst, _ = SC.small_system(f"honeycomb:4x4:{J2}")
        r = ed_sz.ground_energy(*ed_sz.instance_bonds(inst), log=None)
        _, nn, nnn = honeycomb_reference(4, 4)
        bonds = [(*sorted(b), 1.0) for b in nn] + ([(*sorted(b), J2) for b in nnn] if J2 else [])
        r2 = ed_sz.ground_energy(32, bonds, log=None, seed=1)
        assert abs(r["energy"] - r2["energy"]) < 1e-9 * abs(r2["energy"]), (J2, r, r2)
        print(f"  honeycomb 4x4 J2={J2}: verifier bonds {r['energy']:.12f} ({r['energy'] / 32:.8f} per site), "
              f"independent bonds {r2['energy']:.12f}", flush=True)


if __name__ == "__main__":
    failed = 0
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    if "--netket" in sys.argv:
        import jax
        jax.config.update("jax_enable_x64", True)
        tests += NETKET_TESTS
    if "--big" in sys.argv:
        sys.path.insert(0, str(ROOT / "tests"))
        tests += [big_exact_rows, big_chem_rows]
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1 if failed else 0)
