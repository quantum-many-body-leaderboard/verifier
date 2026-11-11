"""Anderson impurity models of QMBL's Impurity instances (v1.9): VarBench's impurity Hamiltonians.

The parameters are VarBench's files `HamParams/<model>_<Nb>.h5` (varbench/varbench at 390a21e, Apache-2.0, vendored in
`instances/Impurity/HamParams/`): the local one-particle matrix e0 (hloc) per block, the bath (energy eps and
hybridisation vector V per bath site, per impurity orbital and block) and the interaction U, J, Up.

    H = sum_{s, a b} hloc^s_ab d+_as d_bs + sum_{s, a, l} eps^{s a}_l c+_als c_als
        + sum_{s, a, l, b} (V^{s a l}_b d+_bs c_als + h.c.) + H_int

H_int, single band: U n_up n_dn. Three bands, the rotationally invariant Kanamori form as forkTPS (the code behind
every row) builds it, Up = U - 2J:

    U sum_m n_m,up n_m,dn + Up sum_{m != m'} n_m,up n_m',dn + (Up - J) sum_{m < m', s} n_ms n_m's
    - J sum_{m != m'} d+_m,up d_m,dn d+_m',dn d_m',up + J sum_{m != m'} d+_m,up d+_m,dn d_m',dn d_m',up

(the signs of the spin-flip and pair-hopping terms are fixed by forkTPS's own atomic-limit test, test/c++/FTPO_impl.cpp
AIM.atomicLimit_SFPH, which tests/test_impurity.py reproduces). The blocks are the spins ('up'/'dn' or 'up'/'down');
with spin-orbit coupling ('ud_0'/'ud_1') they are forkTPS's pseudo-spin blocks, and the interaction is written in them
as for spins. That last step is not confirmed: DMRG of the spin-orbit Hamiltonian built this way goes below VarBench's
converged forkTPS energies, so `build` refuses the spin-orbit models (SOC_UNCONFIRMED; COVERAGE.md has the plan).

Orbitals: per block, the M impurity orbitals first, then the bath orbitals (a, l) for a = 0 .. M-1, l = 0 .. Nb-1;
NetKet's SpinOrbitalFermions(M (1 + Nb), s=1/2) with the first block ('dn', 'down', 'ud_0') as S_z = -1/2 and the
second ('up', 'ud_1') as +1/2. The particle sector: N = the rows' dof (forkTPS's ground-state particle number), (N_dn, N_up) =
(N // 2, N - N // 2) (`sector`). The rows' einf are VarBench's and are not the trace of this Hamiltonian in any
sector (nor over the Fock space); the verifier does not use them. QMBL gives impurity energies as totals: the
verifier's per-site divisor is 1.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

HAMPARAMS = pathlib.Path(__file__).resolve().parent.parent / "instances" / "Impurity" / "HamParams"
FIRST, SECOND = ("dn", "down", "ud_0"), ("up", "ud_1")



def file_for(inst: dict) -> pathlib.Path:
    """VarBench's parameter file of an instance. QMBL's SB-DMFT-MI-HF_10 was run on SB-DMFT-MI_9.h5 (its relabel
    note: the script discretises with Nb + 1 = 10 bath sites); every other id maps to <lattice>_<Nb>.h5."""
    lat, nb = inst["lattice"], int(inst["n_sites"])
    if lat == "SB-DMFT-MI-HF":
        return HAMPARAMS / "SB-DMFT-MI_9.h5"
    return HAMPARAMS / f"{lat}_{nb}.h5"


def read(path) -> dict:
    import h5py

    def cplx(a):
        a = np.asarray(a)
        return a[..., 0] + 1j * a[..., 1]

    with h5py.File(path, "r") as f:
        blocks, M = [], None
        for k in sorted(f["e0/gf_struct"], key=int):
            name = f[f"e0/gf_struct/{k}/0"][()]
            blocks.append(name.decode() if isinstance(name, bytes) else str(name))
            M = int(f[f"e0/gf_struct/{k}/1"][()])
        hloc = {b: cplx(f[f"e0/hloc/{b}"][()]) for b in blocks}
        bath = {}
        for b in blocks:
            g = f[f"bath/bath/{b}"]
            per = []
            for a in sorted(g, key=int):
                sites = []
                for l in sorted(g[a], key=int):
                    sites.append((float(g[a][l]["eps"][()]), cplx(g[a][l]["V"][()])))
                per.append(sites)
            bath[b] = per
        U = float(f["U"][()])
        J = float(f["J"][()]) if "J" in f else 0.0
        Up = float(f["Up"][()]) if "Up" in f else U - 2 * J
        soc = bool(f["e0/SpinOrbit"][()])
    order = [b for b in blocks if b in FIRST] + [b for b in blocks if b in SECOND]
    if len(order) != 2:
        raise ValueError(f"{path}: blocks {blocks}")
    nb = len(bath[order[0]][0])
    return {"blocks": order, "M": M, "Nb": nb, "hloc": hloc, "bath": bath, "U": U, "J": J, "Up": Up, "soc": soc}


def orbital(par: dict, kind: str, a: int, l: int = 0) -> int:
    """Spatial orbital index within a block: impurity orbital a, or bath orbital (a, l)."""
    M, nb = par["M"], par["Nb"]
    return a if kind == "imp" else M + a * nb + l


def one_body(par: dict) -> dict:
    """{block: (n, n) Hermitian one-particle matrix} over the block's M (1 + Nb) orbitals."""
    M, nb = par["M"], par["Nb"]
    n = M * (1 + nb)
    out = {}
    for b in par["blocks"]:
        T = np.zeros((n, n), dtype=complex)
        T[:M, :M] = par["hloc"][b]
        for a in range(M):
            for l, (eps, V) in enumerate(par["bath"][b][a]):
                k = orbital(par, "bath", a, l)
                T[k, k] += eps
                for bb in range(M):
                    T[bb, k] += V[bb]                # V_b d+_b c_(a l)
                    T[k, bb] += np.conj(V[bb])       # h.c.
        if not np.allclose(T, T.conj().T, atol=1e-12):
            raise ValueError(f"block {b}: hloc is not Hermitian")
        out[b] = T
    return out


def terms(par: dict):
    """FermionOperator2nd terms and weights (orbital of block index s (0 = first, S_z -1/2) and spatial k: s n + k)."""
    M, nb = par["M"], par["Nb"]
    n = M * (1 + nb)
    T = one_body(par)
    dn, up = (lambda k: k), (lambda k: n + k)
    tm, w = [], []
    for s, b in enumerate(par["blocks"]):
        o = dn if s == 0 else up
        for i in range(n):
            for j in range(n):
                if abs(T[b][i, j]) > 1e-14:
                    tm.append(((o(i), 1), (o(j), 0)))
                    w.append(T[b][i, j])
    U, J, Up = par["U"], par["J"], par["Up"]
    for m in range(M):
        tm.append(((up(m), 1), (up(m), 0), (dn(m), 1), (dn(m), 0)))
        w.append(U)
    for m in range(M):
        for mp in range(M):
            if m == mp:
                continue
            tm.append(((up(m), 1), (up(m), 0), (dn(mp), 1), (dn(mp), 0)))
            w.append(Up)
            if mp > m:
                for o in (up, dn):
                    tm.append(((o(m), 1), (o(m), 0), (o(mp), 1), (o(mp), 0)))
                    w.append(Up - J)
            if J:
                tm.append(((up(m), 1), (dn(m), 0), (dn(mp), 1), (up(mp), 0)))      # spin flip
                w.append(-J)
                tm.append(((up(m), 1), (dn(m), 1), (dn(mp), 0), (up(mp), 0)))      # pair hopping
                w.append(J)
    return n, tm, w


def n_electrons(inst: dict) -> int:
    """The electron number of the rows: QMBL's (VarBench's) dof. forkTPS minimises over the particle sectors, so this
    is the ground state's."""
    dofs = {r.get("dof") for r in inst["rows"] if r.get("dof") is not None}
    if len(dofs) != 1:
        raise ValueError(f"{inst['instance_id']}: the rows state no single dof ({dofs})")
    return int(dofs.pop())


def sector(inst: dict) -> tuple[int, int]:
    """(N_dn, N_up) = (N // 2, N - N // 2): the two blocks' electron numbers. The Hamiltonian conserves each
    (spin, or forkTPS's pseudo-spin block with spin-orbit coupling); the DMRG cross-check (tests/xcheck_impurity.json,
    S_z free) finds every ground state at block imbalance 0 (even N), which is this sector."""
    N = n_electrons(inst)
    return N // 2, N - N // 2


SOC_UNCONFIRMED = ("the spin-orbit impurity models (TB-DMFT-SOC) are refused: DMRG of this verifier's Hamiltonian "
                   "at VarBench's particle number reaches energies below VarBench's converged forkTPS rows "
                   "(TB-DMFT-SOC_9: -112.87198 vs -112.87177, both hybridisation conventions), so it is not yet "
                   "forkTPS's Hamiltonian (COVERAGE.md)")


def build(inst: dict, jax_operator: bool = True, allow_soc: bool = False):
    """(hilbert, H, graph over the orbitals, params) of an Impurity instance. The spin-orbit models are refused
    (SOC_UNCONFIRMED) unless allow_soc (tests only)."""
    import netket as nk
    par = read(file_for(inst))
    if par["soc"] and not allow_soc:
        raise ValueError(f"{inst['instance_id']}: {SOC_UNCONFIRMED}")
    n, tm, w = terms(par)
    nd, nu = sector(inst)
    hi = nk.hilbert.SpinOrbitalFermions(n_orbitals=n, s=0.5, n_fermions_per_spin=(nd, nu))
    w = np.array(w, dtype=complex)
    dtype = complex if np.any(np.abs(w.imag) > 0) else float
    if dtype is float:
        w = w.real
    H = nk.operator.FermionOperator2nd(hi, terms=tm, weights=w, dtype=dtype)
    if jax_operator:
        from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd as PNS
        H = PNS.from_fermionoperator2nd(H)
    M, nb = par["M"], par["Nb"]
    edges = [[a, M + a2 * nb + l] for a in range(M) for a2 in range(M) for l in range(nb)
             if a == a2 or par["soc"]] + [[a, b] for a in range(M) for b in range(a + 1, M)]
    g = nk.graph.Graph(edges=edges, n_nodes=n)
    return hi, H, g, dict(par, n_orbitals=n, n_electrons=nd + nu, n_dn=nd, n_up=nu)
