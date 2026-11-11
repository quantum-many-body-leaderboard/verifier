"""Hydrogen chains and sheets in a Gaussian basis (v1.9): QMBL track 3 instances whose `basis` is not "none".

QMBL's Hamiltonian (RULES.md 13.1-13.2) projected onto the instance's basis set, in Hartree, with the nuclear
repulsion included:

    H = E_nuc + sum_{pq, s} h_pq c+_ps c_qs + 1/2 sum_{pqrs, s s'} (pq|rs) c+_ps c+_rs' c_ss' c_qs

Geometry (13.1, as qrec-verifier's continuum path): `chain` = N protons at (0, 0, I R), I = 0 .. N-1;
`sheet` (model Hsheet) = L x L protons at (i R, j R, 0), atom I = i L + j. R in bohr. Neutral, N_up / N_dn from the
instance (S_z = 0 for even N). Open boundary only (a finite molecule).

Orbitals: the Loewdin-orthogonalised atomic orbitals, phi_p = sum_mu chi_mu (S^-1/2)_mu,p, with chi_mu pyscf's
atomic orbitals of the basis in pyscf's order (atom by atom along the chain; within an atom by shell, as
`pyscf.gto.Mole.ao_labels()` lists them; cartesian = False). They are fixed by the geometry and the basis alone (no
self-consistent field, no phase or degeneracy choices), local, and orthonormal; a state written in any other
orbital basis must be transformed to these. The basis set is pyscf's of the same name ('STO-6G', 'cc-pVDZ', ...).

Configurations: NetKet's SpinOrbitalFermions(n_orb, s=1/2, n_fermions_per_spin=(N_dn, N_up)), i.e. 2 n_orb
occupation numbers, the n_orb down orbitals first, then the n_orb up orbitals, as for the Hubbard model; the
Hamiltonian is NetKet's ParticleNumberAndSpinConservingFermioperator2nd of pyscf's integrals in these orbitals
(NetKet's `from_pyscf_molecule` with the Loewdin coefficients), in NetKet's fermion sign convention.

Size: the operator stores the two-electron integrals, n_orb^4 numbers before the cutoff; `MAX_ORBITALS` bounds what
the verifier builds (above it, it refuses with the estimate). Checks (tests/test_physics.py, nk_chem and big_chem_rows in --big): pyscf's FCI in the
canonical RHF basis equals FCI with these integrals (basis invariance) and the ED of the NetKet operator; QMBL's
exact (FCI) rows.
"""

from __future__ import annotations

import math
import re

import numpy as np

MAX_ORBITALS = 128           # n_orb^4 doubles: 2.1 GB of two-electron integrals at 128
CUTOFF = 1e-12               # integrals below this (absolute, hartree) are dropped from the operator


def is_chem(inst: dict) -> bool:
    """A hydrogen chain or sheet in a Gaussian basis (basis not 'none')."""
    return inst.get("model") in ("Hchain", "Hsheet") and str((inst.get("params") or {}).get("basis", "none")).lower() != "none"


def atoms(inst: dict) -> np.ndarray:
    n, R = int(inst["n_sites"]), float(inst["params"]["R"])
    if inst["model"] == "Hchain" and inst["lattice"] == "chain":
        return np.array([[0.0, 0.0, i * R] for i in range(n)])
    if inst["model"] == "Hsheet" and inst["lattice"] == "square":
        L = int(round(math.sqrt(n)))
        if L * L != n:
            raise ValueError(f"a square sheet needs N = L x L, got {n}")
        return np.array([[i * R, j * R, 0.0] for i in range(L) for j in range(L)])
    raise ValueError(f"unsupported geometry {inst['model']}/{inst['lattice']}")


def params(inst: dict) -> dict:
    if not is_chem(inst):
        raise ValueError(f"{inst.get('instance_id')}: not a hydrogen chain or sheet in a Gaussian basis")
    if inst.get("boundary", "O") != "O":
        raise ValueError("hydrogen in a Gaussian basis: only open (finite) chains and sheets")
    p = inst["params"]
    n = int(inst["n_sites"])
    nu, nd = int(p.get("N_up", (n + 1) // 2)), int(p.get("N_dn", n // 2))
    if nu + nd != n:
        raise ValueError(f"N_up + N_dn = {nu + nd}, the neutral molecule has {n} electrons")
    return {"basis": str(p["basis"]), "R": float(p["R"]), "n_up": nu, "n_dn": nd}


def molecule(inst: dict):
    from pyscf import gto
    p = params(inst)
    return gto.M(atom=[("H", tuple(a)) for a in atoms(inst)], unit="Bohr", basis=p["basis"],
                 spin=p["n_up"] - p["n_dn"], charge=0, cart=False, verbose=0)


def n_orbitals(inst: dict) -> int:
    return int(molecule(inst).nao)


def loewdin(S: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(S)
    if w.min() < 1e-10:
        raise ValueError(f"the basis is nearly linearly dependent (smallest overlap eigenvalue {w.min():.1e})")
    return (U / np.sqrt(w)) @ U.T


def integrals(inst: dict, max_orbitals: int = MAX_ORBITALS) -> dict:
    """E_nuc, h (n, n) and the chemists' (pq|rs) (n, n, n, n) in the Loewdin orbitals."""
    from pyscf import ao2mo
    mol = molecule(inst)
    n = int(mol.nao)
    if n > max_orbitals:
        raise ValueError(f"{inst['instance_id']}: {n} orbitals; the verifier builds at most {max_orbitals} "
                         f"(the two-electron integrals alone are {8 * n ** 4 / 1e9:.1f} GB)")
    X = loewdin(mol.intor("int1e_ovlp"))
    h = X.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ X
    eri = ao2mo.restore(1, ao2mo.kernel(mol, X, compact=True), n)
    # exact index symmetry (h_pq = h_qp, the 8-fold symmetry of real (pq|rs)): NetKet's sparse operator builder
    # pairs every element with its partner and fails on rounding-level asymmetry
    h = 0.5 * (h + h.T)
    eri = (eri + eri.transpose(1, 0, 2, 3) + eri.transpose(0, 1, 3, 2) + eri.transpose(1, 0, 3, 2)
           + eri.transpose(2, 3, 0, 1) + eri.transpose(3, 2, 0, 1) + eri.transpose(2, 3, 1, 0)
           + eri.transpose(3, 2, 1, 0)) / 8.0
    return {"e_nuc": float(mol.energy_nuc()), "h": h, "eri": eri, "n_orb": n, "X": X, "mol": mol,
            "ao_labels": mol.ao_labels()}


def hilbert(n_orb: int, n_up: int, n_dn: int):
    import netket as nk
    return nk.hilbert.SpinOrbitalFermions(n_orbitals=n_orb, s=0.5, n_fermions_per_spin=(n_dn, n_up))


def operator(mol, X: np.ndarray, cutoff: float = CUTOFF):
    """NetKet's ParticleNumberAndSpinConservingFermioperator2nd of the Hamiltonian in the orbitals X (columns),
    from pyscf's atomic-orbital integrals transformed by NetKet (`from_pyscf_molecule`): E_nuc + sum h_pq c+_p c_q
    + 1/2 sum (pq|rs) c+_p c+_r c_s c_q, spin summed. The tests compare it with the integrals transformed by pyscf
    (`integrals`, pyscf's FCI on them) and with pyscf's FCI in canonical orbitals."""
    from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd as PNS
    return PNS.from_pyscf_molecule(mol, X, cutoff=cutoff)


def build(inst: dict):
    """(hilbert, H, graph over the orbitals, integrals dict, params) of a Gaussian-basis instance."""
    import netket as nk
    p = params(inst)
    ints = integrals(inst)
    n = ints["n_orb"]
    if p["n_up"] != p["n_dn"]:
        raise ValueError("hydrogen in a Gaussian basis: only N_up = N_dn (every open instance has an even N)")
    hi = hilbert(n, p["n_up"], p["n_dn"])
    H = operator(ints["mol"], ints["X"])
    if H.hilbert != hi:
        raise RuntimeError("NetKet's molecular Hilbert space differs from the verifier's")
    # hops for the sampler: every pair of orbitals whose one-electron element is not negligible
    edges = [[a, b] for a in range(n) for b in range(a + 1, n) if abs(ints["h"][a, b]) > 1e-6]
    g = nk.graph.Graph(edges=edges, n_nodes=n)
    return hi, H, g, ints, p


def fci_energy(inst: dict, ints: dict | None = None) -> dict:
    """pyscf's FCI with the verifier's integrals (direct_spin1 on h, (pq|rs)), and pyscf's own FCI from its RHF
    (canonical orbitals): the two must agree, since the FCI energy does not depend on the orbital basis. pyscf's
    default 100 Davidson steps do not converge the Loewdin-basis FCI from 12 orbitals on (an error of 1.6e-7 Ha
    on H12 STO-3G), so both solvers get 1000 and an unconverged solve raises."""
    from pyscf import fci, scf
    p = params(inst)
    ints = ints or integrals(inst)
    n = ints["n_orb"]
    cis = fci.direct_spin1.FCI()
    cis.conv_tol, cis.max_cycle = 1e-12, 1000
    e_l, _ = cis.kernel(ints["h"], ints["eri"], n, (p["n_up"], p["n_dn"]), ecore=ints["e_nuc"])
    mol = ints["mol"]
    mf = (scf.RHF(mol) if p["n_up"] == p["n_dn"] else scf.ROHF(mol)).run(conv_tol=1e-12)
    c2 = fci.FCI(mf)
    c2.conv_tol, c2.max_cycle = 1e-12, 1000
    e_c, _ = c2.kernel()
    if not (np.all(cis.converged) and np.all(c2.converged)):
        raise RuntimeError(f"{inst.get('instance_id')}: pyscf FCI did not converge in 1000 Davidson steps")
    return {"fci_loewdin": float(e_l), "fci_canonical": float(e_c), "rhf": float(mf.e_tot)}
