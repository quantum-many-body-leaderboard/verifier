"""Exact evaluation of a matrix product state (MPS) submitted as a file (format: MPS_FORMAT.md).

The verifier reads the tensors, builds the instance's Hamiltonian as a matrix product operator (MPO)
itself, from its own bond lists (`hamiltonians.square_bonds`) and in the file's site order, with
Jordan-Wigner strings for fermions, and contracts <psi|H|psi> / <psi|psi> exactly. No energy, operator
or Hamiltonian from the submitter enters the result. Further gates: the state's particle numbers (or
total S^z) must be those of the instance's sector, a stated energy must be reproduced, and a sample
drawn from the MPS (perfect sampling) must give the same energy with NetKet's Hamiltonian
(`hamiltonians.build`), which uses its own fermion ordering. numpy and h5py only, except the sampling
check, which needs NetKet.

Conventions of the MPO (MPS_FORMAT.md states them for submitters):
- site k of the chain is lattice site order[k], i = x * ly + y;
- spin-half: local index 0 = up (S^z = +1/2, NetKet's +1), 1 = down;
- electron: local index n_up + 2 n_dn, i.e. [empty, up, down, up+down], with
  |up+down> = c+_up c+_dn |0>; a Fock state of the chain is
  c+_(0,up)^n c+_(0,dn)^n c+_(1,up)^n ... |0>, the creation operators ordered by chain position and,
  within a site, up before down.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import instance as I
from .hamiltonians import square_bonds

FORMAT = "qmbl-mps"
FORMAT_VERSION = 1
BASES = {"spin-half": 2, "electron": 4}
LIMITS = {"max_gb": 48.0,          # peak memory of the contraction
          "max_tflop": 2000.0,     # floating-point operations of <H>, in 1e12
          "max_gb_variance": 24.0,  # <H^2> needs chi^2 D^2; above this it is skipped (not a gate)
          "max_gb_sampling": 8.0}   # the sampling cross-check holds all tensors and right environments
SECTOR_TOL = 1e-8                  # <(N - N0)^2>, summed over the conserved charges
CLAIM_RTOL = 1e-8                  # stated energy vs the verifier's, relative
SAMPLING_Z = 4.0                   # sampled mean vs exact energy, in standard errors


def basis_for(inst: dict) -> str:
    if inst["model"] in ("J1J2", "Heisenberg"):
        return "spin-half"
    if inst["model"] == "Hubbard":
        return "electron"
    raise ValueError(f"model {inst['model']} has no MPS route yet")


# ------------------------------------------------------------------ file

@dataclass
class MPSFile:
    path: str
    sha256: str
    basis: str
    n_sites: int
    order: np.ndarray                 # order[k] = lattice site of chain position k
    bond_dims: list                   # chi_0 = 1, ..., chi_N = 1
    dtype: str
    attrs: dict = field(default_factory=dict)

    def tensor(self, k: int) -> np.ndarray:
        import h5py
        with h5py.File(self.path, "r") as f:
            return _as_working(f["tensors"][str(k)][()])

    def tensors(self) -> list:
        import h5py
        with h5py.File(self.path, "r") as f:
            return [_as_working(f["tensors"][str(k)][()]) for k in range(self.n_sites)]

    @property
    def chi_max(self) -> int:
        return int(max(self.bond_dims))


def _as_working(a: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(a):
        return np.ascontiguousarray(a, dtype=np.complex128)
    return np.ascontiguousarray(a, dtype=np.float64)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def load(path: str, inst: dict | None = None) -> MPSFile:
    """Open and validate a file; the tensors stay on disk (read site by site). Raises ValueError
    with the reason when the file does not follow MPS_FORMAT.md or does not fit the instance."""
    import h5py
    with h5py.File(path, "r") as f:
        attrs = {k: _attr(v) for k, v in f.attrs.items()}
        if attrs.get("format") != FORMAT:
            raise ValueError(f"attribute format must be {FORMAT!r}, found {attrs.get('format')!r}")
        if int(attrs.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"format_version must be {FORMAT_VERSION}")
        basis = attrs.get("local_basis")
        if basis not in BASES:
            raise ValueError(f"local_basis must be one of {sorted(BASES)}, found {basis!r}")
        d = BASES[basis]
        if "site_order" not in f or "tensors" not in f:
            raise ValueError("the file needs a dataset 'site_order' and a group 'tensors'")
        order = np.asarray(f["site_order"][()]).astype(np.int64).ravel()
        n = len(order)
        if int(attrs.get("n_sites", n)) != n:
            raise ValueError("n_sites does not match the length of site_order")
        if sorted(order.tolist()) != list(range(n)):
            raise ValueError("site_order must be a permutation of 0..n_sites-1")
        g = f["tensors"]
        names = set(g.keys())
        if names != {str(k) for k in range(n)}:
            raise ValueError(f"group 'tensors' must hold datasets '0'..'{n - 1}'")
        dims, dtypes = [1], set()
        for k in range(n):
            ds = g[str(k)]
            if len(ds.shape) != 3:
                raise ValueError(f"tensor {k}: shape {ds.shape}, expected (left bond, physical, right bond)")
            cl, dk, cr = ds.shape
            if dk != d:
                raise ValueError(f"tensor {k}: physical dimension {dk}, {basis} needs {d}")
            if cl != dims[-1]:
                raise ValueError(f"tensor {k}: left bond {cl} does not match the right bond {dims[-1]} of tensor {k - 1}")
            dims.append(cr)
            kind = ds.dtype.kind
            if kind not in "fc" or ds.dtype.itemsize not in (4, 8, 16):
                raise ValueError(f"tensor {k}: dtype {ds.dtype}; use float64 or complex128")
            dtypes.add(str(ds.dtype))
        if dims[-1] != 1:
            raise ValueError(f"the last tensor's right bond is {dims[-1]}, must be 1")
    if inst is not None:
        want = basis_for(inst)
        if basis != want:
            raise ValueError(f"instance {inst['instance_id']} needs local_basis {want!r}, file has {basis!r}")
        if n != int(inst["n_sites"]):
            raise ValueError(f"file has {n} sites, instance {inst['instance_id']} has {inst['n_sites']}")
        iid = attrs.get("instance_id")
        if iid and iid != inst["instance_id"]:
            raise ValueError(f"file is for instance {iid!r}, not {inst['instance_id']!r}")
    dtype = "complex128" if any("complex" in t for t in dtypes) else "float64"
    return MPSFile(str(path), file_sha256(path), basis, n, order, dims, dtype, attrs)


# ------------------------------------------------------------------ local operators

def local_ops(basis: str) -> dict:
    """Matrices O[out, in] in the file's local basis."""
    if basis == "spin-half":
        sz = np.diag([1.0, -1.0])
        sp = np.array([[0.0, 1.0], [0.0, 0.0]])        # |up><down|
        return {"I": np.eye(2), "F": np.eye(2), "z": sz, "+": sp, "-": sp.T.copy(), "Sz": sz / 2}
    a_up = np.zeros((4, 4))
    a_up[0, 1] = a_up[2, 3] = 1.0                       # |0><up|, |dn><updn|
    a_dn = np.zeros((4, 4))
    a_dn[0, 2] = a_dn[1, 3] = 1.0                       # |0><dn|, |up><updn|
    p_up, p_dn = np.diag([1.0, -1.0, 1.0, -1.0]), np.diag([1.0, 1.0, -1.0, -1.0])
    n_up, n_dn = np.diag([0.0, 1.0, 0.0, 1.0]), np.diag([0.0, 0.0, 1.0, 1.0])
    return {"I": np.eye(4), "F": p_up @ p_dn, "a_up": a_up, "a_dn": a_dn, "p_up": p_up, "p_dn": p_dn,
            "n_up": n_up, "n_dn": n_dn, "n": n_up + n_dn, "nn": n_up @ n_dn}


# ------------------------------------------------------------------ Hamiltonian as an MPO

@dataclass
class MPO:
    W: list                 # W[k][a, b] = operator (d x d, [out, in]); shape (Dl, Dr, d, d)
    start: int = 0          # left boundary picks row `start` of W[0]
    done: int = 1           # <H> is the `done` component at the right end, <psi|psi> the `start` one
    description: str = ""

    @property
    def bond_dims(self) -> list:
        return [w.shape[0] for w in self.W] + [self.W[-1].shape[1]]


def hamiltonian_terms(inst: dict, overrides: dict | None = None) -> tuple[str, list, list]:
    """The instance's Hamiltonian as (basis, onsite, pairs) on lattice sites, from the verifier's
    bond lists. onsite: (i, op name, coef); pairs: (i, j, kind, coef), kind 'heis' (Pauli s_i.s_j),
    'hop' (sum_sigma c+_i c_j + h.c.) or 'nn' (n_i n_j)."""
    lx, ly = I.lattice_dims(inst)
    axes = I.boundary_axes(inst)
    nn, nnn = square_bonds(lx, ly, axes)
    model = inst["model"]
    overrides = overrides or {}
    if model in ("J1J2", "Heisenberg"):
        if any(s != 1 for *_, s in nn + nnn):
            raise ValueError("antiperiodic boundaries are not defined for the spin models here")
        J2 = float(inst["params"].get("J2", 0.0)) if model == "J1J2" else 0.0
        J2 = overrides.get("J2", J2)
        pairs = [(i, j, "heis", 1.0) for i, j, _ in nn]
        if J2 != 0:
            pairs += [(i, j, "heis", J2) for i, j, _ in nnn]
        return "spin-half", [], pairs
    if model == "Hubbard":
        hp = I.hubbard_params(inst)
        hp.update(overrides)
        pairs = []
        for bonds, t in ((nn, hp["t1"]), (nnn, hp["t2"])):
            if t != 0:
                pairs += [(i, j, "hop", -t * s) for i, j, s in bonds]     # hamiltonians.hubbard: -t s_b
        for bonds, V in ((nn, hp["V1"]), (nnn, hp["V2"])):
            if V != 0:
                pairs += [(i, j, "nn", V) for i, j, _ in bonds]
        onsite = [(i, "nn", hp["U"]) for i in range(lx * ly)] if hp["U"] != 0 else []
        return "electron", onsite, pairs
    raise ValueError(f"model {model} has no MPS route yet")


# Pair terms as sums of (left operator carried with a string, right operator, factor). With modes
# ordered (k, up), (k, dn), (k+1, up), ... and p < q chain positions:
#   c+_(p,up) c_(q,up) = [a+_up P_dn]_p F...F [a_up]_q        c+_(q,up) c_(p,up) = [a_up P_dn]_p F..F [a+_up]_q
#   c+_(p,dn) c_(q,dn) = [a+_dn]_p F...F [P_up a_dn]_q        c+_(q,dn) c_(p,dn) = [a_dn]_p F..F [P_up a+_dn]_q
# with F = P_up P_dn the site parity on every site strictly between p and q.
def _pair_channels(basis: str, ops: dict) -> dict:
    if basis == "spin-half":
        return {"heis": [("z", ops["z"], ops["z"], 1.0, False), ("+", ops["+"], ops["-"], 2.0, False),
                         ("-", ops["-"], ops["+"], 2.0, False)]}
    au, ad, pu, pd = ops["a_up"], ops["a_dn"], ops["p_up"], ops["p_dn"]
    return {"hop": [("cu+", au.T @ pd, au, 1.0, True), ("cu-", au @ pd, au.T, 1.0, True),
                    ("cd+", ad.T, pu @ ad, 1.0, True), ("cd-", ad, pu @ ad.T, 1.0, True)],
            "nn": [("n", ops["n"], ops["n"], 1.0, False)]}


def build_mpo(inst: dict, order, overrides: dict | None = None) -> MPO:
    """Finite-state-machine MPO of the instance's Hamiltonian along the chain order[k]. Channels at a
    cut are (left site, operator) pairs with a partner beyond the cut; D = 2 + channels."""
    basis, onsite, pairs = hamiltonian_terms(inst, overrides)
    ops = local_ops(basis)
    d = BASES[basis]
    order = np.asarray(order)
    n = len(order)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    chans = _pair_channels(basis, ops)
    # closing[(p, name)] = list of (q, coef, right op); a channel opened at p with partner q > p
    closing: dict = {}
    left_op, fermionic = {}, {}
    for i, j, kind, coef in pairs:
        p, q = sorted((int(pos[i]), int(pos[j])))
        if p == q:
            raise ValueError("a bond joins a site to itself")
        for name, lop, rop, fac, ferm in chans[kind]:
            key = (p, name)
            closing.setdefault(key, []).append((q, coef * fac, rop))
            left_op[key], fermionic[key] = lop, ferm
    onsite_at = [np.zeros((d, d)) for _ in range(n)]
    for i, name, coef in onsite:
        onsite_at[int(pos[i])] += coef * ops[name]
    # open channels on the bond right of position k
    last = {key: max(q for q, *_ in v) for key, v in closing.items()}
    cut = [[key for key in sorted(closing) if key[0] <= k < last[key]] for k in range(n)]
    W = []
    for k in range(n):
        lch = cut[k - 1] if k > 0 else []
        rch = cut[k]
        li = {key: 2 + a for a, key in enumerate(lch)}
        ri = {key: 2 + a for a, key in enumerate(rch)}
        w = np.zeros((2 + len(lch), 2 + len(rch), d, d))
        w[0, 0] = ops["I"]
        w[1, 1] = ops["I"]
        w[0, 1] = onsite_at[k]
        for key in rch:
            if key[0] == k:
                w[0, ri[key]] = left_op[key]
        for key in lch:
            if key in ri:
                w[li[key], ri[key]] = ops["F"] if fermionic[key] else ops["I"]
            for q, c, rop in closing[key]:
                if q == k:
                    w[li[key], 1] += c * rop
        W.append(w)
    # first and last tensors: boundary rows / columns kept (row 0 at the left; columns 0 and 1 at
    # the right give <psi|psi> and <psi|H|psi> in one pass)
    W[0] = W[0][:1]
    W[-1] = W[-1][:, :2]
    return MPO(W, 0, 1, f"{inst['instance_id']}: {len(pairs)} pair terms, {len(onsite)} on-site terms, "
                        f"max D {max(w.shape[1] for w in W)}")


def charge_mpo(basis: str, n: int, targets: dict) -> MPO:
    """MPO of sum_c (Q_c - target_c)^2 for the conserved charges (N_up, N_dn; or total S^z)."""
    ops = local_ops(basis)
    names = ("n_up", "n_dn") if basis == "electron" else ("Sz",)
    d = BASES[basis]
    W = []
    for k in range(n):
        w = np.zeros((2 + len(names), 2 + len(names), d, d))
        w[0, 0] = w[1, 1] = ops["I"]
        for a, name in enumerate(names):
            b = ops[name] - targets[name] / n * ops["I"]
            w[0, 2 + a] = b
            w[2 + a, 2 + a] = ops["I"]
            w[2 + a, 1] = 2.0 * b
            w[0, 1] += b @ b
        W.append(w)
    W[0] = W[0][:1]
    W[-1] = W[-1][:, :2]
    return MPO(W, 0, 1, "charge variance")


def sector_targets(inst: dict) -> dict:
    if basis_for(inst) == "electron":
        hp = I.hubbard_params(inst)
        return {"n_up": hp["n_up"], "n_dn": hp["n_dn"]}
    return {"Sz": 0.0}


# ------------------------------------------------------------------ contraction

def _env_step(L: np.ndarray, A: np.ndarray, W: np.ndarray, chunk: int) -> np.ndarray:
    """L(a, w, b) -> L'(a', w', b'): bra index a (conjugated), ket index b. Chunked over b'."""
    cl, d, cr = A.shape
    Ac = A.conj()
    Wt = W.transpose(0, 3, 1, 2)                       # (w, t, w', s)
    out = np.empty((cr, W.shape[1], cr), dtype=np.result_type(L, A, W))   # (a', w', b')
    for b0 in range(0, cr, chunk):
        Ab = A[:, :, b0:b0 + chunk]                    # (b, t, b'c)
        T1 = np.tensordot(L, Ab, axes=(2, 0))          # (a, w, t, b'c)
        T2 = np.tensordot(T1, Wt, axes=([1, 2], [0, 1]))   # (a, b'c, w', s)
        del T1
        T3 = np.tensordot(Ac, T2, axes=([0, 1], [0, 3]))   # (a', b'c, w')
        out[:, :, b0:b0 + chunk] = T3.transpose(0, 2, 1)
    return out


def _env_step2(L: np.ndarray, A: np.ndarray, W: np.ndarray, chunk: int) -> np.ndarray:
    """Two MPO layers: L(a, w1, w2, b) -> L'(a', w1', w2', b'), for <H^2>."""
    cl, d, cr = A.shape
    Ac = A.conj()
    Wt = W.transpose(0, 3, 1, 2)                       # (w, t, w', s)
    D = W.shape[1]
    out = np.empty((cr, D, D, cr), dtype=np.result_type(L, A, W))
    for b0 in range(0, cr, chunk):
        Ab = A[:, :, b0:b0 + chunk]
        T1 = np.tensordot(L, Ab, axes=(3, 0))          # (a, w1, w2, t, b'c)
        T2 = np.tensordot(T1, Wt, axes=([2, 3], [0, 1]))   # (a, w1, b'c, w2', u)
        del T1
        T3 = np.tensordot(T2, Wt, axes=([1, 4], [0, 1]))   # (a, b'c, w2', w1', s)
        del T2
        T4 = np.tensordot(Ac, T3, axes=([0, 1], [0, 4]))   # (a', b'c, w2', w1')
        out[:, :, :, b0:b0 + chunk] = T4.transpose(0, 3, 2, 1)
    return out


def _chunk(cl: int, D: int, d: int, budget_bytes: float, layers: int = 1) -> int:
    per_col = 16.0 * cl * (D ** layers) * d * 2
    return int(max(1, min(1 << 30, budget_bytes // max(per_col, 1.0))))


def contract(mps, mpo: MPO, layers: int = 1, chunk_bytes: float = 2e9, log=None) -> dict:
    """<psi|O|psi> and <psi|psi> (layers=1), or <psi|O O|psi> (layers=2), in one sweep. `mps` is an
    MPSFile (tensors read site by site) or a list of arrays. Environments are rescaled at every
    site; the scale is kept as a logarithm, so unnormalised states cannot overflow."""
    n = mps.n_sites if isinstance(mps, MPSFile) else len(mps)
    get = mps.tensor if isinstance(mps, MPSFile) else (lambda k: mps[k])
    L = np.ones((1,) + (1,) * layers + (1,))
    logscale = 0.0
    t0 = time.perf_counter()
    for k in range(n):
        A = get(k)
        W = mpo.W[k]
        if layers == 1:
            L = _env_step(L, A, W, _chunk(A.shape[0], W.shape[1], A.shape[1], chunk_bytes))
        else:
            L = _env_step2(L, A, W, _chunk(A.shape[0], W.shape[1], A.shape[1], chunk_bytes, 2))
        s = float(np.max(np.abs(L)))
        if s == 0 or not math.isfinite(s):
            raise FloatingPointError(f"environment is {s} at site {k}: the state is zero or not finite")
        L /= s
        logscale += math.log(s)
        if log and (k % 8 == 7 or k == n - 1):
            log(f"[mps] contraction layers={layers}: site {k + 1}/{n}, {time.perf_counter() - t0:.1f} s")
    if layers == 1:
        norm, val = L[0, 0, 0], L[0, 1, 0]
    else:
        norm, val = L[0, 0, 0, 0], L[0, 1, 1, 0]
    ratio = val / norm
    return {"value": float(np.real(ratio)), "imag": float(np.imag(ratio)),
            "log_norm": float(np.log(np.abs(norm)) + logscale), "wall_s": time.perf_counter() - t0}


def cost_estimate(bond_dims: list, mpo_dims: list, d: int, itemsize: int = 8, chunk_bytes: float = 2e9) -> dict:
    """Floating-point operations and peak memory of <H> (and of <H^2>). Leading terms per site:
    2 x chi_l^2 chi_r d D (two tensordots) for <H>, x D for <H^2>; memory two environments
    chi^2 D (chi^2 D^2) plus the site tensor and a chunk buffer."""
    flop, flop2, mem, mem2, mpsb = 0.0, 0.0, 0.0, 0.0, 0.0
    for k in range(len(bond_dims) - 1):
        cl, cr = bond_dims[k], bond_dims[k + 1]
        Dl, Dr = mpo_dims[k], mpo_dims[k + 1]
        flop += 2 * 2 * cl * cl * cr * d * Dl + 2 * 2 * cl * cr * cr * d * Dr + 2 * cl * cr * Dl * Dr * d * d
        flop2 += 2 * 2 * cl * cl * cr * d * Dl * Dl + 2 * 2 * cl * cr * cr * d * Dr * Dr \
            + 4 * cl * cr * Dl * Dl * Dr * d * d
        a = cl * d * cr * itemsize
        mpsb += a
        mem = max(mem, (cl * cl * Dl + cr * cr * Dr) * itemsize + a + 2 * chunk_bytes)
        mem2 = max(mem2, (cl * cl * Dl * Dl + cr * cr * Dr * Dr) * itemsize + a + 2 * chunk_bytes)
    return {"tflop_energy": flop / 1e12, "tflop_variance": flop2 / 1e12, "gb_energy": mem / 1e9,
            "gb_variance": mem2 / 1e9, "gb_mps": mpsb / 1e9, "chi_max": int(max(bond_dims)),
            "mpo_D_max": int(max(mpo_dims)), "scaling": "flop ~ 4 N chi^3 d D (<H>), 4 N chi^3 d D^2 (<H^2>); "
                                                          "memory ~ 2 chi^2 D (<H>), 2 chi^2 D^2 (<H^2>)"}


# ------------------------------------------------------------------ amplitudes and sampling

def amplitudes(tensors: list, idx: np.ndarray) -> np.ndarray:
    """psi(n) for configurations idx (n_configs, N) of local indices in chain order."""
    idx = np.asarray(idx)
    m = idx.shape[0]
    v = np.ones((m, 1), dtype=np.result_type(*tensors))
    for k, A in enumerate(tensors):
        new = np.empty((m, A.shape[2]), dtype=v.dtype)
        for s in range(A.shape[1]):
            sel = idx[:, k] == s
            if np.any(sel):
                new[sel] = v[sel] @ A[:, s, :]
        v = new
    return v[:, 0]


def right_canonical(tensors: list) -> tuple[list, float]:
    """Right-canonical copy (QR from the right) and log of the norm."""
    out = [None] * len(tensors)
    carry = np.ones((1, 1), dtype=np.result_type(*tensors))
    lognorm = 0.0
    for k in range(len(tensors) - 1, -1, -1):
        A = np.tensordot(tensors[k], carry, axes=(2, 0))        # (cl, d, cr')
        cl, d, cr = A.shape
        M = A.reshape(cl, d * cr)
        q, r = np.linalg.qr(M.conj().T)                         # M^dagger = q r  ->  M = r^dagger q^dagger
        B = q.conj().T.reshape(-1, d, cr)
        carry = r.conj().T                                      # (cl, chi')
        if k == 0:
            s = complex(carry[0, 0]) if np.iscomplexobj(carry) else float(carry[0, 0])
            lognorm += math.log(abs(s))
            B = B * (s / abs(s))
        else:
            nrm = np.linalg.norm(carry)
            carry = carry / nrm
            lognorm += math.log(nrm)
        out[k] = B
    return out, lognorm


def perfect_samples(tensors_rc: list, n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent samples of |psi|^2 from a right-canonical MPS (Ferris and Vidal 2012)."""
    v = np.ones((n, 1), dtype=tensors_rc[0].dtype)
    out = np.empty((n, len(tensors_rc)), dtype=np.int64)
    for k, B in enumerate(tensors_rc):
        w = np.einsum("nb,bsc->nsc", v, B)
        p = np.sum(np.abs(w) ** 2, axis=2)
        p /= p.sum(axis=1, keepdims=True)
        c = np.cumsum(p, axis=1)
        s = np.minimum((rng.random(n)[:, None] > c).sum(axis=1), B.shape[1] - 1)
        out[:, k] = s
        v = w[np.arange(n), s]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return out


# ------------------------------------------------------------------ NetKet's basis

class NetKetMap:
    """Maps between chain configurations (local indices in chain order) and the verifier's NetKet
    configurations, with the fermion sign between the two orderings: for occupied modes listed in
    chain order (site k up, site k down, site k+1 up, ...), psi_netket(x) = psi_chain(n) x (-1)^inv,
    inv = the number of pairs of occupied modes whose NetKet orbital indices are out of order.
    (NetKet's own ascending/descending product order changes only a global sign at fixed N.)"""

    def __init__(self, system, order: np.ndarray, basis: str):
        self.basis, self.order, self.n = basis, np.asarray(order), len(order)
        if basis == "electron":
            from .hamiltonians import orbital_table
            orb = orbital_table(system.hilbert, self.n)
            self.cols = np.array([[orb[(int(i), 0.5)], orb[(int(i), -0.5)]] for i in self.order])  # (N, 2)
            key = self.cols.reshape(-1)                         # NetKet orbital of mode (k, up/dn)
            a = np.arange(len(key))
            self.G = ((a[:, None] < a[None, :]) & (key[:, None] > key[None, :])).astype(np.float64)

    def to_chain(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        if self.basis == "spin-half":
            return np.where(x[:, self.order] > 0, 0, 1), np.ones(len(x))
        occ = np.rint(x[:, self.cols.reshape(-1)]).astype(np.int64)        # (m, 2N) modes in chain order
        idx = occ[:, 0::2] + 2 * occ[:, 1::2]
        o = occ.astype(np.float64)
        inv = np.einsum("ma,ab,mb->m", o, self.G, o)
        return idx, 1.0 - 2.0 * (np.rint(inv).astype(np.int64) % 2)

    def to_netket(self, idx: np.ndarray, dtype=np.float64) -> np.ndarray:
        idx = np.asarray(idx)
        m = idx.shape[0]
        if self.basis == "spin-half":
            x = np.empty((m, self.n), dtype=dtype)
            x[:, self.order] = np.where(idx == 0, 1.0, -1.0)
            return x
        x = np.zeros((m, 2 * self.n), dtype=dtype)
        x[:, self.cols[:, 0]] = idx % 2
        x[:, self.cols[:, 1]] = idx // 2
        return x


def sampling_check(inst: dict, mps: MPSFile, e_exact: float, n_samples: int, seed: int,
                   system=None, chunk: int = 64, log=None) -> dict:
    """Perfect samples from the MPS, local energies with the verifier's NetKet Hamiltonian and the
    MPS amplitudes in NetKet's basis; their mean must agree with the exact energy."""
    import jax.numpy as jnp
    from . import hamiltonians as Hm
    t0 = time.perf_counter()
    system = system or Hm.build(inst)
    tensors = mps.tensors()
    rc, _ = right_canonical(tensors)
    del tensors
    rng = np.random.default_rng(seed)
    idx = perfect_samples(rc, n_samples, rng)
    nmap = NetKetMap(system, mps.order, mps.basis)
    x = nmap.to_netket(idx, dtype=np.asarray(system.hilbert.numbers_to_states(np.array([0]))).dtype)
    e_loc = np.empty(n_samples, dtype=complex)
    H = system.hamiltonian
    import netket as nk
    jax_op = isinstance(H, nk.operator.DiscreteJaxOperator)     # numba operators take numpy arrays
    for a in range(0, n_samples, chunk):
        xb = x[a:a + chunk]
        xp, mels = H.get_conn_padded(jnp.asarray(xb) if jax_op else xb)
        xp, mels = np.asarray(xp), np.asarray(mels)
        m, c, _ = xp.shape
        i0, s0 = nmap.to_chain(xb)
        i1, s1 = nmap.to_chain(xp.reshape(m * c, -1))
        psi0 = amplitudes(rc, i0) * s0
        psi1 = (amplitudes(rc, i1) * s1).reshape(m, c)
        e_loc[a:a + chunk] = np.sum(mels * psi1, axis=1) / psi0
    e = np.real(e_loc)
    mean, sd = float(e.mean()), float(e.std(ddof=1))
    se = sd / math.sqrt(n_samples)
    z = (mean - e_exact) / se if se > 0 else (0.0 if abs(mean - e_exact) < 1e-9 * max(1, abs(e_exact)) else math.inf)
    if log:
        log(f"[mps] sampling check: {n_samples} perfect samples, mean {mean:.8f} +- {se:.2g} vs exact {e_exact:.8f} (z {z:+.2f})")
    return {"n_samples": n_samples, "seed": seed, "mean_total": mean, "stderr_total": se,
            "sample_variance_total": sd ** 2, "z": z, "imag_max": float(np.max(np.abs(np.imag(e_loc)))),
            "wall_s": time.perf_counter() - t0}


# ------------------------------------------------------------------ the scorecard

def evaluate(inst: dict, path: str, *, seed: int = 0, n_samples: int = 512, limits: dict | None = None,
             variance: str = "auto", log=print) -> dict:
    """Score one MPS file on one instance: exact <H>, sector, stated energy, sampling cross-check
    and, where it fits, the variance. Returns a scorecard for `op_verify.py combine`."""
    lim = dict(LIMITS, **(limits or {}))
    t0 = time.perf_counter()
    div = I.per_site_divisor(inst)
    gates = []

    def gate(name, status, detail):
        gates.append({"name": name, "status": status, "detail": detail})

    card = {"instance_id": inst["instance_id"], "kind": "mps", "deterministic": True, "seed": seed,
            "state": str(path)}
    try:
        mps = load(path, inst)
    except (ValueError, OSError, KeyError) as exc:
        gate("format", "fail", str(exc))
        return _finish(card, gates, None, div, t0, inst)
    card.update({"state_hash": mps.sha256, "chi_max": mps.chi_max, "bond_dims": mps.bond_dims,
                 "dtype": mps.dtype, "site_order": mps.order.tolist(), "local_basis": mps.basis,
                 "producer": mps.attrs.get("producer")})
    gate("format", "pass", f"{mps.n_sites} sites, {mps.basis}, chi_max {mps.chi_max}, {mps.dtype}")
    mpo = build_mpo(inst, mps.order)
    d = BASES[mps.basis]
    item = 16 if mps.dtype == "complex128" else 8
    cost = cost_estimate(mps.bond_dims, mpo.bond_dims, d, item)
    card["cost"] = cost
    card["mpo"] = mpo.description
    log(f"[mps] {inst['instance_id']}: {mpo.description}; chi_max {mps.chi_max}; <H> needs "
        f"{cost['tflop_energy']:.3g} Tflop, {cost['gb_energy']:.3g} GB; file {cost['gb_mps']:.3g} GB")
    if cost["gb_energy"] > lim["max_gb"] or cost["tflop_energy"] > lim["max_tflop"]:
        gate("size", "fail", f"<H> needs {cost['gb_energy']:.3g} GB and {cost['tflop_energy']:.3g} Tflop; "
             f"limits {lim['max_gb']} GB, {lim['max_tflop']} Tflop (raise them with --max-gb / --max-tflop)")
        return _finish(card, gates, None, div, t0, inst)
    gate("size", "pass", f"{cost['gb_energy']:.3g} GB, {cost['tflop_energy']:.3g} Tflop")

    r = contract(mps, mpo, log=log)
    E = r["value"]
    card["contraction_wall_s"] = r["wall_s"]
    card["log_norm"] = r["log_norm"]
    tol_im = 1e-10 * max(1.0, abs(E))
    gate("real", "pass" if abs(r["imag"]) <= tol_im else "fail", f"Im <H>/<psi|psi> = {r['imag']:.2e}")
    if not math.isfinite(E):
        gate("finite", "fail", "the energy is not finite")
        return _finish(card, gates, None, div, t0, inst)

    q = contract(mps, charge_mpo(mps.basis, mps.n_sites, sector_targets(inst)))
    card["sector_deviation"] = q["value"]
    what = "<(N_up - n_up)^2> + <(N_dn - n_dn)^2>" if mps.basis == "electron" else "<(S^z_total)^2>"
    gate("sector", "pass" if q["value"] <= SECTOR_TOL else "fail",
         f"{what} = {q['value']:.2e} (limit {SECTOR_TOL:g}; the instance's sector is {sector_targets(inst)})")

    claim = mps.attrs.get("energy")
    if claim is None:
        gate("stated energy", "warn", "the file states no energy; site order and basis conventions are "
             "checked only through a stated energy")
    else:
        claim = float(claim)
        dev = abs(E - claim) / max(1.0, abs(claim))
        card["stated_energy_total"] = claim
        gate("stated energy", "pass" if dev <= CLAIM_RTOL else "fail",
             f"verifier {E:.12f}, file {claim:.12f}, relative difference {dev:.2e} (limit {CLAIM_RTOL:g}); "
             + ("" if dev <= CLAIM_RTOL else "a wrong site order, local basis or fermion ordering, or a "
                "different Hamiltonian, gives this"))

    # sampling cross-check with NetKet's Hamiltonian
    gb_s = 2 * cost["gb_mps"] + 1e-9 * 16 * n_samples * mps.chi_max * 64
    if n_samples <= 0:
        gate("sampling", "warn", "skipped (n_samples = 0)")
    elif gb_s > lim["max_gb_sampling"]:
        gate("sampling", "warn", f"skipped: needs about {gb_s:.3g} GB > {lim['max_gb_sampling']} GB")
    else:
        try:
            s = sampling_check(inst, mps, E, n_samples, seed, log=log)
            card["sampling"] = s
            ok = abs(s["z"]) <= SAMPLING_Z and s["imag_max"] <= 1e-6 * max(1.0, abs(E))
            gate("sampling", "pass" if ok else "fail",
                 f"{s['n_samples']} perfect samples, NetKet local energies: {s['mean_total']:.8f} +- "
                 f"{s['stderr_total']:.2g} vs exact {E:.8f}, z {s['z']:+.2f} (limit {SAMPLING_Z})")
        except ImportError as exc:
            gate("sampling", "warn", f"skipped: {exc}")

    want = variance == "yes" or (variance == "auto" and cost["gb_variance"] <= lim["max_gb_variance"]
                                 and cost["tflop_variance"] <= lim["max_tflop"])
    if want:
        h2 = contract(mps, mpo, layers=2, log=log)
        var = h2["value"] - E * E
        card["variance_total"] = var
        card["variance_wall_s"] = h2["wall_s"]
        dof, einf = I.expected_dof(inst), I.expected_einf(inst)
        card["v_score"] = dof * var / (E - einf) ** 2 if (dof and einf is not None) else None
        card["einf"], card["dof"] = einf, dof
        gate("variance", "pass" if var >= -1e-8 * max(1.0, E * E) else "fail", f"<H^2> - <H>^2 = {var:.6e}")
    else:
        card["variance_total"] = None
        card["variance_skipped"] = (f"<H^2> needs {cost['gb_variance']:.3g} GB, {cost['tflop_variance']:.3g} Tflop "
                                    f"(limit {lim['max_gb_variance']} GB)")
    return _finish(card, gates, E, div, t0, inst)


def _finish(card: dict, gates: list, E, div, t0, inst) -> dict:
    status = "fail" if any(g["status"] == "fail" for g in gates) else \
        "warn" if any(g["status"] == "warn" for g in gates) else "pass"
    card["gates"] = gates
    card["status"] = status
    card["qualified"] = status != "fail"
    card["energy_total"] = E
    card["sigma_total"] = 0.0
    card["energy_per_site"] = None if E is None else E / div
    card["sigma_per_site"] = 0.0
    card["method"] = "exact MPS contraction"
    card["wall_s"] = time.perf_counter() - t0
    card["n_samples"] = card.get("sampling", {}).get("n_samples")
    rec = I.target(inst)
    if rec is not None and E is not None:
        card["record"] = {"energy_total": rec["energy"], "energy_per_site": rec["energy"] / div,
                          "method": rec.get("method"), "bound_type": rec.get("bound_type")}
        card["margin_per_site"] = (rec["energy"] - E) / div
        ex = I.exact_record(inst)
        if ex is not None and E < ex["energy"] - 1e-9 * abs(ex["energy"]):
            card["plausibility"] = "BELOW EXACT: bug until shown otherwise"
    return card
