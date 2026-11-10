"""Continuous space: the homogeneous electron gas (QMBL track 2, RULES.md 12).

A state is a function of the electron positions: x has shape (batch, N * dim), Cartesian
coordinates in bohr inside the simulation cell, the N_up spin-up electrons first. It returns log psi
(complex where the twist or the state makes psi complex) or a pair (sign, log|psi|) as fvmc's
wave functions do. The verifier builds everything else itself:

- the cell from the instance (lattice, N_up, N_dn, rs), scaled as QMBL's checks/heg-hf/hf.mjs does;
- the Coulomb energy by an Ewald sum with the uniform background and the Madelung term, in 2D and
  3D, E = 1/2 sum_{i != j} phi(r_i - r_j) + N xi / 2 (the convention of hf.mjs and of QMBL's rows);
- the kinetic energy -1/2 (lap log psi + (grad log psi)^2) by the forward Laplacian
  (qrec_verify/_vendor/fwdlap.py, Y. Chen, Apache-2.0; jax.hessian as the cross-check);
- a Metropolis sampler with all-electron Gaussian moves, its step tuned during burn-in only;
- gates on the state itself: antisymmetry under the exchange of two electrons of the same spin, and
  the twist phase when an electron moves by a cell vector; on the Hamiltonian: the Madelung term
  against the hf.mjs formula at another splitting parameter.

Energies are totals in hartree; QMBL's per-site energy is the energy per electron.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.special import erfc as _erfc

HEG_MODELS = ("HEG2D", "HEG3D")

# hf.mjs cellVectors: rows are the lattice vectors of the unit cell before scaling.
_UNIT = {
    "sc": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "bcc": [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
    "fcc": [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    "square": [[1, 0], [0, 1]],
    "triangular": [[1, 0], [0.5, math.sqrt(3) / 2]],
}


def unit_cell(dim: int, lattice: str) -> np.ndarray:
    """QMBL's lattice names: sc, bcc, fcc, square, triangular, rect-AxB[xC] (side ratios)."""
    if lattice.startswith("rect-"):
        r = [float(v) for v in lattice[5:].split("x")]
        A = np.diag(r)
    elif lattice in _UNIT:
        A = np.array(_UNIT[lattice], dtype=float)
    else:
        raise ValueError(f"lattice {lattice!r} not supported")
    if A.shape != (dim, dim):
        raise ValueError(f"lattice {lattice} is not {dim}D")
    return A


def reciprocal(A: np.ndarray) -> np.ndarray:
    """Rows B_j with A_i . B_j = 2 pi delta_ij."""
    return 2 * np.pi * np.linalg.inv(A).T


@dataclass(frozen=True)
class Cell:
    dim: int
    n_up: int
    n_dn: int
    rs: float
    A: np.ndarray          # (dim, dim), rows are the cell vectors in bohr
    twist: np.ndarray      # fractions of the reciprocal vectors

    @property
    def n(self) -> int:
        return self.n_up + self.n_dn

    @property
    def B(self) -> np.ndarray:
        return reciprocal(self.A)

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.A)))

    @property
    def ks(self) -> np.ndarray:
        return self.twist @ self.B


def make_cell(dim: int, lattice: str, n_up: int, n_dn: int, rs: float, twist=None) -> Cell:
    unit = unit_cell(dim, lattice)
    n = n_up + n_dn
    omega = n * (4 * np.pi / 3) * rs ** 3 if dim == 3 else n * np.pi * rs ** 2
    s = (omega / abs(np.linalg.det(unit))) ** (1.0 / dim)
    tw = np.zeros(dim) if twist is None else np.asarray(twist, dtype=float)
    return Cell(dim, n_up, n_dn, float(rs), unit * s, tw)


def cell_from_instance(inst: dict) -> Cell:
    if inst["model"] not in HEG_MODELS:
        raise ValueError(f"{inst['model']} is not an electron gas")
    p = inst["params"]
    tw = p.get("twist", "Gamma")
    twist = None if tw in (None, "Gamma") else [float(t) for t in tw]
    dim = 2 if inst["model"] == "HEG2D" else 3
    cell = make_cell(dim, inst["lattice"], int(p["N_up"]), int(p["N_dn"]), float(p["rs"]), twist)
    if cell.n != int(inst["n_sites"]):
        raise ValueError("N_up + N_dn differs from n_sites")
    return cell


# ------------------------------------------------------------------ Ewald (numpy, float64)

def _grid(n: int, dim: int) -> np.ndarray:
    r = np.arange(-n, n + 1)
    return np.stack(np.meshgrid(*([r] * dim), indexing="ij"), -1).reshape(-1, dim)


def madelung_hfmjs(A: np.ndarray) -> float:
    """hf.mjs madelung(): kappa = 2.8 / Omega^(1/dim), 10 shells of real and reciprocal vectors.
    The Hamiltonian gate compares the verifier's own xi (another kappa) with this."""
    dim = A.shape[0]
    omega = abs(np.linalg.det(A))
    B = reciprocal(A)
    kappa = 2.8 / omega ** (1.0 / dim)
    m = _grid(10, dim)
    m = m[np.any(m != 0, axis=1)]
    R = np.linalg.norm(m @ A, axis=1)
    G = np.linalg.norm(m @ B, axis=1)
    real = np.sum(_erfc(kappa * R) / R)
    if dim == 3:
        recip = np.sum((4 * np.pi / omega) * np.exp(-G ** 2 / (4 * kappa ** 2)) / G ** 2)
        return float(real + recip - 2 * kappa / np.sqrt(np.pi) - np.pi / (kappa ** 2 * omega))
    recip = np.sum((2 * np.pi / omega) * _erfc(G / (2 * kappa)) / G)
    return float(real + recip - 2 * kappa / np.sqrt(np.pi) - 2 * np.sqrt(np.pi) / (kappa * omega))


@dataclass(frozen=True)
class Ewald:
    """Real-space images, half of the reciprocal vectors with their weights, and the constants.

    The splitting kappa is set by the cell: displacements are reduced to fractional coordinates
    in [-1/2, 1/2), the images |m|_inf <= n_img then cover every distance up to
    (n_img - 1/2) d_min (d_min the smallest spacing of lattice planes), and kappa makes erfc tiny
    there; the reciprocal sum runs to |G| = 2 kappa x 6.1 (exp(-G^2 / 4 kappa^2) < 1e-16)."""
    A: np.ndarray
    kappa: float
    images: np.ndarray     # (n_images, dim) lattice vectors R
    G: np.ndarray          # (n_g, dim), one of each +-G pair
    wG: np.ndarray         # (n_g,) weight of the pair: 2 c(G)
    bg: float              # background term per pair
    xi: float              # Madelung term

    @classmethod
    def build(cls, A: np.ndarray, n_img: int = 2, tail: float = 6.0) -> "Ewald":
        dim = A.shape[0]
        omega = abs(np.linalg.det(A))
        B = reciprocal(A)
        d_min = float(np.min(2 * np.pi / np.linalg.norm(B, axis=1)))
        kappa = tail / ((n_img - 0.5) * d_min)
        images = _grid(n_img, dim) @ A
        g_max = 2 * kappa * 6.1
        mmax = [int(math.ceil(g_max * np.linalg.norm(A[k]) / (2 * np.pi))) + 1 for k in range(dim)]
        axes = [np.arange(-mk, mk + 1) for mk in mmax]
        m = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
        G = m @ B
        g = np.linalg.norm(G, axis=1)
        keep = (g > 0) & (g <= g_max)
        # one of each pair: the first nonzero component of m positive
        nz = m != 0
        first = np.where(nz.any(1), m[np.arange(len(m)), nz.argmax(1)], 0)
        keep &= first > 0
        G, g = G[keep], g[keep]
        if dim == 3:
            c = (4 * np.pi / omega) * np.exp(-g ** 2 / (4 * kappa ** 2)) / g ** 2
            bg = np.pi / (kappa ** 2 * omega)
        else:
            c = (2 * np.pi / omega) * _erfc(g / (2 * kappa)) / g
            bg = 2 * np.sqrt(np.pi) / (kappa * omega)
        R = np.linalg.norm(images, axis=1)
        R = R[R > 0]
        xi = float(np.sum(_erfc(kappa * R) / R) + 2 * np.sum(c) - 2 * kappa / np.sqrt(np.pi) - bg)
        return cls(A, float(kappa), images, G, 2 * c, float(bg), xi)

    def energy_fn(self, n: int) -> Callable:
        """JAX function of one configuration x (n * dim,) -> Coulomb energy (hartree)."""
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import erfc
        dim = self.A.shape[0]
        A = jnp.asarray(self.A)
        Ainv = jnp.asarray(np.linalg.inv(self.A))
        images = jnp.asarray(self.images)
        G = jnp.asarray(self.G)
        wG = jnp.asarray(self.wG)
        kappa, bg, xi = self.kappa, self.bg, self.xi
        iu = np.triu_indices(n, 1)
        const = -0.5 * n * (n - 1) * bg + 0.5 * n * xi - 0.5 * n * float(np.sum(self.wG))

        def energy(x):
            r = x.reshape(n, dim)
            d = (r[:, None, :] - r[None, :, :])[iu]               # (n_pairs, dim)
            f = d @ Ainv
            d = (f - jnp.round(f)) @ A
            dd = d[:, None, :] + images[None, :, :]
            dist = jnp.sqrt(jnp.sum(dd ** 2, axis=-1))
            e_real = jnp.sum(erfc(kappa * dist) / dist)
            phase = r @ G.T                                         # (n, n_g)
            rho2 = jnp.sum(jnp.cos(phase), 0) ** 2 + jnp.sum(jnp.sin(phase), 0) ** 2
            e_recip = 0.5 * jnp.sum(wG * rho2)
            return e_real + e_recip + const
        return energy


# ------------------------------------------------------------------ the state

def as_logpsi(raw):
    """log psi from a state's output: a complex or real array, or a pair (sign, log|psi|)."""
    import jax.numpy as jnp
    if isinstance(raw, (tuple, list)):
        sign, logabs = raw
        return logabs.astype(jnp.complex128) + jnp.log(sign.astype(jnp.complex128))
    return raw.astype(jnp.complex128)


def logpsi_fn(model, variables) -> Callable:
    """Batched log psi: x (batch, n * dim) -> (batch,) complex."""
    def f(x):
        return as_logpsi(model.apply(variables, x))
    return f


def kinetic_fn(logpsi_single: Callable, ncoord: int, method: str = "fwdlap") -> Callable:
    """-1/2 (lap psi) / psi for one configuration, from log psi (complex): lap log psi + (grad log psi)^2."""
    import jax
    import jax.numpy as jnp

    def reim(x):
        v = logpsi_single(x)
        return jnp.stack([jnp.real(v), jnp.imag(v)])

    if method == "fwdlap":
        from ._vendor import fwdlap

        def ke(x):
            eye = jnp.eye(ncoord, dtype=x.dtype)
            zero = fwdlap.zero_tangent_from_primal(x)
            _, jac, lap = fwdlap.lap(reim, (x,), (eye,), (zero,))   # jac (ncoord, 2), lap (2,)
            g = jac[:, 0] + 1j * jac[:, 1]
            return -0.5 * (lap[0] + 1j * lap[1] + jnp.sum(g * g))
        return ke

    def ke(x):
        g = jax.jacfwd(reim)(x)                   # (2, ncoord)
        h = jax.hessian(reim)(x)                  # (2, ncoord, ncoord)
        gc = g[0] + 1j * g[1]
        return -0.5 * (jnp.trace(h[0]) + 1j * jnp.trace(h[1]) + jnp.sum(gc * gc))
    return ke


def local_energy_fn(model, variables, cell: Cell, ewald: Ewald, method: str = "fwdlap") -> Callable:
    """Batched local energy x (batch, n * dim) -> (batch,) complex: kinetic + Coulomb."""
    import jax
    n, dim = cell.n, cell.dim
    lp = logpsi_fn(model, variables)
    single = lambda x: lp(x[None])[0]                     # noqa: E731
    ke = kinetic_fn(single, n * dim, method)
    pe = ewald.energy_fn(n)
    return jax.vmap(lambda x: ke(x) + pe(x))


# ------------------------------------------------------------------ the known state: Hartree-Fock

def occupied_k(cell: Cell) -> tuple[np.ndarray, np.ndarray, bool]:
    """hf.mjs fill(): each spin fills its N_s lowest |k + k_s|^2; open_shell when the last occupied
    shell is only partly filled."""
    dim = cell.dim
    m = _grid(12, dim)
    k = m @ cell.B + cell.ks
    order = np.argsort(np.sum(k ** 2, 1), kind="stable")
    k = k[order]
    e = np.sum(k ** 2, 1)
    out, open_shell = [], False
    for ns in (cell.n_up, cell.n_dn):
        out.append(k[:ns])
        if ns and abs(e[ns] - e[ns - 1]) < 1e-9 * max(1.0, e[ns - 1]):
            open_shell = True
    return out[0], out[1], open_shell


def hf_energy(cell: Cell) -> dict:
    """hf.mjs hf(): the plane-wave determinant's energy per electron, kinetic, exchange, xi."""
    ku, kd, open_shell = occupied_k(cell)
    kin = 0.5 * (np.sum(ku ** 2) + np.sum(kd ** 2))
    ex = 0.0
    for occ in (ku, kd):
        q = occ[:, None, :] - occ[None, :, :]
        qn = np.linalg.norm(q, axis=-1)
        mask = ~np.eye(len(occ), dtype=bool)
        v = (4 * np.pi / qn[mask] ** 2) if cell.dim == 3 else (2 * np.pi / qn[mask])
        ex -= np.sum(v) / (2 * cell.volume)
    xi = madelung_hfmjs(cell.A)
    total = kin + ex + cell.n * xi / 2
    return {"E_HF_per_electron": total / cell.n, "kinetic_per_electron": kin / cell.n,
            "exchange_per_electron": ex / cell.n, "xi": xi, "open_shell": open_shell}


def slater_model(cell: Cell):
    """The plane-wave Slater determinant (the HF state) as a flax module with no parameters."""
    import flax.linen as nn
    import jax.numpy as jnp
    ku, kd, _ = occupied_k(cell)
    n_up, dim = cell.n_up, cell.dim

    class PlaneWaveSlater(nn.Module):
        @nn.compact
        def __call__(self, x):
            r = x.reshape(x.shape[0], -1, dim)

            def logdet(rs, ks):
                if ks.shape[0] == 0:
                    return jnp.zeros(r.shape[0], jnp.complex128)
                M = jnp.exp(1j * jnp.einsum("bnd,kd->bnk", rs, jnp.asarray(ks)))
                s, la = jnp.linalg.slogdet(M)
                return la + jnp.log(s)
            return logdet(r[:, :n_up], ku) + logdet(r[:, n_up:], kd)
    return PlaneWaveSlater()


# ------------------------------------------------------------------ state gates

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def state_gates(lp: Callable, cell: Cell, x: np.ndarray, tol: float = 1e-6) -> list[dict]:
    """Antisymmetry and twist checks on configurations x (batch, n * dim)."""
    n, dim = cell.n, cell.dim
    x = np.asarray(x)
    base = np.asarray(lp(x))
    gates = []
    worst = 0.0
    swaps = []
    for lo, ns in ((0, cell.n_up), (cell.n_up, cell.n_dn)):
        if ns >= 2:
            swaps.append((lo, lo + 1))
    for i, j in swaps:
        y = x.reshape(len(x), n, dim).copy()
        y[:, [i, j]] = y[:, [j, i]]
        d = np.asarray(lp(y.reshape(len(x), -1))) - base
        worst = max(worst, float(np.max(np.abs(np.real(d)))), float(np.max(np.abs(_wrap_angle(np.imag(d) - np.pi)))))
    gates.append({"name": "antisymmetry", "value": worst, "pass": bool(worst <= tol),
                  "detail": "log psi after swapping two electrons of the same spin = log psi + i pi"})
    worst = 0.0
    for k in range(dim):
        y = x.reshape(len(x), n, dim).copy()
        y[:, 0] += cell.A[k]
        d = np.asarray(lp(y.reshape(len(x), -1))) - base
        want = float(cell.ks @ cell.A[k])
        worst = max(worst, float(np.max(np.abs(np.real(d)))), float(np.max(np.abs(_wrap_angle(np.imag(d) - want)))))
    gates.append({"name": "twist", "value": worst, "pass": bool(worst <= tol),
                  "detail": "log psi after moving one electron by a cell vector A_k = log psi + i k_s . A_k"})
    return gates


def hamiltonian_gate(cell: Cell, ewald: Ewald) -> dict:
    ref = madelung_hfmjs(cell.A)
    diff = abs(ewald.xi - ref)
    return {"name": "madelung", "value": diff, "pass": bool(diff <= 1e-10 * max(1.0, abs(ref))),
            "detail": f"xi {ewald.xi:.15g} (kappa {ewald.kappa:.6g}) vs hf.mjs formula {ref:.15g}"}


# ------------------------------------------------------------------ sampler and collection

# thin and n_burn are minima: evaluate() raises them to N and 20 N all-electron moves, since an
# electron needs of order N small moves to cross its own neighbourhood.
HEG = dict(n_chains=128, n_burn=2000, thin=8, block=32, min_samples=65536, target_acc=0.5)


def _wrap_positions(x, A, Ainv, n, dim):
    import jax.numpy as jnp
    r = x.reshape(x.shape[0], n, dim)
    f = r @ Ainv
    return ((f - jnp.floor(f)) @ A).reshape(x.shape[0], -1)


def collect(model, variables, cell: Cell, *, seed: int, n_chains: int, n_burn: int, thin: int,
            block: int, min_samples: int, max_seconds: float, chunk: int = 64,
            method: str = "fwdlap", target_acc: float = 0.5, log: Callable = print) -> dict:
    """Chains from uniform random positions; returns e_loc, logabs (n_chains, n_steps), the
    fraction of recorded samples that moved per chain and block, the step and the gate samples."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    n, dim = cell.n, cell.dim
    A, Ainv = jnp.asarray(cell.A), jnp.asarray(np.linalg.inv(cell.A))
    ewald = Ewald.build(cell.A)
    lp = jax.jit(logpsi_fn(model, variables))
    eloc = jax.jit(local_energy_fn(model, variables, cell, ewald, method))

    def logabs(x):
        return jnp.real(logpsi_fn(model, variables)(x))

    @jax.jit
    def move(key, x, la, step):
        k1, k2 = jax.random.split(key)
        y = _wrap_positions(x + step * jax.random.normal(k1, x.shape), A, Ainv, n, dim)
        lb = logabs(y)
        acc = jnp.log(jax.random.uniform(k2, la.shape)) < 2 * (lb - la)
        return jnp.where(acc[:, None], y, x), jnp.where(acc, lb, la), acc

    def run(key, x, la, step, k):
        def body(c, kk):
            x, la = c
            x, la, a = move(kk, x, la, step)
            return (x, la), a
        (x, la), a = jax.lax.scan(body, (x, la), jax.random.split(key, k))
        return x, la, a.mean(0)
    run = jax.jit(run, static_argnums=4)

    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    x = (jax.random.uniform(k0, (n_chains, n, dim)) @ A).reshape(n_chains, -1)
    la = logabs(x)
    step = 0.3 * float(np.mean(np.linalg.norm(cell.A, axis=1))) / n ** (1.0 / dim)
    t0 = time.perf_counter()
    done = 0
    while done < n_burn:                          # step tuned during burn-in only
        key, k = jax.random.split(key)
        x, la, acc = run(k, x, la, step, 50)
        step *= float(np.exp(np.mean(np.asarray(acc)) - target_acc))
        done += 50
    log(f"[heg] burn-in {n_burn} moves, step {step:.4g} bohr, {time.perf_counter() - t0:.1f} s")

    def eval_e(xs):
        out = []
        for i in range(0, len(xs), chunk):
            out.append(np.asarray(eloc(xs[i:i + chunk])))
        return np.concatenate(out)

    e_blocks, l_blocks, m_blocks = [], [], []
    prev = np.asarray(x)
    stop = "budget"
    while sum(b.shape[1] for b in e_blocks) * n_chains < min_samples:
        xs, ls = [], []
        for _ in range(block):
            key, k = jax.random.split(key)
            x, la, _ = run(k, x, la, step, thin)
            xs.append(np.asarray(x))
            ls.append(np.asarray(la))
        xs = np.stack(xs, 1)                                   # (chains, block, n * dim)
        seq = np.concatenate([prev[:, None], xs], 1)
        moved = np.any(seq[:, 1:] != seq[:, :-1], axis=-1)            # (chains, block)
        q = max(block // 4, 1)                                      # mobility per quarter block
        m_blocks.extend(moved[:, i:i + q].mean(1) for i in range(0, block, q))
        prev = xs[:, -1]
        e_blocks.append(eval_e(xs.reshape(-1, n * dim)).reshape(n_chains, block))
        l_blocks.append(np.stack(ls, 1))
        if time.perf_counter() - t0 > max_seconds:
            stop = "time"
            break
    e = np.concatenate(e_blocks, 1)
    return {"e_loc": np.real(e), "logabs": np.concatenate(l_blocks, 1), "acc": np.stack(m_blocks, 1),
            "imag_max": float(np.max(np.abs(np.imag(e)))), "wall_s": time.perf_counter() - t0,
            "step": step, "stop": stop, "seed": seed, "n_burn": n_burn, "thin": thin, "block": block,
            "gate_samples": prev[: min(16, n_chains)], "ewald": ewald, "logpsi": lp,
            "device": str(jax.devices()[0]), "jax": jax.__version__}


def evaluate(inst: dict, model, variables, *, seed: int, max_hours: float = 12.0,
             min_samples: int | None = None, method: str = "fwdlap", state_id: str = "",
             log: Callable = print, **overrides) -> dict:
    """Score one electron-gas state: the scorecard of measure.scorecard plus the physics gates."""
    from . import measure as Me
    cell = cell_from_instance(inst)
    cfg = dict(HEG, **overrides)
    cfg["thin"] = max(cfg["thin"], cell.n)
    cfg["n_burn"] = max(cfg["n_burn"], 20 * cell.n)
    budget = min_samples or cfg["min_samples"]
    desc = (f"all-electron Gaussian Metropolis, {cfg['n_chains']} chains, {cfg['thin']} moves per sample, "
            f"step tuned to acceptance {cfg['target_acc']} in burn-in")
    log(f"[heg] {inst['instance_id']}: {desc}; budget {budget} samples; cap {max_hours} h; seed {seed}")
    ser = collect(model, variables, cell, seed=seed, n_chains=cfg["n_chains"], n_burn=cfg["n_burn"],
                  thin=cfg["thin"], block=cfg["block"], min_samples=budget, max_seconds=max_hours * 3600,
                  method=method, target_acc=cfg["target_acc"], log=log)
    gates = [hamiltonian_gate(cell, ser["ewald"])] + state_gates(ser["logpsi"], cell, ser["gate_samples"])
    card = Me.scorecard(inst, ser, state_id=state_id or Me.state_hash(variables), sampler_desc=desc)
    card["state_hash"] = Me.state_hash(variables)
    card["physics_gates"] = gates
    card["stop"] = ser["stop"]
    card["step_bohr"] = ser["step"]
    card["local_energy"] = f"verifier Ewald (kappa {ser['ewald'].kappa:.6g}) + kinetic by {method}"
    card["budget_samples"] = budget
    if not all(g["pass"] for g in gates):
        card["status"] = "fail"
    card["qualified"] = bool(card["status"] != "fail" and card["stop"] == "budget")
    card["__METRIC__"] = card["energy_per_site"]
    card["scorer"] = "qrec_verify.continuum v1.8"
    return card
