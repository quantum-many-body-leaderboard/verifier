"""Hydrogen chains and sheets in continuous space (v1.10): QMBL track 3 with `basis: none`.

Ported from qrec-verifier v2.1 (its molecule.py and molecular_score.py) into the public line's continuous-space
path, next to the electron gas (continuum.py), with the same state interface: a state is a function of the electron
positions, x of shape (batch, N * 3), Cartesian coordinates in bohr, the N_up spin-up electrons first; it returns
log psi (complex or real) or a pair (sign, log|psi|). The verifier builds everything else itself.

Hamiltonian (QMBL RULES.md 13.1-13.2; DATA.md: total energy in hartree, nuclear repulsion included, per atom = E / N):

    H = -1/2 sum_i lap_i - sum_{i, I} 1 / |r_i - R_I| + sum_{i < j} 1 / |r_i - r_j| + sum_{I < J} 1 / |R_I - R_J|

with point protons of charge 1 and no pseudopotential. Geometry as the Gaussian-basis instances (chem.atoms, DATA.md's
order): `chain` = N protons at (0, 0, I R), I = 0 .. N-1; `sheet` (model Hsheet, lattice square) = L x L protons at
(i R, j R, 0), atom I = i L + j. R in bohr (params.R). Neutral: N_up and N_dn from params (default ceil(N/2),
floor(N/2)). Open boundary only: every `basis: none` instance of QMBL is a finite molecule (boundary O); a periodic
cell would need the Ewald sum of continuum.py with protons, which no instance asks for yet, so it is refused.

Local energy: the potential by this module's own pair sums (JAX), the kinetic energy -1/2 (lap log psi +
(grad log psi)^2) by the forward Laplacian (continuum.py's vendored fwdlap), never by the submitter's code.

Sampler (qrec-verifier v2.1's settings): all-electron Gaussian Metropolis moves from the verifier's own starts (each
electron on a proton of a random permutation plus a Gaussian of width 0.5 bohr), the step tuned to an acceptance of
0.5 during the first 80 % of the burn-in and then frozen; 50 moves between recorded samples.

Gates on every score (physics_gates on the card):
  - coulomb: the JAX potential on the verifier's own samples against pyscf's point-charge Coulomb sum
    (gto.mole.energy_nuc with the protons at +1 and the electrons at -1; an independent code), and the nuclear
    repulsion against pyscf's Mole.energy_nuc; numpy pair sums if pyscf is not installed (requirements-chem.txt);
  - kinetic_paths: the forward Laplacian against jax.hessian on 8 samples (relative 1e-8);
  - kinetic_estimators: the mean of -1/2 (lap log psi + (grad log psi)^2) against the mean of 1/2 |grad log psi|^2
    (integration by parts; z < 3 passes, z >= 5 fails), on every sample;
  - antisymmetry: log psi after swapping two electrons of the same spin = log psi + i pi;
  - decay: log|psi| falls as one electron moves 20, 40, 80 bohr beyond the molecule (a normalisable state).
Exact references (tests/test_continuum.py, tests/test_physics.py): the hydrogen atom's 1s state has E_loc = -1/2 at
every point; H2 at R = 1.4 bohr has E = -1.1744757142204434 (K. Pachucki, PRA 82, 032509 (2010), Table 4), a floor no
state may go below; the two-electron FCI wave function of H2 in cc-pVDZ scores pyscf's FCI energy, and RHF
determinants of chains score pyscf's RHF energies (the known state, `rhf_check`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

H_ATOM_EXACT = -0.5
H2_EXACT = {1.4: -1.1744757142204434}          # Pachucki 2010, Table 4 (Born-Oppenheimer, nuclear repulsion included)

MOLECULE = dict(n_chains=512, n_burn=4000, thin=50, block=16, min_samples=2 ** 20, target_acc=0.5,
                start_width=0.5, max_step=2.0)
KINETIC_PATHS_RTOL = 1e-8
KINETIC_Z_FAIL = 5.0


def is_continuum_hydrogen(inst: dict) -> bool:
    """A hydrogen chain or sheet in continuous space (basis 'none')."""
    return inst.get("model") in ("Hchain", "Hsheet") and \
        str((inst.get("params") or {}).get("basis", "none")).lower() == "none"


@dataclass(frozen=True)
class Molecule:
    atoms: np.ndarray          # (n_atoms, 3) bohr
    charges: np.ndarray        # (n_atoms,)
    n_up: int
    n_dn: int
    dim: int = 3

    @property
    def n(self) -> int:
        return self.n_up + self.n_dn


def atoms_of(model: str, lattice: str, n_atoms: int, R: float) -> np.ndarray:
    """DATA.md's geometry (the same as chem.atoms for the Gaussian-basis instances)."""
    if model == "Hchain" and lattice == "chain":
        return np.array([[0.0, 0.0, i * R] for i in range(n_atoms)])
    if model == "Hsheet" and lattice == "square":
        L = int(round(math.sqrt(n_atoms)))
        if L * L != n_atoms:
            raise ValueError(f"a square sheet needs N = L x L atoms, got {n_atoms}")
        return np.array([[i * R, j * R, 0.0] for i in range(L) for j in range(L)])
    raise ValueError(f"hydrogen geometry {model}/{lattice} is not supported")


def molecule_of(inst: dict) -> Molecule:
    if not is_continuum_hydrogen(inst):
        raise ValueError(f"{inst.get('instance_id')}: not a hydrogen chain or sheet in continuous space (basis none)")
    if inst.get("boundary", "O") != "O":
        raise ValueError(f"{inst.get('instance_id')}: periodic hydrogen cells in continuous space are not built by "
                         "this verifier (no QMBL instance has one; it would need the Ewald sum of continuum.py with "
                         "protons); only open chains and sheets are")
    p = inst["params"]
    n_at = int(inst["n_sites"])
    atoms = atoms_of(inst["model"], inst["lattice"], n_at, float(p["R"]))
    n_up = int(p.get("N_up", (n_at + 1) // 2))
    n_dn = int(p.get("N_dn", n_at // 2))
    if n_up + n_dn != n_at:
        raise ValueError(f"N_up + N_dn = {n_up + n_dn}, but the neutral molecule has {n_at} electrons")
    return Molecule(atoms, np.ones(n_at), n_up, n_dn)


def small_instance(spec: str) -> dict:
    """'h1' = the hydrogen atom (one up electron); 'h2:<R>' = H2 at R bohr; 'hN:<R>' = an N-atom chain."""
    import re
    m = re.fullmatch(r"h(\d+)(?::(\d+(?:\.\d+)?))?", spec)
    if not m:
        raise ValueError(f"unknown small molecule {spec!r} (h1, h2:<R>, hN:<R>)")
    n, R = int(m.group(1)), float(m.group(2) or 1.0)
    return {"instance_id": spec, "model": "Hchain", "lattice": "chain", "n_sites": n, "boundary": "O",
            "params": {"N_up": (n + 1) // 2, "N_dn": n // 2, "R": R, "basis": "none"}, "rows": []}


# ------------------------------------------------------------------ the potential

def nuclear_repulsion(mol: Molecule) -> float:
    e = 0.0
    for I in range(len(mol.atoms)):
        for J in range(I + 1, len(mol.atoms)):
            e += mol.charges[I] * mol.charges[J] / float(np.linalg.norm(mol.atoms[I] - mol.atoms[J]))
    return e


def potential_np(r: np.ndarray, mol: Molecule) -> float:
    """Coulomb energy of one configuration r (N, 3), term by term (numpy)."""
    e = nuclear_repulsion(mol)
    for i in range(len(r)):
        for I in range(len(mol.atoms)):
            e -= mol.charges[I] / float(np.linalg.norm(r[i] - mol.atoms[I]))
        for j in range(i + 1, len(r)):
            e += 1.0 / float(np.linalg.norm(r[i] - r[j]))
    return e


def potential_fn(mol: Molecule) -> Callable:
    """JAX: one configuration x (N * 3,) -> Coulomb energy (hartree), nuclear repulsion included."""
    import jax.numpy as jnp
    atoms, Z = jnp.asarray(mol.atoms), jnp.asarray(mol.charges)
    e_nn = nuclear_repulsion(mol)
    iu = np.triu_indices(mol.n, 1)

    def energy(x):
        r = x.reshape(mol.n, 3)
        ea = -jnp.sum(Z[None, :] / jnp.linalg.norm(r[:, None, :] - atoms[None, :, :], axis=-1))
        ee = jnp.sum(1.0 / jnp.linalg.norm(r[iu[0]] - r[iu[1]], axis=-1)) if len(iu[0]) else 0.0
        return ea + ee + e_nn
    return energy


def coulomb_gate(mol: Molecule, x: np.ndarray, rtol: float = 1e-10) -> dict:
    """The verifier's potential on configurations x (batch, N * 3) against pyscf's point-charge sum (protons +1,
    electrons -1), and the nuclear repulsion against pyscf's Mole.energy_nuc; numpy pair sums without pyscf."""
    import jax
    import jax.numpy as jnp
    v = np.asarray(jax.vmap(potential_fn(mol))(jnp.asarray(x)))
    e_nn = nuclear_repulsion(mol)
    try:
        from pyscf import gto
        pm = gto.M(atom=[("H", tuple(a)) for a in mol.atoms], unit="Bohr", basis="sto-3g",
                   spin=mol.n_up - mol.n_dn, verbose=0)
        ref_nn = float(pm.energy_nuc())
        charges = np.concatenate([mol.charges, -np.ones(mol.n)])
        ref = np.array([float(gto.mole.energy_nuc(pm, charges, np.concatenate([mol.atoms, xx.reshape(mol.n, 3)])))
                        for xx in np.asarray(x)])
        how = "pyscf point charges (gto.mole.energy_nuc)"
    except ImportError:
        ref_nn = e_nn
        ref = np.array([potential_np(xx.reshape(mol.n, 3), mol) for xx in np.asarray(x)])
        how = "numpy pair sums (pyscf not installed)"
    diff = max(float(np.max(np.abs(v - ref) / np.maximum(1.0, np.abs(ref)))), abs(e_nn - ref_nn) / max(1.0, ref_nn))
    return {"name": "coulomb", "value": diff, "pass": bool(diff <= rtol),
            "detail": f"potential on {len(ref)} samples and the nuclear repulsion {e_nn:.12f} vs {how}"}


# ------------------------------------------------------------------ the kinetic energy

def kinetic_parts_fn(logpsi_single: Callable, ncoord: int, method: str = "fwdlap") -> Callable:
    """One configuration -> (T_lap, T_grad): T_lap = -1/2 (lap log psi + (grad log psi)^2) (complex; its real part is
    the local kinetic energy) and T_grad = 1/2 |grad log psi|^2 (real). Forward Laplacian (continuum.py's fwdlap) or
    jax.hessian (method 'hessian', the cross-check)."""
    import jax
    import jax.numpy as jnp

    def reim(x):
        v = logpsi_single(x)
        return jnp.stack([jnp.real(v), jnp.imag(v)])

    if method == "fwdlap":
        from ._vendor import fwdlap

        def parts(x):
            eye = jnp.eye(ncoord, dtype=x.dtype)
            zero = fwdlap.zero_tangent_from_primal(x)
            _, jac, lap = fwdlap.lap(reim, (x,), (eye,), (zero,))
            g = jac[:, 0] + 1j * jac[:, 1]
            return -0.5 * (lap[0] + 1j * lap[1] + jnp.sum(g * g)), 0.5 * jnp.sum(jnp.abs(g) ** 2)
        return parts

    def parts(x):
        g = jax.jacfwd(reim)(x)
        h = jax.hessian(reim)(x)
        gc = g[0] + 1j * g[1]
        return -0.5 * (jnp.trace(h[0]) + 1j * jnp.trace(h[1]) + jnp.sum(gc * gc)), 0.5 * jnp.sum(jnp.abs(gc) ** 2)
    return parts


def local_energy_fn(model, variables, mol: Molecule, method: str = "fwdlap") -> Callable:
    """Batched x (batch, N * 3) -> (E_loc complex, T_lap complex, T_grad real)."""
    import jax
    from .continuum import logpsi_fn
    lp = logpsi_fn(model, variables)
    single = lambda x: lp(x[None])[0]                     # noqa: E731
    kin = kinetic_parts_fn(single, mol.n * 3, method)
    pot = potential_fn(mol)

    def one(x):
        tl, tg = kin(x)
        return tl + pot(x), tl, tg
    return jax.vmap(one)


# ------------------------------------------------------------------ state gates

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def start_positions(rng: np.random.Generator, mol: Molecule, n_conf: int, width: float) -> np.ndarray:
    """Each electron on a proton (a random permutation per configuration), plus a Gaussian of the given width."""
    sites = np.repeat(np.arange(len(mol.atoms)), np.round(mol.charges).astype(int))
    out = np.empty((n_conf, mol.n, 3))
    for c in range(n_conf):
        p = rng.permutation(len(sites))
        out[c] = mol.atoms[sites[p[np.arange(mol.n) % len(sites)]]] + width * rng.standard_normal((mol.n, 3))
    return out.reshape(n_conf, -1)


def state_gates(lp: Callable, mol: Molecule, x: np.ndarray, seed: int = 0, tol: float = 1e-6) -> list[dict]:
    """Antisymmetry (every adjacent same-spin pair and random pairs) and decay far out, on configurations x."""
    rng = np.random.default_rng(seed)
    n = mol.n
    x = np.asarray(x)
    base = np.asarray(lp(x))
    gates = []
    pairs = []
    for lo, hi in ((0, mol.n_up), (mol.n_up, n)):
        if hi - lo >= 2:
            pairs += [(i, i + 1) for i in range(lo, hi - 1)]
            pairs += [tuple(rng.choice(np.arange(lo, hi), 2, replace=False)) for _ in range(4)]
    worst = 0.0
    for i, j in pairs:
        y = x.reshape(len(x), n, 3).copy()
        y[:, [i, j]] = y[:, [j, i]]
        d = np.asarray(lp(y.reshape(len(x), -1))) - base
        worst = max(worst, float(np.max(np.abs(np.real(d)))), float(np.max(np.abs(_wrap_angle(np.imag(d) - np.pi)))))
    gates.append({"name": "antisymmetry", "value": worst, "pass": bool(np.isfinite(worst) and worst <= tol),
                  "detail": f"log psi after swapping two electrons of the same spin = log psi + i pi ({len(pairs)} pairs)"})
    centre = mol.atoms.mean(axis=0)
    reach = float(np.max(np.linalg.norm(mol.atoms - centre, axis=1)))
    ok, worst_drop = True, float("inf")
    for c in range(min(8, len(x))):
        i = int(rng.integers(n))
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        vals = [float(np.real(base[c]))]
        for dist in (20.0, 40.0, 80.0):
            y = x[c].reshape(n, 3).copy()
            y[i] = centre + (reach + dist) * u
            vals.append(float(np.real(np.asarray(lp(y.reshape(1, -1)))[0])))
        # strictly falling; -inf (psi underflows, e.g. Gaussian orbitals far out) counts as fallen
        ok &= bool(np.isfinite(vals[0]) and not any(np.isnan(v) or v == np.inf for v in vals) and
                   all(b < a or (a == -np.inf and b == -np.inf) for a, b in zip(vals[:-1], vals[1:])))
        fin = [v for v in vals if np.isfinite(v)]
        if len(fin) >= 2:
            worst_drop = min(worst_drop, float(-np.max(np.diff(fin))))
    gates.append({"name": "decay", "value": worst_drop, "pass": bool(ok),
                  "detail": "log|psi| falls as one electron moves 20, 40, 80 bohr beyond the molecule (8 samples)"})
    return gates


def kinetic_paths_gate(model, variables, mol: Molecule, x: np.ndarray) -> dict:
    import jax
    import jax.numpy as jnp
    a = jax.jit(local_energy_fn(model, variables, mol, "fwdlap"))(jnp.asarray(x))[1]
    b = jax.jit(local_energy_fn(model, variables, mol, "hessian"))(jnp.asarray(x))[1]
    a, b = np.asarray(a), np.asarray(b)
    rel = float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))
    return {"name": "kinetic_paths", "value": rel, "pass": bool(np.all(np.isfinite(a)) and rel <= KINETIC_PATHS_RTOL),
            "detail": f"forward Laplacian vs jax.hessian on {len(x)} samples"}


def kinetic_estimators_gate(t_lap: np.ndarray, t_grad: np.ndarray) -> dict:
    """<T_lap> = <T_grad> (integration by parts for a normalisable state); sigma of the difference by stats."""
    from . import stats as S
    d = np.real(t_lap) - t_grad
    sig = S.sigma_estimates(d)["sigma"]
    z = float(abs(d.mean()) / sig) if sig > 0 else (0.0 if abs(d.mean()) < 1e-12 else float("inf"))
    status = "pass" if z < 3 else "warn" if z < KINETIC_Z_FAIL else "fail"
    return {"name": "kinetic_estimators", "value": z, "pass": status != "fail", "status": status,
            "detail": f"mean -1/2 (lap + grad^2) log psi {float(np.real(t_lap).mean()):.8f} vs mean 1/2 |grad log psi|^2 "
                      f"{float(t_grad.mean()):.8f} (difference {float(d.mean()):+.2e} +- {sig:.2e})"}


# ------------------------------------------------------------------ sampler and collection

def collect(model, variables, mol: Molecule, *, seed: int, n_chains: int, n_burn: int, thin: int, block: int,
            min_samples: int, max_seconds: float, chunk: int = 64, method: str = "fwdlap",
            cfg: dict | None = None, log: Callable = print) -> dict:
    """Chains from the verifier's starts; e_loc, t_lap, t_grad, logabs (n_chains, n_steps), the fraction of
    recorded samples that moved per chain and quarter block, the step and the gate samples."""
    import jax
    import jax.numpy as jnp
    from .continuum import logpsi_fn
    jax.config.update("jax_enable_x64", True)
    cfg = dict(MOLECULE, **(cfg or {}))
    n = mol.n
    lp = jax.jit(logpsi_fn(model, variables))
    eloc = jax.jit(local_energy_fn(model, variables, mol, method))

    def logabs(x):
        return jnp.real(logpsi_fn(model, variables)(x))

    def run(key, x, la, step, k):
        def body(c, kk):
            x, la = c
            k1, k2 = jax.random.split(kk)
            y = x + step * jax.random.normal(k1, x.shape)
            lb = logabs(y)
            acc = jnp.log(jax.random.uniform(k2, la.shape)) < 2 * (lb - la)
            return (jnp.where(acc[:, None], y, x), jnp.where(acc, lb, la)), acc
        (x, la), a = jax.lax.scan(body, (x, la), jax.random.split(key, k))
        return x, la, a.mean(0)
    run = jax.jit(run, static_argnums=4)

    rng = np.random.default_rng(seed)
    x = jnp.asarray(start_positions(rng, mol, n_chains, cfg["start_width"]))
    key = jax.random.PRNGKey(seed)
    la = logabs(x)
    step = 0.3
    t0 = time.perf_counter()
    done = 0
    while done < n_burn:                     # step tuned in the first 80 % of the burn-in, then frozen
        k = min(100, n_burn - done)
        key, kk = jax.random.split(key)
        x, la, acc = run(kk, x, la, step, k)
        a = float(np.mean(np.asarray(acc)))
        if done < 0.8 * n_burn:
            step = min(step * math.exp(1.5 * (a - cfg["target_acc"])), cfg["max_step"])
        done += k
    log(f"[hydrogen] burn-in {n_burn} moves, step {step:.4g} bohr, {time.perf_counter() - t0:.1f} s")

    def eval_e(xs):
        outs = [jax.tree_util.tree_map(np.asarray, eloc(xs[i:i + chunk])) for i in range(0, len(xs), chunk)]
        return [np.concatenate([o[k] for o in outs]) for k in range(3)]

    e_b, tl_b, tg_b, l_b, m_b = [], [], [], [], []
    prev = np.asarray(x)
    stop = "budget"
    while sum(b.shape[1] for b in e_b) * n_chains < min_samples:
        xs, ls = [], []
        for _ in range(block):
            key, kk = jax.random.split(key)
            x, la, _ = run(kk, x, la, step, thin)
            xs.append(np.asarray(x))
            ls.append(np.asarray(la))
        xs = np.stack(xs, 1)
        seq = np.concatenate([prev[:, None], xs], 1)
        moved = np.any(seq[:, 1:] != seq[:, :-1], axis=-1)
        q = max(block // 4, 1)
        m_b.extend(moved[:, i:i + q].mean(1) for i in range(0, block, q))
        prev = xs[:, -1]
        e, tl, tg = eval_e(xs.reshape(-1, n * 3))
        e_b.append(e.reshape(n_chains, block))
        tl_b.append(tl.reshape(n_chains, block))
        tg_b.append(tg.reshape(n_chains, block))
        l_b.append(np.stack(ls, 1))
        if not np.all(np.isfinite(e_b[-1])):
            stop = "non_finite_local_energy"
            break
        if time.perf_counter() - t0 > max_seconds:
            stop = "time"
            break
    e = np.concatenate(e_b, 1)
    return {"e_loc": np.real(e), "t_lap": np.concatenate(tl_b, 1), "t_grad": np.concatenate(tg_b, 1),
            "logabs": np.concatenate(l_b, 1), "acc": np.stack(m_b, 1),
            "imag_max": float(np.max(np.abs(np.imag(e)))), "wall_s": time.perf_counter() - t0, "step": step,
            "stop": stop, "seed": seed, "n_burn": n_burn, "thin": thin, "block": block,
            "gate_samples": prev[: min(16, n_chains)], "logpsi": lp,
            "device": str(jax.devices()[0]), "jax": jax.__version__}


def evaluate(inst: dict, model, variables, *, seed: int, max_hours: float = 12.0, min_samples: int | None = None,
             method: str = "fwdlap", state_id: str = "", log: Callable = print, **overrides) -> dict:
    """Score one hydrogen state in continuous space: the scorecard of measure.scorecard plus the physics gates."""
    from . import measure as Me
    mol = molecule_of(inst)
    cfg = dict(MOLECULE, **overrides)
    budget = min_samples or cfg["min_samples"]
    desc = (f"all-electron Gaussian Metropolis, {cfg['n_chains']} chains, {cfg['thin']} moves per sample, step tuned "
            f"to acceptance {cfg['target_acc']} in the first 80 % of {cfg['n_burn']} burn-in moves")
    log(f"[hydrogen] {inst['instance_id']}: {desc}; budget {budget} samples; cap {max_hours} h; seed {seed}")
    ser = collect(model, variables, mol, seed=seed, n_chains=cfg["n_chains"], n_burn=cfg["n_burn"],
                  thin=cfg["thin"], block=cfg["block"], min_samples=budget, max_seconds=max_hours * 3600,
                  method=method, cfg=cfg, log=log)
    gx = ser["gate_samples"]
    gates = [coulomb_gate(mol, gx), kinetic_paths_gate(model, variables, mol, gx[:8]),
             kinetic_estimators_gate(ser["t_lap"], ser["t_grad"])] + state_gates(ser["logpsi"], mol, gx, seed=seed)
    card = Me.scorecard(inst, ser, state_id=state_id or Me.state_hash(variables), sampler_desc=desc)
    card["state_hash"] = Me.state_hash(variables)
    card["physics_gates"] = gates
    card["stop"] = ser["stop"]
    card["step_bohr"] = ser["step"]
    card["local_energy"] = f"verifier Coulomb pair sums + kinetic by {method}"
    card["budget_samples"] = budget
    floor = H2_EXACT.get(float(inst["params"]["R"])) if int(inst["n_sites"]) == 2 else None
    if floor is not None and card["energy_total"] < floor - 3 * card["sigma_total"]:
        card["plausibility"] = f"BELOW the exact H2 energy {floor}: bug until shown otherwise"
        card["status"] = "fail"
    if not all(g["pass"] for g in gates):
        card["status"] = "fail"
    elif card["status"] == "pass" and any(g.get("status") == "warn" for g in gates):
        card["status"] = "warn"
    card["qualified"] = bool(card["status"] != "fail" and card["stop"] == "budget")
    card["__METRIC__"] = card["energy_per_site"]
    card["scorer"] = "qrec_verify.molecule v1.10"
    return card


# ------------------------------------------------------------------ Gaussian orbitals: the known states

SPH_NORM = {0: 1.0 / math.sqrt(4 * math.pi), 1: math.sqrt(3.0 / (4 * math.pi))}


def basis_shells(pmol) -> list[dict]:
    """The shells of a pyscf Mole (s and p only, as hydrogen in cc-pVDZ): centre, l, exponents and pyscf's
    contraction coefficients (radial normalisation included); AO order as pyscf's (p: x, y, z)."""
    shells = []
    for i in range(pmol.nbas):
        l = int(pmol.bas_angular(i))
        if l > 1:
            raise ValueError("only s and p shells are supported")
        shells.append({"center": np.asarray(pmol.bas_coord(i), float), "l": l,
                       "exps": np.asarray(pmol.bas_exp(i), float),
                       "coeffs": np.asarray(pmol._libcint_ctr_coeff(i), float)})
    return shells


def ao_fn(shells: list[dict]) -> Callable:
    """JAX: r (3,) -> AO values (n_ao,) in pyscf's order (the verifier's own evaluation)."""
    import jax.numpy as jnp

    def f(r):
        out = []
        for s in shells:
            d = r - jnp.asarray(s["center"])
            rad = jnp.exp(-jnp.asarray(s["exps"])[:, None] * jnp.sum(d * d)) * jnp.asarray(s["coeffs"])
            rad = jnp.sum(rad, axis=0) * SPH_NORM[s["l"]]
            out.append(rad if s["l"] == 0 else (rad[:, None] * d[None, :]).reshape(-1))
        return jnp.concatenate(out)
    return f


def _pyscf_mol(mol: Molecule, basis: str):
    from pyscf import gto
    return gto.M(atom=[("H", tuple(a)) for a in mol.atoms], unit="Bohr", basis=basis, spin=mol.n_up - mol.n_dn,
                 verbose=0)


def rhf_model(inst: dict, basis: str = "cc-pvdz"):
    """pyscf's RHF (ROHF for odd N) determinant of the instance as a flax module without parameters, evaluated by
    the verifier's own Gaussian-orbital code; returns (module, variables, e_rhf, info)."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from pyscf import scf
    mol = molecule_of(inst)
    pm = _pyscf_mol(mol, basis)
    mf = scf.RHF(pm) if mol.n_up == mol.n_dn else scf.ROHF(pm)
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel()
    cu = jnp.asarray(mf.mo_coeff[:, mf.mo_occ > 0][:, :mol.n_up])
    cd = jnp.asarray(mf.mo_coeff[:, mf.mo_occ > 1.5][:, :mol.n_dn])
    aos = jax.vmap(jax.vmap(ao_fn(basis_shells(pm))))
    n_up = mol.n_up

    class RHFDeterminant(nn.Module):
        @nn.compact
        def __call__(self, x):
            r = x.reshape(x.shape[0], -1, 3)
            phi = aos(r)                                            # (batch, N, n_ao)

            def logdet(p, C):
                if C.shape[1] == 0:
                    return jnp.zeros(r.shape[0], jnp.complex128)
                s, la = jnp.linalg.slogdet(p @ C)
                return la + jnp.log(s.astype(jnp.complex128))
            return logdet(phi[:, :n_up], cu) + logdet(phi[:, n_up:], cd)

    return RHFDeterminant(), {}, float(mf.e_tot), {"basis": basis, "converged": bool(mf.converged),
                                                   "n_ao": int(pm.nao)}


def h2_fci_model(R: float, basis: str = "cc-pvdz"):
    """The two-electron FCI ground state of H2 at R bohr in a Gaussian basis (pyscf), psi(r_up, r_dn) =
    sum_ij C_ij phi_i(r_up) phi_j(r_dn) in pyscf's MOs, as a flax module; returns (module, variables, e_fci)."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from pyscf import fci, scf
    mol = molecule_of(small_instance(f"h2:{R}"))
    pm = _pyscf_mol(mol, basis)
    mf = scf.RHF(pm).run(conv_tol=1e-12)
    e, ci = fci.FCI(mf).kernel()
    # one electron per spin: alpha and beta strings are the orbitals in order
    C = jnp.asarray(mf.mo_coeff @ np.asarray(ci) @ mf.mo_coeff.T)    # in the AO basis
    aos = jax.vmap(jax.vmap(ao_fn(basis_shells(pm))))

    class H2FCI(nn.Module):
        @nn.compact
        def __call__(self, x):
            phi = aos(x.reshape(x.shape[0], 2, 3))
            v = jnp.einsum("bi,ij,bj->b", phi[:, 0], C, phi[:, 1])
            return jnp.log(v.astype(jnp.complex128))

    return H2FCI(), {}, float(e)


def hydrogen_1s_model():
    """The exact hydrogen-atom ground state exp(-r) (proton at the origin): E_loc = -1/2 at every point."""
    import flax.linen as nn
    import jax.numpy as jnp

    class H1s(nn.Module):
        @nn.compact
        def __call__(self, x):
            return (-jnp.linalg.norm(x.reshape(x.shape[0], 3), axis=-1)).astype(jnp.complex128)
    return H1s(), {}


def rhf_check(inst: dict, *, seed: int = 7, max_hours: float = 1.0, min_samples: int | None = None,
              basis: str = "cc-pvdz", log: Callable = print, **overrides) -> dict:
    """Known state at full size: the RHF determinant must score pyscf's RHF energy within 3 sigma."""
    import jax
    jax.config.update("jax_enable_x64", True)
    model, variables, e_ref, info = rhf_model(inst, basis)
    card = evaluate(inst, model, variables, seed=seed, max_hours=max_hours, min_samples=min_samples,
                    state_id=f"RHF/{basis}", log=log, **overrides)
    z = (card["energy_total"] - e_ref) / card["sigma_total"] if card["sigma_total"] > 0 else float("inf")
    return {"instance_id": inst["instance_id"], "state": f"RHF determinant ({basis}, pyscf orbitals, the verifier's "
            "Gaussian-orbital evaluation)", "expected_total": e_ref, "energy_total": card["energy_total"],
            "sigma_total": card["sigma_total"], "z": float(z), "rhf_converged": info["converged"],
            "status": card["status"], "physics_gates": card["physics_gates"],
            "pass": bool(abs(z) < 3 and card["status"] != "fail")}
