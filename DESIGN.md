# Verifier design (v1.7, 2026-10-09)

The verifier re-measures a trained state and decides whether its energy beats a QMBL record. It is
separate from every training code base: it builds the Hamiltonian itself from the QMBL instance file,
runs its own chains from random configurations, and computes its own statistics. The submitter supplies
only the wave function: a function returning log psi in any framework plus a parameter checkpoint (and,
for large fermion systems, a bounded local-energy kernel, which the verifier checks against the generic
one on a sample batch).

## States from other frameworks (2026-10-15)

The scorer measures a flax module. `qrec_verify/external.py` turns every other state into one, so the
samplers, the estimator, the gates and the budgets are the same for every entry:
- JAX without flax: `apply_fn(params, x)` and its parameter tree become a flax module that is traced and
  compiled like any other (the parameter tree is the variable `params/tree`).
- PyTorch, C/C++ (ctypes), plain Python and Julia: a host function `logpsi(x int8 (n, n_dof)) ->
  complex128 (n,)`, called from inside the compiled sampler through `jax.pure_callback` (one host call
  per batch, vmap dimensions folded into the batch, chunks of at most 8192 configurations, one call at a
  time). Julia runs in a child process (`qrec_verify/hostproc.py`), because it must stay on the thread
  that started it.
- The state hash of a host state is the sha256 of its files (code or library and parameters), stored
  as the variable `qmbl_state/sha256`; the scorecard records `state_framework`.
`tests/test_adapters.py` checks that the same states written in every framework agree with flax to
1e-10 in log psi and in the full-sum energy and give the same Monte Carlo energy.

Code: `qrec_verify/` (package), `tests/`. Statistics (`stats.py`) are numpy/scipy only and know nothing
about physics; everything physical is in `instance.py`, `hamiltonians.py` and `measure.py`.

## Continuous space: the electron gas (2026-11-10)

QMBL's electron-gas instances (`HEG2D`, `HEG3D`, RULES.md §12) are scored by `qrec_verify/continuum.py`.
A state is a flax module or a JAX function (`--state jax:...`) of the electron positions: `x` of shape
`(batch, N * dim)`, Cartesian coordinates in bohr, the N_up spin-up electrons first. It returns log psi
(complex where needed) or a pair `(sign, log|psi|)`, the output of fvmc's wave functions. Host states
(PyTorch, C, Julia, numpy) are refused: the kinetic energy needs derivatives of log psi.
- Cell: the instance's lattice (`sc`, `bcc`, `fcc`, `square`, `triangular`, `rect-AxBxC`) scaled to
  N electrons at rs, exactly as QMBL's `checks/heg-hf/hf.mjs`.
- Coulomb energy: an Ewald sum with the uniform background, E = 1/2 sum_{i != j} phi(r_i - r_j) + N xi / 2,
  in 2D and 3D, real-space images after reduction to fractional coordinates and the reciprocal sum as a
  structure factor; the splitting is set by the cell so that both sums are converged to 1e-16.
- Kinetic energy: -1/2 (lap log psi + (grad log psi)^2) by the forward Laplacian (`_vendor/fwdlap.py`,
  Y. Chen, Apache-2.0, unchanged at 84c2dc7); `jax.hessian` is the cross-check in the tests.
- Sampler: all-electron Gaussian Metropolis moves from uniform random positions, the step tuned to 50 %
  acceptance during burn-in only (then fixed); at least N moves between recorded samples and 20 N moves
  of burn-in. Mobility is the fraction of recorded samples that moved.
- Gates beyond the statistics: `madelung` (the verifier's xi against the hf.mjs formula at another
  splitting, 1e-10), `antisymmetry` (log psi + i pi after swapping two same-spin electrons) and `twist`
  (log psi + i k_s . A_k after moving one electron by a cell vector), 1e-6 on samples of the chains.
- Known state (`op_verify.py hf-check`): the plane-wave Slater determinant, scored by the verifier, gives
  QMBL's Hartree-Fock energy within 4 sigma, and its kinetic local energy is the constant sum k^2 / 2.
- Checked against independent codes on six cells (sc, bcc, fcc, rect 3x3x2, square, triangular):
  hf.mjs (xi and Hartree-Fock energies, `tests/test_continuum.py`), FermiNet's `pbc/ewald.py`
  (5e-15 absolute, `tests/xcheck_ferminet.json`), fvmc's Ewald sum (derived from PyQMC; 3e-9, the
  limit of its reciprocal cut, `tests/xcheck_fvmc.json`) and the verifier's own sum at a second
  splitting (1e-11).

## v1.9 (2026-11-11): the lattices of qrec-verifier, t-J, hydrogen in a Gaussian basis, impurity models

The public line and qrec-verifier (the QMBL record loops' own verifier) split after v1.7. The public line went on
with MPS files, framework adapters and the electron gas (its v1.8, above); qrec-verifier added the triangular lattice
(its v1.8), kagome, pyrochlore and shuriken (its v1.9), Shastry-Sutherland (its v1.10) and the transverse-field Ising
model (its v1.11). The version numbers collide; the merge is v1.9 of the public line. What came from where:

| part | from | where |
|---|---|---|
| MPS files, adapters, electron gas, statistics, scorecard | public line v1.7-v1.8, unchanged | `mps.py`, `external.py`, `continuum.py`, `stats.py`, `measure.py` |
| triangular L x L and 3 m^2 (C6v) clusters, stripe known state | qrec-verifier v1.8 | `hamiltonians.triangular_cluster`, `score.stripe_check` |
| kagome, pyrochlore (primitive and 16-site cubic cells), shuriken, configuration-pair known state | qrec-verifier v1.9 | `hamiltonians.frustrated_cluster`, `score.frustrated_check` |
| Shastry-Sutherland on the torus, Neel-pair known state | qrec-verifier v1.10 | `hamiltonians.shastry_dimers`, `score.shastry_check` |
| Shastry-Sutherland on the open square (setup B of arXiv:2309.10955) | new | `hamiltonians.shastry_dimers(L, "O")` |
| transverse-field Ising, ferromagnetic-pair known state, single-flip sampler | qrec-verifier v1.11 | `hamiltonians.tfising`, `score.tfising_check` |
| honeycomb L x M tori (Heisenberg, J1-J2) | new | `frustrated_cluster("honeycomb", ...)` |
| t-J (clusters, Hilbert space without double occupancy, operator, sampler) | new | `tj.py` |
| hydrogen chains in a Gaussian basis | new | `chem.py` |
| Anderson impurity models (VarBench's parameters, forkTPS's Kanamori form) | new | `impurity.py`, `instances/Impurity/HamParams/` (VarBench, Apache-2.0) |
| MPOs of every new spin lattice, of TFIsing and of t-J | new | `mps.py` (`spin_lattice`, channels `zz`, `tjhop`, `tjJ`) |

Not taken from qrec-verifier: its continuum hydrogen path (`molecule.py`, `molecular_score.py`, its v2.1, for the
`basis: none` instances) and the record loops' own bookkeeping. Comments ported from qrec-verifier are rewritten for
the public command line (`op_verify.py`).

One bond list per lattice: `hamiltonians.spin_lattice(inst)` returns the bonds and couplings of every Heisenberg-type
instance (square, rectangular, triangular, kagome, pyrochlore, shuriken, honeycomb, Shastry-Sutherland); the sampled
NetKet Hamiltonian, the spin sampler's exchange moves, the MPO and the exact diagonalisations of the tests are all
built from it. Square and Hubbard systems are unchanged bit for bit.

MPS route (`mps-score`): the new spin lattices, TFIsing and t-J are built as MPOs (checked against NetKet's operators
in `tests/test_mps.py`). Hydrogen (an MPO of the full two-electron Hamiltonian has bond dimension of order n_orb^2) and
the impurity models (spin-dependent complex hybridisation, Kanamori terms) are refused with that reason.

Gates of the new models (the statistics are unchanged):
- Hamiltonian identity: every new lattice or model reproduces QMBL's exact rows (1e-8 relative, or half a unit of the
  row's last printed decimal when it is printed to fewer digits) where a diagonalisation fits: NetKet's Lanczos up to
  about 24 sites, `tests/ed_sz.py` (matrix-free S^z-sector Lanczos on the verifier's bond lists) up to 32 spins,
  `tests/ed_tj.py` (matrix-free t-J Lanczos) up to about 2e8 states, `tests/dmrg_mpo.py` (DMRG on the verifier's own
  MPO) for long chains, pyscf's FCI for hydrogen. Where QMBL has no exact row, a small cluster of the same
  construction against an independent one: NetKet's lattices and operators (honeycomb, open Shastry-Sutherland,
  t-J matrix elements), pyscf's own FCI in canonical orbitals (hydrogen), TeNPy's independently built model on a
  truncated bath, forkTPS's atomic-limit test and full-size DMRG against VarBench's rows (three-band impurity
  models). The spin-orbit impurity models fail the last check (DMRG of the verifier's Hamiltonian goes below
  VarBench's converged energy) and are refused. The exact rows that are not reproduced
  (clusters beyond a plain S^z Lanczos) are listed in `COVERAGE.md`.
- Known state at full size (`op_verify.py neel-check`): a state whose energy follows in closed form from the geometry
  or the parameters: the stripe pair (triangular), configuration pairs (kagome, pyrochlore, shuriken, honeycomb), the
  Neel pair (Shastry-Sutherland), the ferromagnetic pair (TFIsing), and single configurations for t-J, hydrogen and
  impurity models (the diagonal of H, by the Slater-Condon rule from the integrals for hydrogen).
- Sector: t-J configurations never hold a doubly occupied site (constrained Hilbert space; the sampler only exchanges
  site occupations; the MPS sector gate counts doubly occupied sites). Hydrogen and impurity states are sampled at
  fixed (N_dn, N_up).
- Size: hydrogen is built with NetKet's operator up to 128 orbitals (n_orb^4 two-electron integrals); above, v1.10
  factorises the integrals (below). Continuous-space hydrogen (`basis: none`) is scored by v1.10's molecule.py.
- Units and per-site divisors (DATA.md): Pauli totals for spin models (TFIsing per site E / N), units of t for t-J
  (E / N), hartree per atom with the nuclear repulsion for hydrogen, totals for impurity models (divisor 1).

## v1.10 (2026-11-11): spin-orbit impurity models, hydrogen in continuous space and in large Gaussian bases

What came from where:

| part | from | where |
|---|---|---|
| hydrogen in continuous space (`basis: none`): geometry, Coulomb sums, the RHF known state, sampler settings | qrec-verifier v2.1 (`molecule.py`, `molecular_score.py`), rewritten on the public line's continuous-space interface | `molecule.py` |
| kinetic energy for hydrogen (forward Laplacian) | public line v1.8 (`continuum.py`'s vendored fwdlap) | `molecule.kinetic_parts_fn` |
| spin-orbit impurity models: forkTPS's Kanamori roles, total-N space | new (forkTPS's `AIM_SpinOrbit.cpp`) | `impurity.roles`, `impurity.build_from_params` |
| particle moves between the spin blocks | new | `score.particle_move_rule` (total-N spaces only) |
| hydrogen above 128 orbitals: Cholesky integrals, determinant kernel, `det:` states | new | `chem_large.py`, `external._load_det` |
| symmetry-resolved Lanczos for the big exact rows | new | `tests/ed_sym.py` |

Spin-orbit impurity models (TB-DMFT-SOC). forkTPS orders the six impurity sites ud_0[0], ud_1[0], ud_1[1], ud_0[1],
ud_1[2], ud_0[2] and writes the Kanamori interaction by site position (odd "up", even "down"), so orbital 0 has its
blocks' roles swapped relative to orbitals 1 and 2. v1.9 wrote ud_1 as "up" for every orbital; that Hamiltonian's DMRG
went below VarBench's converged forkTPS rows and the models were refused. With forkTPS's roles the spin-flip and pair-
hopping terms between orbital 0 and orbitals 1, 2 move electrons between the blocks: only the total N is conserved,
the Hilbert space is the total-N space, and the sampler adds moves of one electron between any two spin-orbitals.
Gate: full-size DMRG (TeNPy, independently built) of TB-DMFT-SOC_9 lands 4.2e-7 above VarBench's lowest row (3.7e-9
relative; TB-DMFT_9: 8.7e-7 above); the conjugate hybridisation convention falls 1.0e-3 below, so the check
discriminates. Truncated bath: ED of the verifier's operator = TeNPy DMRG to 6e-14.

Hydrogen in continuous space. Same contract as the electron gas (a state of electron positions, the verifier's own
local energy, Metropolis sampler, statistics, scorecard). Gates on every score: Coulomb sums against pyscf's
point-charge sum (independent code), forward Laplacian against jax.hessian, the Laplacian and gradient kinetic
estimators against each other (integration by parts), antisymmetry, decay far out. References in the tests: the
hydrogen atom's exact state (E_loc = -1/2 at every point, sigma 0), the two-electron FCI wave function of H2 in
cc-pVDZ (scores pyscf's FCI energy, above the exact H2 energy of Pachucki 2010), RHF determinants (score pyscf's
RHF energies, above the sources' near-exact rows). Periodic cells are refused (no QMBL instance has one).

Hydrogen above 128 orbitals. The Hamiltonian is unchanged (chem.py's Loewdin orbitals); the two-electron integrals are
stored as a pivoted Cholesky decomposition of pyscf's AO integrals (residual diagonal below 1e-10) transformed to the
Loewdin orbitals. Determinant states are scored with the verifier's kernel (mixed Green's function and Wick's theorem,
exact for a determinant); general states are not (about 10^6 connected configurations per sample). Gates: up to 128
orbitals the Cholesky integrals equal pyscf's (5e-9), the kernel equals NetKet's generic estimator with the dense
operator and the verifier's own Slater-Condon engine (`chem_large.connected`) equals NetKet's get_conn; above,
pyscf's RHF energy is reproduced exactly by the determinant's density-matrix energy and by Monte Carlo of the kernel,
<H> of a random CI vector in an active space equals pyscf's direct_spin1 contraction with pyscf's own integrals, and
every score checks the kernel against `connected` on one sample (up to 1e6 connected configurations). The hop graph
of the hydrogen sampler is now complete (v1.9 linked orbitals with |h_ab| > 1e-6, which leaves sigma and pi orbitals
of a chain unconnected: a chain starting with an electron where a sigma determinant vanishes could not leave).
Refused: Cholesky vectors above 64 GB (above 850 orbitals) and nearly singular overlaps.

Big exact rows. `tests/ed_sym.py` diagonalises the trivial sector of the full automorphism group of the verifier's
bond list times the spin flip, or the sector of any real 1D character of that group (`--character K FLIP`; the
dimer-singlet character of the Shastry-Sutherland torus is one of them, `--dimer-sector`); TFIsing (no U(1)) over
all 2^n configurations, where the trivial sector holds the ground state (stoquastic). The lowest energy of a sector is
an upper bound; equality with a row to 1e-8 reproduces it. Checked on 16-site rows and on a 12-site ring (every real
sector against brute-force projection, 3e-14). Reproduced: Heisenberg triangular 36 P, TFIsing square 36 P (h = 3)
and the Shastry-Sutherland square 36 P rows listed in COVERAGE.md (1e-14 to 3e-11). Not attempted: the 48-site rows
(triangular, kagome-4x4), whose sectors hold about 3e10 states (triangular, |G| = 1152) and 6e10 (kagome); the 36-site matrix alone takes about 7 GB per 6e7 states.

## Input to the statistics

Arrays shaped `(n_chains, n_steps)` in sampling order, after the declared burn-in:
- `e_loc`: local energies (total energy units; the caller passes the per-site divisor);
- `logabs`: log|psi| of each sample (optional, for mode coverage and R-hat on the state itself);
- `acc`: acceptance rate per chain per block (optional).

## Gates

| gate | quantity | pass | warn | fail |
|---|---|---|---|---|
| chains | number of independent chains M | M >= 32 | 16 <= M < 32 | M < 16 |
| mixing | rank-normalised split R-hat, bulk and folded, on e_loc and log|psi| (Vehtari et al. 2021) | <= 1.01 | <= 1.05 | > 1.05 |
| mixing | ESS bulk / ESS tail of e_loc (Geyer initial monotone sequence, chains combined as in Stan) | >= 1e4 / >= 1e3 | lower | |
| mobility | fraction of consecutive samples (one sweep apart) that changed, per chain and block; chains frozen in the last quarter; last-quarter / first-quarter ratio | > 0.005, no frozen chain, ratio >= 0.5 | ratio < 0.5 | frozen chain or mean <= 0.005 |
| modes | chains whose median log|psi| sits Delta above the median of chain medians: weight = fraction x exp(2 Delta) | weight <= 0.05 | weight > 0.05 | weight > 0.05 and Delta >= 5 |
| thermalisation | Geweke z (first 10 % vs last 50 % of every chain; sigmas from the spread of chain means, valid at any autocorrelation); first vs second half z; MSER-5 truncation of the chain-averaged series and the z of the cut part against the rest | \|z\| < 3, MSER cut < 10 % | MSER cut >= 10 % with \|z\| < 3 | any \|z\| >= 3 |
| error bar | sigma from (a) variance x integrated autocorrelation time, (b) Jonsson's automated blocking of the chains laid end to end, (c) spread of chain means | max/min <= 1.5; the reported sigma is the largest | > 1.5 | |
| tails | Hill tail index alpha of \|e_loc - median\| (k = sqrt(n) and k = 0.1 % of n; the smaller counts) | alpha >= 3 | alpha < 3; below 2 the variance is infinite and sigma unreliable (a warning since 2026-10-08: claims rest on two agreeing measurements) | |
| precision | reported sigma per site against the caller's target; entries have no target, so the gate is not applied to them | <= target | | > target |

The physics gates live in `measure.py` and the scorecard: Hamiltonian identity (exact diagonalisation on
a small instance of the same family to 1e-10, a known state at full size, `dof`/`einf` recomputed and
compared with the QMBL row), local-energy identity (bounded kernel vs generic, 1e-10), plausibility
against exact / fixed-node / extrapolated references, and the record decision.

## Deterministic route: MPS files (2026-10-15)

A matrix product state is submitted as a file (`MPS_FORMAT.md`) and evaluated exactly
(`qrec_verify/mps.py`, `op_verify.py mps-score`). The verifier takes only the tensors and the declared
site order from the file. It builds the Hamiltonian as an MPO from its own bond lists (`square_bonds`),
with Jordan-Wigner strings in the format's fixed fermion order, and contracts `<psi|H|psi>/<psi|psi>`
and, where it fits, `<H^2>`. Any state evaluated this way is a true upper bound, whatever conventions
the submitter used, so convention errors cost the entrant energy but cannot fake a record. Gates: format,
size, real energy, sector (`<(Q - Q0)^2> <= 1e-8`), stated energy (relative 1e-8), sampling cross-check
(perfect samples from the MPS, local energies with NetKet's Hamiltonian and its own fermion ordering,
|z| <= 4), variance >= 0. The card has `sigma_per_site` 0; `combine` requires two cards of the same file
with different seeds and equal energies (relative 1e-11). `tests/test_mps.py` checks the MPO against
NetKet's sparse Hamiltonian on random states in the sector (1e-10) and DMRG states from TeNPy and
ITensors.jl against QMBL's exact energies.

## Decision

- One state is measured twice with independent sampler seeds, each with a fixed sample budget. The
  entry verifies when both measurements qualify (every gate passes or only warns) and agree within
  2.5 sigma_combined; the entry's energy is their inverse-variance mean (`op_verify.py combine`).
- Every entry is listed. One that does not verify carries its problems as flags and takes no
  record mark. No error bar is asked of the entrant: the verifier measures its own.
- The target is QMBL's state-of-the-art variational energy (the exact energy where one exists), never
  an extrapolation or a projection. An entry more than 2 sigma_combined below it is a new record
  (QMBL's tie rule, `RULES.md` §6).
- Below an exact energy: bug. Below a projected or extrapolated number: allowed and listed, with a
  second measurement with a different sampler and local-energy implementation before any public claim.

## Scorecard

One JSON per measurement: instance, state hash, sampler and seeds, energy and sigma (total and per site),
variance and V-score in QMBL's convention, every gate value with pass/warn/fail, the three sigma
estimates, R-hat, ESS, IAT, acceptance, tail indices, margin and z against the record, the share of the
gap to the best reference closed, wall time and device.
