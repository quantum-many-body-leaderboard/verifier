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
