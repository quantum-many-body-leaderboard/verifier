# Verifier

The re-measurement of [QMBL](https://qmbl.org) and QMBL: Open Problems. `qrec_verify/` is the
scorer (v1.7, 2026-10-09; v1.9, 2026-11-11, adds the lattices and models below; v1.10 the spin-orbit impurity
models, hydrogen in continuous space and in large Gaussian bases); `DESIGN.md` lists its gates
and what came from where. `op_verify.py` is the command line: it
reads the instance from a QMBL instance file and turns two measurements into one entry.
The tests in `tests/` need NetKet: `python tests/test_stats.py`, `python tests/test_physics.py`,
`python tests/test_score.py --cli`, for the framework adapters `python tests/test_adapters.py --cli`
(`tests/bench_adapters.py` times them), for MPS files `python tests/test_mps.py --netket --tenpy [--itensor]`.
Apache-2.0.

## What it does with a state

It builds the Hamiltonian itself from the QMBL instance (model, lattice, boundary, couplings),
starts its own Markov chains from random configurations, samples a fixed budget (65,536 samples
for spins, 131,072 for fermions) and computes its own statistics:
split R-hat on the local energy and on log|psi|, effective sample size, autocorrelation,
thermalisation, mode coverage, tails, and three independent error-bar estimates, of which it
reports the largest. A measurement qualifies when every gate passes or only warns; its error bar
is reported as reached, with no precision required. An entry that does not verify is listed with
the reasons as flags.

## The state you submit

A function that returns log psi(x), complex, for a batch of configurations `x`, plus the file
with its parameters. Any framework works: the verifier calls the function and does everything
else (sampling, local energies, statistics) itself, the same way for every entry. Any sign rule
(Marshall, stripes) is part of the state; the Hamiltonian has none.

`--state` names the framework (details and complete examples: `adapters/README.md`):

| framework | `--state` | `--params` |
|---|---|---|
| flax (default) | `file.py:build_model` | flax msgpack of the variables |
| PyTorch | `torch:file.py:NAME` (an `nn.Module` class or factory) | `torch.save(model.state_dict())` |
| JAX without flax (Equinox, Haiku, plain) | `jax:file.py:NAME` (returns `apply_fn(params, x)`) | `.npz` or `.msgpack` pytree |
| Julia | `julia:file.jl` (defines `logpsi(x::Matrix{Int8})::Vector{ComplexF64}`) | optional, passed to `init(path)` |
| C / C++ | `c:libstate.so` (exports `logpsi`, `adapters/qmbl_state.h`) | optional, passed to `qmbl_init(path)` |
| any other (numpy host function) | `python:file.py:NAME` (returns `logpsi(x)`) | optional, passed to `NAME` |
| a Slater determinant, hydrogen in a Gaussian basis (v1.10) | `det:state.npz` (arrays `phi_dn`, `phi_up`) | none |

`NAME` gets the QMBL instance as a dict when it takes a positional argument. Every framework sees
the same input: `x` is `int8` of shape `(n_conf, n_dof)`, one configuration per row in the
conventions below; the output is complex128 of shape `(n_conf,)`. flax and JAX states are
compiled into the samplers; the others are called from them through `jax.pure_callback`, which
costs time (`adapters/README.md`, "Speed"). Before you submit, check that
`op_verify.py logpsi-check` reproduces your own log psi on a few configurations.

For large fermion systems you may add a bounded local-energy kernel
(`file.py:factory(model, instance) -> fn(variables, x) -> e_loc`); it is used only after it
agrees with NetKet's generic estimator on the verifier's own samples.

Conventions (`qrec_verify/hamiltonians.py`): sites are numbered `i = x * ly + y` on an `lx x ly`
lattice. Spin models use `netket.hilbert.Spin(s=0.5, N, total_sz=0)` (entries +-1) and the
Heisenberg operator in Pauli units without a sign rule. Hubbard models use
`netket.hilbert.SpinOrbitalFermions(N, s=0.5, n_fermions_per_spin=(n_down, n_up))`, NetKet's
orbital order, hopping -t, and a minus sign on bonds that wrap an antiperiodic axis.

Before an open instance, run your state's family on a solved one: `op_verify.py ed-gate` scores a
state built for a small lattice against exact diagonalisation, and QMBL lists 243 instances with an
exact energy (qmbl.org/instances/).

## An electron-gas state

QMBL's electron-gas instances (`HEG2D`, `HEG3D`) are scored in continuous space
(`qrec_verify/continuum.py`, `DESIGN.md`). The state is a flax module (`file.py:build_model`) or a JAX
function (`jax:file.py:NAME`) of the electron positions: `x` is float64 of shape `(n_conf, N * dim)`,
Cartesian coordinates in bohr inside the simulation cell, the N_up spin-up electrons first. It returns
log psi (complex128 where psi is complex) or a pair `(sign, log|psi|)`, the form fvmc's wave functions
use. Host states (PyTorch, C, Julia, numpy) are not accepted here, because the kinetic energy needs
derivatives of log psi.

The cell is the instance's lattice (`sc`, `bcc`, `fcc`, `square`, `triangular`, `rect-AxBxC`) scaled to
N electrons at rs, as QMBL's `checks/heg-hf/hf.mjs` builds it; `op_verify.py info` prints its vectors.
psi must be antisymmetric under the exchange of two electrons of the same spin and pick up the
phase exp(i k_s . A) when an electron moves by a cell vector A (k_s = 0 at Gamma, the twist of every
instance so far); both are gates. Energies are hartree per electron, the Coulomb energy an Ewald sum
with the uniform background and the Madelung term.

    python op_verify.py hf-check --instance-file F      (the Hartree-Fock determinant, scored by the verifier)
    python op_verify.py score    --instance-file F --state jax:state.py:build --params p.npz --seed 1 --out score_1.json

## A hydrogen state in continuous space (v1.10)

`Hchain` / `Hsheet` instances with `params.basis: none` are scored in continuous space (`qrec_verify/molecule.py`),
with the electron gas's state interface: a flax module or a JAX function of `x`, float64 of shape `(n_conf, N * 3)`,
Cartesian electron coordinates in bohr, the N_up spin-up electrons first, returning log psi or `(sign, log|psi|)`.
The protons sit where the Gaussian-basis instances put them (chain: (0, 0, I R); sheet: (i R, j R, 0), atom
I = i L + j; R = `params.R` in bohr). Energies are hartree with the nuclear repulsion, per atom E / N.

The verifier's local energy is its own: Coulomb pair sums (electrons, protons, nuclear repulsion) and the kinetic
energy by the forward Laplacian. Gates on every score: the potential against pyscf's point-charge Coulomb sum, the
forward Laplacian against `jax.hessian`, the two kinetic estimators (Laplacian and gradient forms) against each
other, antisymmetry under the exchange of two same-spin electrons, and the decay of log|psi| when one electron
moves far out. The sampler is all-electron Gaussian Metropolis from the verifier's own starts (each electron near a
proton). Only open chains and sheets exist in QMBL; a periodic cell is refused with the reason.

    python op_verify.py neel-check --instance-file F    (the RHF determinant in cc-pVDZ against pyscf's RHF energy)
    python op_verify.py score      --instance-file F --state jax:state.py:build --params p.npz --seed 1 --out score_1.json

## Lattices and models (v1.9)

Every lattice state sees `x` as above: `int8` of shape `(n_conf, n_dof)`. What a row means per model
(the code that builds each cluster is named):

| model | lattice / cluster | site order (index i) | `n_dof` | local values |
|---|---|---|---|---|
| Heisenberg, J1-J2 | square, rectangular | `x * ly + y` | N | +-1 (spin up / down) |
| Heisenberg, J1-J2 | triangular L x L (N = L^2, `P` or `O`) | `x * L + y` at `x a1 + y a2` (NetKet's `Triangular(extent=[L, L])`) | N | +-1 |
| Heisenberg, J1-J2 | triangular C6v (N = 3 m^2, `P`) | three sublattice sites per cell, `hamiltonians.triangular_cluster` | N | +-1 |
| Heisenberg | `kagome-LxM` | NetKet's `Kagome(extent=[L, M])` | N | +-1 |
| Heisenberg | `pyrochlore-AxBxC` | NetKet's `Pyrochlore(extent=[A, B, C])` (N = 4ABC), or the 16-site cubic cells (N = 16ABC) | N | +-1 |
| Heisenberg | `shuriken` | L x L square cells of six sites (`hamiltonians.FRUSTRATED`) | N | +-1 |
| Heisenberg, J1-J2 | `honeycomb-LxM` | NetKet's `Honeycomb(extent=[L, M])` | N | +-1 |
| ShastrySutherland | square L x L, `P` or `O` | `x * L + y` | N | +-1 |
| TFIsing | square, rectangular, chain | `x * ly + y` | N | +-1, no S^z constraint |
| Hubbard | square, rectangular | `x * ly + y` | 2N | occupations, NetKet's orbital order |
| tJ | square (L x L, or the tilted square torus), `rectangular-WxL`, `ladder-WxL`, `honeycomb-flake24` | `qrec_verify/tj.py` | 2N | occupations as Hubbard, never two on a site |
| Hchain, Hsheet (Gaussian basis) | open chain, square sheet | Loewdin orbitals in pyscf's AO order (below) | 2 n_orb | occupations as Hubbard |
| Impurity | VarBench's models | per block: the impurity orbitals, then the bath orbitals (`qrec_verify/impurity.py`) | 2 n_orb | occupations as Hubbard; with spin-orbit coupling only the total N is fixed (v1.10) |

Units (QMBL's, DATA.md): spin models in Pauli totals (S.S per site = E / 4N; TFIsing E / N), t-J in units of t
(per site E / N), hydrogen in hartree with the nuclear repulsion (per atom E / N), impurity models as totals.

### A hydrogen state in a Gaussian basis

For `Hchain` / `Hsheet` instances whose `params.basis` names a Gaussian basis (STO-3G, STO-6G, cc-pVDZ, ...), the
verifier builds the ab-initio Hamiltonian of the N atoms in that basis from pyscf's integrals
(`qrec_verify/chem.py`; `requirements-chem.txt`), in hartree with the nuclear repulsion:

    H = E_nuc + sum_{pq,s} h_pq c+_ps c_qs + 1/2 sum_{pqrs,s s'} (pq|rs) c+_ps c+_rs' c_ss' c_qs

The orbitals p are the Loewdin-orthogonalised atomic orbitals, `S^-1/2` applied to pyscf's atomic orbitals of the
basis in pyscf's order (atom by atom along the chain, `pyscf.gto.Mole.ao_labels()`; spherical functions). They depend on
the geometry and the basis only: no SCF, no phase choice. Atoms sit at (0, 0, I R), I = 0 .. N-1, R in bohr as stored
in `params.R`. A configuration is a row of 2 n_orb occupations: the n_orb spin-down orbitals first, then the n_orb
spin-up ones (NetKet's `SpinOrbitalFermions(n_orb, s=1/2, n_fermions_per_spin=(N_dn, N_up))`, its fermion sign
convention, as for Hubbard). A state written in another orbital basis (canonical RHF orbitals, natural orbitals) must
be rotated to these first; `op_verify.py neel-check` scores one Slater determinant of Loewdin orbitals, and
`op_verify.py ed-gate --small hchain:N:R:BASIS` scores your state's family on a small chain against exact
diagonalisation. MPS files of these models are refused with the reason.

Above 128 orbitals (v1.10; `qrec_verify/chem_large.py`) the verifier stores the two-electron integrals as Cholesky
vectors and scores Slater determinants: `--state det:state.npz` with `phi_dn` (n_orb x N_dn) and `phi_up`
(n_orb x N_up), real, in the Loewdin orbitals above; the local energy is the verifier's own determinant kernel. Other
states are not scored there (each sample would need psi on about 10^6 connected configurations). Refused with the
reason: bases whose Cholesky vectors would exceed 64 GB (above 850 orbitals) and bases whose overlap matrix is
nearly singular (smallest eigenvalue below 1e-10: no well-defined Loewdin orbitals).

## Submitting an MPS

A matrix product state (from DMRG or any other tensor-network code) can be submitted as a file instead
of a log psi function. The verifier evaluates it exactly: it builds the instance's Hamiltonian as a matrix
product operator from its own bond lists, in the file's site order, and contracts `<psi|H|psi>/<psi|psi>`.
The energy has no error bar (sigma 0). The format is in `MPS_FORMAT.md`: tensors (left bond, physical,
right bond) in HDF5, an explicit site order (any snake), a stated local basis and fermion ordering
(the one ITensor and TeNPy use), and optionally the energy your code computes, which the verifier must
reproduce. Extra packages: `requirements-mps.txt` (h5py; TeNPy only for its writer).

From TeNPy:

```python
from adapters.mps_tenpy import write_tenpy
from adapters.mps_numpy import snake_order
E = model.H_MPO.expectation_value(psi)          # spins: x 4 for QMBL's Pauli units
write_tenpy("state.h5", psi, snake_order(lx, ly), energy=E, instance_id="Hubbard/square_16_P_4_8")
```

From ITensors.jl:

```julia
include("adapters/mps_itensor.jl")
E = inner(psi', H, psi) / inner(psi, psi)
write_qmbl_mps("state.h5", psi, site_order; energy = E, instance_id = "Hubbard/square_16_P_4_8")
```

From any code that can dump its tensors as arrays:

```python
from adapters.mps_numpy import write_mps
write_mps("state.h5", tensors, site_order, "electron", energy=E, instance_id="Hubbard/square_16_P_4_8")
```

Score it twice (the seeds drive only the sampling cross-check) and combine, as for a sampled state:

    python op_verify.py mps-score --instance-file F --mps state.h5 --seed 1 --out score_1.json
    python op_verify.py mps-score --instance-file F --mps state.h5 --seed 2 --out score_2.json
    python op_verify.py combine   --instance-file F score_1.json score_2.json

The scorecard reports the bond dimensions, the cost of the contraction and the gates (format,
sector, stated energy, a sampling cross-check against NetKet's Hamiltonian, and the variance where
`chi^2 D^2` fits in memory). Cost: about `4 N chi^3 d D` operations and `2 chi^2 D` numbers of memory
for the energy, with d = 2 or 4 and D the MPO's bond dimension (about 30 to 40 for an 8-wide cylinder
with next-nearest-neighbour bonds). Measured on 8 CPU cores (2026-10-15, dense random tensors, 8x8
cylinder with J2 or t' bonds): at chi 1024 the energy takes 51 s for J1-J2 (D 32) and 134 s for
Hubbard (D 42), about 0.5 Tflop/s. Extrapolated to chi 10^4: 11 h and 57 GB (J1-J2; file 62 GB) and
32 h and 74 GB (Hubbard; file 163 GB). The variance (`chi^2 D^2` memory) is skipped at that size.
Above `--max-gb` (default 48) / `--max-tflop` (default 2000) the verifier refuses with the estimate. Before an open instance, check your writer on a solved one, e.g.
`instances/Hubbard/square_16_P_4_8.json`: the stated-energy gate catches a wrong site order or fermion
convention. It cannot catch a wrong Hamiltonian in your own code, because your stated energy comes
from it: in TeNPy 1.1, `add_coupling_term` adds no Jordan-Wigner string, so a hand-built fermion
model must put it in itself (e.g. `"Cdu JW"` on the first site).

## Coverage

By model and lattice (v1.10; `DESIGN.md` has the checks behind each):
- Heisenberg and J1-J2: square and rectangular (periodic or open), triangular (L x L rhombus, periodic or open, and the
  3 m^2 C6v clusters), honeycomb L x M tori; Heisenberg also on kagome L x M, pyrochlore (primitive and 16-site cubic
  cells) and shuriken.
- Shastry-Sutherland: square L x L, torus or open.
- Transverse-field Ising: chain, square, rectangular.
- Hubbard with t, t', U, V1, V2: square and rectangular (periodic, antiperiodic or open axes).
- t-J (with the n n / 4 term): square (L x L and the tilted tori), rectangular, ladders (periodic or antiperiodic),
  the 24-site honeycomb flake.
- Hydrogen chains and sheets in a Gaussian basis (STO-nG, cc-pVXZ, aug-cc-pVXZ): any state up to 128 orbitals;
  Slater determinants up to 850 orbitals (v1.10) (`requirements-chem.txt`).
- Hydrogen chains and sheets in continuous space (`basis: none`, v1.10), open boundary.
- Anderson impurity models: VarBench's single-band and three-band models, with spin-orbit coupling since v1.10.
- The electron gas (since v1.8): 2D and 3D cells.

Not covered: periodic hydrogen cells in continuous space; states other than Slater determinants above 128 orbitals,
Gaussian bases above 850 orbitals or with a nearly singular overlap; MPS files for hydrogen and impurity
models.

## Environment

`requirements.txt`: the versions the tests pass with; a flax or JAX state needs nothing else.
`requirements-torch.txt` and `requirements-julia.txt` add the PyTorch and Julia adapters,
`requirements-mps.txt` the MPS route, `requirements-chem.txt` (pyscf) hydrogen in a Gaussian basis; a C/C++
state needs a compiler only on the entrant's side. A model that needs other packages: say so in
the pull request.
