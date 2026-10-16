# Verifier

The re-measurement of [QMBL](https://qmbl.org) and QMBL: Open Problems. `qrec_verify/` is the
scorer (v1.7, 2026-10-09); `DESIGN.md` lists its gates. `op_verify.py` is the command line: it
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

Square and rectangular lattices: J1-J2 and Heisenberg (periodic or open), Hubbard with t, t',
U, V1, V2 (periodic, antiperiodic or open axes). That is 118 of the 152 open problems on
2026-10-12. Triangular,
kagome, pyrochlore, shuriken and chain lattices, the transverse-field Ising model and the impurity
models are not covered yet.

## Environment

`requirements.txt`: the versions the tests pass with; a flax or JAX state needs nothing else.
`requirements-torch.txt` and `requirements-julia.txt` add the PyTorch and Julia adapters,
`requirements-mps.txt` the MPS route; a C/C++
state needs a compiler only on the entrant's side. A model that needs other packages: say so in
the pull request.
