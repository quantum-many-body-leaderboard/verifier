# Submitting a state from any framework

The verifier needs one thing from a state: log psi(x) for a batch of configurations. It samples
with its own chains, computes local energies with its own Hamiltonian and applies the same gates
to every entry (`../DESIGN.md`). This directory holds the C header, a helper for JAX parameter
files and the same two reference states written in every supported framework (`examples/`);
`tests/test_adapters.py` checks that they agree with the flax versions to 1e-10.

## The contract (every framework)

- Input `x`: `int8`, shape `(n_conf, n_dof)`, one configuration per row.
  - Spin models (J1-J2, Heisenberg): `n_dof` = number of sites, entries +1/-1, total S_z = 0.
  - Hubbard models: `n_dof` = 2 x number of sites, entries 0/1 per orbital in NetKet's
    `SpinOrbitalFermions(N, s=1/2)` order.
  - Site index `i = x * ly + y` on an `lx x ly` lattice (`../README.md`, "Conventions").
- Output: log psi(x), `complex128`, shape `(n_conf,)`. The real part is log|psi|, the imaginary
  part the phase (read modulo 2 pi). A real output means a positive amplitude.
- The function is called many times, with up to 8192 configurations per call (`--host-chunk`),
  one call at a time. It must be deterministic and keep no state between calls. Use double
  precision: `logpsi-check` and the tests compare at 1e-6 and 1e-10.
- Print to stderr only (stdout carries the JSON of some commands).
- `NAME` in a spec is called with the QMBL instance (a dict: `n_sites`, `lattice`, `boundary`,
  `params`, ...) when it has a positional parameter without a default, else with no argument.

The verifier runs the state's code: state files are code, as in the flax case.

## flax (default)

    --state model.py:build_model --params state.msgpack

`build_model()` returns a flax module; `model.apply(variables, x)` gives log psi. The msgpack is
`flax.serialization.msgpack_serialize(variables)` (or of the bare parameter tree). Example:
`examples/rbm_flax.py`.

## PyTorch

    --state torch:model.py:MyNet --params state.pt

1. `model.py` defines an `nn.Module` class (or a factory function) `MyNet`. Its `forward(x)` gets
   `x` as a `torch.int8` tensor on the model's device and returns log psi, `complex128` (or
   `float64` for a positive state), shape `(n_conf,)`. Convert `x` yourself, e.g.
   `x.to(torch.float64)`.
2. Save the weights: `torch.save(model.state_dict(), "state.pt")`. The verifier builds the module,
   loads the state_dict with `weights_only=True` and calls it under `torch.inference_mode()`.
3. Device: CUDA when available, else CPU; `QMBL_TORCH_DEVICE=cpu` forces one.

Examples: `examples/rbm_torch.py`, `examples/slater_jastrow_torch.py`. Needs
`../requirements-torch.txt` on the verifier's side.

## JAX without flax (Equinox, Haiku, plain functions)

    --state jax:model.py:build --params state.npz

1. `build()` (or `build(instance)`) returns a pure function `apply_fn(params, x) -> log psi`,
   `x` int8 of shape `(n_conf, n_dof)`. It is traced and compiled into the samplers, like a flax
   module: no callback, same speed.
2. Parameters: `adapters/jax_npz.py: save_tree("state.npz", params)` for nested dicts of arrays
   (keys joined with `/`), or a flax msgpack. Any other format: define `load_params(path)` in
   `model.py`; the verifier calls it after `build` and passes the result as `params`.

Haiku: `build` returns `hk.without_apply_rng(hk.transform(f)).apply`; Haiku parameters are nested
dicts, so `save_tree` works. Equinox: split the model with `eqx.partition(model, eqx.is_array)`
in `build`, return `lambda p, x: jax.vmap(eqx.combine(p, static))(x)`, and read the array part in
`load_params` (e.g. `eqx.tree_deserialise_leaves`). The tests cover plain functions
(`examples/rbm_jax.py`, `examples/slater_jastrow_jax.py`), not Haiku or Equinox themselves.

## Julia

    --state julia:state.jl [--params state.jld2]

1. `state.jl` defines `logpsi(x::Matrix{Int8})::Vector{ComplexF64}`; `x` has size
   `(n_conf, n_dof)`, one configuration per row as everywhere else (`permutedims(x)` gives one
   per column).
2. With `--params PATH`, the file must also define `init(path::String)`; it is called once,
   before the first `logpsi`, and loads whatever the file needs (the examples use JLD2).
3. The file is included into a fresh module in a child process of the verifier (Julia must stay
   on the thread that started it); configurations go over a pipe.
4. Packages: the verifier's Julia environment has JLD2 (`../requirements-julia.txt`). A state
   that needs other packages: say so in the pull request.

Examples: `examples/rbm.jl`, `examples/slater_jastrow.jl`.

## C and C++

    --state c:libstate.so [--params state.bin]

1. Export the functions of `qmbl_state.h` with C linkage:
   `void logpsi(const int8_t* x, int n_conf, int n_sites, double* re, double* im)`, and
   `int qmbl_init(const char* path)` if the state takes `--params` (return 0 on success).
   `n_sites` is the length of one configuration (`n_dof`, twice the sites for Hubbard).
2. Build a shared library, e.g. `c++ -O2 -shared -fPIC -I adapters examples/rbm.cpp -o librbm.so`.
3. The library is loaded with ctypes and called from one thread at a time.

Examples: `examples/rbm.cpp` (about 40 lines), `examples/slater_jastrow.cpp`.

## Any other language: a numpy host function

    --state python:host.py:build [--params FILE]

`build(params_path)` (or `build(params_path, instance)`) returns `logpsi(x) -> np.ndarray` with the
contract above. Use it to bind a code base that has its own Python bindings. Examples:
`examples/rbm_numpy.py`, `examples/slater_jastrow_numpy.py`.

## Check your export

Evaluate your state with your own code on a few configurations in the verifier's convention and
store them: `np.savez("cfg.npz", x=x, logpsi=logpsi)`. Then

    python op_verify.py logpsi-check --instance-file F --state SPEC --params P --configs cfg.npz

must pass (1e-6 in log|psi| and in the phase), and `op_verify.py ed-gate` on a small lattice of the
same family checks the state against exact diagonalisation (`../README.md`).

## Speed

Host states (PyTorch, Julia, C/C++, Python) are called through `jax.pure_callback`: every
Metropolis step of the sampler leaves the compiled loop, so a host state is slower than the same
state in flax or plain JAX. For states of realistic size the state's own arithmetic dominates.
`tests/bench_adapters.py`, 2026-10-15: the reference RBM (alpha = 1, complex) on the 10x10 J1-J2
model, the frozen spin sampler (256 chains x 16 replicas), 8 CPU cores, no GPU. A sweep is 100
Metropolis steps, each one log psi call on 4096 configurations; a block is 16 sweeps plus the local
energies of its 4096 samples; one frozen score is 4352 sweeps and 16 blocks.

| adapter | s per sweep | s local energies per block | samples per s | one score (h, estimate) |
|---|---|---|---|---|
| flax (compiled) | 2.78 | 4.6 | 84 | 3.4 |
| jax (compiled) | 2.55 | 4.8 | 90 | 3.1 |
| torch (callback) | 2.27 | 7.1 | 94 | 2.8 |
| c, plain loops (callback) | 3.31 | 26.9 | 51 | 4.1 |
| julia (callback, child process) | 3.90 | 14.5 | 53 | 4.8 |
| python, numpy (callback) | 6.56 | 43.8 | 28 | 8.1 |

The callback itself costs little: PyTorch is as fast as flax here. The C example is plain loops
without BLAS, and numpy's complex `exp` and `log` are slow; both show the state's own cost, not the
adapter's. On a GPU, flax, JAX and PyTorch states stay on the device for their arithmetic; a host
state still moves each batch through host memory.
