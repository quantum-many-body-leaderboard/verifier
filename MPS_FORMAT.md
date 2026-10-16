# MPS file format (qmbl-mps, version 1, 2026-10-15)

A matrix product state for a QMBL instance, as one HDF5 file. The verifier (`op_verify.py mps-score`)
reads the tensors, builds the instance's Hamiltonian itself and contracts the energy exactly. Writers:
`adapters/mps_tenpy.py` (TeNPy), `adapters/mps_itensor.jl` (ITensors.jl), `adapters/mps_numpy.py`
(plain arrays from any code).

## Layout

| path | kind | content |
|---|---|---|
| `/` attribute `format` | string | `"qmbl-mps"` |
| `/` attribute `format_version` | integer | `1` |
| `/` attribute `local_basis` | string | `"spin-half"` (J1-J2, Heisenberg) or `"electron"` (Hubbard) |
| `/` attribute `n_sites` | integer | N, the number of lattice sites |
| `/` attribute `instance_id` | string, optional | the QMBL instance, e.g. `"Hubbard/square_16_P_4_8"`; checked when present |
| `/` attribute `energy` | float, optional | the energy of these tensors as the producing code computes it, total, in QMBL's units (below); checked when present |
| `/` attribute `producer` | string, optional | code and version |
| `/site_order` | int64, shape (N,) | `site_order[k]` = the QMBL site of chain position k |
| `/tensors/0` ... `/tensors/{N-1}` | float64 or complex128, shape (chi_l, d, chi_r) | the tensor of chain position k |
| `/quantum_numbers/...` | anything, optional | ignored by the verifier |

- **Site order.** QMBL numbers the sites of an `lx x ly` lattice `i = x * ly + y` (x the first axis
  of the lattice name, `rectangular-AxB` has lx = A). `site_order` is any permutation of 0..N-1:
  QMBL's own order (`site_order[k] = k`, y fastest, the usual cylinder order when y is the periodic
  axis), a snake, or anything else. The verifier builds the Hamiltonian in this order; long-range
  bonds only make its MPO larger.
- **Tensors.** Index order (left bond, physical, right bond) as a row-major (C, h5py) reader sees it;
  the first tensor has left bond 1, the last right bond 1. The state is the product
  `psi(s_0, ..., s_{N-1}) = A0[:, s_0, :] A1[:, s_1, :] ... A{N-1}[:, s_{N-1}, :]`. Neither
  normalisation nor a canonical form is needed. float32 and complex64 are read and converted to
  double precision; dense storage only (quantum-number blocks are written densely; Julia writers store
  the transposed array so that row-major readers see this order, as `adapters/mps_itensor.jl` does).
- **Sector.** The state must lie in the instance's sector: total S^z = 0 for spins; N_up = N_dn =
  `Nf` for Hubbard. The verifier measures `<(Q - Q0)^2>` summed over the conserved charges and fails
  the state above 1e-8. DMRG with conserved quantum numbers gives 0 to rounding.

## Local bases

**spin-half** (d = 2): index 0 = up (S^z = +1/2; +1 in the verifier's NetKet configurations), index
1 = down. The Hamiltonian is QMBL's: `H = J1 sum_<ij> s_i.s_j + J2 sum_<<ij>> s_i.s_j` with Pauli
matrices s (4 x the S.S form), no sign rule. A state with a Marshall or other sign rule carries it in
its tensors. ITensor `"S=1/2"` (Up, Dn) and TeNPy `SpinHalfSite` ('up', 'down') map one to one.

**electron** (d = 4): index `n_up + 2 n_dn`, i.e. `[empty, up, down, up+down]`. Fermion signs follow
one fixed ordering of the 2N modes along the chain:

    |n> = (c+_{0,up})^{n_0,up} (c+_{0,dn})^{n_0,dn} (c+_{1,up})^{n_1,up} (c+_{1,dn})^{n_1,dn} ... |0>

where `c+_{k,s}` creates an electron of spin s on the lattice site `site_order[k]`. Creation operators
are ordered by chain position, and within a site up comes before down, so `|up+down> = c+_up c+_dn |0>`.
In Jordan-Wigner form, `c_{k,s} = (prod_{modes m before (k,s)} (-1)^{n_m}) a_{k,s}`: the string runs
over every site left of k and, for a down electron, over the site's own up mode. This is the
convention of ITensor's `"Electron"` site type (Emp, Up, Dn, UpDn; its `Cdn` carries the sign of the
site's up electron) and of TeNPy's `SpinHalfFermionSite` ('empty', 'up', 'down', 'full'; TeNPy sorts
these by charge internally, `adapters/mps_tenpy.py` reads them by label). Both therefore map without
any sign change. A code that defines `|up+down> = c+_dn c+_up |0>` multiplies the up+down component by
-1 (`write_mps(..., double_sign=-1)`). Reversing the direction of the Jordan-Wigner string (all
strings to the right) is, at fixed particle number, the same as that within-site swap up to a global
sign.

The Hubbard Hamiltonian is the verifier's (`qrec_verify/hamiltonians.py`):
`H = - sum_s sum_bonds t_b s_b (c+_i c_j + h.c.) + U sum_i n_i,up n_i,dn + V1 sum_<ij> n_i n_j + V2 sum_<<ij>> n_i n_j`,
t_b = t1 on nearest and t2 on next-nearest bonds, s_b = -1 on a bond that wraps an antiperiodic axis.

## Units of the stated energy

`energy` is a total, in QMBL's convention: spins in Pauli units (4 x the S.S total; QMBL's per-site
number is E / 4N), Hubbard the plain total (per site E / N). If the producing code works with S
operators, multiply its energy by 4.

## What the verifier checks (gates on the scorecard)

| gate | check | fail |
|---|---|---|
| format | layout, shapes, bonds, dtype, local basis and size against the instance | any mismatch |
| size | memory and operations of the contraction against `--max-gb`, `--max-tflop` | above the limits (raise them deliberately) |
| real | imaginary part of `<H>/<psi|psi>` | > 1e-10 relative |
| sector | `<(N_up - Nf)^2> + <(N_dn - Nf)^2>` or `<(S^z_total)^2>` | > 1e-8 |
| stated energy | the file's `energy` against the verifier's | relative difference > 1e-8 (warn if no energy is stated) |
| sampling | perfect samples from the MPS, local energies with the verifier's NetKet Hamiltonian, which uses its own fermion ordering; mean against the exact energy | \|z\| > 4 (skipped with a warning above `max_gb_sampling`) |
| variance | `<H^2> - <H>^2 >= 0`, computed where `chi^2 D^2` fits in memory | negative |

A wrong site order, local basis or fermion convention cannot produce a too-low energy: the verifier
evaluates whatever state the file describes with the correct Hamiltonian, which is always an upper
bound. Such a mistake usually makes the energy higher. It shows up only against a stated energy, so
state it. Two mistakes are invisible and harmless: swapping up and down for spins (a global spin flip
of an SU(2)-symmetric Hamiltonian) and a global sign.
