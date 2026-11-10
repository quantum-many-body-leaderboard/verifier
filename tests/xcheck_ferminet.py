# One-time cross-check (2026-11-10, 99problems, CPU): the verifier's Ewald energy against FermiNet's
# ferminet/pbc/ewald.py (google-deepmind/ferminet main, Apache-2.0); mcmc.py reduced to
# map_to_simulation_cell. Result: xcheck_ferminet.json.
import sys, json
sys.path.insert(0, "fpkg"); sys.path.insert(0, "../src")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from ferminet.pbc import ewald as FE
from qrec_verify import continuum as CT
cases = [(3,"sc",7,7,1),(3,"bcc",27,0,5),(3,"fcc",65,65,30),(3,"rect-3x3x2",9,9,2),(2,"square",45,45,30),(2,"triangular",13,13,10)]
out = []
for dim, lat, nu, nd, rs in cases:
    cell = CT.make_cell(dim, lat, nu, nd, rs)
    ew = CT.Ewald.build(cell.A); f = jax.jit(jax.vmap(ew.energy_fn(cell.n)))
    pot = jax.jit(FE.make_ewald_potential(lattice=jnp.asarray(cell.A.T), atoms=jnp.zeros((0, dim)),
                                          charges=jnp.zeros((0,)), truncation_limit=8, ndim=dim))
    x = np.random.default_rng(7).uniform(size=(4, cell.n, dim)) @ cell.A
    mine = np.asarray(f(x.reshape(4, -1)))
    theirs = np.array([float(pot(jnp.zeros((cell.n, 0, dim)), jnp.asarray(xi[:, None] - xi[None, :]))) for xi in x])
    d = float(np.max(np.abs(mine - theirs)))
    out.append({"cell": f"{dim}D {lat} N={cell.n} rs={rs}", "verifier": mine.tolist(), "ferminet": theirs.tolist(), "max_abs_diff": d})
    print(f"{dim}D {lat:11s} N={cell.n:3d} rs={rs:3}: max |diff| {d:.2e}  (E ~ {mine[0]:.6f})")
json.dump(out, open("xcheck_ferminet.json", "w"), indent=1)
