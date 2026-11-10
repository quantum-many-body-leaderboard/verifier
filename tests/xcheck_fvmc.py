# One-time cross-check (2026-11-10, 99problems, CPU), not part of the pytest run: the verifier's
# Ewald energy against fvmc's EwaldSum (fvmc/ewaldsum.py of github.com/agnes-valenti/lambda-jellium-nn,
# derived from PyQMC, MIT). Run from a directory holding pkg/fvmc/{__init__,ewaldsum,utils}.py and the
# verifier at ../src; needs ml-collections. Result: tests/xcheck_fvmc.json.
import sys, json
sys.path.insert(0, "pkg"); sys.path.insert(0, "../src")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from fvmc.ewaldsum import EwaldSum
from qrec_verify import continuum as CT
cases = [(3,"sc",7,7,1),(3,"bcc",27,0,5),(3,"fcc",65,65,30),(3,"rect-3x3x2",9,9,2),(2,"square",45,45,30),(2,"triangular",13,13,10)]
out = []
for dim, lat, nu, nd, rs in cases:
    cell = CT.make_cell(dim, lat, nu, nd, rs)
    ew = CT.Ewald.build(cell.A); f = jax.jit(jax.vmap(ew.energy_fn(cell.n)))
    fv = EwaldSum(jnp.asarray(cell.A), n_lat=2)
    x = np.random.default_rng(7).uniform(size=(4, cell.n, dim)) @ cell.A
    mine = np.asarray(f(x.reshape(4, -1)))
    theirs = np.array([float(fv.energy(-jnp.ones(cell.n), jnp.asarray(xi))) for xi in x])
    d = float(np.max(np.abs(mine - theirs)))
    out.append({"cell": f"{dim}D {lat} N={cell.n} rs={rs}", "verifier": mine.tolist(), "fvmc": theirs.tolist(), "max_abs_diff": d})
    print(f"{dim}D {lat:11s} N={cell.n:3d} rs={rs:3}: max |diff| {d:.2e}  (E ~ {mine[0]:.6f})")
json.dump(out, open("xcheck_fvmc.json", "w"), indent=1)
