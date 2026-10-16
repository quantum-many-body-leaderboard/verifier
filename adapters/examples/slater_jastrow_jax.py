"""The reference Slater-Jastrow state as a plain JAX function.
--state jax:slater_jastrow_jax.py:build --params sj.npz   (npz with M and V)"""

import jax
import jax.numpy as jnp


def build(instance):
    n_f = 2 * int(instance["params"]["Nf"])

    def one(p, xi):
        occ = jnp.nonzero(xi > 0, size=n_f)[0]
        sign, logabs = jnp.linalg.slogdet(p["M"][occ])
        return logabs + jnp.where(sign < 0, 1j * jnp.pi, 0.0) + 0.5 * xi @ p["V"] @ xi

    def apply_fn(p, x):                               # x: int8, (n_conf, 2 N)
        return jax.vmap(lambda xi: one(p, xi))(x.astype(jnp.float64))
    return apply_fn
