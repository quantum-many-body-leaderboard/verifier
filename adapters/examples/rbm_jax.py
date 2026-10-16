"""The reference RBM as a plain JAX function.  --state jax:rbm_jax.py:build --params rbm.npz
(rbm.npz = adapters/jax_npz.save_tree("rbm.npz", {"W": W, "b": b, "a": a}))."""

import jax.numpy as jnp


def build():
    def apply_fn(p, x):                               # x: int8, (n_conf, n_sites)
        x = x.astype(jnp.complex128)
        t = x @ p["W"] + p["b"]
        t = jnp.where(t.real < 0, -t, t)
        return (t + jnp.log(1.0 + jnp.exp(-2.0 * t)) - jnp.log(2.0)).sum(-1) + x @ p["a"]
    return apply_fn
