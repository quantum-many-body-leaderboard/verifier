"""Reference state, flax: a complex RBM, log psi = sum_j log cosh(theta_j) + a.x, theta = x W + b.
The same state is written for every adapter in this directory (tests/test_adapters.py checks
that they agree to 1e-10)."""

import flax.linen as nn
import jax.numpy as jnp


def log_cosh(z):
    z = jnp.where(z.real < 0, -z, z)
    return z + jnp.log(1.0 + jnp.exp(-2.0 * z)) - jnp.log(2.0)


class RBM(nn.Module):
    alpha: int = 1

    @nn.compact
    def __call__(self, x):
        n = x.shape[-1]
        c = jnp.complex128
        W = self.param("W", nn.initializers.zeros, (n, self.alpha * n), c)
        b = self.param("b", nn.initializers.zeros, (self.alpha * n,), c)
        a = self.param("a", nn.initializers.zeros, (n,), c)
        x = x.astype(c)
        return log_cosh(x @ W + b).sum(-1) + x @ a


def build_model():
    return RBM()
