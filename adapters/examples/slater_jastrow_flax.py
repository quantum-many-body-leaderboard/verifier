"""Reference fermion state, flax: a Slater determinant of n_f general spin orbitals times a
Jastrow factor, log psi(n) = log det M[occupied orbitals] + n.V.n / 2, n in {0,1}^(2 N) in
NetKet's orbital order, occupied orbitals in increasing order. Every adapter in this directory
writes the same state."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class SlaterJastrow(nn.Module):
    n_f: int

    @nn.compact
    def __call__(self, x):
        n = x.shape[-1]
        M = jnp.asarray(self.param("M", nn.initializers.zeros, (n, self.n_f), jnp.float64))
        V = jnp.asarray(self.param("V", nn.initializers.zeros, (n, n), jnp.float64))

        def one(xi):
            occ = jnp.nonzero(xi > 0, size=self.n_f)[0]
            sign, logabs = jnp.linalg.slogdet(M[occ])
            return logabs + jnp.where(sign < 0, 1j * jnp.pi, 0.0) + 0.5 * xi @ V @ xi

        xf = x.reshape(-1, n).astype(jnp.float64)
        return jax.vmap(one)(xf).reshape(x.shape[:-1])


def build_model(instance):
    return SlaterJastrow(n_f=2 * int(instance["params"]["Nf"]))     # Nf up + Nf down
