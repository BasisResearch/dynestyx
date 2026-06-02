import jax
import jax.numpy as jnp

from dynestyx.inference.integrations.utils import particles_to_delta_mixtures


def test_particles_to_delta_mixtures():
    T, N, D = 1, 7, 11
    x = jax.random.normal(jax.random.key(1), shape=[T, N, D], dtype="float32")
    z = -1e10 * jnp.ones([T, N], dtype="float32")
    [dist] = particles_to_delta_mixtures(x, z)
    assert jnp.isclose(jnp.exp(dist.log_weights).sum(), 1)
