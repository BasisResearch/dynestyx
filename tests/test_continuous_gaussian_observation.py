import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

from dynestyx.inference.configs.filter import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.inference.integrations.cd_dynamax.continuous_filter import (
    compute_continuous_filter,
)
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    GaussianObservation,
)


def _dynamics(observation_model):
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.3 * x,
            diffusion=FullDiffusion(0.1 * jnp.eye(1)),
        ),
        observation_model=observation_model,
    )


@pytest.mark.parametrize(
    "config",
    [
        ContinuousTimeEKFConfig(),
        ContinuousTimeUKFConfig(),
        ContinuousTimeEnKFConfig(n_particles=16),
        ContinuousTimeDPFConfig(n_particles=16),
    ],
)
def test_continuous_filters_accept_gaussian_observation(config):
    dynamics = _dynamics(
        GaussianObservation(h=lambda x, u, t: x + 0.1 * jnp.sin(x), R=jnp.eye(1))
    )
    posterior = compute_continuous_filter(
        dynamics,
        config,
        jr.PRNGKey(0),
        obs_times=jnp.array([0.0, 0.1, 0.2]),
        obs_values=jnp.zeros((3, 1)),
    )
    assert posterior.filtered_means.shape == (3, 1)
    assert jnp.all(jnp.isfinite(posterior.filtered_means))
    assert jnp.isfinite(posterior.marginal_loglik)


@pytest.mark.parametrize(
    "config",
    [
        ContinuousTimeEKFConfig(),
        ContinuousTimeUKFConfig(),
        ContinuousTimeEnKFConfig(n_particles=16),
    ],
)
def test_continuous_gaussian_filters_reject_generic_observation(config):
    dynamics = _dynamics(lambda x, u, t: dist.Poisson(jnp.exp(x)))
    with pytest.raises(TypeError, match="Use ContinuousTimeDPFConfig"):
        compute_continuous_filter(
            dynamics,
            config,
            jr.PRNGKey(0),
            obs_times=jnp.array([0.0, 0.1, 0.2]),
            obs_values=jnp.zeros((3, 1)),
        )
