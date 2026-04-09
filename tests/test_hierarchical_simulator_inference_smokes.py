"""Smoke tests for hierarchical plate-aware simulator-based inference."""

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer import init_to_value

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, ODESimulator
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import NUTSConfig
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianStateEvolution,
    LinearGaussianObservation,
)
from dynestyx.models.state_evolution import AffineDrift

# ---------------------------------------------------------------------------
# Discrete-time nonlinear model (reused from hierarchical simulator smokes)
# ---------------------------------------------------------------------------


class _PlateNonlinearTransition(eqx.Module):
    beta: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        return 0.75 * x + self.beta * jnp.tanh(x)


def _hierarchical_nonlinear_discrete_model(
    obs_times=None, obs_values=None, predict_times=None, M=3, **kwargs
):
    """Hierarchical nonlinear discrete model with per-trajectory beta."""
    with dsx.plate("trajectories", M):
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 0.4))
        beta = 0.8 * (jnp.tanh(beta_raw) / 2.0)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(1), covariance_matrix=0.2 * jnp.eye(1)
            ),
            state_evolution=GaussianStateEvolution(
                F=_PlateNonlinearTransition(beta=beta),
                cov=0.05 * jnp.eye(1),
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.array([[1.0]]), R=jnp.array([[0.1]])
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def test_hierarchical_discrete_simulator_inference_smoke():
    """DiscreteTimeSimulator conditioning works under plate for MCMC."""
    M = 3
    T = 20
    t = jnp.arange(T, dtype=jnp.float32)

    # Generate synthetic data
    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            _hierarchical_nonlinear_discrete_model(predict_times=t, M=M)
    obs_values = tr["f_observations"]["value"][:, 0]  # (M, T, obs_dim)

    init_values = {
        "beta_raw": jnp.zeros(M),
    }

    with DiscreteTimeSimulator():
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
                init_strategy=init_to_value(values=init_values),
            ),
            model=_hierarchical_nonlinear_discrete_model,
        )
        posterior = inference.run(
            jr.PRNGKey(42),
            t,
            obs_values,
            M=M,
        )

    assert "beta_raw" in posterior
    assert posterior["beta_raw"].shape[-1] == M
    assert not jnp.isnan(posterior["beta_raw"]).any()
    assert not jnp.isinf(posterior["beta_raw"]).any()


# ---------------------------------------------------------------------------
# ODE model for simulator-based inference
# ---------------------------------------------------------------------------


def _hierarchical_ode_model(
    obs_times=None, obs_values=None, predict_times=None, M=3, **kwargs
):
    """Hierarchical ODE model with per-trajectory coupling rho."""
    A_base = jnp.array([[-1.0, 0.0], [0.0, -1.0]])

    with dsx.plate("trajectories", M):
        rho_raw = numpyro.sample("rho_raw", dist.Normal(0.5, 0.5))
        rho = jnn.softplus(rho_raw)
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 1, 0].set(rho)
        drift = AffineDrift(A=A)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2), covariance_matrix=jnp.eye(2)
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=drift, diffusion_coefficient=None
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.array([[0.0, 1.0]]), R=jnp.eye(1)
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def test_hierarchical_ode_simulator_inference_smoke():
    """ODESimulator conditioning works under plate for MCMC."""
    M = 3
    t = jnp.linspace(0.0, 1.0, 10)

    # Generate synthetic data
    with ODESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            _hierarchical_ode_model(predict_times=t, M=M)
    obs_values = tr["f_observations"]["value"][:, 0]  # (M, T, obs_dim)

    init_values = {
        "rho_raw": 0.5 * jnp.ones(M),
    }

    with ODESimulator():
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
                init_strategy=init_to_value(values=init_values),
            ),
            model=_hierarchical_ode_model,
        )
        posterior = inference.run(
            jr.PRNGKey(42),
            t,
            obs_values,
            M=M,
        )

    assert "rho_raw" in posterior
    assert posterior["rho_raw"].shape[-1] == M
    assert not jnp.isnan(posterior["rho_raw"]).any()
    assert not jnp.isinf(posterior["rho_raw"]).any()
