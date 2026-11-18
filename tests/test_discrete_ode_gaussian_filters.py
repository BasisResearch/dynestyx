"""Discrete EKF/UKF integration with ODE flows on irregular observation grids."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import dynestyx as dsx
from dynestyx.discretizers import Discretizer, ODEFlowConfig
from dynestyx.inference.configs.filter import EKFConfig, UKFConfig
from dynestyx.inference.configs.simulator import ODESimulatorConfig
from dynestyx.inference.configs.smoother import EKFSmootherConfig, UKFSmootherConfig


def _exact_posterior(times, controls, observations, jitter, smooth):
    """Condition the joint Gaussian state path on each prefix or on all data."""
    # For dx/dt = t * (x + u), x_next = a * (x + u) - u + noise.
    a = np.exp(np.diff(times**2) / 2)
    n = len(times)
    mean = np.zeros(n)
    loadings = np.eye(n)
    for k in range(n - 1):
        mean[k + 1] = a[k] * (mean[k] + controls[k]) - controls[k]
        loadings[k + 1, : k + 1] = a[k] * loadings[k, : k + 1]
    state_cov = (loadings * np.array([1.0] + [jitter**2] * (n - 1))) @ loadings.T
    obs_cov = state_cov + 0.4 * np.eye(n)
    residual = observations - (mean + times - controls)
    posterior_mean, posterior_var = [], []
    for k in range(n):
        observed = slice(None, n if smooth else k + 1)
        cross = state_cov[k, observed]
        gain = np.linalg.solve(obs_cov[observed, observed], cross)
        posterior_mean.append(mean[k] + gain @ residual[observed])
        posterior_var.append(state_cov[k, k] - gain @ cross)
    loglik = -0.5 * (
        n * np.log(2 * np.pi)
        + np.linalg.slogdet(obs_cov)[1]
        + residual @ np.linalg.solve(obs_cov, residual)
    )
    return posterior_mean, posterior_var, loglik


@pytest.mark.parametrize("jitter", [0.0, 0.3])
@pytest.mark.parametrize(
    "config",
    [
        EKFConfig(filter_source="cd_dynamax"),
        UKFConfig(),
        EKFSmootherConfig(filter_source="cd_dynamax"),
        UKFSmootherConfig(),
    ],
)
def test_irregular_controlled_ode(config, jitter):
    times = jnp.array([0.1, 0.22, 0.5, 0.61])
    controls = jnp.array([-0.4, 0.8, 0.2, -0.5])
    observations = jnp.array([0.2, -0.1, 1.7, 0.7])
    model = dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: t * (x + u)
        ),
        observation_model=dsx.GaussianObservation(
            h=lambda x, u, t: x + t - jnp.asarray(u), R=0.4 * jnp.eye(1)
        ),
        control_dim=1,
    )
    smoothing = isinstance(config, (EKFSmootherConfig, UKFSmootherConfig))
    handler = dsx.Smoother(config) if smoothing else dsx.Filter(config)
    flow = ODEFlowConfig(
        simulator_config=ODESimulatorConfig(
            adjoint=dfx.DirectAdjoint(), dt0=0.05, max_steps=20
        ),
        jitter_scale=jitter,
    )
    with handler, Discretizer(flow):
        posterior = dsx.condition(
            "x",
            model,
            obs_times=times,
            obs_values=observations[:, None],
            ctrl_times=times,
            ctrl_values=controls[:, None],
        )
    assert posterior.dists is not None and posterior.marginal_loglik is not None
    means, variances, ll = _exact_posterior(
        np.asarray(times),
        np.asarray(controls),
        np.asarray(observations),
        jitter,
        smoothing,
    )
    np.testing.assert_allclose(
        [d.mean.item() for d in posterior.dists], means, atol=2e-5
    )
    np.testing.assert_allclose(
        [d.variance.item() for d in posterior.dists], variances, atol=2e-5
    )
    np.testing.assert_allclose(posterior.marginal_loglik, ll, atol=2e-5)


def test_handwritten_gaussian_transition_receives_interval():
    times = jnp.array([1.0, 1.08, 1.3])
    model = dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=dsx.GaussianStateEvolution(
            F=lambda x, u, t0, t1: x + t1**2 - t0**2, cov=jnp.eye(1)
        ),
        observation_model=dsx.LinearGaussianObservation(H=jnp.eye(1), R=jnp.eye(1)),
    )
    expected = times**2 - times[0] ** 2
    with dsx.Filter(UKFConfig()):
        posterior = dsx.condition(
            "x", model, obs_times=times, obs_values=expected[:, None]
        )
    assert posterior.dists is not None
    np.testing.assert_allclose(
        [d.mean.item() for d in posterior.dists], expected, atol=2e-6
    )


def test_nonlinear_ode_ukf_likelihood_gradient():
    times = jnp.array([0.0, 0.1, 0.3])
    observations = jnp.array([[0.3], [0.15], [-0.1]])

    def likelihood(rate, times, observations):
        model = dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
            state_evolution=dsx.ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -rate * x**3
            ),
            observation_model=dsx.LinearGaussianObservation(H=jnp.eye(1), R=jnp.eye(1)),
        )
        with dsx.Filter(UKFConfig()), Discretizer(ODEFlowConfig(jitter_scale=1e-3)):
            return dsx.sample(
                "x", model, obs_times=times, obs_values=observations
            ).marginal_loglik

    ll, grad = jax.jit(jax.value_and_grad(likelihood))(1.0, times, observations)
    eps = 0.01
    compiled = jax.jit(likelihood)
    difference = (
        compiled(1.0 + eps, times, observations)
        - compiled(1.0 - eps, times, observations)
    ) / (2 * eps)
    assert np.isfinite(ll)
    np.testing.assert_allclose(grad, difference, rtol=2e-3, atol=2e-5)
    # A single observation updates the initial distribution without an ODE interval.
    expected = dist.Normal(0.0, jnp.sqrt(2.0)).log_prob(observations[0, 0])
    np.testing.assert_allclose(
        compiled(1.0, times[:1], observations[:1]), expected, atol=2e-6
    )
