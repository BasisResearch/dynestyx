"""Regression coverage for automatic EnKF update-space selection."""

import dataclasses
from unittest.mock import patch

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from cuthbert.ensemble_kalman import ensemble_kalman_filter

import dynestyx as dsx
from dynestyx.inference.configs.filter import (
    ContinuousTimeEnKFConfig,
    EnKFConfig,
    EnKFLocalizationFunctions,
)
from dynestyx.inference.configs.smoother import EnRTSSmootherConfig
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
    compute_cuthbert_filter,
)


@pytest.mark.parametrize(
    ("config_type", "n_particles", "option", "localized", "overrides", "expected"),
    [
        (EnKFConfig, 3, None, False, {}, True),
        (EnKFConfig, 4, None, False, {}, False),
        (EnKFConfig, 5, None, False, {}, False),
        (EnKFConfig, 3, None, True, {}, False),
        (EnKFConfig, 5, True, False, {}, True),
        (EnKFConfig, 3, False, False, {}, False),
        (EnKFConfig, 3, False, True, {}, False),
        (EnKFConfig, 3, True, True, {}, None),
        (ContinuousTimeEnKFConfig, 3, True, False, {}, None),
        (EnKFConfig, 5, None, False, {"n_particles": 3}, True),
        (EnKFConfig, 3, True, False, {"ensemble_subspace": False}, False),
        (EnRTSSmootherConfig, 3, None, False, {}, True),
    ],
)
def test_enkf_ensemble_subspace(
    config_type, n_particles, option, localized, overrides, expected
):
    localization = (
        EnKFLocalizationFunctions(
            modify_cross_covariance=lambda cross_covariance, model_inputs: (
                cross_covariance
            )
        )
        if localized
        else None
    )
    config_kwargs = dict(
        n_particles=n_particles,
        ensemble_subspace=option,
        localization=localization,
    )
    if expected is None:
        with pytest.raises(ValueError, match="ensemble_subspace=True"):
            config_type(**config_kwargs)
        return

    config = config_type(**config_kwargs)
    dynamics = dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(3), jnp.eye(3)),
        state_evolution=dsx.LinearGaussianStateEvolution(
            A=0.9 * jnp.eye(3), cov=0.1 * jnp.eye(3)
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
            ),
            R=0.2 * jnp.eye(4),
        ),
    )
    key = jr.PRNGKey(7)
    with patch.object(
        ensemble_kalman_filter,
        "build_filter",
        wraps=ensemble_kalman_filter.build_filter,
    ) as builder:
        build_cuthbert_filter(
            dynamics, config, key, want_parallel=False, extra_filter_kwargs=overrides
        )
    assert builder.call_args.kwargs["ensemble_subspace"] is expected
    effective_particles = overrides.get("n_particles", n_particles)
    assert builder.call_args.kwargs["n_particles"] == effective_particles

    effective_config = dataclasses.replace(
        config,
        n_particles=effective_particles,
        ensemble_subspace=overrides.get("ensemble_subspace", option),
    )
    obs_times = jnp.arange(3.0)
    obs_values = jnp.array(
        [[0.1, 0.2, -0.1, 0.2], [0.0, 0.1, 0.2, 0.3], [0.2, -0.1, 0.0, 0.1]]
    )
    with patch.object(
        ensemble_kalman_filter,
        "build_filter",
        wraps=ensemble_kalman_filter.build_filter,
    ) as builder:
        loglik, states = compute_cuthbert_filter(
            dynamics, effective_config, key, obs_times=obs_times, obs_values=obs_values
        )
    assert builder.call_args.kwargs["ensemble_subspace"] is expected
    assert jnp.isfinite(loglik)
    assert states.ensemble.shape == (3, effective_particles, 3)
    assert jnp.all(jnp.isfinite(states.ensemble))

    explicit_config = dataclasses.replace(effective_config, ensemble_subspace=expected)
    explicit_loglik, explicit_states = compute_cuthbert_filter(
        dynamics, explicit_config, key, obs_times=obs_times, obs_values=obs_values
    )
    assert jnp.allclose(loglik, explicit_loglik)
    assert jnp.allclose(states.ensemble, explicit_states.ensemble)
