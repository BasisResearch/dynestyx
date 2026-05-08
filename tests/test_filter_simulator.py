"""
Fast tests for Filter + Simulator integration.

Verifies that Filter and Simulator work correctly together for SDE, ODE, and
discrete-time models with known parameters. These tests use trace-only evaluation
(no MCMC) for speed.
"""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from numpyro.handlers import seed, trace

from dynestyx import (
    DiscreteTimeSimulator,
    Filter,
    SDESimulator,
    Simulator,
)
from dynestyx.inference.filter_configs import (
    ContinuousTimeEnKFConfig,
    PFConfig,
)
from tests.fixtures import (
    _squeeze_sim_dims,
    data_conditioned_jumpy_controls,
    data_conditioned_jumpy_controls_ode,
    data_conditioned_jumpy_controls_sde,
)


def test_filter_sdesimulator_known_params():
    """Filter + SDESimulator: filtered means track observations with known dynamics."""
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_sde()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_obs = synthetic["observations"]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_obs.shape == filtered_means.shape
    assert jnp.allclose(synthetic_obs, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_obs - filtered_means)) < 3.5e-2


def test_filter_odesimulator_known_params():
    """Filter + ODESimulator: filtered means track observations with known dynamics."""
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_ode()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_obs = synthetic["observations"]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_obs.shape == filtered_means.shape
    assert jnp.allclose(synthetic_obs, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_obs - filtered_means)) < 0.1


@pytest.mark.parametrize("filter_type", ["kf", "ekf"])
def test_filter_discretetimesimulator_known_params(filter_type):
    """Filter + DiscreteTimeSimulator: filtered means track observations."""
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls(
        filter_type=filter_type,
        filter_source="cuthbert",
    )
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_obs = synthetic["observations"]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_obs.shape == filtered_means.shape
    assert jnp.allclose(synthetic_obs, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_obs - filtered_means)) < 1e-1


@pytest.mark.parametrize("source", ["diffrax", "em_scan"])
def test_filter_simulator_sde_explicit(source):
    """Filter + SDESimulator: explicit SDESimulator (not Simulator) works."""
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_sde()
    rng_key = jr.PRNGKey(42)
    with SDESimulator(source=source):
        with trace() as tr, seed(rng_seed=rng_key):
            data_conditioned_model()
    assert "f_filtered_states_mean" in tr
    assert "f_marginal_loglik" in tr


def test_filter_simulator_ode_explicit():
    """Filter + ODESimulator: explicit ODESimulator (via Simulator) works."""
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_ode()
    rng_key = jr.PRNGKey(42)
    with Simulator():
        with trace() as tr, seed(rng_seed=rng_key):
            data_conditioned_model()
    assert "f_filtered_states_mean" in tr
    assert "f_marginal_loglik" in tr


@pytest.mark.parametrize("source", ["diffrax", "em_scan"])
def test_filter_sdesimulator_predict_times_n_simulations(source):
    """Filter + SDESimulator with predict_times and n_simulations > 1 produces CI-ready outputs."""
    from numpyro.infer import Predictive

    from tests.models import continuous_time_stochastic_l63_model

    rng_key = jr.PRNGKey(42)
    obs_times = jnp.linspace(0.0, 1.0, 6)  # sparse for speed
    predict_times = jnp.linspace(0.0, 1.5, 10)
    true_rho = 28.0

    # Generate observations
    with SDESimulator(n_simulations=1, source=source):
        pred = Predictive(
            continuous_time_stochastic_l63_model,
            params={"rho": jnp.array(true_rho)},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(rng_key, predict_times=obs_times)
    obs_values = jnp.array(_squeeze_sim_dims(sim["f_observations"]))

    # Filter + SDESimulator with n_simulations
    substituted = numpyro.handlers.substitute(
        continuous_time_stochastic_l63_model, data={"rho": jnp.array(true_rho)}
    )
    n_sim = 2
    with SDESimulator(n_simulations=n_sim, source=source):
        with Filter(
            filter_config=ContinuousTimeEnKFConfig(
                n_particles=8, record_filtered_states_mean=True
            )
        ):
            with trace() as tr, seed(rng_seed=0):
                substituted(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    predict_times=predict_times,
                )

    assert "f_predicted_states" in tr
    assert "f_predicted_times" in tr
    assert "f_predicted_observations" in tr
    assert "f_filtered_states_mean" in tr
    pred_states = tr["f_predicted_states"]["value"]
    assert pred_states.shape == (n_sim, len(predict_times), 3)


def test_filter_discretetimesimulator_predict_times_n_simulations():
    """Filter + DiscreteTimeSimulator with predict_times and n_simulations produces CI-ready outputs."""
    from numpyro.infer import Predictive

    from dynestyx.inference.filter_configs import KFConfig
    from tests.models import discrete_time_lti_simplified_model

    rng_key = jr.PRNGKey(42)
    obs_times = jnp.arange(0.0, 6.0, 1.0)  # 6 points
    predict_times = jnp.arange(0.0, 6.0, 1.0)  # same grid for simplicity
    true_alpha = 0.35

    # Generate observations
    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            discrete_time_lti_simplified_model,
            params={"alpha": jnp.array(true_alpha)},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(rng_key, predict_times=predict_times)
    obs_values = jnp.array(_squeeze_sim_dims(sim["f_observations"]))

    # Filter + DiscreteTimeSimulator with n_simulations and predict_times
    substituted = numpyro.handlers.substitute(
        discrete_time_lti_simplified_model, data={"alpha": jnp.array(true_alpha)}
    )
    n_sim = 2
    with DiscreteTimeSimulator(n_simulations=n_sim):
        with Filter(
            filter_config=KFConfig(
                record_filtered_states_mean=True, filter_source="cuthbert"
            )
        ):
            with trace() as tr, seed(rng_seed=0):
                substituted(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    predict_times=predict_times,
                )

    assert "f_predicted_states" in tr
    assert "f_predicted_times" in tr
    assert "f_predicted_observations" in tr
    assert "f_filtered_states_mean" in tr
    pred_states = tr["f_predicted_states"]["value"]
    assert pred_states.shape == (n_sim, len(predict_times), 2), pred_states.shape


def test_filter_discretetimesimulator_pf_predict_times_particle_sampling():
    """PF + DiscreteTimeSimulator uses stochastic particle-mixture rollout."""
    from numpyro.infer import Predictive

    from tests.models import discrete_time_lti_simplified_model

    rng_key = jr.PRNGKey(7)
    obs_times = jnp.arange(0.0, 6.0, 1.0)
    predict_times = jnp.arange(0.0, 9.0, 1.0)
    true_alpha = 0.35

    # Generate training observations once.
    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            discrete_time_lti_simplified_model,
            params={"alpha": jnp.array(true_alpha)},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(rng_key, predict_times=obs_times)
    obs_values = jnp.array(_squeeze_sim_dims(sim["f_observations"]))

    substituted = numpyro.handlers.substitute(
        discrete_time_lti_simplified_model, data={"alpha": jnp.array(true_alpha)}
    )
    n_sim = 4
    with DiscreteTimeSimulator(n_simulations=n_sim):
        with Filter(filter_config=PFConfig(n_particles=128, filter_source="cuthbert")):
            with trace() as tr, seed(rng_seed=0):
                substituted(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    predict_times=predict_times,
                )

    pred_states = tr["f_predicted_states"]["value"]
    assert pred_states.shape == (n_sim, len(predict_times), 2)

    # Segment initial states are sampled from the PF posterior mixture.
    x0_keys = [k for k in tr if k.startswith("f_") and k.endswith("_x_0")]
    assert x0_keys, "Expected rollout segment initial-state sites."
    x0 = tr[x0_keys[0]]["value"]
    assert x0.shape[0] == n_sim
    assert not jnp.allclose(x0[0], x0[1])
