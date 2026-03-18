import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from dynestyx.inference.filter_configs import (
    MarginalPFConfig,
    PFConfig,
    PFResamplingConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.simulators import DiscreteTimeSimulator
from tests.models import discrete_time_l63_model


def test_discrete_pfjax_particle_filter_records_outputs() -> None:
    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.1)
    true_params = {"rho": jnp.array(28.0)}

    predictive = Predictive(
        discrete_time_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(0), obs_times=obs_times)

    obs_values = synthetic["observations"].squeeze(0)

    def data_conditioned_model():
        with Filter(filter_config=PFConfig(n_particles=64, filter_source="pfjax")):
            return discrete_time_l63_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
        data_conditioned_model()

    assert "f_marginal_loglik" in tr
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])

    particles = tr["f_filtered_particles"]["value"]
    log_weights = tr["f_filtered_log_weights"]["value"]
    assert particles.shape == (obs_times.shape[0], 64, 3)
    assert log_weights.shape == (obs_times.shape[0], 64)


def test_discrete_pfjax_marginal_particle_filter_records_outputs() -> None:
    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.1)
    true_params = {"rho": jnp.array(28.0)}

    predictive = Predictive(
        discrete_time_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(2), obs_times=obs_times)

    obs_values = synthetic["observations"].squeeze(0)

    def data_conditioned_model():
        with Filter(filter_config=MarginalPFConfig(n_particles=64)):
            return discrete_time_l63_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(3)):
        data_conditioned_model()

    assert "f_marginal_loglik" in tr
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])

    particles = tr["f_filtered_particles"]["value"]
    log_weights = tr["f_filtered_log_weights"]["value"]
    assert particles.shape == (obs_times.shape[0], 64, 3)
    assert log_weights.shape == (obs_times.shape[0], 64)


def test_discrete_pfjax_stop_gradient_forces_always_resample() -> None:
    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.1)
    true_params = {"rho": jnp.array(28.0)}

    predictive = Predictive(
        discrete_time_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(4), obs_times=obs_times)

    obs_values = synthetic["observations"].squeeze(0)

    def data_conditioned_model():
        with Filter(
            filter_config=PFConfig(
                n_particles=32,
                filter_source="pfjax",
                ess_threshold_ratio=0.0,
                resampling_method=PFResamplingConfig(
                    differential_method="stop_gradient"
                ),
            )
        ):
            return discrete_time_l63_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    with pytest.warns(UserWarning, match="requires resampling at every step"):
        with trace(), seed(rng_seed=jr.PRNGKey(5)):
            data_conditioned_model()
