"""Tests verifying that the BlackJAX integration does not use Common Random Numbers.

The CRN fix ensures that each MCMC step seeds the model with a fresh PRNG key,
so stochastic filters (EnKF, particle filter) use independent noise at every
iteration rather than the same fixed key throughout the chain.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import Predictive, init_to_median

from dynestyx.inference.filter_configs import (
    ContinuousTimeEnKFConfig,
    EKFConfig,
    PFConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.inference.integrations.blackjax.mcmc import init_model
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import (
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)
from dynestyx.simulators import Simulator
from tests.fixtures import _squeeze_sim_dims
from tests.models import (
    continuous_time_stochastic_l63_model,
    discrete_time_lti_simplified_model,
)

# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


def _make_continuous_data():
    predict_times = jnp.arange(start=0.0, stop=2.0, step=0.05)
    obs_times = predict_times
    true_params = {"rho": jnp.array(28.0)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with Simulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=predict_times)
    return obs_times, _squeeze_sim_dims(synthetic["f_observations"])


def _make_discrete_data():
    obs_times = jnp.arange(start=0.0, stop=30.0, step=1.0)
    true_params = {"alpha": jnp.array(0.35)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with Simulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=obs_times)
    return obs_times, _squeeze_sim_dims(synthetic["f_observations"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_single_position(init_params):
    """Strip leading chain axis from init_params.z if present.

    ``initialize_model`` returns params with a chain axis when given a batch of
    keys.  The potential function operates on a single (chain-axis-free)
    position, so we take the first chain.
    """
    return jax.tree_util.tree_map(lambda x: x[0] if x.ndim > 0 else x, init_params.z)


# ---------------------------------------------------------------------------
# Core CRN correctness tests
# ---------------------------------------------------------------------------


def _model_recording_prng_key(
    obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None
):
    """Minimal model that records the PRNG key available via numpyro.prng_key()."""
    numpyro.sample("rho", dist.Uniform(10.0, 40.0))
    k = numpyro.prng_key()
    # Store as float so it can be extracted from the deterministic trace
    numpyro.deterministic("prng_key_0", k[0].astype(jnp.float32))
    numpyro.deterministic("prng_key_1", k[1].astype(jnp.float32))


class TestStochasticPotentialFn:
    """Verify init_model returns a well-behaved stochastic potential."""

    def test_density_key_propagates_to_numpyro_prng_key(self):
        """Verify the core mechanism: handlers.seed(model, density_key) provides
        density_key-derived PRNG keys to numpyro.prng_key() inside the model.

        This confirms that stochastic filters, which call numpyro.prng_key() to
        obtain their random seed, will see a distinct key at each MCMC step once
        the CRN fix is in place.
        """
        obs_times = jnp.zeros(5)

        def _get_prng_key_in_model(density_key):
            """Run the recording model and return the recorded prng_key values."""
            seeded = handlers.seed(_model_recording_prng_key, density_key)
            t = handlers.trace(seeded).get_trace(obs_times, None)
            return jnp.stack([t["prng_key_0"]["value"], t["prng_key_1"]["value"]])

        key1 = _get_prng_key_in_model(jr.PRNGKey(1))
        key2 = _get_prng_key_in_model(jr.PRNGKey(2))

        assert not jnp.array_equal(key1, key2), (
            "numpyro.prng_key() returned the same value for different density_keys — "
            "the seed is not being propagated correctly."
        )

    def test_same_density_key_gives_same_value(self):
        """Evaluating the potential function twice with the *same* density key
        must be deterministic (reproducible results)."""
        obs_times, obs_values = _make_continuous_data()

        with Filter(filter_config=ContinuousTimeEnKFConfig(n_particles=20)):
            init_params, stochastic_potential_fn_gen, _ = init_model(
                rng_key=jr.split(jr.PRNGKey(0), 1),
                model=continuous_time_stochastic_l63_model,
                model_args=(obs_times, obs_values, None, None),
                model_kwargs={},
                init_strategy=init_to_median,
            )
            position = _get_single_position(init_params)
            potential_fn = stochastic_potential_fn_gen(obs_times, obs_values)

            val1 = potential_fn(position, jr.PRNGKey(7))
            val2 = potential_fn(position, jr.PRNGKey(7))

        assert jnp.isclose(val1, val2), (
            f"Expected identical potential values for the same density key, "
            f"got val1={val1}, val2={val2}"
        )

    def test_deterministic_filter_insensitive_to_key(self):
        """For a deterministic filter (EKF), the key should not affect the
        potential-energy value (since EKF has no internal randomness)."""
        obs_times, obs_values = _make_discrete_data()

        with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
            init_params, stochastic_potential_fn_gen, _ = init_model(
                rng_key=jr.split(jr.PRNGKey(0), 1),
                model=discrete_time_lti_simplified_model,
                model_args=(obs_times, obs_values, None, None),
                model_kwargs={},
                init_strategy=init_to_median,
            )
            position = _get_single_position(init_params)
            potential_fn = stochastic_potential_fn_gen(obs_times, obs_values)

            val1 = potential_fn(position, jr.PRNGKey(1))
            val2 = potential_fn(position, jr.PRNGKey(99))

        assert jnp.isclose(val1, val2, atol=1e-4), (
            f"EKF (deterministic) should give the same potential for any density key, "
            f"got val1={val1}, val2={val2}"
        )


# ---------------------------------------------------------------------------
# BlackJAX MCMC integration smoke tests
# (verify the fix does not break existing functionality)
# ---------------------------------------------------------------------------


class TestBlackJAXMCMCNoSeedSmokes:
    """Smoke tests confirming the full BlackJAX MCMC pipeline still runs."""

    def test_nuts_continuous(self):
        obs_times, obs_values = _make_continuous_data()
        with Filter():
            inference = MCMCInference(
                mcmc_config=NUTSConfig(
                    num_samples=8, num_warmup=8, num_chains=1, mcmc_source="blackjax"
                ),
                model=continuous_time_stochastic_l63_model,
            )
            samples = inference.run(jr.PRNGKey(10), obs_times, obs_values)
        assert "rho" in samples
        assert samples["rho"].shape == (1, 8)

    def test_hmc_continuous(self):
        obs_times, obs_values = _make_continuous_data()
        with Filter():
            inference = MCMCInference(
                mcmc_config=HMCConfig(
                    num_samples=8,
                    num_warmup=8,
                    num_chains=1,
                    mcmc_source="blackjax",
                    step_size=5e-3,
                    num_steps=8,
                ),
                model=continuous_time_stochastic_l63_model,
            )
            samples = inference.run(jr.PRNGKey(11), obs_times, obs_values)
        assert "rho" in samples

    def test_hmc_discrete(self):
        obs_times, obs_values = _make_discrete_data()
        with Filter():
            inference = MCMCInference(
                mcmc_config=HMCConfig(
                    num_samples=8,
                    num_warmup=8,
                    num_chains=1,
                    mcmc_source="blackjax",
                    step_size=5e-3,
                    num_steps=8,
                ),
                model=discrete_time_lti_simplified_model,
            )
            samples = inference.run(jr.PRNGKey(12), obs_times, obs_values)
        assert "alpha" in samples

    def test_sgld_continuous(self):
        obs_times, obs_values = _make_continuous_data()
        with Filter():
            inference = MCMCInference(
                mcmc_config=SGLDConfig(
                    num_samples=8,
                    num_warmup=8,
                    num_chains=1,
                    mcmc_source="blackjax",
                    step_size=1e-4,
                    schedule_power=0.55,
                ),
                model=continuous_time_stochastic_l63_model,
            )
            samples = inference.run(jr.PRNGKey(13), obs_times, obs_values)
        assert "rho" in samples

    def test_mala_continuous(self):
        obs_times, obs_values = _make_continuous_data()
        with Filter():
            inference = MCMCInference(
                mcmc_config=MALAConfig(
                    num_samples=8,
                    num_warmup=8,
                    num_chains=1,
                    mcmc_source="blackjax",
                    step_size=1e-3,
                ),
                model=continuous_time_stochastic_l63_model,
            )
            samples = inference.run(jr.PRNGKey(14), obs_times, obs_values)
        assert "rho" in samples

    def test_hmc_with_particle_filter(self):
        """Verify that the particle filter (which requires a non-None key) works
        correctly inside BlackJAX MCMC after the CRN fix."""
        obs_times, obs_values = _make_discrete_data()
        with Filter(filter_config=PFConfig(n_particles=50)):
            inference = MCMCInference(
                mcmc_config=HMCConfig(
                    num_samples=8,
                    num_warmup=8,
                    num_chains=1,
                    mcmc_source="blackjax",
                    step_size=1e-3,
                    num_steps=4,
                ),
                model=discrete_time_lti_simplified_model,
            )
            samples = inference.run(jr.PRNGKey(15), obs_times, obs_values)
        assert "alpha" in samples


# ---------------------------------------------------------------------------
# CRN fix: verify filter sees different seeds across MCMC iterations
# ---------------------------------------------------------------------------


class TestFilterSeedVariation:
    """Verify the filter key actually varies between MCMC steps.

    We simulate what the MCMC scan does: evaluate the stochastic potential
    function at the same position with a sequence of density keys and confirm
    that the resulting log-density values are not all identical (which would
    indicate CRNs).
    """

    def test_pf_log_density_varies_over_simulated_chain(self):
        """Simulated MCMC chain: log-density at the same position should vary
        across steps when fresh density keys are used (PF is sensitive to key).

        The particle filter has inherent Monte Carlo variance, so evaluating the
        same position with different density keys must produce distinct estimates.
        A CRN regression would cause all values to be identical.
        """
        obs_times, obs_values = _make_discrete_data()

        with Filter(filter_config=PFConfig(n_particles=50)):
            init_params, stochastic_potential_fn_gen, _ = init_model(
                rng_key=jr.split(jr.PRNGKey(0), 1),
                model=discrete_time_lti_simplified_model,
                model_args=(obs_times, obs_values, None, None),
                model_kwargs={},
                init_strategy=init_to_median,
            )
            position = _get_single_position(init_params)
            potential_fn = stochastic_potential_fn_gen(obs_times, obs_values)

            # Simulate 10 "MCMC steps" at the same position with different density keys
            density_keys = jr.split(jr.PRNGKey(99), 10)
            values = jax.vmap(lambda dk: potential_fn(position, dk))(density_keys)

        # Values should NOT all be the same (CRN would make them equal)
        assert not jnp.allclose(values, values[0], atol=1e-4), (
            "All log-density evaluations returned the same value — "
            "this suggests Common Random Numbers are still in use."
        )
        assert jnp.all(jnp.isfinite(values))

    def test_pf_log_density_varies_over_chain(self):
        """Particle-filter log-density should vary with different density keys."""
        obs_times, obs_values = _make_discrete_data()

        with Filter(filter_config=PFConfig(n_particles=50)):
            init_params, stochastic_potential_fn_gen, _ = init_model(
                rng_key=jr.split(jr.PRNGKey(0), 1),
                model=discrete_time_lti_simplified_model,
                model_args=(obs_times, obs_values, None, None),
                model_kwargs={},
                init_strategy=init_to_median,
            )
            position = _get_single_position(init_params)
            potential_fn = stochastic_potential_fn_gen(obs_times, obs_values)

            density_keys = jr.split(jr.PRNGKey(77), 10)
            values = jax.vmap(lambda dk: potential_fn(position, dk))(density_keys)

        assert not jnp.allclose(values, values[0], atol=1e-4), (
            "Particle-filter log-density is identical across density keys — CRNs present."
        )
        assert jnp.all(jnp.isfinite(values))
