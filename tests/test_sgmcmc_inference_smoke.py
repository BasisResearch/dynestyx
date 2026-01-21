import jax.numpy as jnp
import jax.random as jr
import jax

import pytest

from tests.fixtures import data_conditioned_continuous_time_stochastic_l63  # noqa: F401
from numpyro.infer.util import initialize_model
import blackjax


@pytest.mark.parametrize("num_samples", [5])
def test_sgmcmc_inference_smoke(
    data_conditioned_continuous_time_stochastic_l63,  # noqa: F811
    num_samples,
):
    """Smoke test version - minimal samples to verify code runs without errors."""
    (
        data_conditioned_model,
        true_params,
        synthetic,
    ) = data_conditioned_continuous_time_stochastic_l63

    rng_key, init_key = jax.random.split(jax.random.PRNGKey(0))
    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        init_key,
        data_conditioned_model,
        model_args=(),
        dynamic_args=True,
    )

    def logdensity_fn(position):
        return -potential_fn_gen()(position)

    def grad_estimator(x, _):
        return jax.grad(logdensity_fn)(x)

    initial_position = init_params.z

    sgld = blackjax.sgld(grad_estimator)
    position = sgld.init(initial_position)

    def schedule_fn(k):
        return 1e-4 * jnp.ones(k.shape)

    num_steps = num_samples
    schedule = schedule_fn(jnp.arange(1, num_steps + 1))

    def inference_loop(rng_key, step, initial_position, num_samples, step_sizes):
        position = initial_position
        positions = []

        _step = jax.jit(step)
        for i in range(num_samples):
            rng_key, step_key = jr.split(rng_key)
            position = step(step_key, position, None, step_sizes[i])
            positions.append(position)

        return positions

    positions = inference_loop(
        jax.random.PRNGKey(0), sgld.step, position, num_steps, schedule
    )
    posterior_samples = {
        k: jnp.stack([positions[i][k] for i in range(len(positions))])
        for k in positions[0].keys()
    }

    assert "rho" in posterior_samples
    posterior_rho = posterior_samples["rho"]
    assert len(posterior_rho) == num_samples
    assert not jnp.isnan(posterior_rho).any()
    assert not jnp.isinf(posterior_rho).any()
