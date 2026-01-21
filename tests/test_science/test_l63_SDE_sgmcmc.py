import jax.numpy as jnp
import jax.random as jr
import jax

import arviz as az
import pytest

from tests.fixtures import data_conditioned_continuous_time_stochastic_l63  # noqa: F401
from tests.test_utils import get_output_dir

from numpyro.infer.util import initialize_model
import blackjax

SAVE_FIG = True
OUTPUT_DIR = get_output_dir("test_l63_SDE_sgmcmc")


@pytest.mark.parametrize("num_samples", [250])
def test_mcmc_inference(data_conditioned_continuous_time_stochastic_l63, num_samples):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
    ) = data_conditioned_continuous_time_stochastic_l63

    obs_times = synthetic["times"]

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        plt.plot(
            obs_times.squeeze(0),
            synthetic["states"].squeeze(0)[:, 0],
            label="x[0]",
        )
        plt.plot(
            obs_times.squeeze(0),
            synthetic["observations"].squeeze(0)[:, 0],
            label="observations",
            linestyle="--",
        )
        plt.legend()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

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

    sgld = blackjax.sgld(grad_estimator)  # , 25)
    position = sgld.init(initial_position)

    def schedule_fn(k):
        return 1e-4 * jnp.ones(k.shape)

    num_steps = 500
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

    if SAVE_FIG and OUTPUT_DIR is not None:
        az.plot_posterior(
            posterior_rho, hdi_prob=0.95, ref_val=true_params["rho"].item()
        )
        plt.savefig(OUTPUT_DIR / "posterior_rho.png", dpi=150, bbox_inches="tight")
        plt.close()

    assert jnp.abs(posterior_rho.mean() - true_params["rho"]) < 5.0

    hdi_data = az.hdi(posterior_rho, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_params["rho"] <= hdi_max, (
        f"True rho {true_params['rho']} not in HDI {hdi_min}, {hdi_max}"
    )
