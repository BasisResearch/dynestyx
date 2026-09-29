import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS

from dynestyx.evaluation.plotting_utils import plot_hmm_states_and_observations
from tests.arviz_utils import hdi_bounds, save_posterior_plot
from tests.fixtures import data_conditioned_hmm  # noqa: F401
from tests.test_utils import get_output_dir

SAVE_FIG = True


@pytest.mark.parametrize("num_samples", [250])
def test_mcmc_inference(data_conditioned_hmm, num_samples):  # noqa: F811
    data_conditioned_model, true_params, synthetic, use_controls = data_conditioned_hmm

    # Set output dir based on whether controls are used
    output_dir_name = "test_hmm" + ("_controlled" if use_controls else "")
    OUTPUT_DIR = get_output_dir(output_dir_name)

    obs_times = synthetic["times"]

    if SAVE_FIG and OUTPUT_DIR is not None:
        plot_hmm_states_and_observations(
            times=obs_times,
            x=synthetic["states"],
            y=synthetic["observations"],
            save_path=OUTPUT_DIR / "data_generation.png",
        )

    mcmc_key = jr.PRNGKey(0)
    nuts_kernel = NUTS(data_conditioned_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_samples)
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

    assert "sigma" in posterior_samples
    posterior_sigma = posterior_samples["sigma"]
    assert len(posterior_sigma) == num_samples
    assert not jnp.isnan(posterior_sigma).any()
    assert not jnp.isinf(posterior_sigma).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        save_posterior_plot(
            posterior_sigma,
            name="sigma",
            output_path=OUTPUT_DIR / "posterior_sigma.png",
            ref_val=true_params["sigma"],
        )

    assert jnp.abs(posterior_sigma.mean() - true_params["sigma"]) < 2.0

    hdi_min, hdi_max = hdi_bounds(posterior_sigma)
    assert hdi_min <= true_params["sigma"] <= hdi_max, (
        f"True sigma {true_params['sigma']} not in HDI {hdi_min}, {hdi_max}"
    )
