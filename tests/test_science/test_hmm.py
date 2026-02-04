import arviz as az
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS

from dsx.plotters import plot_hmm_states_and_observations
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
            times=obs_times.squeeze(0),
            x=synthetic["states"].squeeze(0),
            y=synthetic["observations"].squeeze(0),
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
        import matplotlib.pyplot as plt

        az.plot_posterior(posterior_sigma, hdi_prob=0.95, ref_val=true_params["sigma"])
        plt.savefig(OUTPUT_DIR / "posterior_sigma.png", dpi=150, bbox_inches="tight")
        plt.close()

    assert jnp.abs(posterior_sigma.mean() - true_params["sigma"]) < 2.0

    hdi_data = az.hdi(posterior_sigma, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_params["sigma"] <= hdi_max, (
        f"True sigma {true_params['sigma']} not in HDI {hdi_min}, {hdi_max}"
    )
