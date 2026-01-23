import jax.numpy as jnp
import jax.random as jr

import arviz as az
from numpyro.infer import MCMC, NUTS
import pytest

from tests.fixtures import data_conditioned_continuous_time_stochastic_l63  # noqa: F401
from tests.test_utils import get_output_dir


SAVE_FIG = True


@pytest.mark.parametrize("num_samples", [250])
def test_mcmc_inference(data_conditioned_continuous_time_stochastic_l63, num_samples):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
    ) = data_conditioned_continuous_time_stochastic_l63

    # Set output dir based on whether controls are used
    output_dir_name = "test_l63_SDE_mcmc" + ("_controlled" if use_controls else "")
    OUTPUT_DIR = get_output_dir(output_dir_name)

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

    mcmc_key = jr.PRNGKey(0)
    nuts_kernel = NUTS(data_conditioned_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_samples)
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

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
