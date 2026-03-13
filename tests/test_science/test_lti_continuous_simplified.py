"""Science tests for LTI_continuous: exact KF and DPF."""

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS, BarkerMH

from tests.fixtures import (
    data_conditioned_continuous_time_lti_simplified_science,  # noqa: F401
)
from tests.test_utils import get_output_dir

SAVE_FIG = True


@pytest.mark.parametrize("num_samples", [250])
def test_mcmc_inference(
    data_conditioned_continuous_time_lti_simplified_science,  # noqa: F811
    num_samples,
):
    """LTI_continuous: run exact KF and DPF, check posterior recovers true rho."""
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
        filter_type,
    ) = data_conditioned_continuous_time_lti_simplified_science

    output_dir_name = (
        "test_lti_continuous_simplified"
        + ("_controlled" if use_controls else "")
        + f"_filter_{filter_type}"
    )
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
            synthetic["states"].squeeze(0)[:, 1],
            label="x[1]",
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
    if filter_type == "kf":
        mcmc = MCMC(
            NUTS(data_conditioned_model),
            num_samples=num_samples,
            num_warmup=num_samples,
        )
    else:
        mcmc = MCMC(
            BarkerMH(data_conditioned_model),
            num_samples=num_samples,
            num_warmup=num_samples,
        )
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

    assert "rho" in posterior_samples
    posterior_rho = posterior_samples["rho"]
    assert len(posterior_rho) == num_samples
    assert not jnp.isnan(posterior_rho).any()
    assert not jnp.isinf(posterior_rho).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(
            posterior_rho, hdi_prob=0.95, ref_val=true_params["rho"].item()
        )
        plt.savefig(OUTPUT_DIR / "posterior_rho.png", dpi=150, bbox_inches="tight")
        plt.close()

    true_rho = true_params["rho"]
    # KF is exact; DPF has more variance
    tol = 2.0 if filter_type == "kf" else 2.5
    assert jnp.abs(posterior_rho.mean() - true_rho) < tol

    hdi_data = az.hdi(posterior_rho, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_rho <= hdi_max, (
        f"True rho {true_rho} not in HDI {hdi_min}, {hdi_max}"
    )
