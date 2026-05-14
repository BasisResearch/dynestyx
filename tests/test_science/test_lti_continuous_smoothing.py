"""Science tests for LTI_continuous smoothing with continuous-time KF."""

import os

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS, Predictive

from dynestyx.inference.smoother_configs import ContinuousTimeKFSmootherConfig
from dynestyx.inference.smoothers import Smoother
from dynestyx.simulators import Simulator
from tests.models import continuous_time_lti_simplified_model
from tests.test_utils import get_output_dir

SAVE_FIG = True


def _profiled_arange(
    *, fast_stop: float, fast_step: float, science_stop: float, science_step: float
):
    fast_tests = os.environ.get("DYNESTYX_FAST_TESTS", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    stop = fast_stop if fast_tests else science_stop
    step = fast_step if fast_tests else science_step
    return jnp.arange(start=0.0, stop=stop, step=step)


@pytest.mark.parametrize("num_samples", [250])
def test_continuous_lti_smoother_mcmc_science(num_samples: int):
    """Infer rho with Smoother and validate smoothed trajectories."""
    rng_key = jr.PRNGKey(0)
    data_init_key, mcmc_key, post_key = jr.split(rng_key, 3)

    true_rho = 2.0
    obs_times = _profiled_arange(
        fast_stop=3.0,
        fast_step=0.2,
        science_stop=10.0,
        science_step=0.05,
    )

    predictive = Predictive(
        continuous_time_lti_simplified_model,
        params={"rho": jnp.array(true_rho)},
        num_samples=1,
        exclude_deterministic=False,
    )

    with Simulator():
        synthetic = predictive(data_init_key, predict_times=obs_times)

    obs_values = synthetic["f_observations"].squeeze(0).squeeze(0)
    true_states = synthetic["f_states"].squeeze(0).squeeze(0)
    plot_times = synthetic["f_times"].squeeze(0).squeeze(0)

    output_dir = get_output_dir("test_lti_continuous_smoothing_simplified")

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        plt.plot(plot_times, true_states[:, 0], label="true x[0]")
        plt.plot(plot_times, true_states[:, 1], label="true x[1]")
        plt.plot(plot_times, obs_values[:, 0], "--", label="observations")
        plt.legend()
        plt.savefig(output_dir / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    config = ContinuousTimeKFSmootherConfig(
        record_smoothed_states_mean=True,
        record_smoothed_states_cov_diag=True,
    )

    def data_conditioned_model():
        with Smoother(smoother_config=config):
            return continuous_time_lti_simplified_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    mcmc = MCMC(
        NUTS(data_conditioned_model),
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

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(posterior_rho, hdi_prob=0.95, ref_val=true_rho)
        plt.savefig(output_dir / "posterior_rho.png", dpi=150, bbox_inches="tight")
        plt.close()

    post_pred = Predictive(
        data_conditioned_model,
        params={"rho": posterior_rho.mean()},
        num_samples=1,
        exclude_deterministic=False,
    )
    post_out = post_pred(post_key)

    assert "f_smoothed_states_mean" in post_out
    assert "f_smoothed_states_cov_diag" in post_out

    smoothed_mean = post_out["f_smoothed_states_mean"].squeeze(0)
    smoothed_cov_diag = post_out["f_smoothed_states_cov_diag"].squeeze(0)
    assert smoothed_mean.shape == true_states.shape
    assert smoothed_cov_diag.shape == true_states.shape
    assert jnp.all(jnp.isfinite(smoothed_mean))
    assert jnp.all(jnp.isfinite(smoothed_cov_diag))

    rmse_x1 = jnp.sqrt(jnp.mean((smoothed_mean[:, 1] - true_states[:, 1]) ** 2))

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        sigma_x1 = jnp.sqrt(jnp.maximum(smoothed_cov_diag[:, 1], 1e-9))
        plt.plot(plot_times, true_states[:, 1], label="true x[1]")
        plt.plot(plot_times, smoothed_mean[:, 1], label="smoothed x[1]")
        plt.plot(plot_times, obs_values[:, 0], "--", label="observations")
        plt.fill_between(
            plot_times,
            smoothed_mean[:, 1] - 2.0 * sigma_x1,
            smoothed_mean[:, 1] + 2.0 * sigma_x1,
            alpha=0.2,
            label="smoothed +-2 sigma",
        )
        plt.legend()
        plt.savefig(output_dir / "smoothed_state_x1.png", dpi=150, bbox_inches="tight")
        plt.close()

    assert jnp.abs(posterior_rho.mean() - true_rho) < 2.0
    assert float(rmse_x1) < 2.0

    hdi_data = az.hdi(posterior_rho, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_rho <= hdi_max, (
        f"True rho {true_rho} not in HDI {hdi_min}, {hdi_max}"
    )
