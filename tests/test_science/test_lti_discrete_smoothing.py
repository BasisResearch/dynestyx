"""Science tests for LTI_discrete smoothing with KF and PF backends."""

import os

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS, BarkerMH, Predictive

from dynestyx.inference.smoother_configs import KFSmootherConfig, PFSmootherConfig
from dynestyx.inference.smoothers import Smoother
from dynestyx.simulators import DiscreteTimeSimulator
from tests.models import discrete_time_lti_simplified_model
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


@pytest.mark.parametrize("smoother_type", ["kf", "pf"])
@pytest.mark.parametrize("num_samples", [250])
def test_discrete_lti_smoother_mcmc_science(smoother_type: str, num_samples: int):
    """Infer alpha with Smoother and validate smoothed trajectories."""
    rng_key = jr.PRNGKey(0)
    data_init_key, mcmc_key, post_key = jr.split(rng_key, 3)

    true_alpha = 0.4
    obs_times = _profiled_arange(
        fast_stop=30.0,
        fast_step=1.0,
        science_stop=200.0,
        science_step=1.0,
    )

    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params={"alpha": jnp.array(true_alpha)},
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        synthetic = predictive(data_init_key, predict_times=obs_times)

    obs_values = synthetic["f_observations"].squeeze(0).squeeze(0)
    true_states = synthetic["f_states"].squeeze(0).squeeze(0)
    plot_times = synthetic["f_times"].squeeze(0).squeeze(0)

    output_dir_name = f"test_lti_discrete_smoothing_simplified_smoother_{smoother_type}"
    output_dir = get_output_dir(output_dir_name)

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        plt.plot(plot_times, true_states[:, 0], label="true x[0]")
        plt.plot(plot_times, true_states[:, 1], label="true x[1]")
        plt.plot(
            plot_times,
            obs_values[:, 0],
            label="observations",
            linestyle="--",
        )
        plt.legend()
        plt.savefig(output_dir / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    config: KFSmootherConfig | PFSmootherConfig
    kernel_cls: type[NUTS] | type[BarkerMH]
    if smoother_type == "kf":
        config = KFSmootherConfig(
            filter_source="cd_dynamax",
            record_smoothed_states_mean=True,
            record_smoothed_states_cov_diag=True,
        )
        kernel_cls = NUTS
    else:
        config = PFSmootherConfig(
            filter_source="cuthbert",
            n_particles=256,
            record_smoothed_states_mean=True,
            record_smoothed_states_cov_diag=True,
        )
        kernel_cls = BarkerMH

    def data_conditioned_model():
        with Smoother(smoother_config=config):
            return discrete_time_lti_simplified_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    mcmc = MCMC(
        kernel_cls(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()
    assert "alpha" in posterior_samples

    posterior_alpha = posterior_samples["alpha"]
    assert len(posterior_alpha) == num_samples
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(posterior_alpha, hdi_prob=0.95, ref_val=true_alpha)
        plt.savefig(output_dir / "posterior_alpha.png", dpi=150, bbox_inches="tight")
        plt.close()

    post_pred = Predictive(
        data_conditioned_model,
        params={"alpha": posterior_alpha.mean()},
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

    rmse_x0 = jnp.sqrt(jnp.mean((smoothed_mean[:, 0] - true_states[:, 0]) ** 2))

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        sigma_x0 = jnp.sqrt(jnp.maximum(smoothed_cov_diag[:, 0], 1e-9))
        plt.plot(plot_times, true_states[:, 0], label="true x[0]")
        plt.plot(plot_times, smoothed_mean[:, 0], label="smoothed x[0]")
        plt.plot(plot_times, obs_values[:, 0], "--", label="observations")
        plt.fill_between(
            plot_times,
            smoothed_mean[:, 0] - 2.0 * sigma_x0,
            smoothed_mean[:, 0] + 2.0 * sigma_x0,
            alpha=0.2,
            label="smoothed +-2 sigma",
        )
        plt.legend()
        plt.savefig(output_dir / "smoothed_state_x0.png", dpi=150, bbox_inches="tight")
        plt.close()

    alpha_tol = 0.15 if smoother_type == "kf" else 0.25
    rmse_tol = 0.8 if smoother_type == "kf" else 1.2

    assert jnp.abs(posterior_alpha.mean() - true_alpha) < alpha_tol
    assert float(rmse_x0) < rmse_tol

    hdi_data = az.hdi(posterior_alpha, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_alpha <= hdi_max, (
        f"True alpha {true_alpha} not in HDI {hdi_min}, {hdi_max}"
    )
