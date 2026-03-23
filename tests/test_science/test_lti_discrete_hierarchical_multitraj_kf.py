"""Science test: hierarchical multi-trajectory linear-Gaussian KF inference."""

import arviz as az
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx.inference.filter_configs import KFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models.lti_dynamics import LTI_discrete
from dynestyx.simulators import DiscreteTimeSimulator
from tests.test_utils import get_output_dir

SAVE_FIG = True


def hierarchical_multitraj_lti_model(
    *,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M: int = 12,
):
    """Hierarchical LTI model with population + per-trajectory latent effects."""
    state_dim = 2
    obs_dim = 1
    A_base = jnp.array([[0.0, 0.1], [0.1, 0.85]])
    Q = 0.12 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = 0.2 * jnp.eye(obs_dim)

    mu_raw = numpyro.sample("mu_raw", dist.Normal(0.5, 1.5))
    sigma_raw = numpyro.sample("sigma_raw", dist.HalfNormal(3.0))

    with dsx.plate("trajectories", M):
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(mu_raw, sigma_raw))
        alpha = jnn.sigmoid(alpha_raw)
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
        if obs_values is None:
            times = predict_times if predict_times is not None else obs_times
            dsx.sample("f", dynamics, predict_times=times)
        else:
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )


@pytest.mark.parametrize("num_samples", [200])
def test_hierarchical_multitraj_lti_kf_science(num_samples: int):
    """Recover population mean/scale from multi-trajectory data under exact KF."""
    rng_key = jr.PRNGKey(0)
    data_key, mcmc_key = jr.split(rng_key, 2)

    n_traj = 24
    obs_times = jnp.arange(0.0, 200.0, 1.0)
    true_params = {
        "mu_raw": jnp.array(2.0),
        "sigma_raw": jnp.array(2.0),
    }

    predictive = Predictive(
        hierarchical_multitraj_lti_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(data_key, predict_times=obs_times, M=n_traj)

    output_dir_name = "test_lti_discrete_hierarchical_multitraj_kf"
    output_dir = get_output_dir(output_dir_name)

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        t = synthetic["f_times"][0, 0, 0]
        x = synthetic["f_states"][0, 0, 0]
        y = synthetic["f_observations"][0, 0, 0]
        plt.plot(t, x[:, 0], label="x[0]")
        plt.plot(t, x[:, 1], label="x[1]")
        plt.plot(t, y[:, 0], "--", label="obs")
        plt.legend()
        plt.savefig(output_dir / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    # f_observations: (num_samples, M, n_sim, T, obs_dim)
    obs_values = synthetic["f_observations"][0, :, 0]

    def data_conditioned_model():
        with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
            return hierarchical_multitraj_lti_model(
                obs_times=obs_times,
                obs_values=obs_values,
                M=n_traj,
            )

    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )
    mcmc.run(mcmc_key)

    posterior = mcmc.get_samples()
    assert "mu_raw" in posterior
    assert "sigma_raw" in posterior
    assert "alpha_raw" in posterior

    mu_post = posterior["mu_raw"]
    sigma_post = posterior["sigma_raw"]
    alpha_raw_post = posterior["alpha_raw"]
    assert len(mu_post) == num_samples
    assert len(sigma_post) == num_samples
    assert not jnp.isnan(mu_post).any()
    assert not jnp.isnan(sigma_post).any()
    assert not jnp.isnan(alpha_raw_post).any()
    assert not jnp.isinf(mu_post).any()
    assert not jnp.isinf(sigma_post).any()
    assert not jnp.isinf(alpha_raw_post).any()
    assert alpha_raw_post.shape == (num_samples, n_traj)

    mu_true = true_params["mu_raw"]
    sigma_true = true_params["sigma_raw"]
    alpha_raw_true = synthetic["alpha_raw"][0]

    # On the dynamics scale alpha=sigmoid(mu_raw), require reasonable recovery.
    alpha_mean_post = jnn.sigmoid(mu_post).mean()
    alpha_mean_true = jnn.sigmoid(mu_true)
    assert jnp.abs(alpha_mean_post - alpha_mean_true) < 0.15

    # Conservative tolerance: hierarchical latent effects add posterior variance.
    assert jnp.abs(sigma_post.mean() - sigma_true) < 0.25

    mu_q = jnp.quantile(mu_post, jnp.array([0.025, 0.975]))
    sigma_q = jnp.quantile(sigma_post, jnp.array([0.025, 0.975]))
    assert mu_q[0] <= mu_true <= mu_q[1]
    assert sigma_q[0] <= sigma_true <= sigma_q[1]

    alpha_true = jnn.sigmoid(alpha_raw_true)
    alpha_post_mean = jnn.sigmoid(alpha_raw_post).mean(axis=0)
    assert jnp.mean(jnp.abs(alpha_post_mean - alpha_true)) < 0.2

    alpha_raw_q = jnp.quantile(alpha_raw_post, jnp.array([0.05, 0.95]), axis=0)
    alpha_raw_coverage = jnp.mean(
        (alpha_raw_true >= alpha_raw_q[0]) & (alpha_raw_true <= alpha_raw_q[1])
    )
    assert alpha_raw_coverage > 0.55

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(mu_post, hdi_prob=0.95, ref_val=float(mu_true))
        plt.savefig(output_dir / "posterior_mu_raw.png", dpi=150, bbox_inches="tight")
        plt.close()

        az.plot_posterior(sigma_post, hdi_prob=0.95, ref_val=float(sigma_true))
        plt.savefig(
            output_dir / "posterior_sigma_raw.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 4))
        traj_idx = jnp.arange(n_traj)
        parts = ax.violinplot(alpha_raw_post, positions=traj_idx, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_alpha(0.35)
        ax.scatter(
            traj_idx,
            alpha_raw_true,
            color="black",
            s=16,
            label="true alpha_raw",
            zorder=3,
        )
        ax.set_xlabel("trajectory index")
        ax.set_ylabel("alpha_raw")
        ax.set_title("Trajectory-level posterior: alpha_raw")
        ax.legend()
        plt.savefig(
            output_dir / "posterior_alpha_raw_by_trajectory.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
