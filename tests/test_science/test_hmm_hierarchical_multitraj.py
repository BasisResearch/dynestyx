"""Science test: hierarchical multi-trajectory HMM inference."""

from typing import cast

import arviz as az
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx.inference.filter_configs import HMMConfig
from dynestyx.inference.filters import Filter
from dynestyx.models import DynamicalModel
from dynestyx.simulators import DiscreteTimeSimulator
from tests.test_utils import get_output_dir

SAVE_FIG = True


class _HMMTransition(eqx.Module):
    trans: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        return dist.Categorical(probs=self.trans[x])


class _HMMEmission(eqx.Module):
    means: jnp.ndarray
    sigma: jnp.ndarray

    def __call__(self, x, u, t):
        return dist.Normal(self.means[x], self.sigma)


def hierarchical_hmm_model(
    *,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M: int = 16,
):
    """Hierarchical HMM: global sigma, per-trajectory p_stay via population params."""
    K = 2
    means = jnp.array([-1.0, 1.0])

    mu_raw = numpyro.sample("mu_raw", dist.Normal(1.0, 1.0))
    sigma_raw = numpyro.sample("sigma_raw", dist.HalfNormal(2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    with dsx.plate("trajectories", M):
        p_stay_raw = numpyro.sample("p_stay_raw", dist.Normal(mu_raw, sigma_raw))
        p_stay = jnn.sigmoid(p_stay_raw)
        trans = jnp.stack(
            [
                jnp.stack([p_stay, 1 - p_stay], axis=-1),
                jnp.stack([1 - p_stay, p_stay], axis=-1),
            ],
            axis=-2,
        )

        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.Categorical(probs=jnp.ones(K) / K),
            state_evolution=_HMMTransition(trans=trans),
            observation_model=_HMMEmission(means=means, sigma=sigma),  # type: ignore[arg-type]
        )
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


@pytest.mark.parametrize("num_samples", [250])
def test_hierarchical_hmm_science(num_samples: int):
    """Recover population mean/scale from multi-trajectory HMM data."""
    rng_key = jr.PRNGKey(0)
    data_key, mcmc_key = jr.split(rng_key, 2)

    n_traj = 16
    obs_times = jnp.arange(0.0, 200.0, 1.0)
    true_params = {
        "mu_raw": jnp.array(1.0),
        "sigma_raw": jnp.array(0.5),
        "sigma": jnp.array(0.3),
    }

    predictive = Predictive(
        hierarchical_hmm_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(data_key, predict_times=obs_times, M=n_traj)

    output_dir = get_output_dir("test_hmm_hierarchical_multitraj")

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        t = synthetic["f_times"][0, 0, 0]
        y = synthetic["f_observations"][0, 0, 0]
        plt.plot(t, y, ".", markersize=2, label="obs")
        plt.legend()
        plt.savefig(output_dir / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    # f_observations: (num_samples, M, n_sim, T)
    obs_values = synthetic["f_observations"][0, :, 0]

    def data_conditioned_model():
        with Filter(filter_config=HMMConfig()):
            return hierarchical_hmm_model(
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
    assert "sigma" in posterior
    assert "p_stay_raw" in posterior

    mu_post = posterior["mu_raw"]
    sigma_raw_post = posterior["sigma_raw"]
    sigma_post = posterior["sigma"]
    p_stay_raw_post = posterior["p_stay_raw"]

    assert len(mu_post) == num_samples
    assert len(sigma_raw_post) == num_samples
    assert len(sigma_post) == num_samples
    assert p_stay_raw_post.shape == (num_samples, n_traj)
    assert not jnp.isnan(mu_post).any()
    assert not jnp.isnan(sigma_raw_post).any()
    assert not jnp.isnan(sigma_post).any()
    assert not jnp.isnan(p_stay_raw_post).any()
    assert not jnp.isinf(mu_post).any()
    assert not jnp.isinf(sigma_raw_post).any()
    assert not jnp.isinf(sigma_post).any()
    assert not jnp.isinf(p_stay_raw_post).any()

    mu_true = true_params["mu_raw"]
    sigma_raw_true = true_params["sigma_raw"]
    sigma_true = true_params["sigma"]
    p_stay_raw_true = synthetic["p_stay_raw"][0]

    # Population mean on p_stay scale
    p_stay_mean_post = jnn.sigmoid(mu_post).mean()
    p_stay_mean_true = jnn.sigmoid(mu_true)
    assert jnp.abs(p_stay_mean_post - p_stay_mean_true) < 0.15

    # Sigma_raw recovery
    assert jnp.abs(sigma_raw_post.mean() - sigma_raw_true) < 0.3

    # Emission noise recovery
    assert jnp.abs(sigma_post.mean() - sigma_true) < 0.25

    # Quantile coverage for population params
    mu_q = jnp.quantile(mu_post, jnp.array([0.025, 0.975]))
    sigma_raw_q = jnp.quantile(sigma_raw_post, jnp.array([0.025, 0.975]))
    assert mu_q[0] <= mu_true <= mu_q[1]
    assert sigma_raw_q[0] <= sigma_raw_true <= sigma_raw_q[1]

    # Per-trajectory recovery
    p_stay_true = jnn.sigmoid(p_stay_raw_true)
    p_stay_post_mean = jnn.sigmoid(p_stay_raw_post).mean(axis=0)
    assert jnp.mean(jnp.abs(p_stay_post_mean - p_stay_true)) < 0.2

    p_stay_raw_q = jnp.quantile(p_stay_raw_post, jnp.array([0.05, 0.95]), axis=0)
    p_stay_raw_coverage = jnp.mean(
        (p_stay_raw_true >= p_stay_raw_q[0]) & (p_stay_raw_true <= p_stay_raw_q[1])
    )
    assert p_stay_raw_coverage > 0.55

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(mu_post, hdi_prob=0.95, ref_val=float(mu_true))
        plt.savefig(output_dir / "posterior_mu_raw.png", dpi=150, bbox_inches="tight")
        plt.close()

        az.plot_posterior(sigma_raw_post, hdi_prob=0.95, ref_val=float(sigma_raw_true))
        plt.savefig(
            output_dir / "posterior_sigma_raw.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        az.plot_posterior(sigma_post, hdi_prob=0.95, ref_val=float(sigma_true))
        plt.savefig(output_dir / "posterior_sigma.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 4))
        traj_idx = jnp.arange(n_traj)
        parts = ax.violinplot(p_stay_raw_post, positions=traj_idx, widths=0.8)
        for pc in cast(list, parts["bodies"]):
            pc.set_alpha(0.35)
        ax.scatter(
            traj_idx,
            p_stay_raw_true,
            color="black",
            s=16,
            label="true p_stay_raw",
            zorder=3,
        )
        ax.set_xlabel("trajectory index")
        ax.set_ylabel("p_stay_raw")
        ax.set_title("Trajectory-level posterior: p_stay_raw")
        ax.legend()
        plt.savefig(
            output_dir / "posterior_p_stay_raw_by_trajectory.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
