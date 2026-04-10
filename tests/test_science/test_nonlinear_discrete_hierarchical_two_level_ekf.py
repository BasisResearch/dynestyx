"""Science test: two-level hierarchical nonlinear multi-trajectory EKF inference."""

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
from dynestyx.inference.filter_configs import EKFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models import DynamicalModel, GaussianStateEvolution
from dynestyx.models.observations import LinearGaussianObservation
from dynestyx.simulators import DiscreteTimeSimulator
from tests.test_utils import get_output_dir

SAVE_FIG = True


def _beta_from_raw(beta_raw) -> jnp.ndarray:
    beta_raw_arr = jnp.asarray(beta_raw)
    return 0.8 * (jnn.sigmoid(beta_raw_arr) - 0.5)


class _NestedNonlinearTransition(eqx.Module):
    beta: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        return 0.70 * x + self.beta * jnp.tanh(x) + 0.05 * jnp.sin(x)


def _make_nonlinear_dynamics(beta: jnp.ndarray) -> DynamicalModel:
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), 0.35 * jnp.eye(1)),
        state_evolution=GaussianStateEvolution(
            F=_NestedNonlinearTransition(beta=beta),
            cov=0.06 * jnp.eye(1),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.12**2]]),
        ),
    )


def hierarchical_nonlinear_two_level_model(
    *,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    G: int = 3,
    M: int = 4,
):
    """Nested hierarchy: global -> group -> trajectory for nonlinear beta."""
    mu_raw = numpyro.sample("mu_raw", dist.Normal(0.0, 1.8))
    sigma_group = numpyro.sample("sigma_group", dist.HalfNormal(2.0))

    with dsx.plate("groups", G):
        group_raw = numpyro.sample("group_raw", dist.Normal(mu_raw, sigma_group))
        sigma_member = numpyro.sample("sigma_member", dist.HalfNormal(0.35))
        with dsx.plate("trajectories", M):
            beta_raw = numpyro.sample("beta_raw", dist.Normal(group_raw, sigma_member))
            beta = _beta_from_raw(beta_raw)
            dynamics = _make_nonlinear_dynamics(beta)
            if obs_values is None:
                times = predict_times if predict_times is not None else obs_times
                dsx.sample("f", dynamics, predict_times=times)
            else:
                dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)


@pytest.mark.parametrize("num_samples", [400])
def test_hierarchical_nonlinear_two_level_ekf_science(num_samples: int):
    """Recover global/group/trajectory nonlinear params in a nested hierarchy."""
    rng_key = jr.PRNGKey(0)
    data_key, mcmc_key = jr.split(rng_key, 2)

    n_groups = 3
    n_traj_per_group = 4
    obs_times = jnp.arange(0.0, 200.0, 1.0)
    true_params = {
        "mu_raw": jnp.array(0.0),
        "sigma_group": jnp.array(1.8),
        # Fix group centers to create visually separated clusters in posterior plots.
        "group_raw": jnp.array([-1.8, 0.0, 1.8]),
        # Keep within-group spread tight so group separation is visible.
        "sigma_member": jnp.array([0.25, 0.25, 0.25]),
    }

    predictive = Predictive(
        hierarchical_nonlinear_two_level_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(
            data_key,
            predict_times=obs_times,
            G=n_groups,
            M=n_traj_per_group,
        )

    output_dir = get_output_dir("test_nonlinear_discrete_hierarchical_two_level_ekf")
    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        t = synthetic["f_times"][0, 0, 0, 0]
        x = synthetic["f_states"][0, 0, 0, 0]
        y = synthetic["f_observations"][0, 0, 0, 0]
        plt.plot(t, x[:, 0], label="x[0]")
        plt.plot(t, y[:, 0], "--", label="obs")
        plt.legend()
        plt.savefig(output_dir / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    # f_observations: (num_samples, M, G, n_sim, T, obs_dim)
    obs_values = synthetic["f_observations"][0, :, :, 0]

    def data_conditioned_model():
        with Filter(filter_config=EKFConfig(filter_source="cd_dynamax")):
            return hierarchical_nonlinear_two_level_model(
                obs_times=obs_times,
                obs_values=obs_values,
                G=n_groups,
                M=n_traj_per_group,
            )

    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )
    mcmc.run(mcmc_key)
    posterior = mcmc.get_samples()

    assert "mu_raw" in posterior
    assert "sigma_group" in posterior
    assert "group_raw" in posterior
    assert "sigma_member" in posterior
    assert "beta_raw" in posterior

    mu_post = posterior["mu_raw"]
    sigma_group_post = posterior["sigma_group"]
    group_raw_post = posterior["group_raw"]
    sigma_member_post = posterior["sigma_member"]
    beta_raw_post = posterior["beta_raw"]

    group_raw_true = synthetic["group_raw"][0]
    beta_raw_true = synthetic["beta_raw"][0]

    assert len(mu_post) == num_samples
    assert len(sigma_group_post) == num_samples
    assert group_raw_post.shape == (num_samples, n_groups)
    assert sigma_member_post.shape == (num_samples, n_groups)
    assert beta_raw_post.shape == (num_samples, n_traj_per_group, n_groups)
    assert not jnp.isnan(mu_post).any()
    assert not jnp.isnan(sigma_group_post).any()
    assert not jnp.isnan(group_raw_post).any()
    assert not jnp.isnan(sigma_member_post).any()
    assert not jnp.isnan(beta_raw_post).any()

    mu_true = true_params["mu_raw"]
    sigma_group_true = true_params["sigma_group"]

    beta_pop_post = _beta_from_raw(mu_post).mean()
    beta_pop_true = _beta_from_raw(mu_true)
    assert jnp.abs(beta_pop_post - beta_pop_true) < 0.25
    assert jnp.abs(sigma_group_post.mean() - sigma_group_true) < 0.9

    # EKF is approximate for this nonlinear model; use a wider interval check.
    mu_q = jnp.quantile(mu_post, jnp.array([0.01, 0.99]))
    sigma_group_q = jnp.quantile(sigma_group_post, jnp.array([0.01, 0.99]))
    assert mu_q[0] <= mu_true <= mu_q[1]
    assert sigma_group_q[0] <= sigma_group_true <= sigma_group_q[1]

    group_mae = jnp.mean(jnp.abs(group_raw_post.mean(axis=0) - group_raw_true))
    assert group_mae < 0.4

    beta_post = _beta_from_raw(beta_raw_post)
    beta_true = _beta_from_raw(beta_raw_true)
    beta_mae = jnp.mean(jnp.abs(beta_post.mean(axis=0) - beta_true))
    assert beta_mae < 0.15

    beta_raw_q = jnp.quantile(beta_raw_post, jnp.array([0.05, 0.95]), axis=0)
    beta_raw_coverage = jnp.mean(
        (beta_raw_true >= beta_raw_q[0]) & (beta_raw_true <= beta_raw_q[1])
    )
    assert beta_raw_coverage > 0.7

    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(mu_post, hdi_prob=0.95, ref_val=float(mu_true))
        plt.savefig(output_dir / "posterior_mu_raw.png", dpi=150, bbox_inches="tight")
        plt.close()

        az.plot_posterior(
            sigma_group_post, hdi_prob=0.95, ref_val=float(sigma_group_true)
        )
        plt.savefig(
            output_dir / "posterior_sigma_group.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        group_idx = jnp.arange(n_groups)
        parts = ax.violinplot(group_raw_post, positions=group_idx, widths=0.7)
        for pc in cast(list, parts["bodies"]):
            pc.set_alpha(0.35)
        ax.scatter(
            group_idx,
            group_raw_true,
            color="black",
            s=20,
            label="true group_raw",
            zorder=3,
        )
        ax.set_xlabel("group index")
        ax.set_ylabel("group_raw")
        ax.set_title("Group-level posterior: group_raw")
        ax.legend()
        plt.savefig(
            output_dir / "posterior_group_raw_by_group.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        beta_raw_post_flat = beta_raw_post.reshape(num_samples, -1)
        beta_raw_true_flat = beta_raw_true.reshape(-1)
        fig, ax = plt.subplots(figsize=(12, 4))
        member_idx = jnp.arange(beta_raw_true_flat.shape[0])
        parts = ax.violinplot(beta_raw_post_flat, positions=member_idx, widths=0.8)
        for pc in cast(list, parts["bodies"]):
            pc.set_alpha(0.35)
        ax.scatter(
            member_idx,
            beta_raw_true_flat,
            color="black",
            s=16,
            label="true beta_raw",
            zorder=3,
        )
        ax.set_xlabel("flattened member index (trajectory, group)")
        ax.set_ylabel("beta_raw")
        ax.set_title("Trajectory-level posterior: beta_raw (two-level hierarchy)")
        ax.legend()
        plt.savefig(
            output_dir / "posterior_beta_raw_by_member.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
