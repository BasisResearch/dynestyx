"""Science test: hierarchical nonlinear multi-trajectory inference with EKF."""

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
    """Map unconstrained raw params to stable nonlinear transition gain."""
    beta_raw_arr = jnp.asarray(beta_raw)
    return 0.8 * (jnn.sigmoid(beta_raw_arr) - 0.5)


class _PlateNonlinearTransition(eqx.Module):
    beta: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        # Nonlinear dynamics: tanh term makes transition non-affine in state.
        return 0.75 * x + self.beta * jnp.tanh(x)


def _make_nonlinear_dynamics(beta: jnp.ndarray) -> DynamicalModel:
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), 0.35 * jnp.eye(1)),
        state_evolution=GaussianStateEvolution(
            F=_PlateNonlinearTransition(beta=beta), cov=0.06 * jnp.eye(1)
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.12**2]]),
        ),
    )


def hierarchical_nonlinear_multitraj_model(
    *,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M: int = 16,
):
    """Hierarchical nonlinear model with one trajectory-level beta per member."""
    mu_raw = numpyro.sample("mu_raw", dist.Normal(1.0, 1.0))
    sigma_raw = numpyro.sample("sigma_raw", dist.HalfNormal(2.0))

    with dsx.plate("trajectories", M):
        beta_raw = numpyro.sample("beta_raw", dist.Normal(mu_raw, sigma_raw))
        beta = _beta_from_raw(beta_raw)
        dynamics = _make_nonlinear_dynamics(beta)
        if obs_values is None:
            times = predict_times if predict_times is not None else obs_times
            dsx.sample("f", dynamics, predict_times=times)
        else:
            dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)


@pytest.mark.parametrize("num_samples", [250])
def test_hierarchical_nonlinear_multitraj_ekf_science(num_samples: int):
    """Recover hierarchical nonlinear parameters from multi-trajectory observations."""
    rng_key = jr.PRNGKey(0)
    data_key, mcmc_key = jr.split(rng_key, 2)

    n_traj = 16
    obs_times = jnp.arange(0.0, 100.0, 1.0)
    true_params = {
        "mu_raw": jnp.array(2.0),
        "sigma_raw": jnp.array(2.0),
    }

    predictive = Predictive(
        hierarchical_nonlinear_multitraj_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(data_key, predict_times=obs_times, M=n_traj)

    output_dir = get_output_dir("test_nonlinear_discrete_hierarchical_multitraj_ekf")
    if SAVE_FIG and output_dir is not None:
        import matplotlib.pyplot as plt

        t = synthetic["f_times"][0, 0, 0]
        x = synthetic["f_states"][0, 0, 0]
        y = synthetic["f_observations"][0, 0, 0]
        plt.plot(t, x[:, 0], label="x[0]")
        plt.plot(t, y[:, 0], "--", label="obs")
        plt.legend()
        plt.savefig(output_dir / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    # f_observations: (num_samples, M, n_sim, T, obs_dim)
    obs_values = synthetic["f_observations"][0, :, 0]

    def data_conditioned_model():
        with Filter(filter_config=EKFConfig(filter_source="cd_dynamax")):
            return hierarchical_nonlinear_multitraj_model(
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
    assert "beta_raw" in posterior

    mu_post = posterior["mu_raw"]
    sigma_post = posterior["sigma_raw"]
    beta_raw_post = posterior["beta_raw"]
    beta_raw_true = synthetic["beta_raw"][0]

    assert len(mu_post) == num_samples
    assert len(sigma_post) == num_samples
    assert beta_raw_post.shape == (num_samples, n_traj)
    assert not jnp.isnan(mu_post).any()
    assert not jnp.isnan(sigma_post).any()
    assert not jnp.isnan(beta_raw_post).any()
    assert not jnp.isinf(mu_post).any()
    assert not jnp.isinf(sigma_post).any()
    assert not jnp.isinf(beta_raw_post).any()

    mu_true = true_params["mu_raw"]
    sigma_true = true_params["sigma_raw"]

    beta_pop_post = _beta_from_raw(mu_post).mean()
    beta_pop_true = _beta_from_raw(mu_true)
    assert jnp.abs(beta_pop_post - beta_pop_true) < 0.2
    assert jnp.abs(sigma_post.mean() - sigma_true) < 0.25

    # EKF is approximate for this nonlinear model; use a wider interval check.
    mu_q = jnp.quantile(mu_post, jnp.array([0.01, 0.99]))
    sigma_q = jnp.quantile(sigma_post, jnp.array([0.01, 0.99]))
    assert mu_q[0] <= mu_true <= mu_q[1]
    assert sigma_q[0] <= sigma_true <= sigma_q[1]

    beta_post = _beta_from_raw(beta_raw_post)
    beta_true = _beta_from_raw(beta_raw_true)
    assert jnp.mean(jnp.abs(beta_post.mean(axis=0) - beta_true)) < 0.2

    beta_raw_q = jnp.quantile(beta_raw_post, jnp.array([0.05, 0.95]), axis=0)
    beta_raw_coverage = jnp.mean(
        (beta_raw_true >= beta_raw_q[0]) & (beta_raw_true <= beta_raw_q[1])
    )
    assert beta_raw_coverage > 0.5

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

        traj_idx = jnp.arange(n_traj)
        fig, ax = plt.subplots(figsize=(12, 4))
        parts = ax.violinplot(beta_raw_post, positions=traj_idx, widths=0.8)
        for pc in cast(list, parts["bodies"]):
            pc.set_alpha(0.35)
        ax.scatter(
            traj_idx,
            beta_raw_true,
            color="black",
            s=16,
            label="true beta_raw",
            zorder=3,
        )
        ax.set_xlabel("trajectory index")
        ax.set_ylabel("beta_raw")
        ax.set_title("Trajectory-level posterior: beta_raw")
        ax.legend()
        plt.savefig(
            output_dir / "posterior_beta_raw_by_trajectory.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
