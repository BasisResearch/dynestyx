"""Smoke tests for hierarchical plate-aware variational inference (SVI)."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

import dynestyx as dsx
from dynestyx.inference.filter_configs import KFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models.lti_dynamics import LTI_discrete


def _make_hierarchical_lti_data(M=3, T=10, seed=0):
    """Generate M trajectories from LTI models with per-trajectory alpha."""
    key = jr.PRNGKey(seed)
    state_dim = 2
    obs_dim = 1
    alphas = jnp.array([0.3, 0.5, 0.2])[:M]
    obs_times = jnp.arange(T, dtype=jnp.float32)

    all_obs = []
    for i in range(M):
        k1, k2, key = jr.split(key, 3)
        A = jnp.array([[alphas[i], 0.1], [0.1, 0.8]])
        Q = 0.1 * jnp.eye(state_dim)
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.25]])

        x = jr.multivariate_normal(k1, jnp.zeros(state_dim), jnp.eye(state_dim))
        ys = []
        for t in range(T):
            k1, k2, key = jr.split(key, 3)
            y = H @ x + jr.multivariate_normal(k2, jnp.zeros(obs_dim), R)
            ys.append(y)
            x = A @ x + jr.multivariate_normal(k1, jnp.zeros(state_dim), Q)
        all_obs.append(jnp.stack(ys))

    obs_values = jnp.stack(all_obs)  # (M, T, obs_dim)
    return obs_times, obs_values


def _hierarchical_lti_model_fixed_noise(obs_times=None, obs_values=None, M=3, **kwargs):
    """Hierarchical LTI model: fixed noise, per-trajectory alpha only."""
    state_dim = 2
    Q = 0.1 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.25]])

    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7))
        A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
        )


def test_hierarchical_lti_svi_smoke():
    """SVI with hierarchical LTI model + KF filter runs without error."""
    M = 3
    obs_times, obs_values = _make_hierarchical_lti_data(M=M)

    def data_conditioned_model():
        with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
            return _hierarchical_lti_model_fixed_noise(
                obs_times=obs_times,
                obs_values=obs_values,
                M=M,
            )

    guide = AutoLowRankMultivariateNormal(data_conditioned_model, rank=2)
    optimizer = optax.adam(learning_rate=1e-3)
    svi = SVI(data_conditioned_model, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(jr.PRNGKey(0), 5)

    posterior = guide.sample_posterior(
        jr.PRNGKey(1), svi_result.params, sample_shape=(2,)
    )

    assert "alpha" in posterior
    assert posterior["alpha"].shape == (2, M)
    assert not jnp.isnan(posterior["alpha"]).any()
    assert not jnp.isinf(posterior["alpha"]).any()
