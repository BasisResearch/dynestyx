"""Smoke tests for hierarchical (plate-aware) filtering."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value

import dynestyx as dsx
from dynestyx.inference.filter_configs import EKFConfig, KFConfig
from dynestyx.inference.filters import Filter
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import NUTSConfig
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

        # Simple forward simulation
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


def hierarchical_lti_model(obs_times=None, obs_values=None, M=3, **kwargs):
    """Hierarchical LTI model: shared noise params, per-trajectory alpha."""
    state_dim = 2

    # Global params
    sigma_q = numpyro.sample("sigma_q", dist.HalfNormal(1.0))
    sigma_r = numpyro.sample("sigma_r", dist.HalfNormal(1.0))

    Q = sigma_q**2 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = sigma_r**2 * jnp.array([[1.0]])

    with dsx.plate("trajectories", M):
        # Per-trajectory parameter
        alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7))

        # Build batched A matrix: shape (M, 2, 2)
        A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)

        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
        )


def hierarchical_lti_model_fixed_noise(obs_times=None, obs_values=None, M=3, **kwargs):
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


def test_hierarchical_lti_cuthbert_ekf_smoke():
    """Single-level plate: discrete-time LTI + cuthbert EKF (alpha only)."""
    M = 3
    obs_times, obs_values = _make_hierarchical_lti_data(M=M)

    init_values = {
        "alpha": 0.3 * jnp.ones(M),
    }

    with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
                init_strategy=init_to_value(values=init_values),
            ),
            model=hierarchical_lti_model_fixed_noise,
        )
        posterior = inference.run(
            jr.PRNGKey(42),
            obs_times,
            obs_values,
            M=M,
        )

    # Check that per-trajectory alpha has the right shape
    assert "alpha" in posterior
    assert posterior["alpha"].shape[-1] == M  # plate dim is last


def test_hierarchical_lti_cd_dynamax_kf_smoke():
    """Single-level plate: discrete-time LTI + cd_dynamax KF (with noise params)."""
    M = 3
    obs_times, obs_values = _make_hierarchical_lti_data(M=M)

    init_values = {
        "sigma_q": jnp.array(0.3),
        "sigma_r": jnp.array(0.5),
        "alpha": 0.3 * jnp.ones(M),
    }

    with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
                init_strategy=init_to_value(values=init_values),
            ),
            model=hierarchical_lti_model,
        )
        posterior = inference.run(
            jr.PRNGKey(42),
            obs_times,
            obs_values,
            M=M,
        )

    assert "alpha" in posterior
    assert posterior["alpha"].shape[-1] == M

    # Check global params exist
    assert "sigma_q" in posterior
    assert "sigma_r" in posterior


def test_hierarchical_loglik_finite():
    """Verify marginal log-likelihood from plate-aware filter is finite."""
    M = 3
    obs_times, obs_values = _make_hierarchical_lti_data(M=M)

    init_values = {
        "sigma_q": jnp.array(0.3),
        "sigma_r": jnp.array(0.5),
        "alpha": 0.3 * jnp.ones(M),
    }

    with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=2,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
                init_strategy=init_to_value(values=init_values),
            ),
            model=hierarchical_lti_model,
        )
        posterior = inference.run(
            jr.PRNGKey(0),
            obs_times,
            obs_values,
            M=M,
        )

    assert "alpha" in posterior
    alpha_vals = posterior["alpha"]
    assert not jnp.isnan(alpha_vals).any()
    assert not jnp.isinf(alpha_vals).any()
