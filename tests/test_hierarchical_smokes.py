"""Smoke tests for hierarchical (plate-aware) filtering."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
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


# ---------------------------------------------------------------------------
# Nested (two-level) plate tests
# ---------------------------------------------------------------------------


def _make_nested_hierarchical_data(G=2, M=3, T=10, seed=0):
    """Generate M trajectories x G groups from LTI models.

    NumPyro nested plates produce shapes where the inner plate is the
    leftmost batch dim: ``alpha ~ (M, G)``.  Data must match.
    """
    key = jr.PRNGKey(seed)
    state_dim = 2
    obs_dim = 1
    obs_times = jnp.arange(T, dtype=jnp.float32)

    # Build (M, G, T, obs_dim) to match numpyro plate ordering
    all_obs = jnp.zeros((M, G, T, obs_dim))
    for m in range(M):
        for g in range(G):
            k1, key = jr.split(key)
            alpha = 0.3 + 0.1 * g + 0.05 * m
            A = jnp.array([[alpha, 0.1], [0.1, 0.8]])
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
            all_obs = all_obs.at[m, g].set(jnp.stack(ys))

    return obs_times, all_obs  # (M, G, T, obs_dim)


def nested_plate_model(obs_times=None, obs_values=None, G=2, M=3, **kwargs):
    """Two-level hierarchy: groups x trajectories, fixed noise.

    NumPyro plate ordering: inner plate (trajectories) → leftmost batch dim,
    outer plate (groups) → next.  So alpha has shape ``(M, G)`` and
    obs_values must be ``(M, G, T, obs_dim)``.
    """
    state_dim = 2
    Q = 0.1 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.25]])

    with dsx.plate("groups", G):
        # Per-group parameter — shape (G,)
        beta = numpyro.sample("beta", dist.Normal(0.0, 0.3))

        with dsx.plate("trajectories", M):
            # Per-trajectory parameter — shape (M, G)
            alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7))

            # Build A with shape (M, G, 2, 2)
            A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
            A = jnp.broadcast_to(A_base, (M, G, 2, 2)).copy()
            A = A.at[:, :, 0, 0].set(alpha)
            A = A.at[:, :, 1, 1].set(0.8 + beta[None, :])

            dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)

            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )


def test_nested_plate_filter_trace_records_batched_loglik_shape():
    """Nested plate filtering should record batched marginal log-likelihood."""
    G, M = 2, 3
    obs_times, obs_values = _make_nested_hierarchical_data(G=G, M=M)

    with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
        tr = numpyro.handlers.trace(
            numpyro.handlers.seed(nested_plate_model, jr.PRNGKey(0))
        ).get_trace(obs_times=obs_times, obs_values=obs_values, G=G, M=M)

    assert "f_marginal_loglik" in tr
    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M, G)
    assert jnp.isfinite(loglik).all()


def test_plate_alignment_error_is_clear_for_unbatched_inputs():
    """Inside dsx.plate, fully unbatched filter inputs should fail early."""
    M = 3
    obs_times, obs_values = _make_hierarchical_lti_data(M=1)
    obs_values = obs_values[0]  # (T, obs_dim) intentionally unbatched for plate M>1

    def _invalid_model(obs_times=None, obs_values=None, M=3, **kwargs):
        state_dim = 2
        Q = 0.1 * jnp.eye(state_dim)
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.25]])
        A = jnp.array([[0.3, 0.1], [0.1, 0.8]])  # unbatched dynamics

        with dsx.plate("trajectories", M):
            dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
            dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
        with pytest.raises(ValueError, match="Plate/data shape alignment failed"):
            numpyro.handlers.seed(_invalid_model, jr.PRNGKey(0))(
                obs_times=obs_times, obs_values=obs_values, M=M
            )


def test_nested_plates_cuthbert_ekf_smoke():
    """Two-level nested plates: groups x trajectories + cuthbert EKF."""
    G, M = 2, 3
    obs_times, obs_values = _make_nested_hierarchical_data(G=G, M=M)

    init_values = {
        "beta": jnp.zeros(G),
        "alpha": 0.3 * jnp.ones((M, G)),
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
            model=nested_plate_model,
        )
        posterior = inference.run(
            jr.PRNGKey(42), obs_times, obs_values, G=G, M=M,
        )

    assert "alpha" in posterior
    assert posterior["alpha"].shape[-2:] == (M, G)
    assert "beta" in posterior
    assert posterior["beta"].shape[-1] == G


def test_nested_plates_cd_dynamax_kf_smoke():
    """Two-level nested plates: groups x trajectories + cd_dynamax KF."""
    G, M = 2, 3
    obs_times, obs_values = _make_nested_hierarchical_data(G=G, M=M)

    init_values = {
        "beta": jnp.zeros(G),
        "alpha": 0.3 * jnp.ones((M, G)),
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
            model=nested_plate_model,
        )
        posterior = inference.run(
            jr.PRNGKey(42), obs_times, obs_values, G=G, M=M,
        )

    assert "alpha" in posterior
    assert posterior["alpha"].shape[-2:] == (M, G)
    assert "beta" in posterior
    assert posterior["beta"].shape[-1] == G
