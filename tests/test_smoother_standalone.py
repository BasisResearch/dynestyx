"""Tests for dsx.condition with Smoother (standalone, no numpyro)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import optax
import pytest

import dynestyx as dsx
from dynestyx.inference.configs.smoother import KFSmootherConfig
from dynestyx.inference.smoothers import Smoother
from dynestyx.types import ConditionedResult


def _make_lti_dynamics(alpha):
    return dsx.LTI_discrete(
        A=jnp.array([[alpha, 0.1], [0.1, 0.8]]),
        Q=0.1 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.25]]),
    )


def _make_data():
    obs_times = jnp.arange(0.0, 10.0, 1.0)
    key = jr.PRNGKey(42)
    obs_values = jr.normal(key, (len(obs_times), 1))
    return obs_times, obs_values


def _make_dirac_ode_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.1 * x
        ),
        observation_model=dsx.DiracIdentityObservation(),
    )


# --- dsx.condition tests (standalone, no numpyro) ---


def test_infer_smoother_returns_infer_result():
    """dsx.condition with Smoother returns an ConditionedResult."""
    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")):
        result = dsx.condition(
            "f", dynamics, obs_times=obs_times, obs_values=obs_values
        )

    assert isinstance(result, ConditionedResult)
    assert result.marginal_loglik is not None
    assert jnp.isfinite(result.marginal_loglik)
    assert result.states is not None
    assert result.dists is not None


def test_infer_smoother_optax_mle():
    """Use dsx.condition + Smoother + optax to do MLE without numpyro."""
    obs_times, obs_values = _make_data()

    def neg_loglik(alpha):
        dynamics = _make_lti_dynamics(alpha)
        with Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")):
            result = dsx.condition(
                "f", dynamics, obs_times=obs_times, obs_values=obs_values
            )
        return -result.marginal_loglik

    optimizer = optax.adam(1e-2)
    alpha = jnp.array(0.3)
    opt_state = optimizer.init(alpha)

    initial_loss = neg_loglik(alpha)
    grad_fn = jax.grad(neg_loglik)

    for _ in range(20):
        grads = grad_fn(alpha)
        updates, opt_state = optimizer.update(grads, opt_state)
        alpha = optax.apply_updates(alpha, updates)

    final_loss = neg_loglik(alpha)
    assert final_loss < initial_loss


def test_infer_smoother_does_not_register_numpyro_sites():
    """dsx.condition with Smoother does NOT register numpyro sites."""
    from numpyro.handlers import seed, trace

    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")):
            result = dsx.condition(
                "f", dynamics, obs_times=obs_times, obs_values=obs_values
            )

    assert isinstance(result, ConditionedResult)
    assert result.marginal_loglik is not None
    assert "f_marginal_loglik" not in tr
    assert "f_marginal_log_likelihood" not in tr


def test_condition_smoother_no_observations():
    """dsx.condition with Smoother and no obs returns ConditionedResult with marginal_loglik=None."""
    dynamics = _make_lti_dynamics(0.5)
    obs_times, obs_values = _make_data()

    with Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")):
        dsx.condition("observed", dynamics, obs_times=obs_times, obs_values=obs_values)
        result = dsx.condition(
            "f",
            dynamics,
            obs_times=None,
            obs_values=None,
            predict_times=jnp.arange(0.0, 5.0, 1.0),
        )

    assert isinstance(result, ConditionedResult)
    assert result.marginal_loglik is None
    assert result.states is None
    assert result.dists is None


def test_smoother_rejects_dirac_ode_inference():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([[0.1], [0.2], [0.3]])

    with pytest.raises(
        ValueError,
        match="Inference/scoring .* DiracIdentityObservation",
    ):
        with Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")):
            dsx.condition(
                "f",
                _make_dirac_ode_dynamics(),
                obs_times=obs_times,
                obs_values=obs_values,
            )


# --- dsx.sample tests (numpyro model) ---


def test_sample_smoother_registers_numpyro_sites():
    """dsx.sample with Smoother registers numpyro sites via the callback."""
    from numpyro.handlers import seed, trace

    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")):
            dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    assert "f_marginal_loglik" in tr
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
