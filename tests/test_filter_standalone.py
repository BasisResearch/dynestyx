"""Tests for dsx.condition (standalone, no numpyro) and dsx.sample (numpyro model)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import dynestyx as dsx
from dynestyx.inference.filter_configs import EnKFConfig, KFConfig
from dynestyx.inference.filters import Filter
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


# --- dsx.condition tests (standalone, no numpyro) ---


def test_infer_returns_infer_result():
    """dsx.condition returns an ConditionedResult with marginal_loglik."""
    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with Filter(filter_config=KFConfig(filter_source="cuthbert")):
        result = dsx.condition(
            "f", dynamics, obs_times=obs_times, obs_values=obs_values
        )

    assert isinstance(result, ConditionedResult)
    assert result.marginal_loglik is not None
    assert jnp.isfinite(result.marginal_loglik)
    assert result.states is not None
    assert result.dists is not None


def test_infer_enkf_with_crn_seed():
    """dsx.condition works with EnKF and explicit crn_seed."""
    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with Filter(
        filter_config=EnKFConfig(n_particles=16, crn_seed=jr.PRNGKey(0)),
    ):
        result = dsx.condition(
            "f", dynamics, obs_times=obs_times, obs_values=obs_values
        )

    assert isinstance(result, ConditionedResult)
    assert result.marginal_loglik is not None
    assert jnp.isfinite(result.marginal_loglik)


def test_infer_optax_mle():
    """Use dsx.condition + optax to do MLE without numpyro."""
    obs_times, obs_values = _make_data()

    def neg_loglik(alpha):
        dynamics = _make_lti_dynamics(alpha)
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
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


def test_infer_does_not_register_numpyro_sites():
    """dsx.condition does NOT register numpyro sites — that's dsx.sample's job."""
    from numpyro.handlers import seed, trace

    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            result = dsx.condition(
                "f", dynamics, obs_times=obs_times, obs_values=obs_values
            )

    assert isinstance(result, ConditionedResult)
    assert result.marginal_loglik is not None
    assert "f_marginal_loglik" not in tr
    assert "f_marginal_log_likelihood" not in tr


def test_condition_no_observations():
    """dsx.condition with no obs returns ConditionedResult with marginal_loglik=None."""
    dynamics = _make_lti_dynamics(0.5)

    with Filter(filter_config=KFConfig(filter_source="cuthbert")):
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


# --- dsx.sample tests (numpyro model) ---


def test_sample_registers_numpyro_sites():
    """dsx.sample registers numpyro sites via the callback."""
    from numpyro.handlers import seed, trace

    obs_times, obs_values = _make_data()
    dynamics = _make_lti_dynamics(0.5)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    assert "f_marginal_loglik" in tr
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
