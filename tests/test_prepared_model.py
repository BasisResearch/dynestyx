"""Prepared model layouts persist across automatic LPB executions and JIT."""

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.handlers import seed
from numpyro.infer import MCMC, NUTS

import dynestyx as dsx


def _dynamics():
    return dsx.LTI_discrete(
        A=jnp.eye(1) * 0.8, Q=jnp.eye(1) * 0.1, H=jnp.eye(1), R=jnp.eye(1) * 0.2
    )


def _model(obs_times, obs_values, missing_obs_metadata=None):
    return dsx.sample(
        "f",
        _dynamics(),
        obs_times=obs_times,
        obs_values=obs_values,
        missing_obs_metadata=missing_obs_metadata,
    ).joint_log_prob


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_prepared_model_eager_then_jit(prepare, missing):
    times = jnp.arange(3.0)
    values = jnp.array([[0.0], [jnp.nan if missing else 0.1], [0.2]])
    model = dsx.PreparedModel(_model)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        if prepare:
            assert model.prepare(times, values) is model
        expected = seed(model, rng_seed=0)(times, values)
        with jax.checking_leaks():
            compiled = jax.jit(seed(model, rng_seed=0))
            actual = compiled(times, values)
    assert jnp.isfinite(actual)
    assert jnp.allclose(actual, expected)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        expected = seed(model, rng_seed=0)(times + 10, values + 0.3)
    assert jnp.allclose(compiled(times + 10, values + 0.3), expected)
    assert inspect.signature(model) == inspect.signature(_model)


def test_prepared_model_isolation_and_fixed_layout():
    times = jnp.arange(3.0)
    full = jnp.zeros((3, 1))
    missing = full.at[1, 0].set(jnp.nan)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        first = dsx.PreparedModel(_model).prepare(times, full)
        second = dsx.PreparedModel(_model).prepare(times, missing)
        compiled = jax.jit(seed(first, rng_seed=0))
        assert jnp.isfinite(compiled(times, full))
        assert jnp.isfinite(jax.jit(seed(second, rng_seed=0))(times, missing))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="does not match.*layout"
    ):
        compiled(times, missing).block_until_ready()
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="does not match.*layout"
    ):
        first.prepare(times, missing)
    assert jnp.isfinite(compiled(times, full))
    with pytest.raises(ValueError, match="needs a fixed.*missingness pattern"):
        jax.jit(seed(_model, rng_seed=0))(times, full)


def test_prepared_model_explicit_metadata_overrides_cached_layout():
    times = jnp.arange(3.0)
    full = jnp.zeros((3, 1))
    missing = full.at[1, 0].set(jnp.nan)
    metadata = dsx.prepare_missing_observation_metadata(
        _dynamics(), obs_times=times, obs_values=missing
    )
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        model = dsx.PreparedModel(_model).prepare(times, full)
        assert jnp.isfinite(seed(model, rng_seed=0)(times, missing, metadata))
        assert jnp.isfinite(jax.jit(seed(model, rng_seed=0))(times, full))


def test_prepared_model_multiple_sites_and_plate_layouts():
    times = jnp.arange(3.0)
    values = jnp.zeros((2, 3, 1)).at[0, 0, 0].set(jnp.nan).at[1, 1, 0].set(jnp.nan)

    def model(obs_times, obs_values):
        first = _model(obs_times, obs_values[0])
        with dsx.plate("members", 2):
            second = dsx.sample(
                "g", _dynamics(), obs_times=obs_times, obs_values=obs_values
            ).joint_log_prob
        assert first is not None and second is not None
        return first + second.sum()

    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        wrapped = dsx.PreparedModel(model).prepare(times, values)
        expected = seed(wrapped, rng_seed=0)(times, values)
        actual = jax.jit(seed(wrapped, rng_seed=0))(times, values)
    assert jnp.isfinite(actual)
    assert jnp.allclose(actual, expected)


def test_prepared_model_jitted_mcmc_arguments():
    times = jnp.arange(3.0)
    values = jnp.zeros((3, 1)).at[1, 0].set(jnp.nan)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        model = dsx.PreparedModel(_model).prepare(times, values)
        mcmc = MCMC(
            NUTS(model),
            num_warmup=2,
            num_samples=2,
            jit_model_args=True,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(0), times, values)
    assert jnp.all(jnp.isfinite(mcmc.get_samples()["f_state_path_params"]))


def test_prepared_model_nested_cache_scope():
    times = jnp.arange(3.0)
    values = jnp.zeros((3, 1))
    missing = values.at[1, 0].set(jnp.nan)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        inner = dsx.PreparedModel(_model).prepare(times, missing)

    def model(obs_times, obs_values):
        first = inner(obs_times, missing)
        second = dsx.sample(
            "outer", _dynamics(), obs_times=obs_times, obs_values=obs_values
        ).joint_log_prob
        assert first is not None and second is not None
        return first + second

    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        outer = dsx.PreparedModel(model).prepare(times, values)
        expected = seed(outer, rng_seed=0)(times, values)
        actual = jax.jit(seed(outer, rng_seed=0))(times, values)
    assert jnp.isfinite(actual)
    assert jnp.allclose(actual, expected)


def test_failed_preparation_discards_layout_and_restores_scope():
    times = jnp.arange(3.0)
    values = jnp.zeros((3, 1))

    def model(obs_times, obs_values, fail=False):
        result = _model(obs_times, obs_values)
        if fail:
            raise ValueError("preparation failed")
        return result

    wrapped = dsx.PreparedModel(model)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        with pytest.raises(ValueError, match="preparation failed"):
            wrapped.prepare(times, values, fail=True)
    for callable_model in (wrapped, _model):
        with pytest.raises(ValueError, match="needs a fixed.*missingness pattern"):
            jax.jit(seed(callable_model, rng_seed=0))(times, values)
