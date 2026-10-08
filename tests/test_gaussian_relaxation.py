"""Numerical and integration coverage for deterministic Gaussian relaxation."""

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from jaxtyping import Array, Float
from numpyro.handlers import seed

import dynestyx as dsx
from dynestyx.inference.configs.filter import EKFConfig, KFConfig
from dynestyx.inference.utils.plate_utils import _slice_dynamics_for_plate_member


def _deterministic_model():
    return dsx.DynamicalModel(
        initial_condition=dist.Delta(jnp.array([1.0, 2.0]), event_dim=1),
        state_evolution=dsx.DeterministicStateEvolution(
            lambda x, u, t, tn: x + (tn - t)
        ),
        observation_model=dsx.DeterministicObservation(lambda x, u, t: 2 * x),
        t0=0.0,
    )


@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
def test_deterministic_event_axes_and_identity(shape):
    x = jnp.ones(shape)
    evo = dsx.DeterministicStateEvolution(lambda x, u, t, tn: x + tn - t)
    obs = dsx.DeterministicObservation(lambda x, u, t: x * 2)
    for distribution, expected in [
        (evo(x, None, 1, 3), x + 2),
        (obs(x, None, 0), 2 * x),
    ]:
        np.testing.assert_array_equal(distribution.sample(jr.key(0)), expected)
        assert distribution.event_shape == (() if not shape else shape[-1:])
        assert distribution.batch_shape == (() if not shape else shape[:-1])
    identity = dsx.DiracIdentityObservation()
    assert isinstance(identity, dsx.DeterministicObservation)
    np.testing.assert_array_equal(identity(x, None, 0).sample(jr.key(0)), x)


def test_deterministic_simulation_is_exact():
    result = dsx.simulate(
        _deterministic_model(),
        rng_key=jr.key(0),
        predict_times=jnp.arange(4.0),
        n_simulations=2,
    )
    assert result.states is not None
    expected = jnp.array([1.0, 2.0]) + jnp.arange(4.0)[:, None]
    np.testing.assert_array_equal(result.states, jnp.broadcast_to(expected, (2, 4, 2)))
    np.testing.assert_array_equal(result.observations, 2 * result.states)


@pytest.mark.parametrize(
    "setting,expected,mode",
    [
        (0.2, 0.2 * jnp.eye(2), "replace"),
        (jnp.array([0.1, 0.3]), jnp.diag(jnp.array([0.1, 0.3])), "add"),
        (
            jnp.array([[0.3, 0.1], [0.1, 0.2]]),
            jnp.array([[0.3, 0.1], [0.1, 0.2]]),
            "replace",
        ),
        (dsx.ScalarCovariance(variance=0.4), 0.4 * jnp.eye(2), "add"),
        (
            dsx.DiagonalCovariance(sd=jnp.array([0.2, 0.3])),
            jnp.diag(jnp.array([0.04, 0.09])),
            "replace",
        ),
    ],
)
def test_relax_deterministic_components(setting, expected, mode):
    model = _deterministic_model()
    relaxed = dsx.relax_dynamics(
        model,
        initial_condition_cov=setting,
        state_evolution_cov=setting,
        observation_model_cov=setting,
        mode=mode,
    )
    x = jnp.array([2.0, 3.0])
    assert isinstance(relaxed.state_evolution, dsx.GaussianStateEvolution)
    assert isinstance(relaxed.observation_model, dsx.GaussianObservation)
    for distribution, mean in [
        (relaxed.initial_condition, jnp.array([1.0, 2.0])),
        (relaxed.state_evolution(x, None, 1, 3), x + 2),
        (relaxed.observation_model(x, None, 0), 2 * x),
    ]:
        np.testing.assert_allclose(distribution.mean, mean)
        np.testing.assert_allclose(distribution.covariance_matrix, expected, rtol=1e-6)
    assert relaxed.t0 is model.t0
    assert isinstance(model.initial_condition, dist.Delta)
    assert isinstance(model.state_evolution, dsx.DeterministicStateEvolution)
    assert dsx.relax_dynamics(model) is model


def test_partial_relaxation_and_metadata():
    model = _deterministic_model()
    relaxed = dsx.relax_dynamics(model, observation_model_cov=0.1)
    assert relaxed.initial_condition is model.initial_condition
    assert relaxed.state_evolution is model.state_evolution
    assert isinstance(relaxed.observation_model, dsx.GaussianObservation)
    assert isinstance(model.observation_model, dsx.DeterministicObservation)
    assert relaxed.observation_model.h is model.observation_model.h
    assert (
        relaxed.state_dim,
        relaxed.observation_dim,
        relaxed.control_dim,
        relaxed.observation_control_alignment,
        relaxed.control_model,
    ) == (
        model.state_dim,
        model.observation_dim,
        model.control_dim,
        model.observation_control_alignment,
        model.control_model,
    )


@pytest.mark.parametrize(
    "initial",
    [dist.Delta(1.0), dist.Normal(1.0, 0.5), dist.Normal(1.0, 0.5).expand((3,))],
)
def test_scalar_initial_events(initial):
    with seed(rng_seed=0), dsx.plate("members", 3):
        model = dsx.DynamicalModel(
            initial,
            dsx.DeterministicStateEvolution(lambda x, u, t, tn: x),
            dsx.DiracIdentityObservation(),
        )
        relaxed = dsx.relax_dynamics(model, initial_condition_cov=0.2, mode="add")
    assert isinstance(relaxed.initial_condition, dist.Normal)
    assert relaxed.initial_condition.event_shape == ()
    np.testing.assert_allclose(
        relaxed.initial_condition.variance,
        0.2 if isinstance(initial, dist.Delta) else 0.45,
        rtol=1e-6,
    )
    assert relaxed.initial_condition.batch_shape == initial.batch_shape


@pytest.mark.parametrize(
    "initial",
    [
        dist.Delta(jnp.array([1.0, 2.0])).to_event(1),
        dist.Normal(jnp.array([1.0, 2.0]), jnp.array([0.2, 0.3])).to_event(1),
        dist.Normal(jnp.array([1.0, 2.0]), jnp.array([0.2, 0.3]))
        .to_event(1)
        .expand((3,)),
        dist.MultivariateNormal(
            jnp.array([1.0, 2.0]), jnp.array([[0.4, 0.1], [0.1, 0.2]])
        ).expand((3,)),
    ],
)
def test_wrapped_vector_initial_conditions(initial):
    with seed(rng_seed=0), dsx.plate("members", 3):
        model = dsx.DynamicalModel(
            initial,
            dsx.DeterministicStateEvolution(lambda x, u, t, tn: x),
            dsx.DiracIdentityObservation(),
        )
        relaxed = dsx.relax_dynamics(model, initial_condition_cov=0.2, mode="add")
    base = initial
    while isinstance(base, (dist.Independent, dist.ExpandedDistribution)):
        base = base.base_dist
    old_cov = (
        jnp.zeros((2, 2))
        if isinstance(base, dist.Delta)
        else (
            base.covariance_matrix
            if isinstance(base, dist.MultivariateNormal)
            else jnp.diag(base.variance)
        )
    )
    np.testing.assert_allclose(relaxed.initial_condition.mean, initial.mean)
    np.testing.assert_allclose(
        relaxed.initial_condition.covariance_matrix,
        jnp.broadcast_to(
            old_cov + 0.2 * jnp.eye(2),
            relaxed.initial_condition.covariance_matrix.shape,
        ),
        rtol=1e-6,
    )
    assert relaxed.initial_condition.event_shape == (2,)


@pytest.mark.parametrize(
    "backend,mode", [("cuthbert", "add"), ("cd_dynamax", "replace")]
)
def test_linear_filter_matches_manual_model(backend, mode):
    model = dsx.LTI_discrete(
        A=0.9 * jnp.eye(2), Q=0.1 * jnp.eye(2), H=jnp.eye(2), R=0.2 * jnp.eye(2)
    )
    settings: dict[str, Any] = dict(
        initial_condition_cov=0.3,
        state_evolution_cov=jnp.array([0.2, 0.4]),
        observation_model_cov=0.5,
        mode=mode,
    )
    relaxed = dsx.relax_dynamics(model, **settings)
    assert isinstance(model.state_evolution, dsx.LinearGaussianStateEvolution)
    manual = dsx.LTI_discrete(
        A=0.9 * jnp.eye(2),
        Q=jnp.diag(jnp.array([0.2, 0.4])) + (0.1 * jnp.eye(2) if mode == "add" else 0),
        H=jnp.eye(2),
        R=(0.7 if mode == "add" else 0.5) * jnp.eye(2),
        initial_cov=(1.3 if mode == "add" else 0.3) * jnp.eye(2),
    )
    arguments = dict(
        obs_times=jnp.arange(4.0),
        obs_values=jnp.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
    )
    config = KFConfig(filter_source=backend)
    with dsx.Filter(config), dsx.GaussianRelaxation(**settings):
        actual = dsx.condition("actual", model, **arguments)
    with dsx.Filter(config):
        expected = dsx.condition("expected", manual, **arguments)
        direct = dsx.condition("direct", relaxed, **arguments)
    np.testing.assert_allclose(
        actual.marginal_loglik, expected.marginal_loglik, rtol=1e-5
    )
    np.testing.assert_allclose(
        direct.marginal_loglik, expected.marginal_loglik, rtol=1e-5
    )
    assert isinstance(relaxed.state_evolution, dsx.LinearGaussianStateEvolution)
    assert isinstance(relaxed.observation_model, dsx.LinearGaussianObservation)


def test_reusable_covariances_in_models_and_lti():
    scalar = dsx.ScalarCovariance(sd=0.2)
    diagonal = dsx.DiagonalCovariance(variance=jnp.array([0.1, 0.3]))
    model = dsx.LTI_discrete(
        A=jnp.eye(2),
        Q=scalar,
        H=jnp.eye(2),
        R=diagonal,
        initial_cov=dsx.FullCovariance(jnp.eye(2)),
    )
    np.testing.assert_allclose(
        model.state_evolution(jnp.ones(2), None, 0, 1).covariance_matrix,
        0.04 * jnp.eye(2),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        model.observation_model(jnp.ones(2), None, 0).covariance_matrix,
        jnp.diag(jnp.array([0.1, 0.3])),
    )
    obs = dsx.GaussianObservation(lambda x, u, t: x**2, diagonal)
    evo = dsx.GaussianStateEvolution(lambda x, u, t, tn: x, scalar)
    np.testing.assert_allclose(
        obs(jnp.ones(2), None, 0).covariance_matrix,
        model.observation_model(jnp.ones(2), None, 0).covariance_matrix,
    )
    np.testing.assert_allclose(
        evo(jnp.ones(2), None, 0, 1).covariance_matrix,
        model.state_evolution(jnp.ones(2), None, 0, 1).covariance_matrix,
    )


@pytest.mark.parametrize("backend", ["cuthbert", "cd_dynamax"])
def test_relaxed_nonlinear_ekf_matches_manual_model(backend):
    model = _deterministic_model()
    manual = dsx.DynamicalModel(
        dist.MultivariateNormal(jnp.array([1.0, 2.0]), 0.1 * jnp.eye(2)),
        dsx.GaussianStateEvolution(lambda x, u, t, tn: x + tn - t, 0.1 * jnp.eye(2)),
        dsx.GaussianObservation(lambda x, u, t: 2 * x, 0.2 * jnp.eye(2)),
        t0=0.0,
    )
    arguments = dict(
        obs_times=jnp.arange(3.0),
        obs_values=jnp.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]),
    )
    config = EKFConfig(filter_source=backend)
    with (
        dsx.Filter(config),
        dsx.GaussianRelaxation(
            initial_condition_cov=0.1,
            state_evolution_cov=0.1,
            observation_model_cov=0.2,
        ),
    ):
        actual = dsx.condition("actual", model, **arguments)
    with dsx.Filter(config):
        expected = dsx.condition("expected", manual, **arguments)
    np.testing.assert_allclose(
        actual.marginal_loglik, expected.marginal_loglik, rtol=1e-5
    )


def test_add_existing_nonlinear_gaussians_and_zero_noise():
    deterministic = _deterministic_model()
    existing = dsx.relax_dynamics(
        deterministic,
        initial_condition_cov=0.2,
        state_evolution_cov=0.3,
        observation_model_cov=0.4,
    )
    zero = dsx.relax_dynamics(
        existing,
        initial_condition_cov=0.0,
        state_evolution_cov=0.0,
        observation_model_cov=0.0,
        mode="add",
    )
    added = dsx.relax_dynamics(
        existing,
        state_evolution_cov=jnp.array([0.1, 0.2]),
        observation_model_cov=jnp.array([[0.3, 0.1], [0.1, 0.2]]),
        mode="add",
    )
    x = jnp.array([1.0, 2.0])
    for component in ["state_evolution", "observation_model"]:
        arguments = (
            (x, None, 0.0, 1.0) if component == "state_evolution" else (x, None, 0.0)
        )
        before = getattr(existing, component)(*arguments)
        after_zero = getattr(zero, component)(*arguments)
        after_added = getattr(added, component)(*arguments)
        np.testing.assert_allclose(before.mean, after_added.mean)
        np.testing.assert_allclose(
            before.covariance_matrix, after_zero.covariance_matrix
        )
        increment = (
            jnp.diag(jnp.array([0.1, 0.2]))
            if component == "state_evolution"
            else jnp.array([[0.3, 0.1], [0.1, 0.2]])
        )
        np.testing.assert_allclose(
            after_added.covariance_matrix, before.covariance_matrix + increment
        )


class _TimeCovariance(eqx.Module):
    variance: jax.Array

    def __call__(self, *times):
        return (self.variance + times[-1]) * jnp.eye(2)


def test_add_preserves_callable_covariance_and_gradients():
    model = dsx.LTI_discrete(A=jnp.eye(2), Q=jnp.eye(2), H=jnp.eye(2), R=jnp.eye(2))
    model = eqx.tree_at(
        lambda m: (m.state_evolution.cov, m.observation_model.R),
        model,
        (_TimeCovariance(jnp.array(0.1)), _TimeCovariance(jnp.array(0.2))),
    )
    relaxed = dsx.relax_dynamics(
        model, state_evolution_cov=0.3, observation_model_cov=0.4, mode="add"
    )
    assert isinstance(relaxed.state_evolution, dsx.LinearGaussianStateEvolution)
    assert isinstance(relaxed.observation_model, dsx.LinearGaussianObservation)
    np.testing.assert_allclose(
        relaxed.state_evolution.params_at(1.0, 2.0).cov, 2.4 * jnp.eye(2), rtol=1e-6
    )
    np.testing.assert_allclose(
        relaxed.observation_model.params_at(3.0).R, 3.6 * jnp.eye(2)
    )

    def objective(added_variance, variance):
        learned = eqx.tree_at(
            lambda m: m.state_evolution.cov, model, _TimeCovariance(variance)
        )
        changed = dsx.relax_dynamics(
            learned, state_evolution_cov=added_variance, mode="add"
        )
        return changed.state_evolution(jnp.ones(2), None, 0.0, 1.0).log_prob(
            jnp.zeros(2)
        )

    gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        jnp.array(0.3), jnp.array(0.1)
    )
    reference = jax.grad(
        lambda added_variance, variance: dist.MultivariateNormal(
            jnp.ones(2), (1.0 + variance + added_variance) * jnp.eye(2)
        ).log_prob(jnp.zeros(2)),
        argnums=(0, 1),
    )(jnp.array(0.3), jnp.array(0.1))
    np.testing.assert_allclose(gradient, reference, rtol=1e-5)


def test_callable_addition_slices_original_and_added_parameters():
    with seed(rng_seed=0), dsx.plate("members", 2):
        model = dsx.DynamicalModel(
            dist.MultivariateNormal(jnp.zeros((2, 2)), jnp.eye(2)),
            dsx.LinearGaussianStateEvolution(
                jnp.eye(2), _TimeCovariance(jnp.array([0.1, 0.2]))
            ),
            dsx.LinearGaussianObservation(jnp.eye(2), jnp.eye(2)),
        )
    noise = dsx.DiagonalCovariance(variance=jnp.array([[0.3, 0.4], [0.5, 0.6]]))
    relaxed = dsx.relax_dynamics(model, state_evolution_cov=noise, mode="add")
    for index in range(2):
        member = _slice_dynamics_for_plate_member(relaxed, (2,), (index,))
        expected = (0.1 + 0.1 * index + 1.0) * jnp.eye(2) + noise.as_matrix(2)[index]
        np.testing.assert_allclose(
            member.state_evolution(jnp.zeros(2), None, 0.0, 1.0).covariance_matrix,
            expected,
        )


def test_member_specific_relaxation_simulation_matches_direct():
    """Plated deterministic relaxation and direct simulation produce identical draws."""
    covariance = dsx.DiagonalCovariance(variance=jnp.array([[0.1, 0.3], [0.2, 0.4]]))
    with seed(rng_seed=0), dsx.plate("members", 2):
        model = dsx.DynamicalModel(
            dist.Delta(jnp.ones((2, 2)), event_dim=1),
            dsx.DeterministicStateEvolution(lambda x, u, t, tn: 0.8 * x),
            dsx.DeterministicObservation(lambda x, u, t: x**2),
        )
    with (
        seed(rng_seed=0),
        dsx.Simulator(),
        dsx.GaussianRelaxation(
            state_evolution_cov=covariance, observation_model_cov=covariance
        ),
        dsx.plate("members", 2),
    ):
        result = dsx.sample("trajectory", model, predict_times=jnp.arange(3.0))
    assert result.states.shape == (2, 1, 3, 2)
    relaxed = dsx.relax_dynamics(
        model, state_evolution_cov=covariance, observation_model_cov=covariance
    )
    with seed(rng_seed=0), dsx.Simulator(), dsx.plate("members", 2):
        direct = dsx.sample("trajectory", relaxed, predict_times=jnp.arange(3.0))
    np.testing.assert_array_equal(result.states, direct.states)
    np.testing.assert_array_equal(result.observations, direct.observations)


def _plate_covariance(kind: str, variance: Float[Array, "*batch"]) -> dsx.Covariance:
    """Build shared or member-specific noise with known event axes."""
    if kind == "scalar":
        return dsx.ScalarCovariance(variance=variance)
    if kind == "diagonal":
        return dsx.DiagonalCovariance(
            variance=variance[..., None] * jnp.array([1.0, 1.8])
        )
    return dsx.FullCovariance(
        variance[..., None, None] * jnp.array([[1.0, 0.2], [0.2, 1.5]])
    )


class _PlateTimeCovariance(eqx.Module):
    """Keep a shared structured covariance visible inside a covariance callable."""

    covariance: dsx.Covariance

    def __call__(self, x, u, t, tn) -> Float[Array, "event_dim event_dim"]:
        """Scale shared noise by the transition duration."""
        return self.covariance.as_matrix(2) * (1 + 0.2 * (tn - t))


@pytest.mark.parametrize("kind", ["scalar", "diagonal", "full"])
@pytest.mark.parametrize("mode", ["replace", "add"])
def test_nested_plate_relaxation_matches_memberwise_filter(kind, mode):
    """Relaxed nested-plate posteriors must equal independently built member models.

    Both plate sizes equal the state/observation dimension. Shared covariance
    event axes must survive, and the transition covariance is the only input
    carrying member-specific axes: initial means, observations, and times are
    all shared. This exercises covariance classification, vmap, and the plate
    alignment guard through both relaxation entry points.
    """
    mean = jnp.array([0.2, -0.1])

    def F(x, u, t, tn):
        return 0.8 * x + 0.05 * (tn - t) * jnp.sin(x)

    def h(x, u, t):
        return x + 0.1 * x**2

    if mode == "replace":
        model = dsx.DynamicalModel(
            dist.Delta(mean, event_dim=1),
            dsx.DeterministicStateEvolution(F),
            dsx.DeterministicObservation(h),
        )
    else:
        model = dsx.DynamicalModel(
            dist.MultivariateNormal(mean, 0.2 * jnp.eye(2)),
            dsx.GaussianStateEvolution(
                F, _PlateTimeCovariance(_plate_covariance(kind, jnp.array(0.1)))
            ),
            dsx.GaussianObservation(h, 0.25 * jnp.eye(2)),
        )
    ic_noise = _plate_covariance(kind, jnp.array(0.15))
    evo_noise = _plate_covariance(kind, jnp.array([[0.04, 0.06], [0.08, 0.11]]))
    obs_noise = _plate_covariance(kind, jnp.array(0.12))
    times = jnp.arange(4.0)
    values = jnp.array([[0.2, -0.1], [0.5, 0.2], [0.7, 0.1], [0.1, -0.2]])
    config = EKFConfig(filter_source="cuthbert")
    with (
        dsx.Filter(config),
        dsx.GaussianRelaxation(
            initial_condition_cov=ic_noise,
            state_evolution_cov=evo_noise,
            observation_model_cov=obs_noise,
            mode=mode,
        ),
        dsx.plate("groups", 2),
        dsx.plate("members", 2),
    ):
        actual = dsx.condition("trajectory", model, obs_times=times, obs_values=values)
    relaxed = dsx.relax_dynamics(
        model,
        initial_condition_cov=ic_noise,
        state_evolution_cov=evo_noise,
        observation_model_cov=obs_noise,
        mode=mode,
    )
    with dsx.Filter(config), dsx.plate("groups", 2), dsx.plate("members", 2):
        direct = dsx.condition("direct", relaxed, obs_times=times, obs_values=values)
    assert actual.marginal_loglik is not None and direct.marginal_loglik is not None
    assert actual.dists is not None and direct.dists is not None
    assert actual.marginal_loglik.shape == (2, 2)
    np.testing.assert_allclose(
        actual.marginal_loglik, direct.marginal_loglik, rtol=1e-6
    )
    for index in np.ndindex(2, 2):
        ic_cov = ic_noise.as_matrix(2) + (0.2 * jnp.eye(2) if mode == "add" else 0)
        evo_cov = evo_noise.as_matrix(2)[index]
        if mode == "add":
            base_cov = _plate_covariance(kind, jnp.array(0.1)).as_matrix(2)

            def member_cov(x, u, t, tn):
                return base_cov * (1 + 0.2 * (tn - t)) + evo_cov

            evolution = dsx.GaussianStateEvolution(F, member_cov)
        else:
            evolution = dsx.GaussianStateEvolution(F, evo_cov)
        obs_cov = obs_noise.as_matrix(2) + (0.25 * jnp.eye(2) if mode == "add" else 0)
        manual = dsx.DynamicalModel(
            dist.MultivariateNormal(mean, ic_cov),
            evolution,
            dsx.GaussianObservation(h, obs_cov),
        )
        with dsx.Filter(config):
            expected = dsx.condition(
                "member", manual, obs_times=times, obs_values=values
            )
        assert expected.marginal_loglik is not None and expected.dists is not None
        np.testing.assert_allclose(
            actual.marginal_loglik[index], expected.marginal_loglik, rtol=1e-5
        )
        for plated_dist, direct_dist, member_dist in zip(
            actual.dists, direct.dists, expected.dists, strict=True
        ):
            assert plated_dist.event_shape == (2,)
            assert plated_dist.batch_shape == (2, 2)
            np.testing.assert_allclose(
                plated_dist.mean[index], member_dist.mean, rtol=1e-5
            )
            np.testing.assert_allclose(
                plated_dist.covariance_matrix[index],
                member_dist.covariance_matrix,
                rtol=1e-5,
            )
            np.testing.assert_allclose(plated_dist.mean, direct_dist.mean, rtol=1e-6)
            np.testing.assert_allclose(
                plated_dist.covariance_matrix, direct_dist.covariance_matrix, rtol=1e-6
            )


@pytest.mark.parametrize(
    "make,match",
    [
        (lambda: dsx.ScalarCovariance(sd=-1.0), "nonnegative"),
        (lambda: dsx.ScalarCovariance(sd=jnp.array(1e30, dtype=jnp.float32)), "finite"),
        (lambda: dsx.DiagonalCovariance(variance=jnp.array([1.0, jnp.nan])), "finite"),
        (lambda: dsx.ScalarCovariance(sd=1.0, variance=1.0), "exactly one"),
        (lambda: dsx.FullCovariance(jnp.ones((2, 3))), "square"),
        (lambda: dsx.FullCovariance(jnp.array([[1.0, 2.0], [0.0, 1.0]])), "symmetric"),
        (
            lambda: dsx.relax_dynamics(
                _deterministic_model(), observation_model_cov=jnp.ones(3)
            ),
            "dimension",
        ),
        (
            lambda: dsx.GaussianRelaxation(state_evolution_cov=jnp.ones((3, 2, 2))),
            "explicit",
        ),
    ],
)
def test_invalid_settings(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_unsupported_and_continuous_inputs():
    with pytest.raises((TypeError, ValueError), match="mode"):
        dsx.GaussianRelaxation(mode=cast(Any, "bad"))
    with pytest.raises(TypeError, match="callables|state_evolution"):
        dsx.GaussianRelaxation(state_evolution_cov=cast(Any, lambda t: jnp.eye(2)))
    model = _deterministic_model()
    unsupported = eqx.tree_at(
        lambda m: m.initial_condition,
        model,
        dist.StudentT(3.0, jnp.zeros(2), 1.0).to_event(1),
    )
    with pytest.raises(TypeError, match="initial_condition"):
        dsx.relax_dynamics(unsupported, initial_condition_cov=0.1)
    custom = eqx.tree_at(
        lambda m: m.state_evolution,
        model,
        lambda x, u, t, tn: dist.Delta(x, event_dim=1),
    )
    with pytest.raises(TypeError, match="state_evolution"):
        dsx.relax_dynamics(custom, state_evolution_cov=0.1)
    continuous = dsx.DynamicalModel(
        model.initial_condition,
        dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: x),
        model.observation_model,
    )
    with pytest.raises(TypeError, match="discrete-time"):
        dsx.relax_dynamics(continuous, state_evolution_cov=0.1)


def test_jit_rejects_invalid_learned_noise():
    @jax.jit
    def objective(variance):
        return dsx.relax_dynamics(
            _deterministic_model(), initial_condition_cov=variance
        ).initial_condition.log_prob(jnp.zeros(2))

    assert jnp.isfinite(objective(jnp.array(0.2)))
    with pytest.raises(Exception, match="nonnegative"):
        objective(jnp.array(-0.2)).block_until_ready()
