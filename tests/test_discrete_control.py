"""Tests for DiscreteControlLoopSimulator and compute_cuthbert_filter_update."""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.control.discrete_controller_simulators import (
    ControlledSimulatedResult,
    DiscreteControlLoopSimulator,
    filter_state_dist,
    filter_state_mean,
)
from dynestyx.control.mppi import MPPI
from dynestyx.discretizers import (
    Discretizer,
    EulerMaruyamaConfig,
    _discretize_state_evolution,
)
from dynestyx.inference.configs.filter import (
    EKFConfig,
    EnKFConfig,
    KFConfig,
    PFConfig,
    UKFConfig,
)
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
    compute_cuthbert_belief_analysis,
    compute_cuthbert_belief_prediction,
    compute_cuthbert_filter,
    compute_cuthbert_filter_update,
)
from dynestyx.inference.integrations.utils import WeightedParticles
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.lti_dynamics import LTI_discrete
from dynestyx.models.observations import LinearGaussianObservation
from tests.fixtures import _n_particles
from tests.test_utils import (
    assert_finite,
    assert_trace_sites_exist_and_field_all_finite,
    value_at_time,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _lti_1d(
    A=1.0, B=1.0, Q=0.05, R=0.1, observation_control_alignment="previous_transition"
):
    """1D linear-Gaussian model matching the tutorial's discrete-time demo.

    Defaults to "previous_transition" explicitly: the only convention
    closed-loop control supports for now, and naming it keeps the
    unspecified-field warning out of the suite. Pass `None` for an
    unspecified model.
    """
    return LTI_discrete(
        A=jnp.array([[A]]),
        Q=Q * jnp.eye(1),
        H=jnp.array([[1.0]]),
        R=R * jnp.eye(1),
        B=jnp.array([[B]]),
        observation_control_alignment=observation_control_alignment,
    )


def _with_alignment(dynamics, alignment):
    """Set a model's observation/control convention.

    The closed loop reads it from the model, but LTI_discrete does not expose
    the field, so it is rewritten here after construction.
    """
    return eqx.tree_at(lambda m: m.observation_control_alignment, dynamics, alignment)


class _LinearPolicy(eqx.Module):
    """u = -K x_hat, as an equinox.Module policy."""

    K: jax.Array

    def __call__(self, x_hat, t_now, t_next, s):
        return -self.K @ filter_state_mean(x_hat), s


def _linear_policy_fn(K):
    """Plain-function equivalent of _LinearPolicy."""

    def policy(x_hat, t_now, t_next, s):
        return -K @ filter_state_mean(x_hat), s

    return policy


class _BlackBoxState:
    """A genuinely black-box transition result: only `.sample()`/`.shape()`,
    no `.log_prob()`/`.mean` anywhere -- standing in for e.g. a MuJoCo step."""

    def __init__(self, x_prev, u, t_now, t_next, state_dim):
        self._x_prev, self._u, self._t_now, self._t_next = x_prev, u, t_now, t_next
        self._state_dim = state_dim

    def sample(self, key):
        dt = self._t_next - self._t_now
        x_next = jnp.tanh(self._x_prev) + self._u * dt
        return x_next + 0.05 * jr.normal(key, x_next.shape)

    def shape(self):
        return (self._state_dim,)


def _black_box_state_evolution(x, u, t_now, t_next):
    return _BlackBoxState(x, u, t_now, t_next, state_dim=x.shape[-1])


def _black_box_dynamics():
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([1.0]), 0.1 * jnp.eye(1)),
        state_evolution=_black_box_state_evolution,
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=0.1 * jnp.eye(1)),
        control_dim=1,
        observation_control_alignment="previous_transition",
    )


def _run_trace(model, *, rng_seed=0):
    with seed(rng_seed=rng_seed):
        return trace(model).get_trace()


# ---------------------------------------------------------------------------
# Group 1: filter_state_mean
# ---------------------------------------------------------------------------


def test_filter_state_mean_uses_mean_property_when_present():
    class _KFLikeState:
        mean = jnp.array([1.0, 2.0])

    assert jnp.allclose(filter_state_mean(_KFLikeState()), jnp.array([1.0, 2.0]))


def test_filter_state_mean_weighted_particle_average():
    class _PFLikeState:
        particles = jnp.array([[0.0], [2.0], [4.0]])  # 3 particles, state_dim=1
        log_weights = jnp.log(jnp.array([0.25, 0.25, 0.5]))

    result = filter_state_mean(_PFLikeState())
    expected = 0.25 * 0.0 + 0.25 * 2.0 + 0.5 * 4.0  # = 2.5
    assert jnp.allclose(result, jnp.array([expected]), atol=1e-5)


def test_filter_state_dist_gaussian_when_chol_cov_present():
    class _KFLikeState:
        mean = jnp.array([1.0, 2.0])
        chol_cov = jnp.array([[1.0, 0.0], [0.5, 1.0]])

    result = filter_state_dist(_KFLikeState(), KFConfig(filter_source="cuthbert"))
    assert isinstance(result, dist.MultivariateNormal)
    assert jnp.allclose(result.mean, _KFLikeState.mean)
    assert jnp.allclose(result.scale_tril, _KFLikeState.chol_cov)


def test_filter_state_dist_weighted_particles_when_particles_present():
    class _PFLikeState:
        particles = jnp.array([[0.0], [2.0], [4.0]])  # 3 particles, state_dim=1
        log_weights = jnp.log(jnp.array([0.25, 0.25, 0.5]))

    result = filter_state_dist(_PFLikeState(), PFConfig(n_particles=3))
    assert isinstance(result, WeightedParticles)
    assert jnp.allclose(result.particles, _PFLikeState.particles)
    assert jnp.allclose(
        jnp.exp(result.log_weights), jnp.array([0.25, 0.25, 0.5]), atol=1e-5
    )


def test_unsupported_filter_state_type_raises():
    class _Neither:
        pass

    with pytest.raises(TypeError, match="Cannot summarize filter state"):
        filter_state_mean(_Neither())


def test_filter_state_dist_rejects_non_cuthbert_backend():
    """`filter_state_dist` dispatches on the backend, not on the state's shape.

    Only cuthbert states carry a Cholesky factor; cd-dynamax reports dense
    covariances, so it needs its own branch rather than being duck-typed into
    this one.
    """
    with pytest.raises(ValueError, match="filter_source='cuthbert' only"):
        filter_state_dist(object(), UKFConfig())


# ---------------------------------------------------------------------------
# Group 2: compute_cuthbert_filter_update core correctness
# ---------------------------------------------------------------------------

_T = 6
_OBS_TIMES = jnp.arange(_T, dtype=jnp.float32)
_CTRL_VALUES = jnp.ones((_T, 1)) * 0.3
_OBS_VALUES = jnp.array([[0.5], [0.4], [0.3], [0.2], [0.1], [0.05]])


def _step_through_filter_update(dynamics, filter_config, *, key_seed=0):
    """Drive compute_cuthbert_filter_update one step at a time over _OBS_VALUES,
    using the same same-index control convention as compute_cuthbert_filter
    (ctrl_values[t] paired with both the transition into t and the observation
    at t), so the result is directly comparable to the whole-trajectory filter.
    """
    filter_obj, _ = build_cuthbert_filter(
        dynamics, filter_config, key=jr.PRNGKey(key_seed), want_parallel=False
    )
    prev_state = None
    means = []
    k = jr.PRNGKey(key_seed)
    for t_idx in range(_T):
        k, sub = jr.split(k)
        u_for_call = None if t_idx == 0 else _CTRL_VALUES[t_idx - 1]
        t_prev = (
            _OBS_TIMES[0] - (_OBS_TIMES[1] - _OBS_TIMES[0])
            if t_idx == 0
            else _OBS_TIMES[t_idx - 1]
        )
        prev_state = compute_cuthbert_filter_update(
            dynamics,
            filter_obj,
            prev_state,
            sub,
            y=_OBS_VALUES[t_idx],
            u=u_for_call,
            t=_OBS_TIMES[t_idx],
            t_prev=t_prev,
        )
        means.append(filter_state_mean(prev_state))
    return jnp.stack(means)


@pytest.mark.parametrize(
    "filter_config", [KFConfig(filter_source="cuthbert"), EKFConfig()]
)
def test_compute_cuthbert_filter_update_matches_whole_trajectory(filter_config):
    dynamics = _lti_1d()
    _, states_batch = compute_cuthbert_filter(
        dynamics,
        filter_config,
        jr.PRNGKey(0),
        obs_times=_OBS_TIMES,
        obs_values=_OBS_VALUES,
        ctrl_values=_CTRL_VALUES,
    )
    means_batch = states_batch.mean.ravel()
    means_step = _step_through_filter_update(
        dynamics, filter_config, key_seed=0
    ).ravel()
    assert jnp.allclose(means_batch, means_step, atol=1e-4)


def test_compute_cuthbert_initial_observation_update_ignores_u():
    """The initial update must skip the nonexistent preceding transition.

    Passing `prev_state=None` sets `is_first_step=True`, so `u` must have no
    effect. This model's transition is control-affine (B=1.0); if a phantom
    transition used `u`, a huge `u` would visibly shift the mean.
    """
    dynamics = _lti_1d()
    filter_obj, _ = build_cuthbert_filter(
        dynamics, EKFConfig(), key=None, want_parallel=False
    )
    y0 = jnp.array([0.5])
    t0 = jnp.array(0.0)

    state_no_u = compute_cuthbert_filter_update(
        dynamics,
        filter_obj,
        None,
        jr.PRNGKey(0),
        y=y0,
        u=None,
        t=t0,
        t_prev=t0 - 1.0,
    )
    state_huge_u = compute_cuthbert_filter_update(
        dynamics,
        filter_obj,
        None,
        jr.PRNGKey(0),
        y=y0,
        u=jnp.array([999.0]),
        t=t0,
        t_prev=t0 - 1.0,
    )
    assert jnp.allclose(state_no_u.mean, state_huge_u.mean, atol=1e-6)


def _euler_maruyama_dynamics():
    # Euler-Maruyama discretization requires an already-resolved
    # StochasticContinuousTimeStateEvolution (with bm_dim metadata filled in), which
    # only happens inside DynamicalModel.__init__ -- matching exactly what
    # Discretizer._sample_ds does internally.
    cte = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: u,
        diffusion=FullDiffusion(0.2 * jnp.eye(1)),
    )
    continuous_dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([1.0]), 0.1 * jnp.eye(1)),
        state_evolution=cte,
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=0.2 * jnp.eye(1)),
        control_dim=1,
    )
    # DynamicalModel.state_evolution is declared as the broader
    # ContinuousTimeStateEvolution | DiscreteStateTransition union; narrow it
    # for the type checker the same way Discretizer._sample_ds does at
    # runtime (dynestyx/discretizers.py), via an isinstance check.
    assert isinstance(
        continuous_dynamics.state_evolution, StochasticContinuousTimeStateEvolution
    )
    return DynamicalModel(
        initial_condition=continuous_dynamics.initial_condition,
        state_evolution=_discretize_state_evolution(
            continuous_dynamics.state_evolution,
            EulerMaruyamaConfig(),
        ),
        observation_model=continuous_dynamics.observation_model,
        control_dim=continuous_dynamics.control_dim,
    )


def test_compute_cuthbert_filter_update_explicit_t_prev_avoids_degeneracy():
    """Regression test for a real bug: for a transition whose covariance
    scales with dt (e.g. an Euler-Maruyama-discretized SDE), omitting
    `t_prev` collapses to dt=0 for the first real step, producing a
    zero-covariance distribution whose NaN log-density leaks through EKF's
    Taylor-linearization gradient (via jnp.where evaluating both branches).
    An explicit, non-degenerate dummy `t_prev` avoids this for callers that
    perform an initial observation update. The closed-loop simulator no longer
    performs such an update because it does not synthesize an uncontrolled
    initial observation.
    """
    dynamics = _euler_maruyama_dynamics()
    filter_obj, _ = build_cuthbert_filter(
        dynamics, EKFConfig(), key=None, want_parallel=False
    )
    t0 = jnp.array(0.0)
    state = compute_cuthbert_filter_update(
        dynamics,
        filter_obj,
        None,
        jr.PRNGKey(0),
        y=jnp.array([0.9]),
        u=None,
        t=t0,
        t_prev=t0 - jnp.array(1.0),
    )
    assert jnp.all(jnp.isfinite(state.mean))


@pytest.mark.parametrize(
    ("filter_config", "tol"),
    [
        (PFConfig(n_particles=_n_particles(500)), 3e-1),
        (EnKFConfig(n_particles=_n_particles(500)), 3e-1),
    ],
)
def test_compute_cuthbert_filter_update_pf_enkf_agree_with_kf_mean(filter_config, tol):
    dynamics = _lti_1d()
    _, kf_states = compute_cuthbert_filter(
        dynamics,
        KFConfig(filter_source="cuthbert"),
        jr.PRNGKey(0),
        obs_times=_OBS_TIMES,
        obs_values=_OBS_VALUES,
        ctrl_values=_CTRL_VALUES,
    )
    kf_means = kf_states.mean.ravel()
    means = _step_through_filter_update(dynamics, filter_config, key_seed=1).ravel()
    assert jnp.mean(jnp.abs(means - kf_means)) < tol


@pytest.mark.parametrize(
    ("filter_config", "expected_dist_type"),
    [
        (KFConfig(filter_source="cuthbert"), dist.MultivariateNormal),
        (EKFConfig(), dist.MultivariateNormal),
        (EnKFConfig(n_particles=_n_particles(50)), dist.MultivariateNormal),
        (PFConfig(n_particles=_n_particles(50)), WeightedParticles),
    ],
)
def test_filter_state_dist_matches_family_and_agrees_with_mean(
    filter_config, expected_dist_type
):
    dynamics = _lti_1d()
    filter_obj, _ = build_cuthbert_filter(
        dynamics, filter_config, key=jr.PRNGKey(0), want_parallel=False
    )
    state = compute_cuthbert_filter_update(
        dynamics,
        filter_obj,
        None,
        jr.PRNGKey(0),
        y=_OBS_VALUES[0],
        u=None,
        t=_OBS_TIMES[0],
        t_prev=_OBS_TIMES[0] - 1.0,
    )
    result = filter_state_dist(state, filter_config)
    assert isinstance(result, expected_dist_type)

    if isinstance(result, WeightedParticles):
        weights = jax.nn.softmax(result.log_weights, axis=-1)
        result_mean = jnp.sum(weights[..., None] * result.particles, axis=-2)
    else:
        result_mean = result.mean
    assert jnp.allclose(result_mean, filter_state_mean(state), atol=1e-5)


# ---------------------------------------------------------------------------
# Group 3: DiscreteControlLoopSimulator validation/error paths
# ---------------------------------------------------------------------------


def _simple_policy():
    return _LinearPolicy(K=jnp.array([[0.5]]))


def test_rejects_continuous_time_dynamics_not_wrapped_in_discretizer():
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: u, diffusion=FullDiffusion(0.1 * jnp.eye(1))
        ),
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=0.1 * jnp.eye(1)),
        control_dim=1,
    )
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy):
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    with pytest.raises(ValueError, match="only supports discrete-time models"):
        _run_trace(model)


def test_rejects_n_simulations_greater_than_one():
    dynamics = _lti_1d()
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy, n_simulations=2):
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    with pytest.raises(NotImplementedError, match="n_simulations"):
        _run_trace(model)


def test_rejects_plated_controlled_simulation_explicitly():
    dynamics = _lti_1d()

    def model():
        with dsx.plate("trajectories", 2):
            dsx.sample("f", dynamics, predict_times=jnp.arange(3.0))

    with pytest.raises(NotImplementedError, match="does not yet support dsx.plate"):
        with dsx.Simulator(control_policy=_simple_policy()):
            _run_trace(model)


def test_rejects_wrong_policy_control_shape():
    class _ScalarPolicy:
        def __call__(self, x_hat, t_now, t_next, s):
            return jnp.array(0.0), s

    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        dsx.simulate(
            _lti_1d(),
            rng_key=jr.PRNGKey(0),
            predict_times=jnp.arange(3.0),
            control_policy=_ScalarPolicy(),
        )


def test_closed_loop_previous_transition_transitions_observations_and_times():
    """Closed loop under "previous_transition", checked at every time on a
    model that reveals which control each step used: x_{k+1} = x_k + u_k and
    y = x + 100 u (near-deterministic). The policy emits u_k = k + 1, so every
    control is distinct and known in advance. The grid is non-uniform so a
    time-shifted field cannot line up by accident.

    The observation model refuses u=None, so this also checks the closed loop
    never emits an observation without a control -- i.e. there is no y_0.
    """

    def state_evolution(x, u, t_now, t_next):
        return dist.MultivariateNormal(x + u, 1e-8 * jnp.eye(1))

    def observation_model(x, u, t):
        if u is None:
            raise ValueError("closed-loop observations require a control")
        return dist.MultivariateNormal(x + 100.0 * u, 1e-8 * jnp.eye(1))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_dim=1,
        observation_control_alignment="previous_transition",
    )

    class _CountingPolicy:
        """u_k = k + 1, independent of the belief."""

        def __call__(self, x_hat, t_now, t_next, s):
            return jnp.reshape(s + 1.0, (1,)), s + 1.0

    predict_times = jnp.array([0.0, 1.0, 2.5, 4.0, 7.0])
    result = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=_CountingPolicy(),
        filter_config=EKFConfig(filter_source="cuthbert"),
        initial_policy_state=jnp.asarray(0.0),
    )
    assert result.times is not None and result.states is not None
    assert result.obs_times is not None and result.observations is not None
    assert result.ctrl_times is not None and result.controls is not None
    times = jnp.asarray(result.times)[0]
    obs_times = jnp.asarray(result.obs_times)[0]
    ctrl_times = jnp.asarray(result.ctrl_times)[0]
    states = jnp.asarray(result.states)[0]
    observations = jnp.asarray(result.observations)[0]
    controls = jnp.asarray(result.controls)[0]

    # Grids: states span every time (x_0 included); controls stop at t_{N-1},
    # the last time the policy has a later time to look ahead to; observations
    # start at t_1, since no control precedes y_0.
    assert jnp.array_equal(times, predict_times)
    assert states.shape[0] == len(times)
    assert jnp.array_equal(ctrl_times, times[:-1])
    assert jnp.array_equal(obs_times, times[1:])
    assert jnp.array_equal(controls, jnp.arange(1.0, len(times))[:, None])

    for t_k, t_next in zip(times[:-1], times[1:], strict=True):
        x_k = value_at_time(states, times, t_k)
        x_next = value_at_time(states, times, t_next)
        u_k = value_at_time(controls, ctrl_times, t_k)
        # Every transition t_k -> t_{k+1} is driven by u_k ...
        assert jnp.allclose(x_next, x_k + u_k, atol=1e-3)
        # ... and y_{k+1} is emitted with that same u_k.
        y_next = value_at_time(observations, obs_times, t_next)
        assert jnp.allclose(y_next, x_next + 100.0 * u_k, atol=1e-3)


def test_previous_transition_policy_sees_the_filtered_belief():
    """Under this convention y_k precedes the choice of u_k, so the policy is
    handed the filtered p_hat_k rather than a predicted belief."""

    class _EchoBeliefPolicy:
        def __call__(self, x_hat, t_now, t_next, s):
            del t_now, t_next, s
            mean = filter_state_mean(x_hat)
            return -0.5 * mean, jnp.ravel(mean)[0]

    result = dsx.simulate(
        _lti_1d(),
        rng_key=jr.PRNGKey(0),
        predict_times=jnp.arange(5.0),
        control_policy=_EchoBeliefPolicy(),
        filter_config=KFConfig(
            filter_source="cuthbert", record_filtered_states_mean=True
        ),
        initial_policy_state=jnp.asarray(0.0),
    )
    seen = jnp.ravel(result.policy_states)
    filtered = jnp.ravel(result.filtered_states_mean[0])
    # p_hat_k for k = 0..N-1; the last belief is never acted on.
    assert jnp.allclose(seen, filtered[:-1], atol=1e-5)


def test_online_control_rejects_an_unknown_alignment():
    """DynamicalModel.__init__ rejects unknown values, but eqx.tree_at rewrites
    the field past that validation -- which is exactly how MPPI edits models --
    so simulate's own dispatch still has to catch it."""
    dynamics = _with_alignment(_lti_1d(), "whenever")
    sim = DiscreteControlLoopSimulator(control_policy=_simple_policy())
    with pytest.raises(ValueError, match="not recognized"):
        sim.simulate(dynamics, rng_key=jr.PRNGKey(0), predict_times=jnp.arange(4.0))


def test_explicit_same_time_online_control_is_not_implemented_yet():
    """Closed-loop "same_time" needs separate prediction and analysis steps that
    cuthbert does not expose. An explicit request must fail loudly rather than
    fall back to "previous_transition"."""
    with pytest.raises(NotImplementedError, match="same_time"):
        dsx.simulate(
            _lti_1d(observation_control_alignment="same_time"),
            rng_key=jr.PRNGKey(0),
            predict_times=jnp.arange(4.0),
            control_policy=_simple_policy(),
        )


def test_unspecified_alignment_resolves_to_previous_transition_with_a_warning():
    predict_times = jnp.arange(4.0)
    with pytest.warns(UserWarning, match="unspecified"):
        result = dsx.simulate(
            _lti_1d(observation_control_alignment=None),
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            control_policy=_simple_policy(),
            filter_config=KFConfig(filter_source="cuthbert"),
        )
    assert result.obs_times is not None
    # previous_transition's signature: no y_0, observations start at t_1.
    assert jnp.array_equal(result.obs_times[0], predict_times[1:])


def test_explicit_previous_transition_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dsx.simulate(
            _lti_1d(),
            rng_key=jr.PRNGKey(0),
            predict_times=jnp.arange(4.0),
            control_policy=_simple_policy(),
            filter_config=KFConfig(filter_source="cuthbert"),
        )


def test_discretizer_carries_an_explicit_same_time_through():
    """An explicit choice on a continuous-time model must survive
    discretization. Dropping it would silently turn "same_time" into the
    unspecified default and so into "previous_transition"."""
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([1.0]), jnp.eye(1)),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: u, diffusion=FullDiffusion(0.1 * jnp.eye(1))
        ),
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=0.1 * jnp.eye(1)),
        control_dim=1,
        observation_control_alignment="same_time",
    )
    sim = DiscreteControlLoopSimulator(
        control_policy=_simple_policy(), filter_config=EKFConfig()
    )

    def model():
        with sim:
            with Discretizer(EulerMaruyamaConfig()):
                return dsx.sample("f", dynamics, predict_times=jnp.arange(4.0))

    with pytest.raises(NotImplementedError, match="same_time"):
        _run_trace(model)


def _filter_step_inputs():
    """A real model, filter, belief and key, so the stubs are reached past the
    runtime type checks rather than failing on their arguments."""
    dynamics = _lti_1d()
    filter_obj, _ = build_cuthbert_filter(
        dynamics,
        KFConfig(filter_source="cuthbert"),
        key=jr.PRNGKey(0),
        want_parallel=False,
    )
    state = compute_cuthbert_filter_update(
        dynamics,
        filter_obj=filter_obj,
        prev_state=None,
        key=jr.PRNGKey(1),
        y=jnp.array([0.5]),
        u=jnp.array([0.0]),
        t=jnp.array(0.0),
        t_prev=jnp.array(-1.0),
    )
    return dynamics, filter_obj, state, jr.PRNGKey(2)


def test_belief_prediction_stub_is_not_implemented_yet():
    dynamics, filter_obj, state, key = _filter_step_inputs()
    with pytest.raises(NotImplementedError, match="prediction-only"):
        compute_cuthbert_belief_prediction(
            dynamics,
            filter_obj,
            state,
            key,
            u=jnp.array([0.0]),
            t=jnp.array(1.0),
            t_prev=jnp.array(0.0),
        )


def test_belief_analysis_stub_is_not_implemented_yet():
    dynamics, filter_obj, state, key = _filter_step_inputs()
    with pytest.raises(NotImplementedError, match="analysis-only"):
        compute_cuthbert_belief_analysis(
            dynamics,
            filter_obj,
            state,
            key,
            y=jnp.array([0.5]),
            u=jnp.array([0.0]),
            t=jnp.array(0.0),
        )


# ---------------------------------------------------------------------------
# Group 4: end-to-end shape & output-key tests
# ---------------------------------------------------------------------------

_ALL_FILTER_CONFIGS = [
    KFConfig(filter_source="cuthbert", record_filtered_states_mean=True),
    EKFConfig(record_filtered_states_mean=True),
    EnKFConfig(n_particles=_n_particles(64), record_filtered_states_mean=True),
    PFConfig(n_particles=_n_particles(64), record_filtered_states_mean=True),
]


@pytest.mark.parametrize("filter_config", _ALL_FILTER_CONFIGS)
def test_end_to_end_shapes_and_finiteness(filter_config):
    dynamics = _lti_1d()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy, filter_config=filter_config
    )
    predict_times = jnp.arange(0.0, 8.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr = _run_trace(model)
    assert_trace_sites_exist_and_field_all_finite(
        tr,
        "f_times",
        "f_states",
        "f_observations",
        "f_controls",
        "f_filtered_states_mean",
        where="end-to-end shapes test",
    )
    # "previous_transition" on [t_0..t_N]: states and filtered beliefs span the
    # whole grid (p_hat_0 is the prior); controls cover t_0..t_{N-1} and the
    # observations they produce t_1..t_N, since no y_0 exists.
    T = len(predict_times)
    assert tr["f_states"]["value"].shape == (1, T, 1)
    assert tr["f_observations"]["value"].shape == (1, T - 1, 1)
    assert tr["f_controls"]["value"].shape == (1, T - 1, 1)
    assert tr["f_filtered_states_mean"]["value"].shape == (1, T, 1)
    assert jnp.array_equal(tr["f_ctrl_times"]["value"][0], predict_times[:-1])
    assert jnp.array_equal(tr["f_obs_times"]["value"][0], predict_times[1:])


@pytest.mark.parametrize(
    ("filter_config", "expect_present"),
    [
        (EKFConfig(record_filtered_states_mean=True), True),
        (EKFConfig(record_filtered_states_mean=False), False),
        (EKFConfig(record_max_elems=0), False),  # default heuristic, capped out
        (EKFConfig(), True),  # default heuristic, uncapped
    ],
)
def test_record_filtered_states_mean_gating(filter_config, expect_present):
    """record_filtered_states_mean=True/False explicitly gates the output;
    left at its default (None), it falls back to a size heuristic --
    mirrors Filter's own _should_record_field convention."""
    dynamics = _lti_1d()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy, filter_config=filter_config
    )

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    tr = _run_trace(model)
    assert ("f_filtered_states_mean" in tr) is expect_present


def test_stateless_policy_runs_without_crashing_and_omits_policy_states():
    """Regression test: a stateless policy (no initial_policy_state given,
    s_0=None) previously crashed (jnp.expand_dims(None, axis=0)) when
    assembling the result dict."""
    dynamics = _lti_1d()
    policy = _linear_policy_fn(jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(control_policy=policy)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    tr = _run_trace(model)
    assert "f_policy_states" not in tr


def test_array_policy_state_preserves_shape_and_values():
    """A non-trivial (non-None) policy state threads through the scan with the
    correct shape and evolution rule.
    """
    dynamics = _lti_1d()

    class _CountingPolicy:
        """s is a running step counter; u is irrelevant to this test."""

        def __call__(self, x_hat, t_now, t_next, s):
            return jnp.zeros(1), s + 1.0

    sim = DiscreteControlLoopSimulator(control_policy=_CountingPolicy())
    predict_times = jnp.arange(0.0, 6.0)

    def model():
        with sim:
            return dsx.sample(
                "f",
                dynamics,
                predict_times=predict_times,
                initial_policy_state=jnp.zeros(1),
            )

    tr = _run_trace(model)
    policy_states = tr["f_policy_states"]["value"]
    T = len(predict_times)
    # One policy call per control, so T-1: the policy cannot act at the final
    # time, having no later time to look ahead to.
    assert policy_states.shape == (1, T - 1, 1)
    assert jnp.array_equal(policy_states[0, :, 0], jnp.arange(1, T, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# Group 5: behavioral/control correctness
# ---------------------------------------------------------------------------


def test_closed_loop_stabilizes_vs_uncontrolled_baseline():
    """u = -K x_hat with A - B*K stable drives the state near 0; K=0 (no
    control) does not, for the same marginally-unstable (A=1) system used in
    the tutorial notebook."""
    dynamics = _lti_1d(A=1.0, B=1.0, Q=0.05, R=0.1)
    predict_times = jnp.arange(0.0, 20.0)

    def run(K):
        policy = _LinearPolicy(K=jnp.array([[K]]))
        sim = DiscreteControlLoopSimulator(control_policy=policy)

        def model():
            with sim:
                return dsx.sample("f", dynamics, predict_times=predict_times)

        return _run_trace(model, rng_seed=0)

    tr_controlled = run(K=0.5)
    tr_uncontrolled = run(K=0.0)

    final_controlled = jnp.abs(tr_controlled["f_states"]["value"][0, -1, 0])
    final_uncontrolled = jnp.abs(tr_uncontrolled["f_states"]["value"][0, -1, 0])

    assert final_controlled < 1.0
    assert final_controlled < final_uncontrolled


def test_first_closed_loop_policy_call_uses_initial_belief():
    dynamics = LTI_discrete(
        A=jnp.eye(1),
        Q=0.1 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.1 * jnp.eye(1),
        B=jnp.eye(1),
        initial_mean=jnp.array([2.0]),
        initial_cov=0.5 * jnp.eye(1),
        observation_control_alignment="previous_transition",
    )

    class _BeliefMeanPolicy:
        def __call__(self, x_hat, t_now, t_next, s):
            del t_now, t_next
            return x_hat.mean, s

    result = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=jnp.arange(3.0),
        control_policy=_BeliefMeanPolicy(),
        filter_config=KFConfig(filter_source="cuthbert"),
    )

    assert result.controls is not None
    assert jnp.allclose(result.controls[0, 0], dynamics.initial_condition.mean)


def test_single_timepoint_closed_loop_returns_only_the_initial_state():
    """A one-point grid leaves the policy no later time to look ahead to, so
    the loop body never runs: one state, nothing observed or controlled."""
    result = dsx.simulate(
        _lti_1d(),
        rng_key=jr.PRNGKey(0),
        predict_times=jnp.array([0.0]),
        control_policy=_simple_policy(),
        filter_config=KFConfig(filter_source="cuthbert"),
    )

    assert_finite(result.states, (1, 1, 1), where="states")
    assert_finite(result.observations, (1, 0, 1), where="observations")
    assert_finite(result.controls, (1, 0, 1), where="controls")
    assert_finite(result.ctrl_times, (1, 0), where="ctrl_times")


def test_determinism_same_seed_reproducible_different_seed_differs():
    dynamics = _lti_1d()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(control_policy=policy)
    predict_times = jnp.arange(0.0, 8.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr_a = _run_trace(model, rng_seed=0)
    tr_b = _run_trace(model, rng_seed=0)
    tr_c = _run_trace(model, rng_seed=1)

    assert jnp.array_equal(tr_a["f_states"]["value"], tr_b["f_states"]["value"])
    assert not jnp.array_equal(tr_a["f_states"]["value"], tr_c["f_states"]["value"])


# ---------------------------------------------------------------------------
# Group 6: continuous-time / Discretizer composition
# ---------------------------------------------------------------------------


def test_discretizer_wrapped_sde_runs_end_to_end():
    cte = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: u,
        diffusion=FullDiffusion(0.1 * jnp.eye(1)),
    )
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([5.0]), 0.1 * jnp.eye(1)),
        state_evolution=cte,
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=0.2 * jnp.eye(1)),
        control_dim=1,
        observation_control_alignment="previous_transition",
    )
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy,
        filter_config=EKFConfig(record_filtered_states_mean=True),
    )
    predict_times = jnp.arange(0.0, 10.0)

    def model():
        with sim:
            with Discretizer(EulerMaruyamaConfig()):
                return dsx.sample("f", dynamics, predict_times=predict_times)

    # The continuous-time model states its convention, and the Discretizer
    # carries it into the discrete-time model, so the closed loop runs it
    # without the unspecified-field warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        tr = _run_trace(model)
    assert jnp.array_equal(tr["f_obs_times"]["value"][0], predict_times[1:])
    assert_trace_sites_exist_and_field_all_finite(
        tr,
        "f_states",
        "f_observations",
        "f_controls",
        "f_filtered_states_mean",
        where="discretizer sde test",
    )


def test_discretizer_wrapped_nonlinear_2d_diverges_uncontrolled_stabilizes_controlled():
    state_dim = control_dim = 2
    A = 0.05 * jnp.eye(state_dim)

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.array([3.0, -2.0]), 0.05 * jnp.eye(state_dim)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: A @ (x**2) + u,
            diffusion=FullDiffusion(0.1 * jnp.eye(state_dim)),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(state_dim), R=0.05 * jnp.eye(state_dim)
        ),
        control_dim=control_dim,
    )
    predict_times = jnp.arange(0.0, 6.0, 0.1)

    def run(k):
        policy = _LinearPolicy(K=k * jnp.eye(control_dim))
        sim = DiscreteControlLoopSimulator(
            control_policy=policy, filter_config=EKFConfig()
        )

        def model():
            with sim:
                with Discretizer(EulerMaruyamaConfig()):
                    return dsx.sample("f", dynamics, predict_times=predict_times)

        return _run_trace(model, rng_seed=0)

    # Unspecified field on a continuous-time model: resolved with a warning.
    with pytest.warns(UserWarning, match="unspecified"):
        tr_controlled = run(k=1.0)
    with pytest.warns(UserWarning, match="unspecified"):
        tr_uncontrolled = run(k=0.0)

    assert_trace_sites_exist_and_field_all_finite(
        tr_controlled, "f_states", where="nonlinear 2d controlled"
    )
    assert_trace_sites_exist_and_field_all_finite(
        tr_uncontrolled, "f_states", where="nonlinear 2d uncontrolled"
    )

    final_controlled = jnp.max(jnp.abs(tr_controlled["f_states"]["value"][0, -1]))
    final_uncontrolled = jnp.max(jnp.abs(tr_uncontrolled["f_states"]["value"][0, -1]))

    assert final_controlled < 1.0
    assert final_uncontrolled > 5.0


# ---------------------------------------------------------------------------
# Group 7: black-box transition compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_config",
    [PFConfig(n_particles=_n_particles(64)), EnKFConfig(n_particles=_n_particles(64))],
)
def test_black_box_transition_runs_under_pf_and_enkf(filter_config):
    dynamics = _black_box_dynamics()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy, filter_config=filter_config
    )
    predict_times = jnp.arange(0.0, 5.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr = _run_trace(model)
    assert_trace_sites_exist_and_field_all_finite(
        tr, "f_states", "f_controls", where="black-box PF/EnKF test"
    )


@pytest.mark.parametrize(
    ("filter_config", "expected_exception"),
    [
        (KFConfig(filter_source="cuthbert"), TypeError),
        (EKFConfig(), ValueError),
    ],
)
def test_black_box_transition_rejected_clearly_by_kf_ekf(
    filter_config, expected_exception
):
    dynamics = _black_box_dynamics()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy, filter_config=filter_config
    )
    predict_times = jnp.arange(0.0, 5.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    with pytest.raises(expected_exception):
        _run_trace(model)


# ---------------------------------------------------------------------------
# Group 8: distribution-returning policies are rejected
# ---------------------------------------------------------------------------


class _GaussianExplorationPolicy:
    """A policy that returns a NumPyro Distribution instead of a raw value --
    no longer supported; DiscreteControlLoopSimulator must reject it clearly
    rather than silently sampling it."""

    def __init__(self, K, std):
        self._K = K
        self._std = std

    def __call__(self, x_hat, t_now, t_next, s):
        mean = -self._K @ filter_state_mean(x_hat)
        return dist.Normal(mean, self._std), s


def test_distribution_returning_policy_raises_clear_error():
    dynamics = _lti_1d()
    policy = _GaussianExplorationPolicy(K=jnp.array([[0.5]]), std=0.1)
    sim = DiscreteControlLoopSimulator(control_policy=policy)
    predict_times = jnp.arange(0.0, 6.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    with pytest.raises(ValueError, match="not yet supported"):
        _run_trace(model)


# ---------------------------------------------------------------------------
# Group 9: dsx.simulate(..., control_policy=...) routing
# ---------------------------------------------------------------------------


def test_simulator_handler_with_control_policy_routes_to_control_loop():
    def model():
        with dsx.Simulator(control_policy=_simple_policy()):
            return dsx.sample("f", _lti_1d(), predict_times=jnp.arange(4.0))

    tr = _run_trace(model)
    assert tr["f_controls"]["value"].shape == (1, 3, 1)


def test_dsx_simulate_with_control_policy_routes_to_control_loop():
    dynamics = _lti_1d(A=1.0, B=1.0, Q=0.05, R=0.1)
    predict_times = jnp.arange(0.0, 20.0)

    result_controlled = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=_LinearPolicy(K=jnp.array([[0.5]])),
    )
    result_uncontrolled = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=_LinearPolicy(K=jnp.array([[0.0]])),
    )

    assert isinstance(result_controlled, ControlledSimulatedResult)
    assert isinstance(result_uncontrolled, ControlledSimulatedResult)
    assert result_controlled.controls is not None
    assert result_controlled.states is not None
    assert result_uncontrolled.states is not None
    assert jnp.abs(result_controlled.states[0, -1, 0]) < 1.0
    assert jnp.abs(result_controlled.states[0, -1, 0]) < jnp.abs(
        result_uncontrolled.states[0, -1, 0]
    )


def test_dsx_simulate_with_control_policy_forwards_filter_config():
    dynamics = _lti_1d()
    predict_times = jnp.arange(0.0, 5.0)

    result = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=_LinearPolicy(K=jnp.array([[0.5]])),
        filter_config=EKFConfig(record_filtered_states_mean=True),
    )
    assert isinstance(result, ControlledSimulatedResult)
    assert result.filtered_states_mean is not None
    assert jnp.all(jnp.isfinite(result.filtered_states_mean))


def test_dsx_simulate_with_control_policy_rejects_ctrl_values():
    dynamics = _lti_1d()
    predict_times = jnp.arange(0.0, 5.0)

    # A control grid that is valid for this "previous_transition" model
    # (predict_times[:-1]), so the closed-loop check is what rejects it.
    with pytest.raises(ValueError, match="computes controls online"):
        dsx.simulate(
            dynamics,
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            ctrl_times=predict_times[:-1],
            ctrl_values=jnp.zeros((4, 1)),
            control_policy=_LinearPolicy(K=jnp.array([[0.5]])),
        )


def test_dsx_simulate_with_control_policy_rejects_simulator_config():
    from dynestyx.inference.configs.simulator import SDESimulatorConfig

    dynamics = _lti_1d()
    predict_times = jnp.arange(0.0, 5.0)

    with pytest.raises(
        ValueError, match="SimulatorConfig together with control_policy"
    ):
        dsx.simulate(
            dynamics,
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            control_policy=_LinearPolicy(K=jnp.array([[0.5]])),
            simulator_config=SDESimulatorConfig(),
        )


def test_initial_policy_state_threads_through_dsx_simulate():
    """dsx.simulate(..., initial_policy_state=...) is used directly as s_0 --
    control_policy is never introspected for an initial_state() method, so a
    stateful policy's initial state must always be passed explicitly."""
    dynamics = _lti_1d()

    class _CountingPolicy:
        """s is a running step counter; u is irrelevant to this test."""

        def __call__(self, x_hat, t_now, t_next, s):
            return jnp.zeros(1), s + 1.0

    predict_times = jnp.arange(0.0, 4.0)
    T = len(predict_times)

    result = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=_CountingPolicy(),
        initial_policy_state=jnp.array([10.0]),
    )

    assert isinstance(result, ControlledSimulatedResult)
    assert result.policy_states is not None
    assert jnp.array_equal(
        result.policy_states[0, :, 0],
        jnp.arange(11, T + 10, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Group 10: MPPI takes one-step dynamics directly; owns its own randomness
# ---------------------------------------------------------------------------


def _mppi_loss(result):
    return jnp.sum(result.states**2) + 0.01 * jnp.sum(result.controls**2)


def test_mppi_runs_end_to_end_without_a_key_argument():
    dynamics = _lti_1d(A=1.05, B=1.0)
    mppi = MPPI(
        dynamics=dynamics,
        loss_fn=_mppi_loss,
        horizon=10,
        noise_std=jnp.array(1.0),
        seed=0,
    )
    predict_times = jnp.arange(0.0, 20.0)

    result = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=mppi,
        filter_config=KFConfig(
            filter_source="cuthbert", record_filtered_states_mean=True
        ),
        initial_policy_state=mppi.initial_state(),
    )
    assert isinstance(result, ControlledSimulatedResult)
    assert result.states is not None
    assert jnp.all(jnp.isfinite(result.states))


def test_mppi_rollout_falls_back_to_sample_for_black_box_dynamics():
    """dynamics.state_evolution here (see _black_box_dynamics) exposes only
    .sample()/.shape(), no .mean -- MPPI's internal rollout must fall back
    to sampling (using its own internally-carried key) rather than crashing
    on a missing .mean."""
    dynamics = _black_box_dynamics()
    mppi = MPPI(
        dynamics=dynamics,
        loss_fn=_mppi_loss,
        horizon=5,
        noise_std=jnp.array(1.0),
        seed=0,
    )
    predict_times = jnp.arange(0.0, 5.0)

    result = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        control_policy=mppi,
        filter_config=PFConfig(
            n_particles=_n_particles(64), record_filtered_states_mean=True
        ),
        initial_policy_state=mppi.initial_state(),
    )
    assert isinstance(result, ControlledSimulatedResult)
    assert result.states is not None
    assert jnp.all(jnp.isfinite(result.states))


def test_mppi_initial_state_and_call_depend_only_on_seed():
    """Unit-level check, isolated from the closed loop (where a different
    outer rng_key also changes x_hat via the real observed trajectory, so
    the chosen control legitimately differs downstream for reasons that have
    nothing to do with MPPI's own randomness). `seed` alone determines the
    key baked into `initial_state()`'s output; `__call__` itself takes no
    key at all, so its output is a pure function of (x_hat, s)."""
    dynamics = _lti_1d(A=1.05, B=1.0)

    x_hat = dist.MultivariateNormal(jnp.array([2.0]), jnp.eye(1))

    def make(seed):
        return MPPI(
            dynamics=dynamics,
            loss_fn=_mppi_loss,
            horizon=10,
            noise_std=jnp.array(1.0),
            seed=seed,
        )

    t0, t1 = jnp.array(0.0), jnp.array(1.0)
    mppi_a1, mppi_a2, mppi_b = make(seed=0), make(seed=0), make(seed=1)
    u_a1, _ = mppi_a1(x_hat, t0, t1, mppi_a1.initial_state())
    u_a2, _ = mppi_a2(x_hat, t0, t1, mppi_a2.initial_state())
    u_b, _ = mppi_b(x_hat, t0, t1, mppi_b.initial_state())

    assert jnp.array_equal(u_a1, u_a2)
    assert not jnp.array_equal(u_a1, u_b)


def test_mppi_masks_non_finite_losses_before_softmax():
    """Regression test: even when some candidate rollouts produce a nan
    loss (e.g. from numerical divergence of an unstable system), MPPI's
    output must stay finite. Unmasked, a single nan loss poisons the entire
    softmax weighting (jax.nn.softmax([1, 2, nan, 3]) is all-nan), unlike a
    lone +inf loss, which softmax already handles gracefully on its own."""
    dynamics = _lti_1d(A=1.05, B=1.0)

    def flaky_loss(result):
        # Deterministically nan for roughly half the candidates (whichever
        # have a positive first control), finite for the rest -- exercises
        # the masking without depending on actual numerical divergence.
        base = jnp.sum(result.states**2) + 0.01 * jnp.sum(result.controls**2)
        return jnp.where(result.controls[0, 0, 0] > 0, jnp.nan, base)

    mppi = MPPI(
        dynamics=dynamics,
        loss_fn=flaky_loss,
        horizon=3,
        n_samples=20,
        noise_std=jnp.array(1.0),
    )

    x_hat = dist.MultivariateNormal(jnp.array([2.0]), jnp.eye(1))

    u0, (next_nominal, _) = mppi(
        x_hat, jnp.array(0.0), jnp.array(1.0), mppi.initial_state()
    )
    assert jnp.all(jnp.isfinite(u0))
    assert jnp.all(jnp.isfinite(next_nominal))
