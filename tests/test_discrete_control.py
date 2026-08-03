"""Tests for DiscreteControlLoopSimulator and compute_cuthbert_filter_update."""

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
    filter_state_mean,
)
from dynestyx.control.mppi import MPPI
from dynestyx.discretizers import Discretizer, euler_maruyama
from dynestyx.inference.configs.filter import EKFConfig, EnKFConfig, KFConfig, PFConfig
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    compute_cuthbert_filter,
    compute_cuthbert_filter_update,
)
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.lti_dynamics import LTI_discrete
from dynestyx.models.observations import LinearGaussianObservation
from tests.fixtures import _n_particles
from tests.test_utils import assert_trace_sites_exist_and_field_all_finite

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _lti_1d(A=1.0, B=1.0, Q=0.05, R=0.1):
    """1D linear-Gaussian model matching the tutorial's discrete-time demo."""
    return LTI_discrete(
        A=jnp.array([[A]]),
        Q=Q * jnp.eye(1),
        H=jnp.array([[1.0]]),
        R=R * jnp.eye(1),
        B=jnp.array([[B]]),
    )


class _LinearPolicy(eqx.Module):
    """u = -K x_hat, as an equinox.Module policy."""

    K: jax.Array

    def __call__(self, x_hat, s):
        return -self.K @ filter_state_mean(x_hat), s


def _linear_policy_fn(K):
    """Plain-function equivalent of _LinearPolicy."""

    def policy(x_hat, s):
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


def test_filter_state_mean_unsupported_type_raises():
    class _Neither:
        pass

    with pytest.raises(TypeError, match="Cannot summarize filter state"):
        filter_state_mean(_Neither())


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
    prev_state = None
    means = []
    k = jr.PRNGKey(key_seed)
    for t_idx in range(_T):
        k, sub = jr.split(k)
        u_for_call = None if t_idx == 0 else _CTRL_VALUES[t_idx - 1]
        t_prev = None if t_idx == 0 else _OBS_TIMES[t_idx - 1]
        prev_state = compute_cuthbert_filter_update(
            dynamics,
            filter_config,
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


def test_compute_cuthbert_filter_update_bootstrap_ignores_u():
    """The bootstrap call (prev_state=None) must skip the transition entirely
    (is_first_step=True), regardless of what `u` is passed. This model's
    transition is control-affine (B=1.0), so if the no-op gating broke and a
    phantom transition used `u`, a huge `u` would visibly shift the mean --
    making this a meaningful check, not just a no-op-by-construction one.
    """
    dynamics = _lti_1d()
    filter_config = EKFConfig()
    y0 = jnp.array([0.5])
    t0 = jnp.array(0.0)

    state_no_u = compute_cuthbert_filter_update(
        dynamics, filter_config, None, jr.PRNGKey(0), y=y0, u=None, t=t0
    )
    state_huge_u = compute_cuthbert_filter_update(
        dynamics, filter_config, None, jr.PRNGKey(0), y=y0, u=jnp.array([999.0]), t=t0
    )
    assert jnp.allclose(state_no_u.mean, state_huge_u.mean, atol=1e-6)


def _euler_maruyama_dynamics():
    # euler_maruyama() requires an already-resolved StochasticContinuousTimeStateEvolution
    # (with bm_dim metadata filled in), which only happens inside DynamicalModel.__init__ --
    # matching exactly what Discretizer._sample_ds does internally.
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
        state_evolution=euler_maruyama(continuous_dynamics.state_evolution),
        observation_model=continuous_dynamics.observation_model,
        control_dim=continuous_dynamics.control_dim,
    )


def test_compute_cuthbert_filter_update_default_t_prev_is_degenerate_for_dt_scaled_transition():
    """Documents the actual failure mode of the bug found and fixed this
    session: for a transition whose covariance scales with dt (e.g. an
    Euler-Maruyama-discretized SDE), omitting `t_prev` collapses to dt=0 for
    the first real step, which produces a zero-covariance distribution whose
    NaN log-density leaks through EKF's Taylor-linearization gradient (via
    jnp.where evaluating both branches). This is why
    DiscreteControlLoopSimulator always supplies an explicit, non-degenerate
    t_prev for its bootstrap call (see the next test).
    """
    dynamics = _euler_maruyama_dynamics()
    state = compute_cuthbert_filter_update(
        dynamics,
        EKFConfig(),
        None,
        jr.PRNGKey(0),
        y=jnp.array([0.9]),
        u=None,
        t=jnp.array(0.0),
        # t_prev omitted -> defaults to t (dt=0)
    )
    assert bool(jnp.any(jnp.isnan(state.mean)))


def test_compute_cuthbert_filter_update_explicit_t_prev_avoids_degeneracy():
    dynamics = _euler_maruyama_dynamics()
    t0 = jnp.array(0.0)
    state = compute_cuthbert_filter_update(
        dynamics,
        EKFConfig(),
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


def test_rejects_ctrl_values():
    dynamics = _lti_1d()
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy):
            return dsx.sample(
                "f",
                dynamics,
                predict_times=jnp.arange(0.0, 5.0),
                ctrl_times=jnp.arange(0.0, 5.0),
                ctrl_values=jnp.zeros((5, 1)),
            )

    with pytest.raises(ValueError, match="computes controls online"):
        _run_trace(model)


def test_rejects_obs_values_conditioning():
    dynamics = _lti_1d()
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy):
            return dsx.sample(
                "f",
                dynamics,
                obs_times=jnp.arange(0.0, 5.0),
                obs_values=jnp.zeros((5, 1)),
            )

    with pytest.raises(ValueError, match="generation-only"):
        _run_trace(model)


def test_rejects_n_simulations_greater_than_one():
    dynamics = _lti_1d()
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy, n_simulations=2):
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    with pytest.raises(NotImplementedError, match="n_simulations"):
        _run_trace(model)


def test_requires_obs_times_or_predict_times():
    dynamics = _lti_1d()
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy):
            return dsx.sample("f", dynamics)

    with pytest.raises(ValueError, match="obs_times or predict_times"):
        _run_trace(model)


def test_requires_seeded_context():
    dynamics = _lti_1d()
    policy = _simple_policy()

    def model():
        with DiscreteControlLoopSimulator(control_policy=policy):
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    with pytest.raises(ValueError, match="PRNG key required"):
        trace(model).get_trace()


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
    T = len(predict_times)
    assert tr["f_states"]["value"].shape == (1, T, 1)
    assert tr["f_observations"]["value"].shape == (1, T, 1)
    assert tr["f_controls"]["value"].shape == (1, T - 1, 1)
    assert tr["f_filtered_states_mean"]["value"].shape == (1, T, 1)


@pytest.mark.parametrize(
    ("record_val", "expect_present"),
    [(True, True), (False, False)],
)
def test_record_filtered_states_mean_explicit_gating(record_val, expect_present):
    dynamics = _lti_1d()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy,
        filter_config=EKFConfig(record_filtered_states_mean=record_val),
    )

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    tr = _run_trace(model)
    assert ("f_filtered_states_mean" in tr) is expect_present


def test_record_filtered_states_mean_default_size_heuristic():
    """record_filtered_states_mean=None (default) records only when the total
    element count is within record_max_elems -- mirrors Filter's own
    _should_record_field convention."""
    dynamics = _lti_1d()
    policy = _LinearPolicy(K=jnp.array([[0.5]]))

    small_cap_sim = DiscreteControlLoopSimulator(
        control_policy=policy,
        filter_config=EKFConfig(record_max_elems=0),
    )

    def small_cap_model():
        with small_cap_sim:
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    tr_small_cap = _run_trace(small_cap_model)
    assert "f_filtered_states_mean" not in tr_small_cap

    default_sim = DiscreteControlLoopSimulator(
        control_policy=policy, filter_config=EKFConfig()
    )

    def default_model():
        with default_sim:
            return dsx.sample("f", dynamics, predict_times=jnp.arange(0.0, 5.0))

    tr_default = _run_trace(default_model)
    assert "f_filtered_states_mean" in tr_default


def test_stateless_policy_runs_without_crashing_and_omits_policy_states():
    """Regression test: a stateless policy (no initial_state()) previously
    crashed (jnp.expand_dims(None, axis=0)) when assembling the result dict."""
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

    Note: a genuinely nested-pytree policy state (e.g. a dict of arrays) is
    NOT supported end-to-end today -- BaseSimulator's shared
    `_run_single_member_simulation` (dynestyx/simulators.py) enforces a
    `dict[str, Array] | None` return type at runtime via jaxtyping, so
    `policy_states` itself must stay a flat `Array`, not a nested structure.
    This is a shared constraint across all simulators, not something specific
    to fix here.
    """
    dynamics = _lti_1d()

    class _CountingPolicy:
        """s is a running step counter; u is irrelevant to this test."""

        def initial_state(self):
            return jnp.zeros(1)

        def __call__(self, x_hat, s):
            return jnp.zeros(1), s + 1.0

    sim = DiscreteControlLoopSimulator(control_policy=_CountingPolicy())
    predict_times = jnp.arange(0.0, 6.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr = _run_trace(model)
    policy_states = tr["f_policy_states"]["value"]
    T = len(predict_times)
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


def test_observation_uses_previous_step_control_not_same_index():
    """Regression test for the control-index convention: y_{k+1} must be
    generated using u_k (the control that drove the transition into x_{k+1}),
    never a same-index u_{k+1} -- which is causally impossible online since
    u_{k+1} is chosen from x_hat_{k+1|k+1}, computed from y_{k+1} itself.
    Uses an observation model whose mean depends on u so a same-index leak
    would be directly visible in the recorded observations.
    """
    control_dim = 1

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([0.0]), 1e-6 * jnp.eye(1)),
        state_evolution=LTI_discrete(
            A=jnp.array([[1.0]]),
            Q=1e-6 * jnp.eye(1),
            H=jnp.array([[1.0]]),
            R=1e-6 * jnp.eye(1),
            B=jnp.array([[0.0]]),
        ).state_evolution,
        observation_model=LinearGaussianObservation(
            H=jnp.zeros((1, 1)),  # observation ignores state entirely
            R=1e-6 * jnp.eye(1),
            D=jnp.array([[1.0]]),  # observation is (near-)exactly u
        ),
        control_dim=control_dim,
    )

    class _GrowingPolicy:
        """A distinct, easily-identified control value at every step."""

        def initial_state(self):
            return jnp.array(0.0)

        def __call__(self, x_hat, s):
            return jnp.reshape(s + 1.0, (1,)), s + 1.0

    sim = DiscreteControlLoopSimulator(control_policy=_GrowingPolicy())
    predict_times = jnp.arange(0.0, 6.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr = _run_trace(model)
    controls = tr["f_controls"]["value"][0, :, 0]
    observations = tr["f_observations"]["value"][0, :, 0]

    # observations[0] has no control yet (bootstrap, u=None -> D contribution 0).
    assert jnp.allclose(observations[0], 0.0, atol=1e-2)
    # observations[k+1] should match controls[k] (u_k), not controls[k+1] (u_{k+1}).
    assert jnp.allclose(observations[1:], controls, atol=1e-2)


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


def test_eqx_module_policy_matches_equivalent_plain_function_policy():
    dynamics = _lti_1d()
    K = jnp.array([[0.5]])
    predict_times = jnp.arange(0.0, 8.0)

    def run(policy):
        sim = DiscreteControlLoopSimulator(control_policy=policy)

        def model():
            with sim:
                return dsx.sample("f", dynamics, predict_times=predict_times)

        return _run_trace(model, rng_seed=0)

    tr_module = run(_LinearPolicy(K=K))
    tr_fn = run(_linear_policy_fn(K))

    assert jnp.array_equal(tr_module["f_states"]["value"], tr_fn["f_states"]["value"])
    assert jnp.array_equal(
        tr_module["f_controls"]["value"], tr_fn["f_controls"]["value"]
    )


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
    )
    policy = _LinearPolicy(K=jnp.array([[0.5]]))
    sim = DiscreteControlLoopSimulator(
        control_policy=policy,
        filter_config=EKFConfig(record_filtered_states_mean=True),
    )
    predict_times = jnp.arange(0.0, 10.0)

    def model():
        with sim:
            with Discretizer(discretize=euler_maruyama):
                return dsx.sample("f", dynamics, predict_times=predict_times)

    tr = _run_trace(model)
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
                with Discretizer(discretize=euler_maruyama):
                    return dsx.sample("f", dynamics, predict_times=predict_times)

        return _run_trace(model, rng_seed=0)

    tr_controlled = run(k=1.0)
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
# Group 8: distribution-returning policies
# ---------------------------------------------------------------------------


class _GaussianExplorationPolicy:
    """A policy that returns a NumPyro Distribution instead of a raw value;
    DiscreteControlLoopSimulator must sample from it."""

    def __init__(self, K, std):
        self._K = K
        self._std = std

    def __call__(self, x_hat, s):
        mean = -self._K @ filter_state_mean(x_hat)
        return dist.Normal(mean, self._std), s


def test_distribution_returning_policy_runs_end_to_end():
    dynamics = _lti_1d()
    policy = _GaussianExplorationPolicy(K=jnp.array([[0.5]]), std=0.1)
    sim = DiscreteControlLoopSimulator(control_policy=policy)
    predict_times = jnp.arange(0.0, 6.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr = _run_trace(model)
    assert_trace_sites_exist_and_field_all_finite(
        tr, "f_states", "f_controls", where="distribution-returning policy test"
    )


def test_distribution_returning_policy_actually_samples():
    """Different keys must produce different controls -- otherwise the
    returned Distribution would be silently ignored rather than sampled."""
    dynamics = _lti_1d()
    policy = _GaussianExplorationPolicy(K=jnp.array([[0.5]]), std=1.0)
    sim = DiscreteControlLoopSimulator(control_policy=policy)
    predict_times = jnp.arange(0.0, 6.0)

    def model():
        with sim:
            return dsx.sample("f", dynamics, predict_times=predict_times)

    tr_a = _run_trace(model, rng_seed=0)
    tr_b = _run_trace(model, rng_seed=1)
    assert not jnp.array_equal(tr_a["f_controls"]["value"], tr_b["f_controls"]["value"])


# ---------------------------------------------------------------------------
# Group 9: dsx.simulate(..., control_policy=...) routing
# ---------------------------------------------------------------------------


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

    with pytest.raises(ValueError, match="computes controls online"):
        dsx.simulate(
            dynamics,
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            ctrl_times=predict_times,
            ctrl_values=jnp.zeros((5, 1)),
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


def test_dsx_simulate_without_control_policy_unchanged():
    """No control_policy given -> falls back to today's type-based routing,
    returning a plain SimulatedResult (no controls field at all), not a
    ControlledSimulatedResult."""
    dynamics = _lti_1d()
    predict_times = jnp.arange(0.0, 5.0)

    result = dsx.simulate(dynamics, rng_key=jr.PRNGKey(0), predict_times=predict_times)
    assert not hasattr(result, "controls")


# ---------------------------------------------------------------------------
# Group 10: MPPI takes one-step dynamics directly; owns its own randomness
# ---------------------------------------------------------------------------


def _mppi_loss(x_seq, u_seq):
    return jnp.sum(x_seq**2) + 0.01 * jnp.sum(u_seq**2)


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
        filter_config=KFConfig(record_filtered_states_mean=True),
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
    )
    assert isinstance(result, ControlledSimulatedResult)
    assert result.states is not None
    assert jnp.all(jnp.isfinite(result.states))


def test_mppi_different_seeds_explore_differently_under_the_same_rng_key():
    """MPPI owns its exploration randomness entirely -- two instances with
    different `seed`s must produce different controls even when driven by
    the identical outer `rng_key`."""
    dynamics = _lti_1d(A=1.05, B=1.0)
    predict_times = jnp.arange(0.0, 20.0)

    def run(seed):
        mppi = MPPI(
            dynamics=dynamics,
            loss_fn=_mppi_loss,
            horizon=10,
            noise_std=jnp.array(1.0),
            seed=seed,
        )
        return dsx.simulate(
            dynamics,
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            control_policy=mppi,
        )

    result_a = run(seed=0)
    result_b = run(seed=1)
    assert isinstance(result_a, ControlledSimulatedResult)
    assert isinstance(result_b, ControlledSimulatedResult)
    assert not jnp.allclose(result_a.controls, result_b.controls)


def test_mppi_initial_state_and_call_depend_only_on_seed():
    """Unit-level check, isolated from the closed loop (where a different
    outer rng_key also changes x_hat via the real observed trajectory, so
    the chosen control legitimately differs downstream for reasons that have
    nothing to do with MPPI's own randomness). `seed` alone determines the
    key baked into `initial_state()`'s output; `__call__` itself takes no
    key at all, so its output is a pure function of (x_hat, s)."""
    dynamics = _lti_1d(A=1.05, B=1.0)

    class _FixedBelief:
        mean = jnp.array([2.0])  # a KF-like belief, not a raw array (whose
        # own .mean is a bound method, not a value -- would collide with
        # filter_state_mean's hasattr(state, "mean") duck-typing check)

    x_hat = _FixedBelief()

    def make(seed):
        return MPPI(
            dynamics=dynamics,
            loss_fn=_mppi_loss,
            horizon=10,
            noise_std=jnp.array(1.0),
            seed=seed,
        )

    mppi_a1, mppi_a2, mppi_b = make(seed=0), make(seed=0), make(seed=1)
    u_a1, _ = mppi_a1(x_hat, mppi_a1.initial_state())
    u_a2, _ = mppi_a2(x_hat, mppi_a2.initial_state())
    u_b, _ = mppi_b(x_hat, mppi_b.initial_state())

    assert jnp.array_equal(u_a1, u_a2)
    assert not jnp.array_equal(u_a1, u_b)


def test_mppi_masks_non_finite_losses_before_softmax():
    """Regression test: even when some candidate rollouts produce a nan
    loss (e.g. from numerical divergence of an unstable system), MPPI's
    output must stay finite. Unmasked, a single nan loss poisons the entire
    softmax weighting (jax.nn.softmax([1, 2, nan, 3]) is all-nan), unlike a
    lone +inf loss, which softmax already handles gracefully on its own."""
    dynamics = _lti_1d(A=1.05, B=1.0)

    def flaky_loss(x_seq, u_seq):
        # Deterministically nan for roughly half the candidates (whichever
        # have a positive first control), finite for the rest -- exercises
        # the masking without depending on actual numerical divergence.
        base = jnp.sum(x_seq**2) + 0.01 * jnp.sum(u_seq**2)
        return jnp.where(u_seq[0, 0] > 0, jnp.nan, base)

    mppi = MPPI(
        dynamics=dynamics,
        loss_fn=flaky_loss,
        horizon=3,
        n_samples=20,
        noise_std=jnp.array(1.0),
    )

    class _FixedBelief:
        mean = jnp.array([2.0])

    u0, (next_nominal, _) = mppi(_FixedBelief(), mppi.initial_state())
    assert jnp.all(jnp.isfinite(u0))
    assert jnp.all(jnp.isfinite(next_nominal))
