"""Closed-loop simulation for controlled discrete-time dynamical models."""

import warnings
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Real
from numpyro.distributions import Delta, Distribution

from dynestyx.inference.configs.filter import BaseFilterConfig, PFConfig
from dynestyx.inference.filters import _default_filter_config
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    CuthbertInputs,
    build_cuthbert_filter,
    compute_cuthbert_belief_analysis,
    compute_cuthbert_belief_prediction,
    compute_cuthbert_filter_update,
)
from dynestyx.inference.utils.distribution_utils import (
    _cholesky_state_sequence_to_dists,
)
from dynestyx.models import DynamicalModel, ObservationControlAlignment
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import _ensure_trailing_dim, _tile_times
from dynestyx.types import SimulatedResult
from dynestyx.utils import _should_record_field


def filter_state_mean(state: Any) -> Real[Array, "..."]:
    """Point-estimate summary of a cuthbert filter state, any family.

    Kalman-family states (`KFConfig`, `EKFConfig`, `EnKFConfig`) expose a
    `.mean` property directly. `PFConfig` states (`ParticleFilterState`) have
    no such property -- they represent the belief as a weighted particle
    cloud (`.particles`, `.log_weights`), so the point estimate is the
    weighted mean instead. Broadcasts over any leading batch/time axis, so it
    works on both a single belief and a whole scanned-out sequence of them.
    """
    if hasattr(state, "mean"):
        return state.mean
    if hasattr(state, "particles") and hasattr(state, "log_weights"):
        weights = jax.nn.softmax(state.log_weights, axis=-1)
        return jnp.sum(weights[..., None] * state.particles, axis=-2)
    raise TypeError(f"Cannot summarize filter state of type {type(state).__name__}")


def filter_state_dist(state: Any, filter_config: BaseFilterConfig) -> Distribution:
    """Full-belief NumPyro distribution for a filter state.

    Only `filter_source="cuthbert"` is supported today, matching
    `DiscreteControlLoopSimulator`'s own restriction. Converts the filter state to a NumPyro `Distribution`
    in the same way that `ConditionedResult.dists` does for the recorded filtered states.

    The shared conversion is time-indexed, so the state is given a leading axis of
    length one and the single distribution unwrapped.

    Args:
        state: A single, unbatched filter state produced by `filter_config`.
        filter_config: The config that produced `state`. Selects the backend and
            carries `recorded_filtered_states_cov_jitter` for ensemble states.

    Returns:
        The belief as a NumPyro `Distribution`.

    Raises:
        ValueError: If `filter_config.filter_source` is not `"cuthbert"`.
    """

    if filter_config.filter_source == "cuthbert":
        # Give the time-indexed conversion a leading axis of one
        with_time_axis = SimpleNamespace(
            **{
                name: jnp.asarray(getattr(state, name))[None]
                for name in ("ensemble", "mean", "chol_cov", "particles", "log_weights")
                if hasattr(state, name)
            }
        )
        return _cholesky_state_sequence_to_dists(
            with_time_axis,
            particle_mode=isinstance(filter_config, PFConfig),
            covariance_jitter=getattr(
                filter_config, "recorded_filtered_states_cov_jitter", 0.0
            ),
        )[0]
    else:
        raise ValueError(
            "filter_state_dist currently only supports filter_source='cuthbert' only, got "
            f"{filter_config.filter_source!r}."
        )


def _validate_policy_control(
    u: Array | Distribution, control_dim: int
) -> Real[Array, " control_dim"]:
    """Normalize one control returned by a policy, rejecting bad shapes."""
    if isinstance(u, Distribution):
        raise ValueError(
            "Returning a distribution is not yet supported, instead "
            "sample from this distribution inside your policy."
        )
    expected_control_shape = (control_dim,)
    if u.shape != expected_control_shape:
        raise ValueError(
            "control_policy must return one control vector with shape "
            f"{expected_control_shape}; got {u.shape}."
        )
    return u


@runtime_checkable
class PolicyCallable(Protocol):
    r"""Structural protocol for a control policy $\pi$.

    $$u_k, s_{k+1} = \pi(\tilde x_k, t_k, t_{k+1}, s_k)$$

    $\tilde x_k$ is the loop's current state estimate. Which state that is
    depends on the convention:

    - under `same_time`, $\tilde x_k$ is the predicted state $\hat{x}_{k|k-1}$
    - under `previous_transition`, $\tilde x_k$ is the filtered state $\hat{x}_{k|k}$
    where $\hat{x}_{k|j}$ is the state estimate at time $t_k$ given observations up to time $t_j$.

    At $k=0$, this is always the model's initial-state distribution $x_0$.

    With `use_true_state=True` there is no filter and no estimate: $\tilde x_k$
    is the true state $x_k$, handed over as a `Delta` at that state, so
    `x_hat.mean` is $x_k$ exactly.

    It is a NumPyro `Distribution`, depending on the filter configuration.
    Use `x_hat.mean` for a family-agnostic point estimate.
    `t_now`/`t_next` are the current and next
    time points. Any plain
    callable matching this signature works, including an `equinox.Module`
    with a matching `__call__`  or a plain Python function.
    `control_policy` must return a concrete
    value.

    `s` is an internal state for the policy (such as a PRNG key), which is
    passed back to it on the next call. A stateless policy can ignore it and
    return `None` for the next state.
    """

    def __call__(
        self,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        t_next: Real[Array, ""],
        s: PyTree,
    ) -> tuple[Real[Array, " control_dim"], PyTree]:
        raise NotImplementedError()


class ControlledSimulatedResult(SimulatedResult):
    r"""`SimulatedResult` extended with the control loop's extra outputs.

    Registered as deterministic sites the same generic way as
    `SimulatedResult`'s own fields.

    Both conventions return $N+1$ states and $N$ controls, so the array
    lengths alone do not say which one ran: consult `obs_times`,
    which is $[t_0, \dots, t_{N-1}]$ under `"same_time"` and
    $[t_1, \dots, t_N]$ under `"previous_transition"`.
    """

    # belief per state).
    controls: Real[Array, "n_simulations ctrl_time control_dim"] | None = None
    filtered_states_mean: (
        Real[Array, "n_simulations filtered_time state_dim"] | None
    ) = None
    policy_states: PyTree | None = None


class DiscreteControlLoopSimulator(BaseSimulator):
    r"""Closed-loop simulator: simulate, observe, filter, and decide controls online.

    Unlike `DiscreteTimeSimulator`, which requires the entire control
    trajectory as a pre-supplied `ctrl_values` array, `DiscreteControlLoopSimulator`
    computes each $u_k$ online from the current belief via `control_policy`.

    Which control an observation sees is set by the model's
    `dynamics.observation_control_alignment` -- the same field that governs
    open-loop simulation.

    With convention `"previous_transition"`, the control $u_k$ is chosen
    *after* seeing the observation $y_k$ (which uses the previous control). The
    policy therefore sees the filtered belief $\hat p_k$. This is the only
    convention the filtered closed loop supports for now, and it is what an
    unspecified (`None`) field resolves to, with a warning.

    With convention `"same_time"`, the control $u_k$ would be chosen *before*
    seeing the observation $y_k$ it drives, so the policy would see the
    predicted belief $\tilde p_k$. That needs separate prediction and analysis
    filter steps, which cuthbert does not expose; an explicit `"same_time"`
    raises `NotImplementedError` until they are built.

    Both loops are written out in full on the
    [Closed-loop control page](https://basisresearch.github.io/dynestyx/stable/api_reference/public/control/).

    With `use_true_state=True` the loop skips filtering altogether and hands
    the policy the true state $x_k$. No
    belief is predicted or updated, so both conventions run, in
    `online_control_loop_same_time_no_filter` and
    `online_control_loop_previous_transition_no_filter`.

    The one-step filter update runs on the cuthbert backend
    (`filter_source="cuthbert"`). See the
    [filters page](https://basisresearch.github.io/dynestyx/stable/api_reference/public/inference/filters/)
    for the available filters, and `build_cuthbert_filter` in
    [`discrete_filter.py`](https://github.com/BasisResearch/dynestyx/blob/main/dynestyx/inference/integrations/cuthbert/discrete_filter.py)
    for which of them the online update supports. Plated controlled simulation
    is not yet supported; see
    [Issue #318](https://github.com/BasisResearch/dynestyx/issues/318).

    Attributes:
        control_policy: Control policy $\pi$; see `PolicyCallable`. Its initial
            state $s_0$ is exactly `simulate`'s `initial_policy_state` argument
            (default `None`, for a stateless policy) -- `control_policy` is
            never introspected for an `initial_state()` method; a stateful
            policy's initial state must always be passed explicitly.
        filter_config: Selects the filtering algorithm; any config
            `build_cuthbert_filter` accepts (see above). Defaults to
            `_default_filter_config(dynamics)` when `None`. The online one-step
            update currently requires `filter_source="cuthbert"`. Its
            `record_filtered_states_mean`/`record_max_elems` fields gate
            whether the `filtered_states_mean` output is recorded, exactly
            as they do for `Filter` (see `dynestyx.utils._should_record_field`).
            Must not be given together with `use_true_state=True`, which
            filters nothing.
        use_true_state: Give the policy the true state $x_k$ instead of a
            filtered belief, and run no filter at all. Defaults to `False`.
            Observations are still emitted and returned, but nothing consumes
            them, and `filtered_states_mean` is always `None`.
        n_simulations: Currently only `1` is supported.
    """

    def __init__(
        self,
        *,
        control_policy: PolicyCallable,
        filter_config: BaseFilterConfig | None = None,
        use_true_state: bool = False,
        n_simulations: int = 1,
    ) -> None:
        super().__init__(n_simulations=n_simulations)
        if use_true_state and filter_config is not None:
            raise ValueError(
                "use_true_state=True runs the closed loop on the true state and "
                "builds no filter, so filter_config has no effect; pass one or "
                "the other."
            )
        self.control_policy = control_policy
        self.filter_config = filter_config
        self.use_true_state = use_true_state

    def _validate_plate_support(self) -> None:
        raise NotImplementedError(
            "DiscreteControlLoopSimulator does not yet support dsx.plate. "
            "Run one controlled model at a time."
        )

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
        predict_times: Real[Array, " predict_time"] | None = None,
        initial_policy_state: PyTree | None = None,
        **kwargs: Any,
    ) -> ControlledSimulatedResult:
        """Simulate one online controlled trajectory.

        Args:
            dynamics: Discrete-time dynamical model.
            rng_key: Root key for environment and fallback filter randomness.
            ctrl_times: Unsupported because controls are selected online.
            ctrl_values: Unsupported because controls are selected online.
            predict_times: Strictly increasing simulation times.
            initial_policy_state: Initial state passed to `control_policy`.
            **kwargs: Additional shared simulator-handler metadata, ignored here.

        Returns:
            States, observations, controls, filtered beliefs and policy states
            on all `predict_times`.
            Under the `"same_time"` convention, the states are of length `len(predict_times)`, but the controls and observations are of length `len(predict_times) - 1`.
            Under the `previous_transition` convention, the states are of length `len(predict_times)`, but the controls and observations are of length `len(predict_times) - 1`.

        Raises:
            ValueError: If inputs are incompatible with online discrete control,
                or `dynamics.observation_control_alignment` is not recognized.
            NotImplementedError: If the requested simulation mode is unsupported,
                including an explicit `observation_control_alignment="same_time"`
                while filtering (`use_true_state=False`).

        Warns:
            UserWarning: If `dynamics.observation_control_alignment` is
                unspecified (`None`); `"previous_transition"` is used.
        """

        del kwargs
        if dynamics.continuous_time:
            raise ValueError(
                "DiscreteControlLoopSimulator only supports discrete-time models "
                "(see class docstring). Wrap continuous-time state evolution "
                "in a Discretizer first."
            )
        if ctrl_times is not None or ctrl_values is not None:
            raise ValueError(
                "DiscreteControlLoopSimulator computes controls online via "
                "`control_policy`; do not pass ctrl_times or ctrl_values. Use a plain "
                "Simulator/DiscreteTimeSimulator instead if you want "
                "open-loop control."
            )
        if self.n_simulations != 1:
            raise NotImplementedError(
                "DiscreteControlLoopSimulator does not yet support n_simulations > 1."
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided")
        T = len(times)
        if T < 1:
            raise ValueError("times must contain at least one timepoint")

        alignment = dynamics.observation_control_alignment
        if alignment is None:
            warnings.warn(
                "dynamics.observation_control_alignment is unspecified; closed-loop "
                "control uses 'previous_transition'. Set it explicitly on the model "
                "to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            alignment = ObservationControlAlignment.PREVIOUS_TRANSITION
        if alignment not in (
            ObservationControlAlignment.SAME_TIME,
            ObservationControlAlignment.PREVIOUS_TRANSITION,
        ):
            # DynamicalModel.__init__ already rejects unknown values, but
            # eqx.tree_at rewrites the field without calling it.
            raise ValueError(
                "observation_control_alignment not recognized, has to be one of "
                f"{[member.value for member in ObservationControlAlignment]}; "
                f"got {alignment!r}."
            )

        # Perfect state knowledge: no filter is built, and both conventions
        # run, the policy reading x_k straight off the trajectory.
        if self.use_true_state:
            no_filter_loop = (
                self.online_control_loop_same_time_no_filter
                if alignment == ObservationControlAlignment.SAME_TIME
                else self.online_control_loop_previous_transition_no_filter
            )
            return no_filter_loop(
                dynamics,
                rng_key=rng_key,
                times=times,
                initial_policy_state=initial_policy_state,
            )

        filter_config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )
        if filter_config.filter_source != "cuthbert":
            # TODO: lift this restriction once cd-dynamax filter sources support
            # online one-step updates -- tracked in
            # https://github.com/BasisResearch/dynestyx/pull/314.
            raise ValueError(
                "DiscreteControlLoopSimulator requires filter_source='cuthbert' "
                "because online one-step updates are not available for "
                f"filter_source={filter_config.filter_source!r}."
            )

        if alignment == ObservationControlAlignment.SAME_TIME:
            raise NotImplementedError(
                "Closed-loop control with observation_control_alignment='same_time' "
                "is not implemented yet -- we are working on it. It needs separate "
                "prediction and analysis filter steps, which cuthbert does not "
                "expose. Use observation_control_alignment='previous_transition', "
                "leave it unspecified, or pass use_true_state=True to run without "
                "a filter. Tracked in "
                "https://github.com/BasisResearch/dynestyx/issues/372."
            )
        return self.online_control_loop_previous_transition(
            dynamics,
            rng_key=rng_key,
            times=times,
            filter_config=filter_config,
            initial_policy_state=initial_policy_state,
        )

    def online_control_loop_same_time(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        times: Real[Array, " predict_time"],
        filter_config: BaseFilterConfig,
        initial_policy_state: PyTree | None = None,
    ) -> ControlledSimulatedResult:
        r"""Run the closed loop under the `"same_time"` convention.

        **Not available yet.** `simulate` raises `NotImplementedError` before
        reaching this method, and calling it directly fails at the first step:
        the analysis and prediction steps below are
        `compute_cuthbert_belief_analysis` and
        `compute_cuthbert_belief_prediction`, which are stubs until cuthbert's
        fused `filter_combine` can be split into separate steps. The body is
        kept as the intended algorithm.

        On $\text{Times} = [t_0, \dots, t_N]$:

        $$
        \begin{aligned}
        &x_0 \sim p_0, \quad \tilde{p}_0 = p_0, \quad s_0 \text{ given}
            && \text{Initialization step} \\
        &\text{for } k = 0, \dots, N-1: \\
        &\quad u_k, s_{k+1} = \pi(\tilde{p}_k, t_k, t_{k+1}, s_k)
            && \text{Select the control} \\
        &\quad y_k \sim p(y_k \mid x_k, u_k, t_k)
            && \text{Emit observation} \\
        &\quad \hat{p}_k = \text{FilterAnalysis}(y_k, \tilde{p}_k, u_k)
            && \text{Update the filtering distribution} \\
        &\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
            && \text{State transition} \\
        &\quad \tilde{p}_{k+1} = \text{PredictionUpdate}(\hat{p}_k, u_k)
            && \text{Predict the filtering distribution}
        \end{aligned}
        $$

        where $\tilde p_k \approx p(x_k \mid y_0, \dots, y_{k-1}, u_0, \dots,
        u_{k-1})$ is the predicted distribution and $\hat p_k \approx p(x_k
        \mid y_0, \dots, y_k, u_0, \dots, u_k)$ the filtered one. The control
        $u_k$ both drives the transition into $x_{k+1}$ and generates $y_k$.

        The loop yields $N+1$ states on the full
        grid but only $N$ observations, controls and filtered beliefs, on
        $[t_0, \dots, t_{N-1}]$: the policy needs $t_{k+1}$ to choose $u_k$, so
        it can never act at $t_N$, and $x_N$ ends up with nothing beside it.
        `ctrl_times` records where the controls actually sit.

        `simulate` is the entry point; it validates the arguments and dispatches
        here. Called directly, `times` and `filter_config` are taken as given.

        Args:
            dynamics: Discrete-time dynamical model.
            rng_key: Root key for environment and fallback filter randomness.
            times: Strictly increasing simulation times, at least one. A single
                time yields one state and nothing else, there being no $t_1$
                for the policy to look ahead to.
            filter_config: Resolved filter configuration; must be a
                `filter_source="cuthbert"` one.
            initial_policy_state: Initial state $s_0$ passed to `control_policy`.

        Returns:
            `times` and `states` of length $N+1$; `observations`, `controls`,
            `ctrl_times`, `filtered_states_mean` and `policy_states` of
            length $N$.
        """
        N = len(times) - 1
        rollout_key, initial_state_key, default_filter_key = jr.split(rng_key, 3)
        online_filter_key = (
            filter_config.crn_seed
            if filter_config.crn_seed is not None
            else default_filter_key
        )
        online_filter_key, initial_filter_state_key = jr.split(online_filter_key)
        filter_obj, _ = build_cuthbert_filter(
            dynamics, filter_config, key=online_filter_key, want_parallel=False
        )

        x_0 = dynamics.initial_condition.sample(initial_state_key)
        initial_dtype = jnp.result_type(jnp.asarray(x_0), times)
        zero_control = jnp.zeros((dynamics.control_dim,), dtype=initial_dtype)
        initial_filter_inputs = CuthbertInputs(
            y=jnp.zeros((dynamics.observation_dim,), dtype=initial_dtype),
            u=zero_control,
            u_prev=zero_control,
            time=times[0],
            time_prev=times[0],
            is_first_step=jnp.asarray(False),
        )

        # The first policy acts on the model prior, which under "same_time" is
        # also the first predicted belief: p_tilde_0 = p_0.
        p_tilde_0 = filter_obj.init_prepare(
            initial_filter_inputs, key=initial_filter_state_key
        )
        s_0 = initial_policy_state

        # --- k = 0 .. N-1, one per transition ---------------------------------
        def _step(carry, t_idx):
            x_k, p_tilde_k, s_k, rollout_key, online_filter_key = carry
            rollout_key, observation_key, transition_key = jr.split(rollout_key, 3)
            online_filter_key, filter_update_key, predict_key = jr.split(
                online_filter_key, 3
            )
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            # u_k = pi(p_tilde_k)
            # The policy takes a distribution, not a raw filter state.
            u_k, s_next = self.control_policy(
                filter_state_dist(p_tilde_k, filter_config), t_now, t_next, s_k
            )
            u_k = _validate_policy_control(u_k, dynamics.control_dim)

            # y_k ~ p(y_k | x_k, u_k)
            y_k = dynamics.observation_model(x_k, u_k, t_now).sample(observation_key)

            # p_hat_k = FilterAnalysis(y_k, p_tilde_k, u_k). Stub for now.
            p_hat_k = compute_cuthbert_belief_analysis(
                dynamics,
                filter_obj=filter_obj,
                prev_state=p_tilde_k,
                key=filter_update_key,
                y=y_k,
                u=u_k,
                t=t_now,
            )

            # x_{k+1} ~ p(x_{k+1} | x_k, u_k)
            x_next = dynamics.state_evolution(x_k, u_k, t_now, t_next).sample(
                transition_key
            )

            # p_tilde_{k+1} = PredictionUpdate(p_hat_k, u_k). Stub for now.
            p_tilde_next = compute_cuthbert_belief_prediction(
                dynamics,
                filter_obj=filter_obj,
                prev_state=p_hat_k,
                key=predict_key,
                u=u_k,
                t=t_next,
                t_prev=t_now,
            )

            new_carry = (
                x_next,
                p_tilde_next,
                s_next,
                rollout_key,
                online_filter_key,
            )
            return new_carry, (x_k, p_hat_k, y_k, s_next, u_k)

        init_carry = (x_0, p_tilde_0, s_0, rollout_key, online_filter_key)
        (x_final, _p_tilde_final, _s_final, _, _), (xs, p_hats, ys, ss, us) = (
            jax.lax.scan(_step, init_carry, jnp.arange(N))
        )

        # The loop runs k = 0..N-1 on a grid [t_0, ..., t_N], so it produces
        # N+1 states but only N of everything else.
        states = jnp.concatenate([xs, jnp.expand_dims(x_final, 0)], axis=0)
        observations = ys
        # y_k is emitted at t_k, so observations and controls share the grid
        obs_times = times[:-1]
        ctrl_times = times[:-1]

        # One filtered belief per observation, so N of them rather than one
        # per state: there is no y_N to condition x_N on.
        mean_shape = filter_state_mean(p_hats).shape[1:]
        record_mean = _should_record_field(
            filter_config.record_filtered_states_mean,
            (N, *mean_shape),
            filter_config.record_max_elems,
        )
        filtered_states_mean = None
        if record_mean:
            filtered_states_mean = _ensure_trailing_dim(
                jnp.expand_dims(filter_state_mean(p_hats), axis=0)
            )

        policy_states = None
        if s_0 is not None:
            # A stateless policy (no initial_policy_state given) has nothing
            # to record; jnp.expand_dims can't be applied to None directly,
            # and there is no meaningful "policy_states" trajectory to report.
            policy_states = jax.tree_util.tree_map(
                lambda leaf: jnp.expand_dims(leaf, axis=0), ss
            )

        return ControlledSimulatedResult(
            times=_tile_times(times, 1),
            x_0=jnp.expand_dims(x_0, axis=0),
            states=_ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            observations=_ensure_trailing_dim(jnp.expand_dims(observations, axis=0)),
            obs_times=_tile_times(obs_times, 1),
            controls=_ensure_trailing_dim(jnp.expand_dims(us, axis=0)),
            ctrl_times=_tile_times(ctrl_times, 1),
            filtered_states_mean=filtered_states_mean,
            policy_states=policy_states,
        )

    def online_control_loop_previous_transition(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        times: Real[Array, " predict_time"],
        filter_config: BaseFilterConfig,
        initial_policy_state: PyTree | None = None,
    ) -> ControlledSimulatedResult:
        r"""Run the closed loop under the `"previous_transition"` convention.

        On $\text{Times} = [t_0, \dots, t_N]$:

        $$
        \begin{aligned}
        &x_0 \sim p_0, \quad \hat{p}_0 = p_0, \quad s_0 \text{ given}
            && \text{Initialization step} \\
        &\text{for } k = 0, \dots, N-1: \\
        &\quad u_k, s_{k+1} = \pi(\hat{p}_k, t_k, t_{k+1}, s_k)
            && \text{Select the control} \\
        &\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
            && \text{State transition} \\
        &\quad y_{k+1} \sim p(y_{k+1} \mid x_{k+1}, u_k, t_{k+1})
            && \text{Emit observation} \\
        &\quad \hat{p}_{k+1} = \text{FilterUpdate}(y_{k+1}, \hat{p}_k, u_k)
            && \text{Update the filtering distribution}
        \end{aligned}
        $$

        where $\hat p_k \approx p(x_k \mid y_1, \dots, y_k, u_0, \dots,
        u_{k-1})$ is the filtered distribution. The control $u_k$ drives the
        transition into $x_{k+1}$ *and* generates $y_{k+1}$, so $y_0$ never
        exists -- there is no $u_{-1}$ to have produced it.

        The policy acts on the filtered belief, having already seen $y_k$. So,
        unlike `"same_time"`, no predicted belief is ever exposed and the
        `FilterUpdate` above stays fused: one filter call per step rather than
        two.

        Shapes differ from `"same_time"` in exactly one place. Both return
        $N+1$ states on $[t_0, \dots, t_N]$ and $N$ controls on
        $[t_0, \dots, t_{N-1}]$; here the $N$ observations sit on
        $[t_1, \dots, t_N]$, dropping the *first* time rather than the last.
        `obs_times` is what distinguishes the two results.

        Args:
            dynamics: Discrete-time dynamical model.
            rng_key: Root key for environment and fallback filter randomness.
            times: Strictly increasing simulation times, at least one. A single
                time yields one state and nothing else.
            filter_config: Resolved filter configuration; must be a
                `filter_source="cuthbert"` one.
            initial_policy_state: Initial state $s_0$ passed to `control_policy`.

        Returns:
            `times`, `states` and `filtered_states_mean` of length $N+1$;
            `observations`, `obs_times`, `controls`, `ctrl_times`
            and `policy_states` of length $N$.
        """
        N = len(times) - 1
        rollout_key, initial_state_key, default_filter_key = jr.split(rng_key, 3)
        online_filter_key = (
            filter_config.crn_seed
            if filter_config.crn_seed is not None
            else default_filter_key
        )
        online_filter_key, initial_filter_state_key = jr.split(online_filter_key)
        filter_obj, _ = build_cuthbert_filter(
            dynamics, filter_config, key=online_filter_key, want_parallel=False
        )

        x_0 = dynamics.initial_condition.sample(initial_state_key)
        initial_dtype = jnp.result_type(jnp.asarray(x_0), times)
        zero_control = jnp.zeros((dynamics.control_dim,), dtype=initial_dtype)
        initial_filter_inputs = CuthbertInputs(
            y=jnp.zeros((dynamics.observation_dim,), dtype=initial_dtype),
            u=zero_control,
            u_prev=zero_control,
            time=times[0],
            time_prev=times[0],
            is_first_step=jnp.asarray(False),
        )
        # The first policy acts on the model prior.
        p_hat_0 = filter_obj.init_prepare(
            initial_filter_inputs, key=initial_filter_state_key
        )
        s_0 = initial_policy_state

        def _step(carry, t_idx):
            x_k, p_hat_k, s_k, rollout_key, online_filter_key = carry
            rollout_key, transition_key, observation_key = jr.split(rollout_key, 3)
            online_filter_key, filter_update_key = jr.split(online_filter_key)
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            # u_k = pi(p_hat_k)
            u_k, s_next = self.control_policy(
                filter_state_dist(p_hat_k, filter_config), t_now, t_next, s_k
            )
            u_k = _validate_policy_control(u_k, dynamics.control_dim)

            # x_{k+1} ~ p(x_{k+1} | x_k, u_k)
            x_next = dynamics.state_evolution(x_k, u_k, t_now, t_next).sample(
                transition_key
            )

            # y_{k+1} ~ p(y_{k+1} | x_{k+1}, u_k)
            y_next = dynamics.observation_model(x_next, u_k, t_next).sample(
                observation_key
            )

            # This filtered update: uses the observation.
            p_hat_next = compute_cuthbert_filter_update(
                dynamics,
                filter_obj=filter_obj,
                prev_state=p_hat_k,
                key=filter_update_key,
                y=y_next,
                u=u_k,
                t=t_next,
                t_prev=t_now,
            )

            new_carry = (x_next, p_hat_next, s_next, rollout_key, online_filter_key)
            return new_carry, (x_next, p_hat_next, y_next, s_next, u_k)

        init_carry = (x_0, p_hat_0, s_0, rollout_key, online_filter_key)
        _, (xs, p_hats, ys, ss, us) = jax.lax.scan(_step, init_carry, jnp.arange(N))

        # x_0 and p_hat_0 precede the loop; every observation follows a
        # transition, so they start at t_1 while controls start at t_0.
        states = jnp.concatenate([jnp.expand_dims(x_0, 0), xs], axis=0)
        p_hats = jax.tree_util.tree_map(
            lambda head, tail: jnp.concatenate(
                [jnp.expand_dims(head, 0), tail], axis=0
            ),
            p_hat_0,
            p_hats,
        )
        obs_times = times[1:]
        ctrl_times = times[:-1]

        mean_shape = filter_state_mean(p_hat_0).shape
        record_mean = _should_record_field(
            filter_config.record_filtered_states_mean,
            (N + 1, *mean_shape),
            filter_config.record_max_elems,
        )
        filtered_states_mean = None
        if record_mean:
            filtered_states_mean = _ensure_trailing_dim(
                jnp.expand_dims(filter_state_mean(p_hats), axis=0)
            )

        policy_states = None
        if s_0 is not None:
            policy_states = jax.tree_util.tree_map(
                lambda leaf: jnp.expand_dims(leaf, axis=0), ss
            )

        return ControlledSimulatedResult(
            times=_tile_times(times, 1),
            x_0=jnp.expand_dims(x_0, axis=0),
            states=_ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            observations=_ensure_trailing_dim(jnp.expand_dims(ys, axis=0)),
            obs_times=_tile_times(obs_times, 1),
            controls=_ensure_trailing_dim(jnp.expand_dims(us, axis=0)),
            ctrl_times=_tile_times(ctrl_times, 1),
            filtered_states_mean=filtered_states_mean,
            policy_states=policy_states,
        )

    def online_control_loop_same_time_no_filter(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        times: Real[Array, " predict_time"],
        initial_policy_state: PyTree | None = None,
    ) -> ControlledSimulatedResult:
        r"""Run the closed loop under `"same_time"` on the true state.

        No filter runs: the policy is handed $x_k$ itself, as a `Delta`.

        On $\text{Times} = [t_0, \dots, t_N]$:

        $$
        \begin{aligned}
        &x_0 \sim p_0, \quad s_0 \text{ given}
            && \text{Initialization step} \\
        &\text{for } k = 0, \dots, N-1: \\
        &\quad u_k, s_{k+1} = \pi(x_k, t_k, t_{k+1}, s_k)
            && \text{Select the control} \\
        &\quad y_k \sim p(y_k \mid x_k, u_k, t_k)
            && \text{Emit observation} \\
        &\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
            && \text{State transition}
        \end{aligned}
        $$

        $u_k$ both generates $y_k$ and drives the transition into $x_{k+1}$.
        The loop yields $N+1$ states on $[t_0, \dots, t_N]$, and $N$ controls
        and observations on $[t_0, \dots, t_{N-1}]$: the policy needs
        $t_{k+1}$ to choose $u_k$, so it never acts at $t_N$, and without $u_N$
        there is no $y_N$.

        Args:
            dynamics: Discrete-time dynamical model.
            rng_key: Root key for the environment's randomness.
            times: Strictly increasing simulation times, at least one. A single
                time yields one state and nothing else.
            initial_policy_state: Initial state $s_0$ passed to `control_policy`.

        Returns:
            `times` and `states` of length $N+1$; `observations`, `obs_times`,
            `controls`, `ctrl_times` and `policy_states` of length $N$.
            `filtered_states_mean` is always `None`.
        """
        N = len(times) - 1
        # Split three ways and drop the filter key the filtered loops take
        # here, so a no-filter run draws the same initial state and the same
        # transition/observation noise as a filtered run from the same seed.
        rollout_key, initial_state_key, _unused_filter_key = jr.split(rng_key, 3)

        x_0 = dynamics.initial_condition.sample(initial_state_key)
        s_0 = initial_policy_state

        def _step(carry, t_idx):
            x_k, s_k, rollout_key = carry
            rollout_key, observation_key, transition_key = jr.split(rollout_key, 3)
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            # u_k = pi(x_k). The policy takes a distribution, so the known
            # state goes in as a Delta: a belief with no uncertainty in it.
            u_k, s_next = self.control_policy(
                Delta(x_k, event_dim=jnp.ndim(x_k)), t_now, t_next, s_k
            )
            u_k = _validate_policy_control(u_k, dynamics.control_dim)

            # y_k ~ p(y_k | x_k, u_k)
            y_k = dynamics.observation_model(x_k, u_k, t_now).sample(observation_key)

            # x_{k+1} ~ p(x_{k+1} | x_k, u_k)
            x_next = dynamics.state_evolution(x_k, u_k, t_now, t_next).sample(
                transition_key
            )

            return (x_next, s_next, rollout_key), (x_k, y_k, s_next, u_k)

        init_carry = (x_0, s_0, rollout_key)
        (x_final, _s_final, _), (xs, ys, ss, us) = jax.lax.scan(
            _step, init_carry, jnp.arange(N)
        )

        # The scan emits x_k, so the final state has to be appended.
        states = jnp.concatenate([xs, jnp.expand_dims(x_final, 0)], axis=0)
        # y_k is emitted at t_k, so observations and controls share the grid.
        obs_times = times[:-1]
        ctrl_times = times[:-1]

        policy_states = None
        if s_0 is not None:
            policy_states = jax.tree_util.tree_map(
                lambda leaf: jnp.expand_dims(leaf, axis=0), ss
            )

        return ControlledSimulatedResult(
            times=_tile_times(times, 1),
            x_0=jnp.expand_dims(x_0, axis=0),
            states=_ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            observations=_ensure_trailing_dim(jnp.expand_dims(ys, axis=0)),
            obs_times=_tile_times(obs_times, 1),
            controls=_ensure_trailing_dim(jnp.expand_dims(us, axis=0)),
            ctrl_times=_tile_times(ctrl_times, 1),
            filtered_states_mean=None,
            policy_states=policy_states,
        )

    def online_control_loop_previous_transition_no_filter(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        times: Real[Array, " predict_time"],
        initial_policy_state: PyTree | None = None,
    ) -> ControlledSimulatedResult:
        r"""Run the closed loop under `"previous_transition"` on the true state.

        No filter runs: the policy is handed $x_k$ itself, as a `Delta`.

        On $\text{Times} = [t_0, \dots, t_N]$:

        $$
        \begin{aligned}
        &x_0 \sim p_0, \quad s_0 \text{ given}
            && \text{Initialization step} \\
        &\text{for } k = 0, \dots, N-1: \\
        &\quad u_k, s_{k+1} = \pi(x_k, t_k, t_{k+1}, s_k)
            && \text{Select the control} \\
        &\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
            && \text{State transition} \\
        &\quad y_{k+1} \sim p(y_{k+1} \mid x_{k+1}, u_k, t_{k+1})
            && \text{Emit observation}
        \end{aligned}
        $$

        $u_k$ both drives the transition into $x_{k+1}$ and generates
        $y_{k+1}$, so there is no $y_0$. The loop yields $N+1$ states on
        $[t_0, \dots, t_N]$, $N$ controls on $[t_0, \dots, t_{N-1}]$ and $N$
        observations on $[t_1, \dots, t_N]$.

        Args:
            dynamics: Discrete-time dynamical model.
            rng_key: Root key for the environment's randomness.
            times: Strictly increasing simulation times, at least one. A single
                time yields one state and nothing else.
            initial_policy_state: Initial state $s_0$ passed to `control_policy`.

        Returns:
            `times` and `states` of length $N+1$; `observations`, `obs_times`,
            `controls`, `ctrl_times` and `policy_states` of length $N$.
            `filtered_states_mean` is always `None`.
        """
        N = len(times) - 1

        # filter_key is unused here, but we still draw it to ensure randomness is consistent
        rollout_key, initial_state_key, _unused_filter_key = jr.split(rng_key, 3)

        x_0 = dynamics.initial_condition.sample(initial_state_key)
        s_0 = initial_policy_state

        def _step(carry, t_idx):
            x_k, s_k, rollout_key = carry
            rollout_key, transition_key, observation_key = jr.split(rollout_key, 3)
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            # u_k = pi(x_k). The policy takes a distribution, so the known
            # state goes in as a Delta.
            u_k, s_next = self.control_policy(
                Delta(x_k, event_dim=jnp.ndim(x_k)), t_now, t_next, s_k
            )
            u_k = _validate_policy_control(u_k, dynamics.control_dim)

            # x_{k+1} ~ p(x_{k+1} | x_k, u_k)
            x_next = dynamics.state_evolution(x_k, u_k, t_now, t_next).sample(
                transition_key
            )

            # y_{k+1} ~ p(y_{k+1} | x_{k+1}, u_k)
            y_next = dynamics.observation_model(x_next, u_k, t_next).sample(
                observation_key
            )

            return (x_next, s_next, rollout_key), (x_next, y_next, s_next, u_k)

        init_carry = (x_0, s_0, rollout_key)
        _, (xs, ys, ss, us) = jax.lax.scan(_step, init_carry, jnp.arange(N))

        # x_0 precedes the loop; every observation follows a transition, so
        # they start at t_1 while controls start at t_0.
        states = jnp.concatenate([jnp.expand_dims(x_0, 0), xs], axis=0)
        obs_times = times[1:]
        ctrl_times = times[:-1]

        policy_states = None
        if s_0 is not None:
            policy_states = jax.tree_util.tree_map(
                lambda leaf: jnp.expand_dims(leaf, axis=0), ss
            )

        return ControlledSimulatedResult(
            times=_tile_times(times, 1),
            x_0=jnp.expand_dims(x_0, axis=0),
            states=_ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            observations=_ensure_trailing_dim(jnp.expand_dims(ys, axis=0)),
            obs_times=_tile_times(obs_times, 1),
            controls=_ensure_trailing_dim(jnp.expand_dims(us, axis=0)),
            ctrl_times=_tile_times(ctrl_times, 1),
            filtered_states_mean=None,
            policy_states=policy_states,
        )


__all__ = [
    "ControlledSimulatedResult",
    "DiscreteControlLoopSimulator",
    "PolicyCallable",
    "filter_state_dist",
    "filter_state_mean",
]
