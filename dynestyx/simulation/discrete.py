"""Discrete-time forward-simulation backend."""

from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import (
    _ensure_trailing_dim,
    _sample_initial_states,
    _sample_observation_path,
    _tile_times,
)
from dynestyx.types import SimulatedResult
from dynestyx.utils import _get_val_or_None, _raise_now_or_error_if


def _align_ctrl_values_to_times(
    *,
    times: Real[Array, " time"],
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None,
) -> Real[Array, "time control_dim"] | Real[Array, " time"] | None:
    """Return control values aligned to the simulator time grid."""
    if ctrl_times is None or ctrl_values is None:
        return ctrl_values

    idx = jnp.searchsorted(ctrl_times, times, side="left")
    max_idx = ctrl_times.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    matched = (idx < ctrl_times.shape[0]) & (ctrl_times[safe_idx] == times)
    safe_idx = _raise_now_or_error_if(
        safe_idx,
        jnp.any(~matched),
        "ctrl_times must contain every discrete simulation time exactly.",
    )
    return ctrl_values[safe_idx]


def _sample_discrete_state_path_from_initial_state(
    dynamics: DynamicalModel,
    *,
    initial_state: Real[Array, " state_dim"] | Real[Array, ""],
    rng_key: PRNGKeyArray,
    times: Real[Array, " time"],
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None,
    include_initial_condition: bool = True,
) -> Real[Array, "state_path_time state_dim"] | Real[Array, " state_path_time"]:
    """Sample one canonical discrete state path from a fixed initial state.

    ctrl_values has its own length ("ctrl_time"), decoupled from `times`: it
    is len(times) for same_time (one entry per transition plus one unused by
    any transition, reserved for the final same_time observation) or
    len(times) - 1 for previous_transition (exactly one entry per
    transition). The returned path's length also depends on
    include_initial_condition, hence the separate "state_path_time" name.

    Returns x_0..x_{T-1} when include_initial_condition is True (same_time
    convention, default). Returns x_1..x_{T-1} only when False
    (previous_transition convention) -- x_0 is the given seed, not re-emitted.
    """
    if len(times) == 1 and include_initial_condition:
        return jnp.expand_dims(initial_state, axis=0)

    state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)

    def _step(carry, t_idx):
        x_prev, key_curr = carry
        key_next, key_transition = jr.split(key_curr)
        transition_dist = state_transition(
            x=x_prev,
            u=_get_val_or_None(ctrl_values, t_idx),
            t_now=times[t_idx],
            t_next=times[t_idx + 1],
        )
        x_t = transition_dist.sample(key_transition)
        return (x_t, key_next), x_t

    (_, _), scan_states = jax.lax.scan(
        _step,
        (initial_state, rng_key),
        jnp.arange(len(times) - 1),
    )
    if include_initial_condition:
        return jnp.concatenate([jnp.expand_dims(initial_state, 0), scan_states], axis=0)
    return scan_states


def _sample_discrete_observation_path(
    dynamics: DynamicalModel,
    *,
    states: Real[Array, "obs_path_time state_dim"] | Real[Array, " obs_path_time"],
    times: Real[Array, " time"],
    ctrl_values: Real[Array, "obs_path_time control_dim"]
    | Real[Array, " obs_path_time"]
    | None,
    rng_key: PRNGKeyArray,
    include_initial_condition: bool = True,
) -> Real[Array, "obs_path_time observation_dim"] | Real[Array, " obs_path_time"]:
    """Sample observations for a discrete state path.

    states/ctrl_values/the return value all share one length ("obs_path_time"),
    decoupled from `times` (always the full predict_times grid, length T):
    that shared length is T for same_time or T-1 for previous_transition.

    include_initial_condition=True (same_time, default): states/times are the
    FULL path (length T, x_0/t_0 included); delegates to the shared
    _sample_observation_path so same_time logic has one source of truth,
    unchanged from today.

    include_initial_condition=False (previous_transition): states is
    x_1..x_{T-1} (length T-1, the output of the state function above with
    include_initial_condition=False); times is still the FULL predict_times
    (length T) so times[k+1] is available. y_{k+1} ~ p(x_{k+1}, ctrl_values[k],
    t_{k+1}), k=0..T-2. y_0 is never sampled -- no wasted draw.
    """
    if include_initial_condition:
        ctrl_eval = (
            (lambda t: ctrl_values[jnp.searchsorted(times, t, side="left")])
            if ctrl_values is not None
            else None
        )
        return _sample_observation_path(
            dynamics,
            states=states,
            times=times,
            rng_key=rng_key,
            control_path_eval=ctrl_eval,
        )

    n = states.shape[0]
    obs_keys = jr.split(rng_key, n)
    future_times = times[1:]

    # Map directly over the pre-sliced arrays (states/ctrl_values/future_times/
    # obs_keys) rather than indexing by a scanned integer inside the mapped
    # body. jax.vmap traces its body once regardless of batch size, so
    # indexing into a genuinely zero-length array (the n=0 edge case, e.g.
    # predict_times of length 1) raises immediately; mapping over already-
    # sliced arrays instead lets vmap's native zero-size handling take over,
    # with no indexing operation in the body at all.
    if ctrl_values is None:

        def _sample_one(x_next, t_next, key):
            obs_dist = dynamics.observation_model(x=x_next, u=None, t=t_next)
            return obs_dist.sample(key)

        return jax.vmap(_sample_one)(states, future_times, obs_keys)

    def _sample_one(x_next, u, t_next, key):
        obs_dist = dynamics.observation_model(x=x_next, u=u, t=t_next)
        return obs_dist.sample(key)

    return jax.vmap(_sample_one)(states, ctrl_values, future_times, obs_keys)


def _sample_discrete_state_path(
    rng_key: PRNGKeyArray,
    *,
    dynamics: DynamicalModel,
    times: Real[Array, " time"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None = None,
) -> Real[Array, "time state_dim"] | Real[Array, " time"]:
    """Sample one state path from the discrete dynamical prior."""
    aligned_ctrl_values = _align_ctrl_values_to_times(
        times=times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    initial_key, transition_key = jr.split(rng_key)
    initial_state = dynamics.initial_condition.sample(initial_key)
    return _sample_discrete_state_path_from_initial_state(
        dynamics,
        initial_state=initial_state,
        rng_key=transition_key,
        times=times,
        ctrl_values=aligned_ctrl_values,
    )


class DiscreteTimeSimulator(BaseSimulator):
    r"""Generate trajectories from a discrete-time dynamical model.

    For prediction times \(t_0,\ldots,t_{T-1}\), this simulator draws
    `n_simulations` independent paths. The observation/control pairing
    depends on `dynamics.observation_control_alignment`:

    For `"same_time"` (default):

    \[
    x_0^{(m)} \sim p_0(x_0), \qquad
    x_{k+1}^{(m)}
      \sim p\!\left(x_{k+1}\mid x_k^{(m)},u_k,t_k,t_{k+1}\right),
    \qquad
    y_k^{(m)} \sim p(y_k\mid x_k^{(m)},u_k,t_k).
    \]

    The first state in the returned path is the initial-condition draw at
    `predict_times[0]`; the simulator then makes one transition draw for each
    adjacent pair of prediction times and samples one observation conditional
    on every realized state, including \(y_0\) (paired with \(u_0\)).

    For `"previous_transition"`:

    \[
    x_0^{(m)} \sim p_0(x_0), \qquad
    x_{k+1}^{(m)} \sim p\!\left(x_{k+1}\mid x_k^{(m)},u_k,t_k,t_{k+1}\right),
    \qquad
    y_{k+1}^{(m)} \sim p(y_{k+1}\mid x_{k+1}^{(m)},u_k,t_{k+1}).
    \]

    Here \(x_0\) is only the seed for the rollout: \(y_0\) is never sampled,
    and neither \(x_0\) nor \(t_0\) appears in the returned result --
    `SimulatedResult.x_0` is `None` and `.times`/`.states`/`.observations` all
    have length \(T-1\), matching the \(T-1\) controls \(u_0,\ldots,u_{T-2}\)
    the caller supplies via `ctrl_values` (aligned to `predict_times[:-1]`,
    not the full `predict_times`). This matches
    [DiscreteControlLoopSimulator][dynestyx.control.discrete_controller_simulators.DiscreteControlLoopSimulator]'s
    closed-loop convention. Only discrete-time models generated through the
    plain `Simulator`/`DiscreteTimeSimulator`/`dsx.simulate` path honor this
    convention today -- see
    [issue #312](https://github.com/BasisResearch/dynestyx/issues/312).

    See
    [DiscreteTimeStateEvolution][dynestyx.models.core.DiscreteTimeStateEvolution]
    for how a discrete transition model is represented in a `DynamicalModel`.

    Use `DiscreteTimeSimulator` as a context manager around a model containing
    `dsx.sample(name, dynamics, predict_times=...)`. The active NumPyro seed
    supplies randomness, while the realized paths are attached to the trace as
    deterministic sites. Use [dsx.simulate][dynestyx.api.simulate] for
    standalone pure-JAX generation without a NumPyro trace.

    Examples:
        >>> def model(predict_times=None):
        ...     dynamics = DynamicalModel(
        ...         initial_condition=initial_dist,
        ...         state_evolution=transition,
        ...         observation_model=observation,
        ...     )
        ...     dsx.sample("f", dynamics, predict_times=predict_times)
        >>> with DiscreteTimeSimulator(n_simulations=3):
        ...     predictive = Predictive(
        ...         model, num_samples=10, exclude_deterministic=False
        ...     )
        ...     draws = predictive(
        ...         jr.PRNGKey(0), predict_times=jnp.arange(20.0)
        ...     )
        >>> draws["f_states"].shape
        (10, 3, 20, state_dim)

        For one direct, pure-JAX model execution:

        >>> result = dsx.simulate(
        ...     dynamics,
        ...     rng_key=jr.PRNGKey(0),
        ...     predict_times=jnp.arange(20.0),
        ...     n_simulations=3,
        ... )

    What this does
    --------------
    The transition distribution is evaluated with the current state, current
    control, and the two adjacent time values. Prediction times therefore need
    not be uniformly spaced, provided the model's transition accepts those
    intervals.

    If controls are supplied, `ctrl_times` must exactly match the grid
    required by `dynamics.observation_control_alignment`: the full
    `predict_times` for `"same_time"` (default) -- `ctrl_values[k]` is used
    for the transition beginning at \(t_k\) and for the observation at
    \(t_k\) -- or `predict_times[:-1]` for `"previous_transition"` --
    `ctrl_values[k]` drives the transition into \(x_{k+1}\) and the
    observation \(y_{k+1}\). The paired control arrays are validated before
    simulation.

    This handler is generation-only and does not condition on `obs_times` or
    `obs_values`. Use
    [LatentPathBuilder][dynestyx.inference.latent.builder.LatentPathBuilder]
    for explicit latent-path inference, or use
    [Filter][dynestyx.inference.filters.Filter] or
    [Smoother][dynestyx.inference.smoothers.Smoother] for marginalized
    inference. Placing this simulator outside a compatible `Filter` or
    `Smoother` draws posterior rollouts at `predict_times`.

    Configuration and defaults
    --------------------------
    Discrete simulation has no solver configuration: transitions are sampled
    directly from `dynamics.state_evolution`. `n_simulations` defaults to one
    and must be at least one. The simulation dimension is retained even when it
    has length one.

    NumPyro trace
    -------------
    For a raw rollout from `dsx.sample("f", ...)`, the following
    `numpyro.deterministic` sites are added:

    - `"f_x_0"`: initial states, shape
      `(*plate_shape, n_simulations, state_dim)`;
    - `"f_times"`: prediction times, shape
      `(*plate_shape, n_simulations, T)`;
    - `"f_states"`: latent states, shape
      `(*plate_shape, n_simulations, T, state_dim)`;
    - `"f_observations"`: sampled observations, shape
      `(*plate_shape, n_simulations, T, observation_dim)`.

    Here `"f"` is replaced by the `name` passed to `dsx.sample`. Under
    `Predictive(..., num_samples=N)`, NumPyro prepends an `N` axis to each
    shape. Because these sites are deterministic, pass
    `exclude_deterministic=False` to `Predictive` (or request the site names
    explicitly) to include them in its returned dictionary.

    When this simulator wraps a `Filter` or `Smoother`, the inner handler
    records its own configured sites and the simulator's aggregate rollout
    sites are instead `"f_predicted_times"`, `"f_predicted_states"`, and
    `"f_predicted_observations"`, with the corresponding time, state, and
    observation shapes above. Each nonempty prediction segment also records
    the state from which that segment starts, with shape
    `(n_simulations, state_dim)`: `"f_0_x_0"` for a segment before the first
    posterior time, and `"f_{j+1}_x_0"` for a segment initialized from the
    posterior at inference-time index `j`. Only segments containing at least
    one requested prediction time are recorded. Inside `dsx.plate`, the segment
    name also identifies the plate member, for example `"f_p0_1_x_0"`.

    If `predict_times` is omitted, no simulator rollout or simulator trace
    sites are produced. Direct calls to
    [DiscreteTimeSimulator().simulate][dynestyx.simulation.discrete.DiscreteTimeSimulator.simulate]
    return `SimulatedResult` without adding NumPyro sites.

    Notes:
        - Use `Simulator` instead when automatic selection among discrete, ODE,
          and SDE backends is desirable.
        - `DiscreteTimeSimulator().simulate(...)` consumes an already allocated
          simulation key. The public [dsx.simulate][dynestyx.api.simulate]
          function splits its root key before dispatch.

    Attributes:
        n_simulations: Number of independent trajectories drawn per model
            execution. Defaults to one and must be greater than or equal to one.
    """

    def __init__(
        self,
        *,
        n_simulations: int = 1,
    ) -> None:
        super().__init__(n_simulations=n_simulations)

    def _simulate_forward_from_initial_state(
        self,
        dynamics: DynamicalModel,
        *,
        initial_state: Real[Array, "n_simulations state_dim"]
        | Real[Array, " n_simulations"],
        rng_key: PRNGKeyArray,
        times: Real[Array, " time"],
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None,
    ) -> SimulatedResult:
        """Run pure forward simulation for a discrete-time model.

        ctrl_values has its own length ("ctrl_time"), decoupled from `times`
        (always the full predict_times grid): len(times) for same_time or
        len(times) - 1 for previous_transition.
        """
        n_sim = initial_state.shape[0]
        sim_keys = jr.split(rng_key, n_sim)
        include_initial_condition = (
            dynamics.observation_control_alignment != "previous_transition"
        )

        def _sim_one_trajectory(
            key: PRNGKeyArray,
            x0: Real[Array, " state_dim"] | Real[Array, ""],
        ):
            key_states, key_obs = jr.split(key)
            states = _sample_discrete_state_path_from_initial_state(
                dynamics,
                initial_state=x0,
                rng_key=key_states,
                times=times,
                ctrl_values=ctrl_values,
                include_initial_condition=include_initial_condition,
            )
            observations = _sample_discrete_observation_path(
                dynamics,
                states=states,
                times=times,
                ctrl_values=ctrl_values,
                rng_key=key_obs,
                include_initial_condition=include_initial_condition,
            )
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(sim_keys, initial_state)
        if include_initial_condition:
            return SimulatedResult(
                times=_tile_times(times, n_sim),
                x_0=initial_state,
                states=_ensure_trailing_dim(states),
                observations=_ensure_trailing_dim(observations),
            )
        return SimulatedResult(
            times=_tile_times(times[1:], n_sim),
            x_0=None,
            states=_ensure_trailing_dim(states),
            observations=_ensure_trailing_dim(observations),
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
        **kwargs,
    ) -> SimulatedResult:
        """Run pure-JAX forward simulation for a discrete-time model.

        Unlike [dsx.simulate][dynestyx.api.simulate], `rng_key` is consumed
        directly as an already-allocated simulation key and is not pre-split.
        Therefore, `dsx.simulate(..., rng_key=root_key)` is equivalent to
        `DiscreteTimeSimulator().simulate(..., rng_key=split_key)`, where
        `split_key = jax.random.split(root_key)[1]`.
        """
        if predict_times is None:
            raise ValueError("predict_times must be provided")

        align_times = (
            predict_times[:-1]
            if dynamics.observation_control_alignment == "previous_transition"
            else predict_times
        )
        aligned_ctrl_values = _align_ctrl_values_to_times(
            times=align_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
        initial_key, rollout_key = jr.split(rng_key)
        initial_state = _sample_initial_states(
            dynamics.initial_condition,
            rng_key=initial_key,
            n_simulations=self.n_simulations,
        )
        return self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=initial_state,
            rng_key=rollout_key,
            times=predict_times,
            ctrl_values=aligned_ctrl_values,
        )
