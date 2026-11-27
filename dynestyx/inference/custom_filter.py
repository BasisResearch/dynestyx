"""Step contracts and observation-aligned execution for user-defined filters."""

from collections.abc import Callable
from typing import cast, overload

import jax
import jax.numpy as jnp
from cuthbert.inference import Filter as CuthbertFilter
from cuthbert.utils import dummy_leading_element
from jaxtyping import Array, PRNGKeyArray, Real, Shaped
from numpyro.distributions import Distribution

from dynestyx.configs.filters import (
    BaseFilterConfig,
    CombineFilterConfig,
    SplitFilterConfig,
)
from dynestyx.inference.filter_types import (
    CDPosterior,
    CombinedSteps,
    CuthbertState,
    FilterHistory,
    SplitKeys,
    SplitSteps,
)
from dynestyx.inference.filter_types import FilterInputs as FilterInputs
from dynestyx.inference.filter_types import FilterOutputs as FilterOutputs
from dynestyx.inference.filter_types import (
    PredictedObservationOutputs as PredictedObservationOutputs,
)
from dynestyx.inference.filter_types import SplitFilter as SplitFilter
from dynestyx.inference.utils.distribution_utils import (
    _cholesky_state_sequence_to_dists,
    _gaussian_sequence_to_dists,
    _particle_sequence_to_dists,
    _posterior_sequence_to_dists,
)
from dynestyx.models import DynamicalModel


def step_inputs(
    dynamics: DynamicalModel,
    times: Real[Array, " time"],
    ys: Shaped[Array, "time observation_dim"],
    ctrl_times: Real[Array, " control_time"] | None,
    controls: Real[Array, "control_time control_dim"] | None,
) -> FilterInputs:
    length = len(times)
    if not length:
        raise ValueError("Filtering requires at least one observation.")
    if controls is None:
        controls = jnp.zeros((length, dynamics.control_dim), dtype=ys.dtype)
    elif len(controls) != length:
        if ctrl_times is None:
            raise ValueError(
                "ctrl_times is required to align controls to observations."
            )
        controls = controls[jnp.searchsorted(ctrl_times, times)]
    return FilterInputs(
        ys,
        controls,
        jnp.concatenate((controls[:1], controls[:-1])),
        times,
        jnp.concatenate((times[:1], times[:-1])),
        jnp.arange(length) == 0,
    )


def _split(key: PRNGKeyArray | None) -> tuple[PRNGKeyArray | None, PRNGKeyArray | None]:
    if key is None:
        return None, None
    left, right = jax.random.split(key)
    return left, right


def _prepend[T](first: T, tail: T) -> T:
    return jax.tree.map(
        lambda x, xs: jnp.concatenate((jnp.asarray(x)[None], xs)), first, tail
    )


@overload
def run_filter_steps[P, S, O](
    algorithm: SplitSteps[P, S],
    inputs: FilterInputs,
    key: PRNGKeyArray | None,
    gather: Callable[[P, S], O],
    *,
    parallel: bool = False,
    include_initial: bool = False,
    split_keys: SplitKeys | None = None,
) -> O: ...


@overload
def run_filter_steps[E, S, O](
    algorithm: CombinedSteps[E, S],
    inputs: FilterInputs,
    key: PRNGKeyArray | None,
    gather: Callable[[S], O],
    *,
    parallel: bool = False,
    include_initial: bool = False,
    split_keys: None = None,
) -> O: ...


def run_filter_steps[P, S, E, O](
    algorithm: SplitSteps[P, S] | CombinedSteps[E, S],
    inputs: FilterInputs,
    key: PRNGKeyArray | None,
    gather: Callable[[P, S], O] | Callable[[S], O],
    *,
    parallel: bool = False,
    include_initial: bool = False,
    split_keys: SplitKeys | None = None,
) -> O:
    """Collect projections in a sequential scan or from associative prefix states."""
    init_key, scan_key = _split(key)
    keys = None if scan_key is None else jax.random.split(scan_key, len(inputs.y))
    if hasattr(algorithm, "filter_combine"):
        fused = cast(CombinedSteps[E, S], algorithm)
        project = cast(Callable[[S], O], gather)
        initial = fused.init_prepare(key=init_key)
        if (
            isinstance(algorithm, CuthbertFilter)
            and hasattr(initial, "_replace")
            and getattr(initial, "model_inputs", False) is None
        ):
            initial = getattr(initial, "_replace")(
                model_inputs=dummy_leading_element(inputs)
            )
        if parallel:
            if not getattr(algorithm, "associative", False):
                raise ValueError("Parallel filtering requires an associative combine.")
            prepared = jax.vmap(
                lambda x, k: fused.filter_prepare(x, key=k),
                in_axes=(0, None if keys is None else 0),
            )(inputs, keys)
            prefixes = jax.lax.associative_scan(
                jax.vmap(fused.filter_combine), _prepend(initial, prepared)
            )
            if not include_initial:
                prefixes = jax.tree.map(lambda xs: xs[1:], prefixes)
            return jax.vmap(project)(prefixes)

        def combine(
            state: S, args: tuple[FilterInputs, PRNGKeyArray | None]
        ) -> tuple[S, O]:
            x, k = args
            updated = fused.filter_combine(state, fused.filter_prepare(x, key=k))
            return updated, project(updated)

        _, history = jax.lax.scan(combine, initial, (inputs, keys))
        return _prepend(project(initial), history) if include_initial else history

    if parallel:
        raise ValueError("Parallel filtering requires an associative combine.")
    if not all(
        callable(getattr(algorithm, name, None))
        for name in ("initialize", "predict", "update")
    ):
        raise TypeError("Expected a split filter or a prepare/combine filter.")
    split = cast(SplitSteps[P, S], algorithm)
    project_split = cast(Callable[[P, S], O], gather)
    if split_keys is not None:
        if key is None:
            raise ValueError("An explicit split key schedule requires a PRNG key.")
        init_key, predict_keys, update_keys = split_keys(key, len(inputs.y))
    elif keys is None:
        predict_keys = update_keys = None
    else:
        pairs = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        predict_keys, update_keys = pairs[:, 0], pairs[:, 1]
    return run_split_steps(
        split, inputs, project_split, init_key, predict_keys, update_keys
    )


def run_split_steps[P, S, O](
    algorithm: SplitSteps[P, S],
    inputs: FilterInputs,
    gather: Callable[[P, S], O],
    init_key: PRNGKeyArray | None,
    predict_keys: Array | None,
    update_keys: Array | None,
) -> O:
    """Run split steps with explicit keys, including backend-specific key schedules."""
    first = jax.tree.map(lambda xs: xs[0], inputs)
    prior = algorithm.initialize(first, key=init_key)
    first_key = None if update_keys is None else update_keys[0]
    if len(inputs.y) == 1:
        updated = algorithm.update(prior, first, key=first_key)
        return jax.tree.map(lambda x: jnp.asarray(x)[None], gather(prior, updated))

    # Prior and filtered pytrees may differ. The first iteration ignores the
    # filtered carry, so infer its shape without executing an extra update.
    shape = jax.eval_shape(lambda: algorithm.update(prior, first, key=first_key))
    initial = jax.tree.map(jnp.zeros_like, shape)

    def step(
        filtered: S, args: tuple[FilterInputs, PRNGKeyArray | None, PRNGKeyArray | None]
    ) -> tuple[S, O]:
        x, pred_key, upd_key = args
        predicted = jax.lax.cond(
            x.is_first_step,
            lambda: prior,
            lambda: algorithm.predict(filtered, x, key=pred_key),
        )
        updated = algorithm.update(predicted, x, key=upd_key)
        return updated, gather(predicted, updated)

    _, history = jax.lax.scan(step, initial, (inputs, predict_keys, update_keys))
    return history


def filter_history(
    model: DynamicalModel,
    config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " time"],
    obs_values: Shaped[Array, "time observation_dim"],
    ctrl_times: Real[Array, " control_time"] | None = None,
    ctrl_values: Real[Array, "control_time control_dim"] | None = None,
    include_initial: bool = False,
) -> tuple[Array, FilterHistory]:
    """Bind one model, scan its steps and collect its selected history."""
    if config.requires_key and key is None:
        raise ValueError(
            f"{type(config).__name__} requires a PRNG key: set 'crn_seed' or run inside a NumPyro seeded context."
        )
    inputs = step_inputs(model, obs_times, obs_values, ctrl_times, ctrl_values)
    algorithm = config.build_filter(model)
    custom = isinstance(config, (SplitFilterConfig, CombineFilterConfig))
    if not custom and config.filter_source == "cuthbert" and len(inputs.time) > 1:
        # Native Cuthbert states expose this historical value in model_inputs.
        inputs = inputs._replace(
            time_prev=inputs.time_prev.at[0].set(
                inputs.time[0] - (inputs.time[1] - inputs.time[0])
            )
        )
    if not custom and config.filter_source == "cd_dynamax" and key is not None:
        key = jax.random.key_data(key)

    def gather(*step):
        output = config.gather_outputs(*step)
        if isinstance(output, (CDPosterior, CuthbertState)):
            return output
        if not isinstance(output, FilterOutputs):
            raise TypeError(
                "gather_outputs must return FilterOutputs or a supported native filter state."
            )
        if jnp.ndim(output.log_normalizing_constant) != 0:
            raise ValueError("Expected a scalar cumulative log likelihood per step.")
        if not config.include_predicted_observations:
            output = output._replace(predicted_observations=None)
        return output

    states = run_filter_steps(
        algorithm,
        inputs,
        key,
        gather,
        parallel=bool(
            getattr(config, "parallel", getattr(config, "associative", False))
        ),
        include_initial=include_initial,
        split_keys=getattr(config, "split_keys", None),
    )
    if isinstance(states, CDPosterior):
        ll = states.marginal_loglik[-1]
        return ll, states._replace(marginal_loglik=ll)
    return states.log_normalizing_constant[-1], states


def filter_distributions(
    outputs: FilterHistory,
    config: BaseFilterConfig,
    plate_shapes: tuple[int, ...] = (),
    *,
    continuous_time: bool,
) -> list[Distribution]:
    """Construct per-time beliefs from the representation selected by the projection."""
    if isinstance(outputs, CDPosterior):
        return _posterior_sequence_to_dists(
            outputs,
            particle_mode=hasattr(outputs, "particles"),
            plate_shapes=plate_shapes,
            missing="raise" if continuous_time or plate_shapes else "empty",
            missing_message="Filtered means/covariances were unavailable for a Gaussian rollout path.",
        )
    if isinstance(outputs, CuthbertState):
        return _cholesky_state_sequence_to_dists(
            outputs,
            particle_mode=hasattr(outputs, "particles"),
            plate_shapes=plate_shapes,
            covariance_jitter=getattr(
                config, "recorded_filtered_states_cov_jitter", 0.0
            ),
        )
    gaussian = (
        outputs.filtered_means is not None,
        outputs.filtered_covariances is not None,
    )
    particles = (outputs.particles is not None, outputs.log_weights is not None)
    if all(gaussian) and not any(particles):
        return _gaussian_sequence_to_dists(
            outputs.filtered_means,
            outputs.filtered_covariances,
            plate_shapes=plate_shapes,
        )
    if (
        outputs.particles is not None
        and outputs.log_weights is not None
        and not any(gaussian)
    ):
        return _particle_sequence_to_dists(
            outputs.particles, outputs.log_weights, plate_shapes=plate_shapes
        )
    raise ValueError(
        "FilterOutputs requires either Gaussian means and covariances or particles and log weights."
    )


def _legacy_filter_result(
    compute,
    name,
    dynamics,
    filter_config,
    key=None,
    *,
    obs_times,
    obs_values,
    ctrl_times=None,
    ctrl_values=None,
    resolved_localization=None,
    **kwargs,
):
    """Support the original low-level run_* functions with the shared belief conversion."""
    options = dict(
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    if resolved_localization is not None:
        options["resolved_localization"] = resolved_localization
    history = compute(dynamics, filter_config, key, **options)
    if isinstance(history, CDPosterior):
        ll, states = history.marginal_loglik, history
    else:
        ll, states = history
    return (
        ll,
        states,
        filter_distributions(
            states, filter_config, continuous_time=dynamics.continuous_time
        ),
    )
