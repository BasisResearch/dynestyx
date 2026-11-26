"""Step contracts and observation-aligned execution for user-defined filters."""

from collections.abc import Callable
from typing import cast, overload

import jax
import jax.numpy as jnp
from cuthbert.inference import Filter as CuthbertFilter
from cuthbert.utils import dummy_leading_element
from jaxtyping import Array, PRNGKeyArray, Real, Shaped
from numpyro.distributions import Distribution

from dynestyx.configs.filters import CombineFilterConfig, SplitFilterConfig
from dynestyx.inference.filter_types import CombinedSteps, SplitSteps
from dynestyx.inference.filter_types import FilterInputs as FilterInputs
from dynestyx.inference.filter_types import FilterOutputs as FilterOutputs
from dynestyx.inference.filter_types import (
    PredictedObservationOutputs as PredictedObservationOutputs,
)
from dynestyx.inference.filter_types import SplitFilter as SplitFilter
from dynestyx.inference.utils.distribution_utils import (
    _gaussian_sequence_to_dists,
    _particle_sequence_to_dists,
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
) -> O: ...


def run_filter_steps[P, S, E, O](
    algorithm: SplitSteps[P, S] | CombinedSteps[E, S],
    inputs: FilterInputs,
    key: PRNGKeyArray | None,
    gather: Callable[[P, S], O] | Callable[[S], O],
    *,
    parallel: bool = False,
    include_initial: bool = False,
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
    if keys is None:
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


def compute_custom_filter(
    dynamics: DynamicalModel,
    config: SplitFilterConfig | CombineFilterConfig,
    times: Real[Array, " time"],
    values: Shaped[Array, "time observation_dim"],
    ctrl_times: Real[Array, " control_time"] | None,
    controls: Real[Array, "control_time control_dim"] | None,
    key: PRNGKeyArray | None,
) -> FilterOutputs:
    inputs = step_inputs(dynamics, times, values, ctrl_times, controls)

    def checked(output: FilterOutputs) -> FilterOutputs:
        if not isinstance(output, FilterOutputs):
            raise TypeError("gather_outputs must return FilterOutputs.")
        if jnp.ndim(output.log_normalizing_constant) != 0:
            raise ValueError("Expected a scalar cumulative log likelihood per step.")
        if not config.include_predicted_observations:
            output = output._replace(predicted_observations=None)
        return output

    if isinstance(config, SplitFilterConfig):
        return run_filter_steps(
            config.build_filter(dynamics),
            inputs,
            key,
            lambda prior, filtered: checked(config.gather_outputs(prior, filtered)),
        )
    return run_filter_steps(
        config.build_filter(dynamics),
        inputs,
        key,
        lambda filtered: checked(config.gather_outputs(filtered)),
        parallel=config.parallel,
    )


def custom_distributions(
    outputs: FilterOutputs, plate_shapes: tuple[int, ...]
) -> list[Distribution]:
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
