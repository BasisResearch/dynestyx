"""Step contracts and observation-aligned execution for user-defined filters."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from cuthbert.inference import Filter as CuthbertFilter
from cuthbert.utils import dummy_leading_element
from jax import Array

from dynestyx.inference.observation_predictions import PredictedObservationOutputs
from dynestyx.inference.utils.distribution_utils import (
    _gaussian_sequence_to_dists,
    _particle_sequence_to_dists,
)


class FilterInputs(NamedTuple):
    """Observation and controls for the interval ending at time."""

    y: Array
    u: Array
    u_prev: Array
    time: Array
    time_prev: Array
    is_first_step: Array


class SplitFilter(NamedTuple):
    """initialize(inputs, *, key) and predict(filtered, inputs, *, key) return priors.

    update(prior, inputs, *, key) returns the filtered carry.
    """

    initialize: Callable
    predict: Callable
    update: Callable


class FilterOutputs(NamedTuple):
    """One gathered step. Supply Gaussian moments or particles and log weights."""

    log_normalizing_constant: Array
    filtered_means: Array | None = None
    filtered_covariances: Array | None = None
    particles: Array | None = None
    log_weights: Array | None = None
    predicted_observations: PredictedObservationOutputs | None = None
    extra_outputs: dict[str, Array] | None = None


def step_inputs(dynamics, times, ys, ctrl_times, controls):
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


def _split(key, count):
    return (None,) * count if key is None else jax.random.split(key, count)


def _prepend(first, tail):
    return jax.tree.map(lambda x, xs: jnp.concatenate((jnp.asarray(x)[None], xs)), first, tail)


def run_filter_steps(
    algorithm, inputs, key, gather, *, parallel=False, include_initial=False
):
    """Collect projections in a sequential scan or from associative prefix states."""
    init_key, scan_key = _split(key, 2)
    keys = None if scan_key is None else jax.random.split(scan_key, len(inputs.y))
    first = jax.tree.map(lambda xs: xs[0], inputs)
    if hasattr(algorithm, "filter_combine"):
        initial = algorithm.init_prepare(key=init_key)
        if isinstance(algorithm, CuthbertFilter) and hasattr(initial, "model_inputs"):
            initial = initial._replace(model_inputs=dummy_leading_element(inputs))
        if parallel:
            if not algorithm.associative:
                raise ValueError("Parallel filtering requires an associative combine.")
            prepared = jax.vmap(
                lambda x, k: algorithm.filter_prepare(x, key=k),
                in_axes=(0, None if keys is None else 0),
            )(inputs, keys)
            prefixes = jax.lax.associative_scan(
                jax.vmap(algorithm.filter_combine), _prepend(initial, prepared)
            )
            if not include_initial:
                prefixes = jax.tree.map(lambda xs: xs[1:], prefixes)
            return jax.vmap(gather)(prefixes)

        def combine(state, args):
            x, k = args
            updated = algorithm.filter_combine(
                state, algorithm.filter_prepare(x, key=k)
            )
            return updated, gather(updated)

        _, history = jax.lax.scan(combine, initial, (inputs, keys))
        return _prepend(gather(initial), history) if include_initial else history

    if parallel:
        raise ValueError("Parallel filtering requires an associative combine.")
    if not all(
        callable(getattr(algorithm, name, None))
        for name in ("initialize", "predict", "update")
    ):
        raise TypeError("Expected a split filter or a prepare/combine filter.")
    prior = algorithm.initialize(first, key=init_key)
    first_key = None if keys is None else keys[0]
    _, update_key = _split(first_key, 2)
    updated = algorithm.update(prior, first, key=update_key)
    first_output = gather(prior, updated)

    def step(filtered, args):
        x, k = args
        pred_key, upd_key = _split(k, 2)
        predicted = algorithm.predict(filtered, x, key=pred_key)
        updated = algorithm.update(predicted, x, key=upd_key)
        return updated, gather(predicted, updated)

    if len(inputs.y) == 1:
        return jax.tree.map(lambda x: jnp.asarray(x)[None], first_output)
    rest = jax.tree.map(lambda xs: xs[1:], (inputs, keys))
    _, history = jax.lax.scan(step, updated, rest)
    return _prepend(first_output, history)


def compute_custom_filter(dynamics, config, times, values, ctrl_times, controls, key):
    inputs = step_inputs(dynamics, times, values, ctrl_times, controls)

    def gather(*states: Any):
        output = config.gather_outputs(*states)
        if not isinstance(output, FilterOutputs):
            raise TypeError("gather_outputs must return FilterOutputs.")
        if jnp.ndim(output.log_normalizing_constant) != 0:
            raise ValueError("Expected a scalar cumulative log likelihood per step.")
        if not config.include_predicted_observations:
            output = output._replace(predicted_observations=None)
        return output

    return run_filter_steps(
        config.build_filter(dynamics), inputs, key, gather, parallel=config.parallel
    )


def custom_distributions(outputs, plate_shapes):
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
    if all(particles) and not any(gaussian):
        return _particle_sequence_to_dists(
            outputs.particles, outputs.log_weights, plate_shapes=plate_shapes
        )
    raise ValueError(
        "FilterOutputs requires either Gaussian means and covariances or particles and log weights."
    )
