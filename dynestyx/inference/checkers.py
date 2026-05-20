"""Validation helpers for inference modules."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Real, Shaped

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    ContinuousTimeConfigs,
    DiscreteTimeConfigs,
    EKFConfig,
    EnKFConfig,
    HMMConfigs,
    KFConfig,
)
from dynestyx.inference.smoother_configs import (
    BaseSmootherConfig,
    ContinuousTimeSmootherConfigs,
    DiscreteTimeSmootherConfigs,
    EKFSmootherConfig,
    KFSmootherConfig,
)
from dynestyx.models import DynamicalModel
from dynestyx.utils import _has_any_batched_plate_source


def _raise_if_missing_detected(
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    has_missing,
    message: str,
) -> None:
    """Raise plainly when possible; fall back to eqx.error_if under tracing."""
    try:
        missing_now = bool(has_missing)
    except jax.errors.TracerBoolConversionError:
        _ = eqx.error_if(obs_values, has_missing, message)
        return

    if missing_now:
        raise ValueError(message)


def _leading_dims(
    arr: Shaped[Array, "..."] | None, n_dims: int
) -> tuple[int, ...] | None:
    """Return up to n_dims leading dimensions for diagnostics."""
    if arr is None:
        return None
    n = min(n_dims, arr.ndim)
    return tuple(int(d) for d in arr.shape[:n])


def _summarize_dynamics_leading_dims(
    dynamics: DynamicalModel, n_dims: int, max_items: int = 6
) -> str:
    """Summarize leading dimensions from JAX-array leaves in a model pytree."""
    shapes: list[tuple[int, ...]] = []
    for leaf in jax.tree.leaves(
        dynamics,
        is_leaf=lambda node: isinstance(node, numpyro.distributions.Distribution),
    ):
        if isinstance(leaf, jax.Array):
            n = min(n_dims, leaf.ndim)
            shapes.append(tuple(int(d) for d in leaf.shape[:n]))

    unique_shapes: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            unique_shapes.append(shape)

    if len(unique_shapes) <= max_items:
        return str(unique_shapes)
    return f"{unique_shapes[:max_items]} (+{len(unique_shapes) - max_items} more)"


def _validate_batched_plate_alignment(
    dynamics: DynamicalModel,
    plate_shapes: tuple[int, ...],
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None,
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None,
) -> None:
    """Raise early when plate_shapes do not align with any batched input source."""
    if _has_any_batched_plate_source(
        dynamics,
        plate_shapes,
        arrays=(obs_times, obs_values, ctrl_times, ctrl_values),
    ):
        return

    n_plates = len(plate_shapes)
    diagnostics = (
        "Plate/data shape alignment failed before batched filtering. "
        f"plate_shapes={plate_shapes}. No dynamics leaves or observed/control arrays "
        "have leading dimensions matching plate_shapes. "
        f"dynamics_leading_dims={_summarize_dynamics_leading_dims(dynamics, n_plates)}; "
        f"obs_times_leading_dims={_leading_dims(obs_times, n_plates)}; "
        f"obs_values_leading_dims={_leading_dims(obs_values, n_plates)}; "
        f"ctrl_times_leading_dims={_leading_dims(ctrl_times, n_plates)}; "
        f"ctrl_values_leading_dims={_leading_dims(ctrl_values, n_plates)}. "
        "Expected at least one batched source to start with plate_shapes."
    )
    raise ValueError(diagnostics)


def _validate_missing_observation_support(
    config: BaseFilterConfig | BaseSmootherConfig,
    *,
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None,
    mode: Literal["filter", "smoother"],
) -> None:
    """Reject unsupported NaN-valued observations for filter/smoother backends."""
    if obs_values is None:
        return

    has_missing = jnp.any(jnp.isnan(obs_values))

    if mode == "filter":
        continuous_types = ContinuousTimeConfigs
        discrete_types = DiscreteTimeConfigs
        exact_supported_types = (KFConfig, EnKFConfig, EKFConfig)
        exact_supported_msg = (
            "NaN-valued obs_values are currently supported only for "
            "cuthbert KFConfig, EnKFConfig, and EKFConfig filters."
        )
        cd_dynamax_msg = (
            "CD-Dynamax filters do not support NaNs in obs_values. "
            "Missing observations via NaNs currently require a cuthbert-backed "
            "discrete-time filter."
        )
        fallback_label = "filter"
    elif mode == "smoother":
        continuous_types = ContinuousTimeSmootherConfigs
        discrete_types = DiscreteTimeSmootherConfigs
        exact_supported_types = (KFSmootherConfig, EKFSmootherConfig)
        exact_supported_msg = (
            "NaN-valued obs_values are currently supported only for "
            "cuthbert KFSmootherConfig and EKFSmootherConfig smoothers."
        )
        cd_dynamax_msg = (
            "CD-Dynamax smoothers do not support NaNs in obs_values. "
            "Missing observations via NaNs currently require a cuthbert-backed "
            "discrete-time smoother."
        )
        fallback_label = "smoother"
    else:
        raise AssertionError(
            f"Unexpected missing-observation validation mode: {mode!r}"
        )
    if isinstance(config, continuous_types):
        _raise_if_missing_detected(
            obs_values,
            has_missing,
            cd_dynamax_msg,
        )
        return

    if mode == "filter" and isinstance(config, HMMConfigs):
        _raise_if_missing_detected(
            obs_values,
            has_missing,
            "HMM filtering does not support NaNs in obs_values.",
        )
        return

    if isinstance(config, discrete_types):
        filter_source = getattr(config, "filter_source", None)
        if filter_source == "cd_dynamax":
            _raise_if_missing_detected(
                obs_values,
                has_missing,
                cd_dynamax_msg,
            )
            return

        if filter_source == "cuthbert":
            if isinstance(config, exact_supported_types):
                return
            _raise_if_missing_detected(
                obs_values,
                has_missing,
                exact_supported_msg,
            )
            return

    _raise_if_missing_detected(
        obs_values,
        has_missing,
        f"NaN-valued obs_values are not supported for {type(config).__name__} {fallback_label}s.",
    )
