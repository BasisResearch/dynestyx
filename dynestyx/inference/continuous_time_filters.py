"""Continuous-time filter dispatcher: delegates to cd-dynamax (and future backends)."""

import jax

from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.inference.cd_dynamax.continuous import (
    _CONTINUOUS_FILTER_TYPES,
    run_continuous_filter,
)

__all__ = ["_filter_continuous_time", "_CONTINUOUS_FILTER_TYPES"]


def _filter_continuous_time(
    name: str,
    filter_type: str,
    dynamics: DynamicalModel,
    context: Context,
    key: jax.Array | None = None,
    filter_kwargs: dict | None = None,
    record_kwargs: dict = {},
) -> None:
    """Continuous-time marginal likelihood via CD-Dynamax.

    Supports: enkf, dpf, ekf, ukf, default (→ EnKF).

    Args:
        name: Name of the factor.
        filter_type: Type of filter to use.
        dynamics: Dynamical model to filter.
        context: Context containing the observations and controls.
        key: Random key for the filter.
        filter_kwargs: Keyword arguments for the filter.
        record_kwargs: Keyword arguments for recording filtered states.
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    if filter_type.lower() not in _CONTINUOUS_FILTER_TYPES:
        raise ValueError(
            f"Invalid filter type: {filter_type}. Valid types: {_CONTINUOUS_FILTER_TYPES}"
        )

    run_continuous_filter(
        name, filter_type, dynamics, context, key, filter_kwargs, record_kwargs
    )
