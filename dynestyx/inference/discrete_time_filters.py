"""Discrete-time filter dispatcher: delegates to cuthbert or cd-dynamax backends."""

import jax

from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.inference.cd_dynamax.discrete import (
    run_discrete_filter as run_cd_dynamax_discrete,
)
from dynestyx.inference.cuthbert.discrete import (
    run_discrete_filter as run_cuthbert_discrete,
)

_DISCRETE_FILTER_TYPES: list[str] = ["default", "taylor_kf", "pf", "kf", "ekf", "ukf"]

_CUTHBERT_FILTER_TYPES: frozenset[str] = frozenset({"default", "taylor_kf", "pf"})
_CD_DYNAMAX_FILTER_TYPES: frozenset[str] = frozenset({"kf", "ekf", "ukf"})


def _filter_discrete_time(
    name: str,
    filter_type: str,
    dynamics: DynamicalModel,
    context: Context,
    key: jax.Array | None = None,
    filter_kwargs: dict | None = None,
    record_kwargs: dict = {},
) -> None:
    """Discrete-time marginal likelihood via cuthbert or cd-dynamax.

    - cuthbert: taylor_kf, pf, default
    - cd-dynamax: kf, ekf, ukf

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

    ft = filter_type.lower()
    if ft in _CD_DYNAMAX_FILTER_TYPES:
        run_cd_dynamax_discrete(
            name, filter_type, dynamics, context, key, filter_kwargs, record_kwargs
        )
    elif ft in _CUTHBERT_FILTER_TYPES:
        run_cuthbert_discrete(
            name, filter_type, dynamics, context, key, filter_kwargs, record_kwargs
        )
    else:
        raise ValueError(
            f"Invalid filter type: {filter_type}. Valid types: {_DISCRETE_FILTER_TYPES}"
        )
