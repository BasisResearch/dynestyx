"""Discrete-time filter dispatcher: delegates to cuthbert or cd-dynamax backends."""

import jax

from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.inference.filter_configs import BaseFilterConfig
from dynestyx.inference.integrations.cd_dynamax.discrete import (
    run_discrete_filter as run_cd_dynamax_discrete,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    run_discrete_filter as run_cuthbert_discrete,
)


def _filter_discrete_time(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
) -> None:
    """Discrete-time marginal likelihood via cuthbert or cd-dynamax.

    Filter type inferred from config class: KFConfig, EKFConfig, UKFConfig (cd-dynamax)
    or EKFConfig (cuthbert), PFConfig (cuthbert).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        context: Context containing the observations and controls.
        filter_config: Configuration for the filter.
    """

    if filter_config.filter_source == "cd_dynamax":
        run_cd_dynamax_discrete(name, dynamics, context, filter_config)
    elif filter_config.filter_source == "cuthbert":
        run_cuthbert_discrete(name, dynamics, context, filter_config, key)
    else:
        raise ValueError(f"Unknown filter source: {filter_config.filter_source}")
