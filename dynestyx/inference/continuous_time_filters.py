"""Continuous-time filter dispatcher: delegates to cd-dynamax (and future backends)."""

import jax

from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.inference.filter_configs import BaseFilterConfig
from dynestyx.inference.integrations.cd_dynamax.continuous import run_continuous_filter


def _filter_continuous_time(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
) -> None:
    """Continuous-time marginal likelihood via CD-Dynamax.

    Supports: EnKF, DPF, EKF, UKF (inferred from config type).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        context: Context containing the observations and controls.
        filter_config: Configuration for the filter.
    """
    run_continuous_filter(name, dynamics, context, filter_config, key)  # type: ignore[arg-type]
