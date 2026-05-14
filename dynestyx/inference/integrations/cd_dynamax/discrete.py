"""Compatibility wrapper for cd-dynamax discrete-time filtering/smoothing."""

from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    _lti_to_lgssm_params,
    _prepare_inputs,
    compute_cd_dynamax_discrete_filter,
    run_discrete_filter,
)
from dynestyx.inference.integrations.cd_dynamax.discrete_smoother import (
    compute_cd_dynamax_discrete_smoother,
    run_discrete_smoother,
)

__all__ = [
    "compute_cd_dynamax_discrete_filter",
    "run_discrete_filter",
    "compute_cd_dynamax_discrete_smoother",
    "run_discrete_smoother",
    "_lti_to_lgssm_params",
    "_prepare_inputs",
]
