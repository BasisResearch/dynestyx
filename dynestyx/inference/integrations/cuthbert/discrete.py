"""Compatibility wrapper for cuthbert discrete-time filtering/smoothing."""

from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    CuthbertInputs,
    compute_cuthbert_filter,
    run_discrete_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete_smoother import (
    compute_cuthbert_smoother,
    run_discrete_smoother,
)

__all__ = [
    "CuthbertInputs",
    "compute_cuthbert_filter",
    "run_discrete_filter",
    "compute_cuthbert_smoother",
    "run_discrete_smoother",
]
