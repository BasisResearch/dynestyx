"""Effectful handlers for evaluating conditioned dynamical-model results."""

from __future__ import annotations

import dataclasses

from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Bool, Real

from dynestyx.evaluation.configs import ObservationScoringConfig
from dynestyx.evaluation.observation_scoring import build_evaluation_result
from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.models import DynamicalModel
from dynestyx.types import (
    ConditionedResult,
    EvaluationResult,
    chain_numpyro_site_registrations,
)
from dynestyx.utils import _raise_now_or_error_if

_MISSING_OBSERVATIONS_ERROR = (
    "Observation scoring does not yet support missing obs_values. "
    "Remove or impute missing observations before using Evaluation."
)


@dataclasses.dataclass
class Evaluation(ObjectInterpretation, HandlesSelf):
    """Evaluate outputs forwarded by an inner conditioning handler."""

    observation_scoring_config: ObservationScoringConfig

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, ...] | None = None,
        obs_values: Real[Array, ...] | None = None,
        _obs_values_filled: Real[Array, ...] | None = None,
        _obs_mask: Bool[Array, ...] | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, ...] | None = None,
        ctrl_values: Real[Array, ...] | None = None,
        filtered_result: ConditionedResult | None = None,
        **kwargs,
    ) -> EvaluationResult:
        if filtered_result is None:
            raise ValueError(
                "Observation scoring requires a filtered ConditionedResult. "
                "Place Evaluation outside Filter:\n\n"
                "with Evaluation(observation_scoring_config=...):\n"
                "    with Filter(filter_config=...):\n"
                "        dsx.condition(...)"
            )

        if obs_values is not None and _obs_mask is not None:
            obs_values = _raise_now_or_error_if(
                obs_values,
                ~_obs_mask.all(),
                _MISSING_OBSERVATIONS_ERROR,
            )
        elif _obs_has_missing:
            raise ValueError(_MISSING_OBSERVATIONS_ERROR)

        evaluation_result = build_evaluation_result(
            filtered_result=filtered_result,
            obs_values=obs_values,
            scoring_config=self.observation_scoring_config,
            plate_shapes=plate_shapes,
        )
        filtered_result.evaluation_result = evaluation_result

        forwarded_result = fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            _obs_values_filled=_obs_values_filled,
            _obs_mask=_obs_mask,
            _obs_has_missing=_obs_has_missing,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            filtered_result=filtered_result,
            evaluation_result=evaluation_result,
            **kwargs,
        )
        evaluation_result._register_numpyro_sites = chain_numpyro_site_registrations(
            evaluation_result._register_numpyro_sites,
            getattr(forwarded_result, "_register_numpyro_sites", None),
        )
        return evaluation_result


__all__ = ["Evaluation"]
