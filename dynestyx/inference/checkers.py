"""Validation helpers for inference modules."""

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Real

from dynestyx.configs.filters import (
    BaseFilterConfig,
    ContinuousTimeConfigs,
    DiscreteTimeConfigs,
    EnKFConfig,
    HMMConfigs,
    KFConfig,
    PFConfig,
)
from dynestyx.configs.smoother import (
    BaseSmootherConfig,
    ContinuousTimeSmootherConfigs,
    DiscreteTimeSmootherConfigs,
    EnRTSSmootherConfig,
    KFSmootherConfig,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
)
from dynestyx.utils.validation import _raise_now_or_error_if


def _validate_missing_observation_support(
    config: BaseFilterConfig | BaseSmootherConfig,
    *,
    obs_values: Real[Array, "*obs_value_shape"],
    mode: Literal["filter", "smoother"],
) -> Real[Array, "*obs_value_shape"]:
    """Reject unsupported NaN-valued observations for filter/smoother backends."""
    has_missing = jnp.any(jnp.isnan(obs_values))

    if mode == "filter":
        continuous_types = ContinuousTimeConfigs
        discrete_types = DiscreteTimeConfigs
        exact_supported_types = (KFConfig, EnKFConfig)
        exact_supported_msg = (
            "NaN-valued obs_values are currently supported only for "
            "cuthbert KFConfig and EnKFConfig filters."
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
        exact_supported_types = (KFSmootherConfig, EnRTSSmootherConfig)
        exact_supported_msg = (
            "NaN-valued obs_values are currently supported only for "
            "cuthbert KFSmootherConfig and EnRTSSmootherConfig smoothers."
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
        return _raise_now_or_error_if(obs_values, has_missing, cd_dynamax_msg)

    if mode == "filter" and isinstance(config, HMMConfigs):
        return obs_values

    if isinstance(config, discrete_types):
        filter_source = getattr(config, "filter_source", None)
        if filter_source == "cd_dynamax":
            return _raise_now_or_error_if(obs_values, has_missing, cd_dynamax_msg)

        if filter_source == "cuthbert":
            if mode == "filter" and isinstance(config, PFConfig):
                return _raise_now_or_error_if(
                    obs_values,
                    has_missing,
                    "PFConfig does not treat NaN-valued obs_values specially. "
                    "Whether missing observations work correctly is up to the "
                    "user-specified transition and observation functions to "
                    "handle NaNs appropriately.",
                    action="warn",
                )
            if isinstance(config, exact_supported_types):
                return obs_values
            return _raise_now_or_error_if(
                obs_values,
                has_missing,
                exact_supported_msg,
            )

    return _raise_now_or_error_if(
        obs_values,
        has_missing,
        f"NaN-valued obs_values are not supported for {type(config).__name__} {fallback_label}s.",
    )


def _validate_inference_supported_model_classes(dynamics: DynamicalModel) -> None:
    """Reject model classes that are valid for simulation but not inference.

    In particular, deterministic continuous-time dynamics paired with
    ``DiracIdentityObservation`` are allowed for forward simulation, but not for
    scoring or inference. In continuous time, exact identity observations turn
    the observation model into a degenerate path constraint rather than a
    regular likelihood term.
    """
    if isinstance(
        dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
    ) and isinstance(dynamics.observation_model, DiracIdentityObservation):
        raise ValueError(
            "Inference/scoring with deterministic continuous-time dynamics "
            "(ODEs) and DiracIdentityObservation is not supported. Forward "
            "simulation is still allowed, but exact-observation likelihoods "
            "are degenerate in continuous time; add observation noise or "
            "reformulate the model."
        )
