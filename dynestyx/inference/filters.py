import dataclasses
import math
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.inference.checkers import (
    _validate_batched_plate_alignment,
    _validate_missing_observation_support,
)
from dynestyx.inference.distribution_utils import (
    _categorical_log_probs_to_dists,
    _cholesky_state_sequence_to_dists,
    _posterior_sequence_to_dists,
)
from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    ContinuousTimeConfigs,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    DiscreteTimeConfigs,
    EKFConfig,
    EnKFConfig,
    HMMConfig,
    HMMConfigs,
    KFConfig,
    PFConfig,
    PFResamplingConfig,
    UKFConfig,
)
from dynestyx.inference.hmm_filters import _filter_hmm, compute_hmm_filter
from dynestyx.inference.integrations.cd_dynamax.continuous import (
    ContinuousTimeFilterConfig,
    compute_continuous_filter,
    run_continuous_filter,
)
from dynestyx.inference.integrations.cd_dynamax.discrete import (
    compute_cd_dynamax_discrete_filter,
)
from dynestyx.inference.integrations.cd_dynamax.discrete import (
    run_discrete_filter as run_cd_dynamax_discrete,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    run_discrete_filter as run_cuthbert_discrete,
)
from dynestyx.inference.plate_utils import (
    _array_plate_axis,
    _canonicalize_plate_dynamics_for_vmap,
    _flatten_array_for_plate_vmap,
    _make_plate_in_axes,
    _reshape_vmap_outputs_to_plate,
    _restore_batched_initial_condition_for_vmap_member,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import FunctionOfTime

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class BaseLogFactorAdder(ObjectInterpretation, HandlesSelf):
    """Base for filter handlers."""

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> FunctionOfTime:
        filtered_dists = None
        if not (obs_times is None or obs_values is None):
            filtered_dists = self._add_log_factors(
                name,
                dynamics,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )

        # Filter consumes obs_times and obs_values, so they are passed forward as None
        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=None,
            obs_values=None,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            filtered_times=obs_times,
            filtered_dists=filtered_dists,
            **kwargs,
        )

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        # Inheritors should implement this method.
        raise NotImplementedError()


def _default_filter_config(dynamics: DynamicalModel):
    """Return appropriate default filter config when none specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEnKFConfig()

    return EnKFConfig()


@dataclasses.dataclass
class Filter(BaseLogFactorAdder):
    r"""Performs Bayesian filtering to compute the filtering distribution $p(x_t | y_{1:t})$ and the marginal likelihood $\log p(y_{1:T})$.

    A `Filter` object should be used as a context manager around a call to a model with a `dsx.sample(...)` statement
    to condition a dynamical model on observations via a filtering algorithm. The filter
    is selected and dispatched according to the `filter_config` argument, which adds the
    marginal log-likelihood as a NumPyro factor, allowing for downstream parameter inference.

    Examples:
        >>> def model(obs_times=None, obs_values=None):
        ...     dynamics = DynamicalModel(...)
        ...     return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)
        >>> def filtered_model(t, y):
        ...     with Filter(filter_config=KFConfig()):
        ...         return model(obs_times=t, obs_values=y)

    What this does
    --------------
    Filtering is the recursive (potentially approximate) computation of the filtering distribution
    \(p(x_t \mid y_{1:t})\). It allows for the computation of the marginal likelihood:

    \[
      \log p(y_{1:T}) = \sum_{t=1}^T \log p(y_t \mid y_{1:t-1}),
    \]

    which in turn can be used to compute the posterior distribution over the parameters $p(\theta | y_{1:T})$.


    Available Filter Configurations
    ----------------------------------
    There are several different filters available in `dynestyx`, each with their own strengths and weaknesses.
    What filters are applicable to a given model depends heavily on any special structure of the model (for example, linear and/or Gaussian observations).
    For a summary table of all config classes and when to use them, see
    [Available filter configurations](../filter_configs.md).

    Defaults
    --------
    If `filter_config=None`, defaults are:

    - `ContinuousTimeEnKFConfig()` for continuous-time models, and
    - `EnKFConfig()` for discrete-time models.

    Notes:
        - If your latent state is *discrete* (an HMM), you must use `HMMConfig`.
        - What gets recorded to the trace (means/covariances, particles/weights,
        etc.) depends on `filter_config.record_*` and the backend implementation.

    Attributes:
        filter_config: Selects the filtering algorithm and its hyperparameters.
            If `None`, a reasonable default is chosen based on whether the model
            is continuous-time or discrete-time.
    """

    filter_config: BaseFilterConfig | None = None

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        """
        Add the marginal log likelihood as a numpyro factor.

        Args:
            name: Name of the factor.
            dynamics: Dynamical model to filter.
            plate_shapes: Tuple of plate sizes from enclosing dsx.plate contexts.
            obs_times: Observation times.
            obs_values: Observed values.
            ctrl_times: Control times (optional).
            ctrl_values: Control values (optional).
        """
        if obs_times is None or obs_values is None:
            raise ValueError("obs_times and obs_values are required for filtering.")

        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )
        if isinstance(config, BaseFilterConfig):
            _validate_missing_observation_support(
                config,
                obs_values=obs_values,
                mode="filter",
            )

        key = numpyro.prng_key() if config.crn_seed is None else config.crn_seed

        if plate_shapes:
            return self._add_log_factors_batched(
                name,
                dynamics,
                config,
                key=key,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    "Continuous-time models require a continuous-time filter config. "
                    "If you want to use a discrete-time filter, nest `Discretizer()` "
                    "inside `Filter()`. "
                    f"Got {type(config).__name__}; valid continuous-time config types: {valid}."
                )
            return _filter_continuous_time(
                name,
                dynamics,
                config,  # type: ignore[arg-type]
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )
        else:
            if isinstance(config, HMMConfigs):
                return _filter_hmm(
                    name,
                    dynamics,
                    cast(HMMConfig, config),
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    **kwargs,
                )
            elif isinstance(config, DiscreteTimeConfigs):
                return _filter_discrete_time(
                    name,
                    dynamics,
                    config,  # type: ignore[arg-type]
                    key=key,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    **kwargs,
                )
            else:
                valid = [c.__name__ for c in HMMConfigs + DiscreteTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )

    def _add_log_factors_batched(
        self,
        name: str,
        dynamics: DynamicalModel,
        config: BaseFilterConfig,
        *,
        key: PRNGKeyArray | None,
        plate_shapes: tuple[int, ...],
        obs_times: Real[Array, "*obs_time_plate obs_time"],
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"],
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
    ) -> list[numpyro.distributions.Distribution]:
        """Compute batched marginal log-likelihoods via vmap for plate contexts.

        Vmaps the pure-JAX compute function over each plate dimension, issues one
        numpyro.factor with batched log-likelihoods, and reconstructs per-time
        filtered distributions with plate-shaped batch dimensions for rollout.
        """
        # Determine the compute function (dispatch before vmap).
        output_kind: str
        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            output_kind = "continuous"

            def compute_output(dyn, ot, ov, ct, cv, k):
                return compute_continuous_filter(
                    dyn,
                    cast(ContinuousTimeFilterConfig, config),
                    k,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_times=ct,
                    ctrl_values=cv,
                )

        elif isinstance(config, HMMConfigs):
            output_kind = "hmm"

            def compute_output(dyn, ot, ov, ct, cv, k):
                return compute_hmm_filter(
                    dyn,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_values=cv,
                )

        elif isinstance(config, DiscreteTimeConfigs):
            if config.filter_source == "cuthbert":
                output_kind = "cuthbert"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cuthbert_filter(
                        dyn,
                        config,
                        k,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            elif config.filter_source == "cd_dynamax":
                output_kind = "cd_dynamax_discrete"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cd_dynamax_discrete_filter(
                        dyn,
                        config,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            else:
                raise ValueError(f"Unknown filter source: {config.filter_source}")
        else:
            raise ValueError(
                f"Unsupported filter config for plate: {type(config).__name__}"
            )

        # Pre-split keys for all plate members (needed for stochastic filters).
        if key is not None:
            # Ensure we use typed PRNG keys so split returns shape (total,)
            # rather than old-style (total, 2).
            if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
                key = jax.random.wrap_key_data(key)
            total = math.prod(plate_shapes)
            keys = jax.random.split(key, total)
        else:
            keys = None

        _validate_batched_plate_alignment(
            dynamics,
            plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
        total = math.prod(plate_shapes)
        canonical_plate_shapes = (total,)
        member_indices = jnp.arange(total)
        original_initial_condition = dynamics.initial_condition
        batched_dynamics = _canonicalize_plate_dynamics_for_vmap(dynamics, plate_shapes)
        flat_obs_times = _flatten_array_for_plate_vmap(
            obs_times,
            plate_shapes,
            min_suffix_ndim=1,
        )
        flat_obs_values = _flatten_array_for_plate_vmap(
            obs_values,
            plate_shapes,
            min_suffix_ndim=1,
        )
        flat_ctrl_times = _flatten_array_for_plate_vmap(
            ctrl_times,
            plate_shapes,
            min_suffix_ndim=1,
        )
        flat_ctrl_values = _flatten_array_for_plate_vmap(
            ctrl_values,
            plate_shapes,
            min_suffix_ndim=1,
        )

        dyn_axes = _make_plate_in_axes(batched_dynamics, canonical_plate_shapes)
        ot_axis = _array_plate_axis(flat_obs_times, canonical_plate_shapes)
        ov_axis = _array_plate_axis(flat_obs_values, canonical_plate_shapes)
        ct_axis = _array_plate_axis(flat_ctrl_times, canonical_plate_shapes)
        cv_axis = _array_plate_axis(flat_ctrl_values, canonical_plate_shapes)
        k_axis = 0 if keys is not None else None

        def compute_output_for_member(dyn, member_idx, ot, ov, ct, cv, k):
            dyn = _restore_batched_initial_condition_for_vmap_member(
                dyn,
                original_initial_condition,
                plate_shapes,
                member_idx,
            )
            return compute_output(dyn, ot, ov, ct, cv, k)

        outputs = jax.vmap(
            compute_output_for_member,
            in_axes=(dyn_axes, 0, ot_axis, ov_axis, ct_axis, cv_axis, k_axis),
        )(
            batched_dynamics,
            member_indices,
            flat_obs_times,
            flat_obs_values,
            flat_ctrl_times,
            flat_ctrl_values,
            keys,
        )
        outputs = _reshape_vmap_outputs_to_plate(outputs, plate_shapes)

        if output_kind in {"continuous", "cd_dynamax_discrete"}:
            marginal_logliks = outputs.marginal_loglik
        elif output_kind == "hmm":
            marginal_logliks, log_filt_seq = outputs
        elif output_kind == "cuthbert":
            marginal_logliks, states = outputs
        else:
            raise ValueError(f"Unsupported batched output kind: {output_kind}")

        numpyro.factor(f"{name}_marginal_log_likelihood", marginal_logliks)
        numpyro.deterministic(f"{name}_marginal_loglik", marginal_logliks)

        if output_kind == "continuous":
            particle_mode = isinstance(config, ContinuousTimeDPFConfig)
            return _posterior_sequence_to_dists(
                outputs,
                means_attr="filtered_means",
                covariances_attr="filtered_covariances",
                plate_shapes=plate_shapes,
                particle_mode=particle_mode,
                missing_message=(
                    "Filtered means/covariances were unavailable for a Gaussian rollout path."
                ),
            )
        if output_kind == "cd_dynamax_discrete":
            return _posterior_sequence_to_dists(
                outputs,
                means_attr="filtered_means",
                covariances_attr="filtered_covariances",
                plate_shapes=plate_shapes,
                particle_mode=False,
                missing_message=(
                    "Filtered means/covariances were unavailable for a Gaussian rollout path."
                ),
            )
        if output_kind == "hmm":
            return _categorical_log_probs_to_dists(
                log_filt_seq,
                plate_shapes=plate_shapes,
            )
        if output_kind == "cuthbert":
            return _cholesky_state_sequence_to_dists(
                states,
                particle_mode=isinstance(config, PFConfig),
                plate_shapes=plate_shapes,
            )

        raise ValueError(f"Unsupported batched output kind: {output_kind}")


def _filter_discrete_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Discrete-time marginal likelihood via cuthbert or cd-dynamax.

    Filter type inferred from config class: KFConfig, EKFConfig, UKFConfig
    (cd-dynamax) or KFConfig, EKFConfig, EnKFConfig, PFConfig (cuthbert).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: Configuration for the filter.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """

    if filter_config.filter_source == "cd_dynamax":
        return run_cd_dynamax_discrete(
            name,
            dynamics,
            filter_config,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    elif filter_config.filter_source == "cuthbert":
        return run_cuthbert_discrete(
            name,
            dynamics,
            filter_config,
            key=key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown filter source: {filter_config.filter_source}")


def _filter_continuous_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Continuous-time marginal likelihood via CD-Dynamax.

    Supports: EnKF, DPF, EKF, UKF (inferred from config type).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: Configuration for the filter.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """
    return run_continuous_filter(
        name,
        dynamics,
        cast(ContinuousTimeFilterConfig, filter_config),
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
    )


__all__ = [
    "ContinuousTimeKFConfig",
    "ContinuousTimeDPFConfig",
    "ContinuousTimeEnKFConfig",
    "ContinuousTimeEKFConfig",
    "ContinuousTimeUKFConfig",
    "EKFConfig",
    "EnKFConfig",
    "Filter",
    "HMMConfig",
    "HMMConfigs",
    "KFConfig",
    "PFConfig",
    "PFResamplingConfig",
    "UKFConfig",
]
