import dataclasses
import math
import warnings
from abc import ABC, abstractmethod
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Bool, PRNGKeyArray, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.checkers import (
    _validate_batched_plate_alignment,
    _validate_inference_supported_model_classes,
    _validate_missing_observation_support,
)
from dynestyx.inference.configs.filter import (
    BaseFilterConfig,
    ConstructCholInnovationCovariance,
    ContinuousTimeConfigs,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    DiscreteTimeConfigs,
    EKFConfig,
    EnKFConfig,
    EnKFLocalizationConfig,
    EnKFLocalizationFunctions,
    HMMConfig,
    HMMConfigs,
    KFConfig,
    ModifyCrossCovariance,
    ModifyPredictedObservationCovariance,
    PFConfig,
    PFResamplingConfig,
    TaperCovarianceFn,
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
from dynestyx.inference.observation_predictions import (
    PredictedObservationOutputs,
    add_observation_prediction_sites,
    extract_filter_predictions,
)
from dynestyx.inference.utils.distribution_utils import (
    _categorical_log_probs_to_dists,
    _cholesky_state_sequence_to_dists,
    _posterior_sequence_to_dists,
)
from dynestyx.inference.utils.numpyro_sites import (
    register_filter_sites,
    register_hmm_filter_sites,
)
from dynestyx.inference.utils.plate_utils import (
    _array_plate_axis,
    _make_plate_in_axes,
    _slice_dist_for_plate_member,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import (
    ConditionedResult,
    FunctionOfTime,
    chain_numpyro_site_registrations,
)
from dynestyx.utils import _dist_has_plate_batch_dims, _ensure_trailing_event_axis

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class BaseLogFactorAdder(ObjectInterpretation, HandlesSelf, ABC):
    """Base for filter handlers."""

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        filtered_result: ConditionedResult | None = None,
        smoothed_result: ConditionedResult | None = None,
        **kwargs,
    ) -> FunctionOfTime:
        if filtered_result is not None or smoothed_result is not None:
            raise ValueError(
                "Filter cannot condition an already conditioned result. Use only "
                "one Filter or Smoother for a dsx.condition/dsx.sample operation."
            )

        filtered_dists = None
        self.marginal_loglik = self.filtered_states = self._filter_config_used = None
        self.predicted_observations = None
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

        result = self._build_infer_result(obs_times, filtered_dists)

        # Observation inputs remain available to outer consumers such as Evaluation.
        forwarded_result = fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            filtered_result=(result if filtered_dists is not None else None),
            **kwargs,
        )

        forwarded_register = getattr(forwarded_result, "_register_numpyro_sites", None)
        result._register_numpyro_sites = chain_numpyro_site_registrations(
            result._register_numpyro_sites,
            forwarded_register,
        )

        return result

    @abstractmethod
    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None: ...

    @abstractmethod
    def _build_infer_result(
        self,
        times: Real[Array, "*time_plate time"] | None,
        filtered_dists: list | None,
    ) -> ConditionedResult: ...


def _default_filter_config(dynamics: DynamicalModel) -> BaseFilterConfig:
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
    [Available filter configurations](configs/filter_configs.md).

    Defaults
    --------
    If `filter_config=None`, defaults are:

    - `ContinuousTimeEnKFConfig()` for continuous-time models, and
    - `EnKFConfig()` for discrete-time models.

    Notes:
        - If your latent state is *discrete* (an HMM), you must use `HMMConfig`.
        - What gets recorded to the trace (means/covariances, particles/weights,
        etc.) depends on `filter_config.record_*` and the backend implementation.
        - Supported one-step-ahead predictive-observation outputs are included
        in `ConditionedResult` by default. The
        `record_predicted_observations_*` fields control NumPyro trace sites.

    Attributes:
        filter_config: Selects the filtering algorithm and its hyperparameters.
            If `None`, a reasonable default is chosen based on whether the model
            is continuous-time or discrete-time.
    """

    filter_config: BaseFilterConfig | None = None
    marginal_loglik: Real[Array, "*plate"] | None = dataclasses.field(
        default=None, repr=False, init=False
    )
    filtered_states: object = dataclasses.field(default=None, repr=False, init=False)
    _filter_config_used: BaseFilterConfig | None = dataclasses.field(
        default=None, repr=False, init=False
    )
    predicted_observations: PredictedObservationOutputs | None = dataclasses.field(
        default=None, repr=False, init=False
    )

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_mask: Bool[Array, "*obs_value_plate obs_time observation_dim"]
        | Bool[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        """Run filtering and store the marginal log-likelihood.

        Pure computation — no numpyro side effects. Site registration
        happens via the callback in ConditionedResult when called through dsx.sample.
        """
        if obs_times is None or obs_values is None:
            raise ValueError("obs_times and obs_values are required for filtering.")
        _validate_inference_supported_model_classes(dynamics)

        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )
        if isinstance(config, BaseFilterConfig):
            obs_values = _validate_missing_observation_support(
                config,
                obs_values=obs_values,
                mode="filter",
            )
        # Resolve PRNG key: use explicit seed from config, fall back to numpyro
        # context (inside a seeded model), or None (deterministic filters don't need one).
        if config.crn_seed is not None:
            key = config.crn_seed
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                key = numpyro.prng_key()  # returns None outside seed handler

        if plate_shapes:
            return self._add_log_factors_batched(
                name,
                dynamics,
                config,
                key=key,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                _obs_has_missing=_obs_has_missing,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

        if not isinstance(config, HMMConfigs):
            obs_values = _ensure_trailing_event_axis(obs_values)
            if ctrl_values is not None:
                ctrl_values = _ensure_trailing_event_axis(ctrl_values)

        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    "Continuous-time models require a continuous-time filter config. "
                    "If you want to use a discrete-time filter, nest `Discretizer()` "
                    "inside `Filter()`. "
                    f"Got {type(config).__name__}; valid continuous-time config types: {valid}."
                )
            marginal_loglik, states, filtered_dists = _filter_continuous_time(
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
        elif isinstance(config, HMMConfigs):
            loglik, log_filt_seq, filtered_dists = _filter_hmm(
                name,
                dynamics,
                cast(HMMConfig, config),
                obs_times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )
            marginal_loglik = loglik
            states = log_filt_seq
        elif isinstance(config, DiscreteTimeConfigs):
            marginal_loglik, states, filtered_dists = _filter_discrete_time(
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

        self.predicted_observations = extract_filter_predictions(
            states,
            dynamics=dynamics,
            filter_config=config,
            obs_times=obs_times,
            ctrl_values=ctrl_values,
        )

        self.marginal_loglik = marginal_loglik
        self.filtered_states = states
        self._filter_config_used = config

        return filtered_dists

    def _build_infer_result(
        self,
        times: Real[Array, "*time_plate time"] | None,
        filtered_dists: list | None,
    ) -> ConditionedResult:
        """Construct a ConditionedResult with deferred NumPyro registration."""
        marginal_loglik = self.marginal_loglik
        states = self.filtered_states
        config = self._filter_config_used
        predictions = self.predicted_observations
        _is_batched = (
            isinstance(marginal_loglik, jax.Array) and marginal_loglik.ndim > 0
        )

        def _register(site_name: str) -> None:
            if marginal_loglik is None or config is None:
                return
            if isinstance(config, HMMConfigs):
                register_hmm_filter_sites(
                    site_name,
                    marginal_loglik,
                    cast(jax.Array, states),
                    cast(HMMConfig, config),
                )
            elif _is_batched:
                # TODO: support per-field recording for batched (plate) states
                numpyro.factor(f"{site_name}_marginal_log_likelihood", marginal_loglik)
                numpyro.deterministic(f"{site_name}_marginal_loglik", marginal_loglik)
            else:
                register_filter_sites(site_name, marginal_loglik, states, config)
            add_observation_prediction_sites(
                site_name,
                filter_config=config,
                predictions=predictions,
            )

        return ConditionedResult(
            marginal_loglik=marginal_loglik,
            times=times,
            states=states,
            dists=filtered_dists,
            predicted_observations=predictions,
            _register_numpyro_sites=_register,
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
        _obs_values_filled: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_mask: Bool[Array, "*obs_value_plate obs_time observation_dim"]
        | Bool[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_has_missing: bool | None = None,
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
        uses_preprocessed_obs = False
        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            output_kind = "continuous"

            def _compute_output(dyn, ot, ov, ovf, om, ct, cv, k):
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
            uses_preprocessed_obs = True

            def _compute_output(dyn, ot, ov, ovf, om, ct, cv, k):
                return compute_hmm_filter(
                    dyn,
                    obs_times=ot,
                    obs_values=ov,
                    _obs_values_filled=ovf,
                    _obs_mask=om,
                    ctrl_values=cv,
                )

        elif isinstance(config, DiscreteTimeConfigs):
            if config.filter_source == "cuthbert":
                output_kind = "cuthbert"

                def _compute_output(dyn, ot, ov, ovf, om, ct, cv, k):
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

                def _compute_output(dyn, ot, ov, ovf, om, ct, cv, k):
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

        def compute_output(dyn, ot, ov, ovf, om, ct, cv, k):
            # Add scalar event axes after vmap removes plate dimensions.
            if not isinstance(config, HMMConfigs):
                ov = _ensure_trailing_event_axis(ov)
                if cv is not None:
                    cv = _ensure_trailing_event_axis(cv)
            return _compute_output(dyn, ot, ov, ovf, om, ct, cv, k)

        # Pre-split keys for all plate members (needed for stochastic filters).
        if key is not None:
            # Ensure we use typed PRNG keys so split returns shape (total,)
            # rather than old-style (total, 2).
            if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
                key = jax.random.wrap_key_data(key)
            total = math.prod(plate_shapes)
            split_keys = jax.random.split(key, total)
            keys = split_keys.reshape(*plate_shapes, *split_keys.shape[1:])
        else:
            keys = None

        # Build in_axes: same axes reused for each nested vmap.
        dyn_axes = _make_plate_in_axes(dynamics, plate_shapes)
        ot_axis = _array_plate_axis(obs_times, plate_shapes)
        ov_axis = _array_plate_axis(obs_values, plate_shapes)
        ct_axis = _array_plate_axis(ctrl_times, plate_shapes)
        cv_axis = _array_plate_axis(ctrl_values, plate_shapes)
        _validate_batched_plate_alignment(
            dynamics,
            plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
        k_axis = 0 if keys is not None else None
        if uses_preprocessed_obs:
            ovf_axis = _array_plate_axis(_obs_values_filled, plate_shapes)
            om_axis = _array_plate_axis(_obs_mask, plate_shapes)
        else:
            ovf_axis = None
            om_axis = None
            _obs_values_filled = None
            _obs_mask = None
        base_axes = (
            dyn_axes,
            ot_axis,
            ov_axis,
            ovf_axis,
            om_axis,
            ct_axis,
            cv_axis,
            k_axis,
        )

        # A plate-batched ``initial_condition`` cannot be sliced by vmap: numpyro
        # keeps ``batch_shape`` in static aux-data, so a vmap-sliced distribution
        # has a stale batch shape and its ``.mean``/``.sample``/``.log_prob``
        # re-expand to the full plate shape. Instead, thread a per-plate-member
        # index through the nested vmap and rebuild the member's initial condition
        # from the clean original (same reconstruction the simulator uses).
        ic_batched = _dist_has_plate_batch_dims(
            dynamics.initial_condition, plate_shapes
        )

        # Nest vmap for each plate dimension.
        # TODO: Allow for partial plate dimensions here.
        if ic_batched:
            orig_ic = dynamics.initial_condition

            def compute_output_member(dyn, ot, ov, ovf, om, ct, cv, k, *idxs):
                member_ic = _slice_dist_for_plate_member(
                    orig_ic, plate_shapes, tuple(idxs)
                )
                dyn = eqx.tree_at(
                    lambda m: m.initial_condition,
                    dyn,
                    member_ic,
                    is_leaf=lambda x: x is None,
                )
                return compute_output(dyn, ot, ov, ovf, om, ct, cv, k)

            idx_arrays = [jnp.arange(s) for s in plate_shapes]
            n_plates = len(plate_shapes)
            vmapped = compute_output_member
            for w in range(n_plates):
                d = n_plates - 1 - w
                # Wrap w (innermost-first) maps plate dim d = n_plates - 1 - w,
                # so the index array for that dim is mapped on axis 0 only at
                # that level while the other per-dimension index arrays stay
                # broadcasted scalars.
                idx_axes = tuple(0 if j == d else None for j in range(n_plates))
                vmapped = jax.vmap(vmapped, in_axes=(*base_axes, *idx_axes))
            outputs = vmapped(
                dynamics,
                obs_times,
                obs_values,
                _obs_values_filled,
                _obs_mask,
                ctrl_times,
                ctrl_values,
                keys,
                *idx_arrays,
            )
        else:
            vmapped = compute_output
            for _ in plate_shapes:
                vmapped = jax.vmap(vmapped, in_axes=base_axes)
            outputs = vmapped(
                dynamics,
                obs_times,
                obs_values,
                _obs_values_filled,
                _obs_mask,
                ctrl_times,
                ctrl_values,
                keys,
            )

        if output_kind in {"continuous", "cd_dynamax_discrete"}:
            marginal_logliks = outputs.marginal_loglik
            states = outputs
        elif output_kind == "hmm":
            marginal_logliks, states = outputs
            log_filt_seq = states
        elif output_kind == "cuthbert":
            marginal_logliks, states = outputs
        else:
            raise ValueError(f"Unsupported batched output kind: {output_kind}")

        self.marginal_loglik = marginal_logliks
        self.filtered_states = states
        self._filter_config_used = config

        self.predicted_observations = extract_filter_predictions(
            states,
            dynamics=dynamics,
            filter_config=config,
            obs_times=obs_times,
            ctrl_values=ctrl_values,
            plate_shapes=plate_shapes,
        )

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
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[
    Real[Array, ""] | None,
    object | None,
    list[numpyro.distributions.Distribution],
]:
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
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[
    Real[Array, ""],
    object,
    list[numpyro.distributions.Distribution],
]:
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
    "ConstructCholInnovationCovariance",
    "ContinuousTimeKFConfig",
    "ContinuousTimeDPFConfig",
    "ContinuousTimeEnKFConfig",
    "ContinuousTimeEKFConfig",
    "ContinuousTimeUKFConfig",
    "EKFConfig",
    "EnKFConfig",
    "EnKFLocalizationConfig",
    "EnKFLocalizationFunctions",
    "Filter",
    "HMMConfig",
    "HMMConfigs",
    "KFConfig",
    "ModifyCrossCovariance",
    "ModifyPredictedObservationCovariance",
    "PFConfig",
    "PFResamplingConfig",
    "TaperCovarianceFn",
    "UKFConfig",
]
