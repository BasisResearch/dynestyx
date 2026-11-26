import dataclasses
import warnings
from typing import cast

import jax
import numpyro
from cuthbert.inference import Filter as CuthbertFilter
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Bool, Real

from dynestyx.configs.filters import (
    BaseFilterConfig,
    CombineFilterConfig,
    ConstructCholInnovationCovariance,
    ContinuousTimeConfigs,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
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
    SplitFilterConfig,
    TaperCovarianceFn,
    UKFConfig,
)
from dynestyx.handlers import (
    HandlesSelf,
    _condition_intp,
    _dynestyx_stack_kind,
    _DynestyxStackKind,
)
from dynestyx.inference.checkers import (
    _validate_inference_supported_model_classes,
    _validate_missing_observation_support,
)
from dynestyx.inference.custom_filter import (
    FilterOutputs,
    compute_custom_filter,
    custom_distributions,
    step_inputs,
)
from dynestyx.inference.enkf_localization import (
    resolve_enkf_localization,
)
from dynestyx.inference.hmm_filters import compute_hmm_filter
from dynestyx.inference.integrations.cd_dynamax.filter_steps import (
    CDFilter,
    CDPosterior,
)
from dynestyx.inference.integrations.cuthbert.discrete_filter import run_cuthbert_steps
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
from dynestyx.inference.utils.plate_utils import _map_inference
from dynestyx.models import DynamicalModel
from dynestyx.types import (
    ConditionedResult,
    chain_numpyro_site_registrations,
)
from dynestyx.utils.arrays import _ensure_trailing_event_axis


def _default_filter_config(dynamics: DynamicalModel) -> BaseFilterConfig:
    """Return appropriate default filter config when none specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEnKFConfig()

    return EnKFConfig()


@dataclasses.dataclass
class Filter(ObjectInterpretation, HandlesSelf):
    r"""Performs Bayesian filtering to compute the filtering distribution $p(x_t | y_{1:t})$ and the marginal likelihood $\log p(y_{1:T})$.

    A `Filter` object should be used as a context manager around a call to a model with a `dsx.sample(...)` statement
    to condition a dynamical model on observations via a filtering algorithm. The filter
    is selected and dispatched according to the `filter_config` argument, which adds the
    marginal log-likelihood as a NumPyro factor, allowing for downstream parameter inference.

    Examples:
        >>> import dynestyx as dsx
        >>> from dynestyx import DynamicalModel, Filter
        >>> from dynestyx.configs import KFConfig
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
    [Available filter configurations](../configurations/filter_configs.md).

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

    @implements(_dynestyx_stack_kind)
    def _stack_kind(self, **kwargs):
        return [_DynestyxStackKind.FILTER, *fwd()]

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
    ) -> ConditionedResult:
        if filtered_result is not None or smoothed_result is not None:
            raise ValueError(
                "Filter cannot condition an already conditioned result. Use only "
                "one Filter or Smoother for a dsx.condition/dsx.sample operation."
            )

        filtered_dists = None
        self.marginal_loglik = self.filtered_states = self._filter_config_used = None
        self.predicted_observations = None
        if obs_times is not None and obs_values is not None:
            filtered_dists = self._filter(
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

    def _filter(
        self,
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, "*obs_time_plate obs_time"],
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"],
        _obs_values_filled: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_mask: Bool[Array, "*obs_value_plate obs_time observation_dim"]
        | Bool[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        """Compute the filter result. NumPyro sites are registered by dsx.sample."""
        _validate_inference_supported_model_classes(dynamics)

        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )
        obs_values = _validate_missing_observation_support(
            config, obs_values=obs_values, mode="filter"
        )
        # Resolve PRNG key: use explicit seed from config, fall back to numpyro
        # context (inside a seeded model), or None (deterministic filters don't need one).
        if config.crn_seed is not None:
            key = config.crn_seed
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                key = numpyro.prng_key()  # returns None outside seed handler

        custom = isinstance(config, (SplitFilterConfig, CombineFilterConfig))
        hmm = isinstance(config, HMMConfigs)
        if (
            dynamics.continuous_time
            and not custom
            and not isinstance(config, ContinuousTimeConfigs)
        ):
            raise ValueError(
                "Continuous-time models require a continuous-time filter config. "
                "To use a discrete-time filter, nest Discretizer() inside Filter()."
            )
        if not dynamics.continuous_time and isinstance(config, ContinuousTimeConfigs):
            raise ValueError(f"Invalid filter config: {type(config).__name__}")
        if (
            not plate_shapes
            and obs_values.shape[0] == 0
            and config.filter_source == "cuthbert"
            and not custom
        ):
            self._filter_config_used = config
            return []

        if key is None and isinstance(
            config,
            (PFConfig, EnKFConfig, ContinuousTimeDPFConfig, ContinuousTimeEnKFConfig),
        ):
            raise ValueError(
                f"{type(config).__name__} requires a PRNG key: set 'crn_seed' or run inside a NumPyro seeded context."
            )

        resolved_localization = None
        binding_config = config
        if isinstance(config, EnKFConfig):
            options = dict(
                config.extra_filter_kwargs,
                store_predicted_ensemble=config.include_predicted_observations,
            )
            if config.localization is not None:
                resolved_localization = resolve_enkf_localization(
                    config.localization,
                    state_dim=dynamics.state_dim,
                    observation_dim=dynamics.observation_dim,
                )
                options["_resolved_enkf_localization"] = resolved_localization
            binding_config = dataclasses.replace(config, extra_filter_kwargs=options)

        def compute_output(dyn, ot, ov, ovf, om, ct, cv, k):
            if hmm:
                return compute_hmm_filter(
                    dyn,
                    obs_times=ot,
                    obs_values=ov,
                    _obs_values_filled=ovf,
                    _obs_mask=om,
                    ctrl_values=cv,
                )
            ov = _ensure_trailing_event_axis(ov)
            if cv is not None:
                cv = _ensure_trailing_event_axis(cv)
            if isinstance(config, (SplitFilterConfig, CombineFilterConfig)):
                outputs = compute_custom_filter(dyn, config, ot, ov, ct, cv, k)
                return outputs.log_normalizing_constant[-1], outputs
            inputs = step_inputs(dyn, ot, ov, ct, cv)
            algorithm = binding_config.build_filter(dyn)
            if isinstance(algorithm, CDFilter):
                posterior = algorithm.run(inputs, k)
                return posterior.marginal_loglik, posterior
            if isinstance(algorithm, CuthbertFilter):
                return run_cuthbert_steps(
                    algorithm,
                    inputs,
                    k,
                    parallel=isinstance(config, KFConfig) and bool(config.associative),
                )
            raise TypeError(
                f"Unsupported built-in filter returned by {type(config).__name__}."
            )

        outputs = _map_inference(
            compute_output,
            dynamics,
            plate_shapes,
            obs_times,
            obs_values,
            ctrl_times,
            ctrl_values,
            key,
            _obs_values_filled if hmm else None,
            _obs_mask if hmm else None,
        )

        marginal_logliks, states = outputs
        self.marginal_loglik = marginal_logliks
        self.filtered_states = states
        self._filter_config_used = config

        if isinstance(states, FilterOutputs):
            self.predicted_observations = states.predicted_observations
            return custom_distributions(states, plate_shapes)

        self.predicted_observations = extract_filter_predictions(
            states,
            dynamics=dynamics,
            filter_config=config,
            obs_times=obs_times,
            ctrl_values=ctrl_values,
            plate_shapes=plate_shapes,
            resolved_localization=resolved_localization,
        )

        if isinstance(states, CDPosterior):
            return _posterior_sequence_to_dists(
                states,
                particle_mode=isinstance(config, ContinuousTimeDPFConfig),
                plate_shapes=plate_shapes,
                missing="empty"
                if not dynamics.continuous_time and not plate_shapes
                else "raise",
                missing_message="Filtered means/covariances were unavailable for a Gaussian rollout path.",
            )
        if hmm:
            return _categorical_log_probs_to_dists(
                states,
                plate_shapes=plate_shapes,
            )
        return _cholesky_state_sequence_to_dists(
            states,
            particle_mode=isinstance(config, PFConfig),
            plate_shapes=plate_shapes,
            covariance_jitter=getattr(
                config, "recorded_filtered_states_cov_jitter", 0.0
            ),
        )

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
            elif _is_batched and not isinstance(
                config, (SplitFilterConfig, CombineFilterConfig)
            ):
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
            extra_outputs=(states.extra_outputs or {})
            if isinstance(states, FilterOutputs)
            else {},
            dists=filtered_dists,
            predicted_observations=predictions,
            _register_numpyro_sites=_register,
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
