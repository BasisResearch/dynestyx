import dataclasses
import math
from abc import ABC, abstractmethod
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
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
from dynestyx.inference.numpyro_sites import (
    register_filter_sites,
    register_hmm_filter_sites,
)
from dynestyx.inference.plate_utils import (
    _array_plate_axis,
    _make_plate_in_axes,
    _slice_dist_for_plate_member,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import ConditionedResult, FunctionOfTime
from dynestyx.utils import _dist_has_plate_batch_dims

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class BaseLogFactorAdder(ObjectInterpretation, HandlesSelf, ABC):
    """Base for filter handlers."""

    @implements(_condition_intp)
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

        fwd(
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

        return self._build_infer_result(name, filtered_dists)

    @abstractmethod
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
    ) -> list[numpyro.distributions.Distribution] | None: ...

    @abstractmethod
    def _build_infer_result(
        self, name: str, filtered_dists: list | None
    ) -> ConditionedResult: ...


def _default_filter_config(dynamics: DynamicalModel):
    if dynamics.continuous_time:
        return ContinuousTimeEnKFConfig()
    return EnKFConfig()


@dataclasses.dataclass
class Filter(BaseLogFactorAdder):
    filter_config: BaseFilterConfig | None = None
    marginal_loglik: jax.Array | None = dataclasses.field(
        default=None, repr=False, init=False
    )
    filtered_states: object = dataclasses.field(default=None, repr=False, init=False)
    _filter_config_used: BaseFilterConfig | None = dataclasses.field(
        default=None, repr=False, init=False
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
        if obs_times is None or obs_values is None:
            raise ValueError("obs_times and obs_values are required for filtering.")

        config = self.filter_config or _default_filter_config(dynamics)
        if isinstance(config, BaseFilterConfig):
            _validate_missing_observation_support(
                config,
                obs_values=obs_values,
                mode="filter",
            )

        if config.crn_seed is not None:
            key = config.crn_seed
        else:
            import warnings  # noqa: PLC0415

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                key = numpyro.prng_key()

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
            marginal_loglik, states, filtered_dists = _filter_continuous_time(
                name,
                dynamics,
                config,
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
                config,
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

        self.marginal_loglik = marginal_loglik
        self.filtered_states = states
        self._filter_config_used = config
        return filtered_dists

    def _build_infer_result(
        self, name: str, filtered_dists: list | None
    ) -> ConditionedResult:
        marginal_loglik = self.marginal_loglik
        states = self.filtered_states
        config = self._filter_config_used
        is_batched = isinstance(marginal_loglik, jax.Array) and marginal_loglik.ndim > 0

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
            elif is_batched:
                numpyro.factor(f"{site_name}_marginal_log_likelihood", marginal_loglik)
                numpyro.deterministic(f"{site_name}_marginal_loglik", marginal_loglik)
            else:
                register_filter_sites(site_name, marginal_loglik, states, config)

        return ConditionedResult(
            marginal_loglik=marginal_loglik,
            states=states,
            dists=filtered_dists,
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
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
    ) -> list[numpyro.distributions.Distribution]:
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

        if key is not None:
            if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
                key = jax.random.wrap_key_data(key)
            total = math.prod(plate_shapes)
            split_keys = jax.random.split(key, total)
            keys = split_keys.reshape(*plate_shapes, *split_keys.shape[1:])
        else:
            keys = None

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
        base_axes = (dyn_axes, ot_axis, ov_axis, ct_axis, cv_axis, k_axis)

        ic_batched = _dist_has_plate_batch_dims(
            dynamics.initial_condition, plate_shapes
        )

        if ic_batched:
            orig_ic = dynamics.initial_condition

            def compute_output_member(dyn, ot, ov, ct, cv, k, *idxs):
                member_ic = _slice_dist_for_plate_member(
                    orig_ic, plate_shapes, tuple(idxs)
                )
                dyn = eqx.tree_at(
                    lambda m: m.initial_condition,
                    dyn,
                    member_ic,
                    is_leaf=lambda x: x is None,
                )
                return compute_output(dyn, ot, ov, ct, cv, k)

            idx_arrays = [jnp.arange(s) for s in plate_shapes]
            n_plates = len(plate_shapes)
            vmapped = compute_output_member
            for w in range(n_plates):
                d = n_plates - 1 - w
                idx_axes = tuple(0 if j == d else None for j in range(n_plates))
                vmapped = jax.vmap(vmapped, in_axes=(*base_axes, *idx_axes))
            outputs = vmapped(
                dynamics,
                obs_times,
                obs_values,
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
                ctrl_times,
                ctrl_values,
                keys,
            )

        if output_kind in {"continuous", "cd_dynamax_discrete"}:
            marginal_logliks = outputs.marginal_loglik
        elif output_kind == "hmm":
            marginal_logliks, log_filt_seq = outputs
        elif output_kind == "cuthbert":
            marginal_logliks, states = outputs
        else:
            raise ValueError(f"Unsupported batched output kind: {output_kind}")

        self.marginal_loglik = marginal_logliks
        if output_kind == "hmm":
            self.filtered_states = log_filt_seq
        elif output_kind == "cuthbert":
            self.filtered_states = states
        else:
            self.filtered_states = outputs
        self._filter_config_used = config

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
) -> tuple[jax.Array | None, object | None, list[numpyro.distributions.Distribution]]:
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
    if filter_config.filter_source == "cuthbert":
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
) -> tuple[jax.Array, object, list[numpyro.distributions.Distribution]]:
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
