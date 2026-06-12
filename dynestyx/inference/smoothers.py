import dataclasses
import math
from abc import ABC, abstractmethod
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.checkers import (
    _validate_batched_plate_alignment,
    _validate_missing_observation_support,
)
from dynestyx.inference.distribution_utils import (
    _cholesky_state_sequence_to_dists,
    _posterior_sequence_to_dists,
)
from dynestyx.inference.integrations.cd_dynamax.continuous_smoother import (
    compute_continuous_smoother,
    run_continuous_smoother,
)
from dynestyx.inference.integrations.cd_dynamax.discrete_smoother import (
    compute_cd_dynamax_discrete_smoother,
)
from dynestyx.inference.integrations.cd_dynamax.discrete_smoother import (
    run_discrete_smoother as run_cd_dynamax_discrete_smoother,
)
from dynestyx.inference.integrations.cuthbert.discrete_smoother import (
    CuthbertSmootherConfig,
    compute_cuthbert_smoother,
)
from dynestyx.inference.integrations.cuthbert.discrete_smoother import (
    run_discrete_smoother as run_cuthbert_discrete_smoother,
)
from dynestyx.inference.numpyro_sites import register_smoother_sites
from dynestyx.inference.plate_utils import (
    _array_plate_axis,
    _make_plate_in_axes,
    _slice_dist_for_plate_member,
)
from dynestyx.inference.smoother_configs import (
    BaseSmootherConfig,
    ContinuousTimeEKFSmootherConfig,
    ContinuousTimeKFSmootherConfig,
    ContinuousTimeSmootherConfigs,
    DiscreteTimeSmootherConfigs,
    EKFSmootherConfig,
    KFSmootherConfig,
    PFSmootherConfig,
    UKFSmootherConfig,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import ConditionedResult, FunctionOfTime
from dynestyx.utils import _dist_has_plate_batch_dims

DiscreteSmootherConfig = (
    KFSmootherConfig | EKFSmootherConfig | UKFSmootherConfig | PFSmootherConfig
)
ContinuousSmootherConfig = (
    ContinuousTimeKFSmootherConfig | ContinuousTimeEKFSmootherConfig
)
SmootherAnyConfig = DiscreteSmootherConfig | ContinuousSmootherConfig


def _default_smoother_config(dynamics: DynamicalModel) -> SmootherAnyConfig:
    """Return an appropriate default smoother config when none is specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEKFSmootherConfig()
    return EKFSmootherConfig(filter_source="cuthbert")


def _valid_smoother_config_names(*, continuous_time: bool) -> list[str]:
    if continuous_time:
        return [c.__name__ for c in ContinuousTimeSmootherConfigs]
    return [c.__name__ for c in DiscreteTimeSmootherConfigs]


def _validate_future_only_predict_times(
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None,
) -> Real[Array, "*predict_time_plate predict_time"] | None:
    """Validate the current smoother prediction contract."""
    if predict_times is None or obs_times is None:
        return predict_times
    obs_end = obs_times[..., -1:]
    _ = eqx.error_if(
        predict_times,
        jnp.any(predict_times < obs_end),
        "Smoother prediction only supports predict_times >= max(obs_times); in-window smoothing predictions are not implemented yet. Please use `Filter` for in-window predictions for now.",
    )
    return predict_times


def _final_obs_times_for_rollout(
    obs_times: Real[Array, "*obs_time_plate obs_time"],
) -> Real[Array, "*obs_time_plate one"]:
    """Return the final observation time while keeping simulator segmentation host-safe."""
    try:
        obs_times_host = np.asarray(jax.device_get(obs_times))
        return jnp.asarray(obs_times_host[..., -1:], dtype=obs_times.dtype)
    except Exception:
        return obs_times[..., -1:]


class BaseSmootherLogFactorAdder(ObjectInterpretation, HandlesSelf, ABC):
    """Base class for smoother handlers."""

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
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        **kwargs,
    ) -> FunctionOfTime:
        smoothed_dists = None
        if not (obs_times is None or obs_values is None):
            smoothed_dists = self._add_log_factors(
                name,
                dynamics,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )

        predict_times = _validate_future_only_predict_times(predict_times, obs_times)
        filtered_times = None
        filtered_dists = None
        posterior_rollout_final_only = False
        smoothed_times = obs_times
        if predict_times is not None and smoothed_dists:
            assert obs_times is not None
            filtered_times = _final_obs_times_for_rollout(obs_times)
            filtered_dists = [smoothed_dists[-1]]
            posterior_rollout_final_only = True
            smoothed_times = None
            smoothed_dists = None

        # fwd() lets handlers above (e.g. Simulator) use smoothed_dists for rollout.
        fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=None,
            obs_values=None,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            filtered_times=filtered_times,
            filtered_dists=filtered_dists,
            smoothed_times=smoothed_times,
            smoothed_dists=smoothed_dists,
            _posterior_rollout_final_only=posterior_rollout_final_only,
            **kwargs,
        )

        return self._build_infer_result(name, smoothed_dists)

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
        self, name: str, smoothed_dists: list | None
    ) -> ConditionedResult: ...


@dataclasses.dataclass
class Smoother(BaseSmootherLogFactorAdder):
    r"""Performs Bayesian smoothing to compute the smoothing distribution p(x_t | y_{1:T})."""

    smoother_config: SmootherAnyConfig | None = None
    marginal_loglik: jax.Array | None = dataclasses.field(
        default=None, repr=False, init=False
    )
    smoothed_states: object = dataclasses.field(default=None, repr=False, init=False)
    _smoother_config_used: BaseSmootherConfig | None = dataclasses.field(
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
            raise ValueError("obs_times and obs_values are required for smoothing.")

        config = (
            self.smoother_config
            if self.smoother_config is not None
            else _default_smoother_config(dynamics)
        )
        if not isinstance(config, BaseSmootherConfig):
            valid = _valid_smoother_config_names(
                continuous_time=dynamics.continuous_time
            )
            raise ValueError(
                f"Invalid smoother config: {type(config).__name__}. "
                "Expected a smoother config class from dynestyx.inference.smoother_configs. "
                f"Valid types: {valid}"
            )
        _validate_missing_observation_support(
            config,
            obs_values=obs_values,
            mode="smoother",
        )

        # Resolve PRNG key: use explicit seed from config, fall back to numpyro
        # context (inside a seeded model), or None (deterministic smoothers don't need one).
        typed_config = config
        if typed_config.crn_seed is not None:
            key = typed_config.crn_seed
        else:
            import warnings  # noqa: PLC0415

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                key = numpyro.prng_key()  # returns None outside seed handler

        if plate_shapes:
            return self._add_log_factors_batched(
                name,
                dynamics,
                typed_config,
                key=key,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

        if dynamics.continuous_time:
            if not isinstance(typed_config, ContinuousTimeSmootherConfigs):
                valid = _valid_smoother_config_names(continuous_time=True)
                raise ValueError(
                    f"Invalid smoother config: {type(typed_config).__name__}. "
                    f"Valid continuous-time config types: {valid}"
                )
            continuous_config = cast(ContinuousSmootherConfig, typed_config)
            marginal_loglik, states, smoothed_dists = _smooth_continuous_time(
                name,
                dynamics,
                continuous_config,
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )
        elif not isinstance(typed_config, DiscreteTimeSmootherConfigs):
            valid = _valid_smoother_config_names(continuous_time=False)
            raise ValueError(
                f"Invalid smoother config: {type(typed_config).__name__}. "
                f"Valid discrete-time config types: {valid}"
            )
        else:
            discrete_config = cast(DiscreteSmootherConfig, typed_config)
            marginal_loglik, states, smoothed_dists = _smooth_discrete_time(
                name,
                dynamics,
                discrete_config,
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )

        self.marginal_loglik = marginal_loglik
        self.smoothed_states = states
        self._smoother_config_used = typed_config

        return smoothed_dists

    def _build_infer_result(
        self, name: str, smoothed_dists: list | None
    ) -> ConditionedResult:
        """Construct ConditionedResult with a deferred numpyro registration callback."""
        marginal_loglik = self.marginal_loglik
        states = self.smoothed_states
        config = self._smoother_config_used
        _is_batched = (
            isinstance(marginal_loglik, jax.Array) and marginal_loglik.ndim > 0
        )

        def _register(site_name: str) -> None:
            if marginal_loglik is None or config is None:
                return
            if _is_batched:
                # TODO: support per-field recording for batched (plate) states
                numpyro.factor(f"{site_name}_marginal_log_likelihood", marginal_loglik)
                numpyro.deterministic(f"{site_name}_marginal_loglik", marginal_loglik)
            else:
                register_smoother_sites(site_name, marginal_loglik, states, config)

        return ConditionedResult(
            marginal_loglik=marginal_loglik,
            states=states,
            dists=smoothed_dists,
            _register_numpyro_sites=_register,
        )

    def _add_log_factors_batched(
        self,
        name: str,
        dynamics: DynamicalModel,
        config: SmootherAnyConfig,
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
        """Compute batched marginal log-likelihoods via vmap for plate contexts."""
        output_kind: str
        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeSmootherConfigs):
                valid = _valid_smoother_config_names(continuous_time=True)
                raise ValueError(
                    f"Invalid smoother config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            continuous_config = cast(ContinuousSmootherConfig, config)
            output_kind = "continuous"

            def compute_output(dyn, ot, ov, ct, cv, k):
                return compute_continuous_smoother(
                    dyn,
                    continuous_config,
                    k,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_times=ct,
                    ctrl_values=cv,
                )

        elif isinstance(config, DiscreteTimeSmootherConfigs):
            discrete_config = cast(DiscreteSmootherConfig, config)
            if discrete_config.filter_source == "cuthbert":
                if isinstance(discrete_config, UKFSmootherConfig):
                    raise ValueError(
                        "UKF smoothing is not available in cuthbert. "
                        "Use UKFSmootherConfig(filter_source='cd_dynamax')."
                    )
                cuthbert_config = cast(CuthbertSmootherConfig, discrete_config)
                output_kind = "cuthbert"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cuthbert_smoother(
                        dyn,
                        cuthbert_config,
                        k,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            elif discrete_config.filter_source == "cd_dynamax":
                output_kind = "cd_dynamax_discrete"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cd_dynamax_discrete_smoother(
                        dyn,
                        discrete_config,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            else:
                raise ValueError(
                    f"Unknown filter source: {discrete_config.filter_source}"
                )
        else:
            raise ValueError(
                f"Unsupported smoother config for plate: {type(config).__name__}"
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

        # A plate-batched ``initial_condition`` cannot be sliced by vmap (numpyro
        # keeps ``batch_shape`` in static aux-data). Thread a per-plate-member index
        # through the nested vmap and rebuild the member's initial condition from the
        # clean original. See ``Filter._add_log_factors_batched`` for details.
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
        elif output_kind == "cuthbert":
            marginal_logliks, states = outputs
        else:
            raise ValueError(f"Unsupported batched output kind: {output_kind}")

        self.marginal_loglik = marginal_logliks
        self.smoothed_states = outputs
        self._smoother_config_used = config

        if output_kind == "continuous":
            return _posterior_sequence_to_dists(
                outputs,
                means_attr="smoothed_means",
                covariances_attr="smoothed_covariances",
                plate_shapes=plate_shapes,
                particle_mode=False,
                missing_message=(
                    "Smoothed means/covariances were unavailable for a Gaussian rollout path."
                ),
            )
        if output_kind == "cd_dynamax_discrete":
            return _posterior_sequence_to_dists(
                outputs,
                means_attr="smoothed_means",
                covariances_attr="smoothed_covariances",
                plate_shapes=plate_shapes,
                particle_mode=False,
                missing_message=(
                    "Smoothed means/covariances were unavailable for a Gaussian rollout path."
                ),
            )
        if output_kind == "cuthbert":
            return _cholesky_state_sequence_to_dists(
                states,
                particle_mode=isinstance(config, PFSmootherConfig),
                plate_shapes=plate_shapes,
            )

        raise ValueError(f"Unsupported batched output kind: {output_kind}")


def _smooth_discrete_time(
    name: str,
    dynamics: DynamicalModel,
    smoother_config: DiscreteSmootherConfig,
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
    """Discrete-time marginal likelihood via cuthbert or cd-dynamax smoothers."""

    if isinstance(smoother_config, UKFSmootherConfig) and (
        smoother_config.filter_source == "cuthbert"
    ):
        raise ValueError(
            "UKF smoothing is not available in cuthbert. "
            "Use UKFSmootherConfig(filter_source='cd_dynamax') or a cuthbert-supported smoother "
            "(KFSmootherConfig, EKFSmootherConfig, PFSmootherConfig)."
        )

    if isinstance(smoother_config, PFSmootherConfig) and (
        smoother_config.filter_source != "cuthbert"
    ):
        raise ValueError(
            "PFSmootherConfig is only supported with filter_source='cuthbert'."
        )

    if smoother_config.filter_source == "cd_dynamax":
        marginal_loglik, states, smoothed_dists = run_cd_dynamax_discrete_smoother(
            name,
            dynamics,
            smoother_config,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    elif smoother_config.filter_source == "cuthbert":
        if isinstance(smoother_config, UKFSmootherConfig):
            raise ValueError(
                "UKF smoothing is not available in cuthbert. "
                "Use UKFSmootherConfig(filter_source='cd_dynamax') or a cuthbert-supported smoother "
                "(KFSmootherConfig, EKFSmootherConfig, PFSmootherConfig)."
            )
        marginal_loglik, states, smoothed_dists = run_cuthbert_discrete_smoother(
            name,
            dynamics,
            smoother_config,
            key=key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown filter source: {smoother_config.filter_source}")

    return marginal_loglik, states, smoothed_dists


def _smooth_continuous_time(
    name: str,
    dynamics: DynamicalModel,
    smoother_config: ContinuousSmootherConfig,
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
    """Continuous-time marginal likelihood via CD-Dynamax smoothers."""
    if smoother_config.filter_source != "cd_dynamax":
        raise ValueError(
            f"{type(smoother_config).__name__} supports only filter_source='cd_dynamax'."
        )

    marginal_loglik, smoothed, smoothed_dists = run_continuous_smoother(
        name,
        dynamics,
        smoother_config,
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
    )
    return marginal_loglik, smoothed, smoothed_dists


__all__ = [
    "ContinuousTimeEKFSmootherConfig",
    "ContinuousTimeKFSmootherConfig",
    "EKFSmootherConfig",
    "KFSmootherConfig",
    "PFSmootherConfig",
    "Smoother",
    "BaseSmootherConfig",
    "UKFSmootherConfig",
]
