import dataclasses
import math

import jax
import jax.numpy as jnp
import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.inference.checkers import _validate_batched_plate_alignment
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
from dynestyx.inference.filters import (
    _array_plate_axis,
    _make_plate_in_axes,
    _particle_to_batched_dists,
    _slice_time_axis,
    _time_len_from_array,
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
    compute_cuthbert_smoother,
)
from dynestyx.inference.integrations.cuthbert.discrete_smoother import (
    run_discrete_smoother as run_cuthbert_discrete_smoother,
)
from dynestyx.inference.smoother_configs import SmootherOptions
from dynestyx.models import DynamicalModel
from dynestyx.types import FunctionOfTime
from dynestyx.utils import _ensure_continuous_bm_dim


def _default_smoother_config(dynamics: DynamicalModel):
    """Return an appropriate default smoother config when none is specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEKFConfig()
    return EKFConfig(filter_source="cuthbert")


def _unsupported_smoother_message(config: BaseFilterConfig) -> str | None:
    """Return an actionable unsupported message when a smoother is out of scope."""
    if isinstance(config, HMMConfig):
        return (
            "HMM smoothing is not supported in this release. "
            "Supported smoothers are for Gaussian/PF state-space models only."
        )

    if isinstance(
        config,
        (ContinuousTimeEnKFConfig, ContinuousTimeUKFConfig, ContinuousTimeDPFConfig),
    ):
        return (
            f"{type(config).__name__} smoothing is not supported in this release. "
            "Supported continuous-time smoothers: ContinuousTimeKFConfig, ContinuousTimeEKFConfig (cd_dynamax)."
        )

    if isinstance(config, EnKFConfig):
        return (
            "EnKF smoothing is not supported for discrete-time models in this release. "
            "Use KFConfig/EKFConfig/UKFConfig with filter_source='cd_dynamax', "
            "or KFConfig/EKFConfig/PFConfig with filter_source='cuthbert'."
        )

    if isinstance(config, UKFConfig) and config.filter_source == "cuthbert":
        return (
            "UKF smoothing is not available in cuthbert. "
            "Use UKFConfig(filter_source='cd_dynamax') or a cuthbert-supported smoother "
            "(KFConfig, EKFConfig, PFConfig)."
        )

    return None


def _smoothed_posterior_to_dists(
    posterior,
    *,
    plate_shapes: tuple[int, ...],
    particle_mode: bool,
):
    """Convert vmapped smoother posterior objects to per-time distributions."""
    if particle_mode:
        particles = posterior.particles
        log_weights = posterior.log_weights
        return _particle_to_batched_dists(
            particles,
            log_weights,
            plate_shapes=plate_shapes,
        )

    means = posterior.smoothed_means
    covs = posterior.smoothed_covariances
    if means is None or covs is None:
        raise ValueError(
            "Smoothed means/covariances were unavailable for a Gaussian rollout path."
        )
    t_len = _time_len_from_array(means, plate_shapes)
    return [
        numpyro.distributions.MultivariateNormal(
            _slice_time_axis(means, t, plate_shapes),
            covariance_matrix=_slice_time_axis(covs, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _cuthbert_smoothed_states_to_dists(
    states,
    config: BaseFilterConfig,
    *,
    plate_shapes: tuple[int, ...],
):
    """Convert vmapped cuthbert smoother outputs to per-time smoothed distributions."""
    if isinstance(config, PFConfig):
        particles = states.particles
        log_weights = states.log_weights
        particles = particles[
            (slice(None),) * len(plate_shapes) + (slice(1, None), ...)
        ]
        log_weights = log_weights[
            (slice(None),) * len(plate_shapes) + (slice(1, None), ...)
        ]
        return _particle_to_batched_dists(
            particles,
            log_weights,
            plate_shapes=plate_shapes,
        )

    mean = states.mean[(slice(None),) * len(plate_shapes) + (slice(1, None), ...)]
    chol_cov = states.chol_cov[
        (slice(None),) * len(plate_shapes) + (slice(1, None), ...)
    ]
    cov = jnp.matmul(chol_cov, jnp.swapaxes(chol_cov, -1, -2))
    t_len = _time_len_from_array(mean, plate_shapes)
    return [
        numpyro.distributions.MultivariateNormal(
            _slice_time_axis(mean, t, plate_shapes),
            covariance_matrix=_slice_time_axis(cov, t, plate_shapes),
        )
        for t in range(t_len)
    ]


class BaseSmootherLogFactorAdder(ObjectInterpretation, HandlesSelf):
    """Base class for smoother handlers."""

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
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

        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=None,
            obs_values=None,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            smoothed_times=obs_times,
            smoothed_dists=smoothed_dists,
            **kwargs,
        )

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        raise NotImplementedError()


@dataclasses.dataclass
class Smoother(BaseSmootherLogFactorAdder):
    r"""Performs Bayesian smoothing to compute the smoothing distribution p(x_t | y_{1:T})."""

    filter_config: BaseFilterConfig | None = None
    smoother_options: SmootherOptions = dataclasses.field(
        default_factory=SmootherOptions
    )

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: jax.Array | None = None,
        obs_values: jax.Array | None = None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        if obs_times is None or obs_values is None:
            raise ValueError("obs_times and obs_values are required for smoothing.")

        dynamics = _ensure_continuous_bm_dim(dynamics)

        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_smoother_config(dynamics)
        )

        unsupported = _unsupported_smoother_message(config)
        if unsupported is not None:
            raise ValueError(unsupported)

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
                    f"Invalid smoother config: {type(config).__name__}. "
                    f"Valid continuous-time config types: {valid}"
                )
            return _smooth_continuous_time(
                name,
                dynamics,
                config,
                self.smoother_options,
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )

        if isinstance(config, HMMConfigs):
            raise ValueError(
                "HMM smoothing is not supported in this release. "
                "Use Filter(HMMConfig()) for filtering only."
            )

        if isinstance(config, DiscreteTimeConfigs):
            return _smooth_discrete_time(
                name,
                dynamics,
                config,
                self.smoother_options,
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )

        valid = [c.__name__ for c in HMMConfigs + DiscreteTimeConfigs]
        raise ValueError(
            f"Invalid smoother config: {type(config).__name__}. "
            f"Valid discrete-time config types: {valid}"
        )

    def _add_log_factors_batched(
        self,
        name: str,
        dynamics: DynamicalModel,
        config: BaseFilterConfig,
        *,
        key: jax.Array | None,
        plate_shapes: tuple[int, ...],
        obs_times: jax.Array,
        obs_values: jax.Array,
        ctrl_times=None,
        ctrl_values=None,
    ) -> list[numpyro.distributions.Distribution]:
        """Compute batched marginal log-likelihoods via vmap for plate contexts."""
        output_kind: str
        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid smoother config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            output_kind = "continuous"

            def compute_output(dyn, ot, ov, ct, cv, k):
                return compute_continuous_smoother(
                    dyn,
                    config,
                    self.smoother_options,
                    k,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_times=ct,
                    ctrl_values=cv,
                )

        elif isinstance(config, DiscreteTimeConfigs):
            if config.filter_source == "cuthbert":
                output_kind = "cuthbert"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cuthbert_smoother(
                        dyn,
                        config,
                        self.smoother_options,
                        k,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            elif config.filter_source == "cd_dynamax":
                output_kind = "cd_dynamax_discrete"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cd_dynamax_discrete_smoother(
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

        vmapped = compute_output
        for _ in plate_shapes:
            vmapped = jax.vmap(
                vmapped,
                in_axes=(dyn_axes, ot_axis, ov_axis, ct_axis, cv_axis, k_axis),
            )

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

        numpyro.factor(f"{name}_marginal_log_likelihood", marginal_logliks)
        numpyro.deterministic(f"{name}_marginal_loglik", marginal_logliks)

        if output_kind == "continuous":
            return _smoothed_posterior_to_dists(
                outputs,
                plate_shapes=plate_shapes,
                particle_mode=False,
            )
        if output_kind == "cd_dynamax_discrete":
            return _smoothed_posterior_to_dists(
                outputs,
                plate_shapes=plate_shapes,
                particle_mode=False,
            )
        if output_kind == "cuthbert":
            return _cuthbert_smoothed_states_to_dists(
                states,
                config,
                plate_shapes=plate_shapes,
            )

        raise ValueError(f"Unsupported batched output kind: {output_kind}")


def _smooth_discrete_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    smoother_options: SmootherOptions,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Discrete-time marginal likelihood via cuthbert or cd-dynamax smoothers."""

    if isinstance(filter_config, EnKFConfig):
        raise ValueError(
            "Discrete EnKF smoothing is not supported. "
            "Use KFConfig/EKFConfig/UKFConfig (cd_dynamax) or KFConfig/EKFConfig/PFConfig (cuthbert)."
        )

    if filter_config.filter_source == "cd_dynamax":
        return run_cd_dynamax_discrete_smoother(
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
        return run_cuthbert_discrete_smoother(
            name,
            dynamics,
            filter_config,
            smoother_options,
            key=key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    raise ValueError(f"Unknown filter source: {filter_config.filter_source}")


def _smooth_continuous_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    smoother_options: SmootherOptions,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Continuous-time marginal likelihood via CD-Dynamax smoothers."""
    if isinstance(
        filter_config,
        (ContinuousTimeEnKFConfig, ContinuousTimeUKFConfig, ContinuousTimeDPFConfig),
    ):
        raise ValueError(
            f"{type(filter_config).__name__} smoothing is not supported. "
            "Supported continuous-time smoothers: ContinuousTimeKFConfig, ContinuousTimeEKFConfig."
        )

    return run_continuous_smoother(
        name,
        dynamics,
        filter_config,
        smoother_options,
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
    "Smoother",
    "SmootherOptions",
    "HMMConfig",
    "HMMConfigs",
    "KFConfig",
    "PFConfig",
    "PFResamplingConfig",
    "UKFConfig",
]
