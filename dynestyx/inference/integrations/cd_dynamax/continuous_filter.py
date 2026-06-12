"""Continuous-time filters via CD-Dynamax: KF, EnKF, DPF, EKF, UKF."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.models import (
    PosteriorGSSMFiltered,
)

from dynestyx.inference.distribution_utils import _posterior_sequence_to_dists
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
)
from dynestyx.models import DynamicalModel

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

ContinuousTimeFilterConfig = (
    ContinuousTimeKFConfig
    | ContinuousTimeEnKFConfig
    | ContinuousTimeDPFConfig
    | ContinuousTimeEKFConfig
    | ContinuousTimeUKFConfig
)


def _config_to_cd_dynamax_filter_kwargs(
    config: ContinuousTimeFilterConfig,
    params,
    obs_values,
    obs_times,
    ctrl_values,
    key,
) -> dict:
    """Build the filter_kwargs dict passed to cd_dynamax_model.filter()."""

    # cd-dynamax uses the legacy PRNG key interface, but newer numpyro uses typed keys.
    # We should convert accordingly.
    # https://docs.jax.dev/en/latest/jax.random.html#module-jax.random
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)

    base = {
        "params": params,
        "emissions": obs_values,
        "t_emissions": obs_times,
        "inputs": ctrl_values,
        "key": key,
        "filter_state_order": config.filter_state_order,
        "filter_state_cov_rescaling": config.cov_rescaling
        if config.cov_rescaling is not None
        else 1.0,
        "diffeqsolve_max_steps": config.diffeqsolve_max_steps,
        "diffeqsolve_dt0": config.diffeqsolve_dt0,
        "diffeqsolve_kwargs": config.diffeqsolve_kwargs,
        "extra_filter_kwargs": config.extra_filter_kwargs,
        "warn": config.warn,
    }
    if isinstance(config, ContinuousTimeEnKFConfig):
        base["filter_type"] = "EnKF"
        base["enkf_N_particles"] = config.n_particles
        base["enkf_inflation_delta"] = (
            config.inflation_delta if config.inflation_delta is not None else 0.0
        )
        base["extra_filter_kwargs"] = {
            "perturb_measurements": config.perturb_measurements
            if config.perturb_measurements is not None
            else True,
            **config.extra_filter_kwargs,
        }
    elif isinstance(config, ContinuousTimeEKFConfig):
        base["filter_type"] = "EKF"
        base["filter_emission_order"] = config.filter_emission_order
        base["filter_num_iter"] = 1
    elif isinstance(config, ContinuousTimeUKFConfig):
        base["filter_type"] = "UKF"
        base["extra_filter_kwargs"] = {
            "alpha": config.alpha,
            "beta": config.beta,
            "kappa": config.kappa,
            **config.extra_filter_kwargs,
        }
    elif isinstance(config, ContinuousTimeDPFConfig):
        if config.resampling_method.base_method != "multinomial":
            raise ValueError(
                "Only multinomial resampling is supported for CD-Dynamax DPF."
            )
        base["filter_type"] = "DPF"
        base["N_particles"] = config.n_particles
        base["extra_filter_kwargs"] = {
            "resample_method": config.resampling_method.differential_method,
            "softness": config.resampling_method.softness,
            "ess_threshold_ratio": config.ess_threshold_ratio,
            **config.extra_filter_kwargs,
        }
    return base


def _run_linear_kf(
    dynamics: DynamicalModel,
    obs_times,
    obs_values,
    ctrl_values,
    filter_config: ContinuousTimeKFConfig,
) -> PosteriorGSSMFiltered:
    """Run exact continuous-discrete KF (AffineLinearDrift + constant diffusion + LinearGaussianObservation)."""
    params = dsx_to_cdlgssm_params(dynamics)
    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=dynamics.state_dim,
        emission_dim=dynamics.observation_dim,
        input_dim=dynamics.control_dim,
    )
    filtered = cd_model.filter(
        params=params,
        emissions=obs_values,
        t_emissions=obs_times,
        inputs=ctrl_values,
        warn=filter_config.warn,
    )
    return filtered


def compute_continuous_filter(
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX continuous-time filter computation (no numpyro side-effects)."""
    obs_times_arr = jnp.asarray(obs_times)
    if obs_times_arr.ndim == 1:
        obs_times_arr = obs_times_arr[:, None]

    control_dim = dynamics.control_dim
    ctrl_vals = (
        ctrl_values
        if ctrl_values is not None
        else jnp.zeros((obs_times_arr.shape[0], control_dim))
    )

    if isinstance(filter_config, ContinuousTimeKFConfig):
        filtered = _run_linear_kf(
            dynamics, obs_times_arr, obs_values, ctrl_vals, filter_config
        )
    else:
        if isinstance(
            filter_config, (ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig)
        ):
            if key is None:
                raise ValueError(
                    f"{type(filter_config).__name__} requires a PRNG key: set 'crn_seed' in the filter config, "
                    "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
                )

        if isinstance(filter_config, ContinuousTimeDPFConfig):
            cd_dynamax_model: SSMType = ContDiscreteNonlinearSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
                input_dim=dynamics.control_dim,
            )
        else:
            cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
                input_dim=dynamics.control_dim,
            )

        params, _ = dsx_to_cd_dynamax(dynamics, cd_model=cd_dynamax_model)
        filter_kwargs = _config_to_cd_dynamax_filter_kwargs(
            filter_config, params, obs_values, obs_times_arr, ctrl_vals, key
        )

        filtered = cd_dynamax_model.filter(**filter_kwargs)  # type: ignore

    return filtered


def run_continuous_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> tuple[jax.Array, object, list[dist.Distribution]]:
    """Run continuous-time filter via CD-Dynamax.

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - filtered_posterior: CD-Dynamax posterior object with filtered_means,
              filtered_covariances, and marginal_loglik attributes.
            - filtered_dists: list of MultivariateNormal distributions p(x_t | y_{1:t})
              at each obs time, for posterior rollout.
    """
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    filtered_dists = _posterior_sequence_to_dists(
        filtered,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=isinstance(filter_config, ContinuousTimeDPFConfig),
        missing_message="Filtered means/covariances unexpectedly None for non-DPF config",
    )
    return filtered.marginal_loglik, filtered, filtered_dists


__all__ = [
    "ContinuousTimeFilterConfig",
    "compute_continuous_filter",
    "run_continuous_filter",
]
