"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF, RBPF."""

import warnings
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    lgssm_filter,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_filter,
)
from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    RBPFiltered,
    rbpfilter,
    rbpfilter_optimal,
)
from jax.experimental import sparse as jax_sparse
from jaxtyping import Array, Bool, PRNGKeyArray, Real

from dynestyx.inference.configs.filter import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    RBPFConfig,
    UKFConfig,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    _require_constant_linear_gaussian_fields,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.utils import squeeze_leading_singletons
from dynestyx.inference.utils.distribution_utils import _posterior_sequence_to_dists
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    MixedStateDistribution,
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)
from dynestyx.observation_missingness import prepare_observation_views


def _prepare_slds_rbpf_inputs(
    dynamics: DynamicalModel,
    obs_values: Real[Array, "obs_time observation_dim"] | Real[Array, " obs_time"],
    obs_times: Real[Array, " obs_time"],
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None,
) -> tuple[
    Real[Array, "obs_time observation_dim"], Real[Array, "obs_time control_dim"]
]:
    emissions = obs_values[:, None] if obs_values.ndim == 1 else obs_values
    t_len = emissions.shape[0]
    if dynamics.control_dim == 0:
        inputs = jnp.zeros((t_len, 0))
    elif ctrl_values is None:
        inputs = jnp.zeros((t_len, dynamics.control_dim))
    elif ctrl_values.shape[0] > t_len:
        inds = jnp.searchsorted(
            cast(Real[Array, " ctrl_time"], ctrl_times), obs_times, side="left"
        )
        inputs = ctrl_values[inds]
    else:
        inputs = ctrl_values
    return emissions, inputs


def _slds_to_dynamax_params(dynamics: DynamicalModel) -> ParamsSLDS:
    """Build cd-dynamax SLDS params from a structured dynestyx SLDS model."""
    state_dim = dynamics.state_dim - 1
    emission_dim = dynamics.observation_dim
    control_dim = dynamics.control_dim

    if (
        isinstance(dynamics.state_evolution, SwitchingLinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, SwitchingLinearGaussianObservation)
        and isinstance(dynamics.initial_condition, MixedStateDistribution)
    ):
        evo = dynamics.state_evolution
        obs = dynamics.observation_model
        ic = dynamics.initial_condition
        num_regimes = evo.num_regimes
        dynamics_input_weights = (
            jnp.zeros((num_regimes, state_dim, control_dim)) if evo.B is None else evo.B
        )
        emission_input_weights = (
            jnp.zeros((num_regimes, emission_dim, control_dim))
            if obs.D is None
            else obs.D
        )
        return ParamsSLDS(
            discrete=DiscreteParamsSLDS(
                initial_distribution=ic.categorical_probs,
                transition_matrix=evo.transition_matrix,
                proposal_transition_matrix=evo.transition_matrix,
            ),
            linear_gaussian=LGParamsSLDS(
                initial_mean=ic.continuous_locs,
                initial_cov=ic.continuous_covs,
                dynamics_weights=evo.A,
                dynamics_cov=evo.cov,
                dynamics_bias=jnp.zeros((num_regimes, state_dim))
                if evo.bias is None
                else evo.bias,
                dynamics_input_weights=dynamics_input_weights,
                emission_weights=obs.H,
                emission_cov=obs.R,
                emission_bias=jnp.zeros((num_regimes, emission_dim))
                if obs.bias is None
                else obs.bias,
                emission_input_weights=emission_input_weights,
                initialized=True,
            ),
        )
    raise TypeError(
        "filter_type='rbpf' expects a DynamicalModel with "
        "SwitchingLinearGaussianStateEvolution and "
        "SwitchingLinearGaussianObservation and initial_condition as "
        "MixedStateDistribution."
    )


def _call_slds_rbpfilter(
    params: ParamsSLDS,
    filter_config: RBPFConfig,
    key: PRNGKeyArray,
    emissions: Real[Array, "obs_time observation_dim"],
    inputs: Real[Array, "obs_time control_dim"],
    emission_mask: Bool[Array, "obs_time observation_dim"],
) -> RBPFiltered:
    """Call cd-dynamax's SLDS RBPF implementation for particle histories."""
    if filter_config.proposal == "prior":
        return rbpfilter(
            filter_config.n_particles,
            params,
            emissions,
            key,
            inputs=inputs,
            ess_threshold=filter_config.ess_threshold_ratio,
            emission_mask=emission_mask,
        )
    if filter_config.proposal == "optimal":
        return rbpfilter_optimal(
            filter_config.n_particles,
            params,
            emissions,
            key,
            inputs=inputs,
            emission_mask=emission_mask,
        )
    raise ValueError(f"Unknown RBPF proposal: {filter_config.proposal!r}")


def _rbpfilter_field(rbpf_output, field: str):
    if isinstance(rbpf_output, dict):
        return rbpf_output.get(field)
    return getattr(rbpf_output, field, None)


def _filter_output_field(posterior, field: str, default=None):
    if isinstance(posterior, dict):
        return posterior.get(field, default)
    return getattr(posterior, field, default)


def _slds_rbpfilter_output_to_filter_output(
    rbpf_output: RBPFiltered | dict[str, Array],
    *,
    num_regimes: int,
) -> dict[str, Array]:
    """Convert cd-dynamax RBPF output to dynestyx's generic filter fields."""
    weights = _rbpfilter_field(rbpf_output, "weights")
    means = _rbpfilter_field(rbpf_output, "means")
    covs = _rbpfilter_field(rbpf_output, "covariances")
    states = _rbpfilter_field(rbpf_output, "states")
    filtered_means = jnp.sum(weights[..., None] * means, axis=1)
    centered = means - filtered_means[:, None, :]
    filtered_covs = jnp.sum(
        weights[..., None, None]
        * (covs + centered[..., :, None] * centered[..., None, :]),
        axis=1,
    )
    regime_probs = jnp.sum(
        weights[..., None] * jax.nn.one_hot(states, num_regimes), axis=1
    )
    particles = jnp.concatenate([states[..., None].astype(means.dtype), means], axis=-1)
    log_weights = jnp.log(weights)
    marginal_loglik = _rbpfilter_field(rbpf_output, "marginal_loglik")
    if marginal_loglik is None:
        raise AttributeError(
            "cd-dynamax SLDS RBPF output must include `marginal_loglik`. "
            "Update cd_dynamax.dynamax.slds.inference.rbpfilter/"
            "rbpfilter_optimal to return RBPFiltered(marginal_loglik=...)."
        )
    return {
        "marginal_loglik": marginal_loglik,
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs,
        "filtered_regime_probs": regime_probs,
        "particles": particles,
        "log_weights": log_weights,
    }


def _lti_to_lgssm_params(dynamics: DynamicalModel):
    """Build dynamax ParamsLGSSM from LinearGaussianSSM.initialize for an LTI model."""
    state_dim = dynamics.state_dim
    emission_dim = dynamics.observation_dim
    control_dim = dynamics.control_dim

    if (
        isinstance(dynamics.state_evolution, LinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, LinearGaussianObservation)
        and isinstance(dynamics.initial_condition, dist.MultivariateNormal)
    ):
        evo = dynamics.state_evolution
        obs = dynamics.observation_model
        ic = dynamics.initial_condition
        _require_constant_linear_gaussian_fields(
            evo,
            ("A", "B", "bias", "cov"),
            where="The cd_dynamax discrete Kalman filter",
        )
        _require_constant_linear_gaussian_fields(
            obs,
            ("H", "D", "bias", "R"),
            where="The cd_dynamax discrete Kalman filter",
        )
        model = LinearGaussianSSM(
            state_dim=state_dim,
            emission_dim=emission_dim,
            input_dim=control_dim,
            has_dynamics_bias=evo.bias is not None,
            has_emissions_bias=obs.bias is not None,
        )
        params, _ = model.initialize(
            initial_mean=squeeze_leading_singletons(ic.loc, 1),
            initial_covariance=squeeze_leading_singletons(ic.covariance_matrix, 2),
            dynamics_weights=evo.A,
            dynamics_bias=evo.bias,
            dynamics_input_weights=evo.B,
            dynamics_covariance=evo.cov,
            emission_weights=obs.H,
            emission_bias=obs.bias,
            emission_input_weights=obs.D,
            emission_covariance=obs.R,
        )
        return params
    raise TypeError(
        "filter_type='kf' expects a DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation and initial_condition as MultivariateNormal."
    )


def _prepare_inputs(
    dynamics: DynamicalModel,
    obs_values: Real[Array, "obs_time observation_dim"],
    obs_times: Real[Array, " obs_time"],
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None,
) -> tuple[
    Real[Array, "obs_time observation_dim"],
    Real[Array, "obs_time control_dim"],
]:
    """Prepare emissions and inputs arrays for cd-dynamax discrete filters."""
    emissions = obs_values
    t1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
        inputs = jnp.zeros((t1, control_dim))
    elif ctrl_values.shape[0] > t1:
        aligned_ctrl_times = cast(Real[Array, " ctrl_time"], ctrl_times)
        inds = jnp.searchsorted(aligned_ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        inputs = ctrl_values
    return emissions, inputs


def compute_cd_dynamax_discrete_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    _obs_values_filled: Array | None = None,
    _obs_mask: Array | None = None,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
) -> Any:
    """Pure-JAX cd-dynamax discrete filter computation (no numpyro side-effects)."""
    if isinstance(filter_config, RBPFConfig):
        if key is None:
            raise ValueError(
                "compute_cd_dynamax_discrete_filter requires a PRNG key for RBPFConfig."
            )
        params = _slds_to_dynamax_params(dynamics)
        if _obs_values_filled is None or _obs_mask is None:
            _obs_values_filled, _obs_mask, _ = prepare_observation_views(
                dynamics, obs_values
            )
        if _obs_values_filled is None or _obs_mask is None:
            raise ValueError("RBPF filtering requires observed values and a mask.")
        rbpf_emissions, rbpf_inputs = _prepare_slds_rbpf_inputs(
            dynamics,
            _obs_values_filled,
            obs_times,
            ctrl_times,
            ctrl_values,
        )
        rbpf_mask = _obs_mask[:, None] if _obs_mask.ndim == 1 else _obs_mask
        rbpf_output = _call_slds_rbpfilter(
            params,
            filter_config,
            key,
            rbpf_emissions,
            rbpf_inputs,
            rbpf_mask,
        )
        return _slds_rbpfilter_output_to_filter_output(
            rbpf_output,
            num_regimes=params.discrete.transition_matrix.shape[0],
        )

    emissions, inputs = _prepare_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )

    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        return lgssm_filter(params, emissions, inputs=inputs)

    # EKF and UKF share the same nonlinear params representation.
    params_nl = gaussian_to_nlgssm_params(dynamics)

    if isinstance(filter_config, EKFConfig):
        obs_model = dynamics.observation_model
        if isinstance(obs_model, LinearGaussianObservation) and isinstance(
            obs_model.H, jax_sparse.JAXSparse
        ):
            warnings.warn(
                "A sparse observation matrix H was passed to EKFConfig. This works "
                "correctly, but likely gives no efficiency gain due to internal"
                "use of automatic differentiation.",
                stacklevel=2,
            )
        return extended_kalman_filter(params_nl, emissions, inputs=inputs)
    if isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return unscented_kalman_filter(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )
    raise ValueError(
        f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
        "Expected KFConfig, EKFConfig, UKFConfig, or RBPFConfig."
    )


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    _obs_values_filled: Array | None = None,
    _obs_mask: Array | None = None,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[Real[Array, ""], object, list[dist.Distribution]]:
    """Run discrete-time filter via cd-dynamax (KF, EKF, UKF, RBPF).

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - posterior: CD-Dynamax posterior object with filtered_means and
              filtered_covariances attributes.
            - filtered_dists: list of MultivariateNormal distributions p(x_t | y_{1:t})
              at each obs time, for posterior rollout.
    """
    posterior = compute_cd_dynamax_discrete_filter(
        dynamics,
        filter_config,
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        _obs_values_filled=_obs_values_filled,
        _obs_mask=_obs_mask,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    filtered_dists = _posterior_sequence_to_dists(
        posterior,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=isinstance(filter_config, RBPFConfig),
        missing="empty",
    )
    marginal_loglik = _filter_output_field(posterior, "marginal_loglik")
    if marginal_loglik is None:
        raise AttributeError("cd-dynamax filter output is missing `marginal_loglik`.")
    return marginal_loglik, posterior, filtered_dists


__all__ = [
    "compute_cd_dynamax_discrete_filter",
    "run_discrete_filter",
    "_lti_to_lgssm_params",
    "_prepare_inputs",
]
