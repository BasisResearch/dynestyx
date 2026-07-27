"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF, RBPF."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    PosteriorGSSMFiltered,
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
    rbpfilter,
    rbpfilter_optimal,
)

from dynestyx.inference.distribution_utils import _posterior_sequence_to_dists
from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    RBPFConfig,
    UKFConfig,
    _config_to_record_kwargs,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    _require_constant_linear_gaussian_fields,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.utils import squeeze_leading_singletons
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    MixedStateDistribution,
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)
from dynestyx.utils import _should_record_field


def _prepare_slds_rbpf_inputs(dynamics, obs_values, obs_times, ctrl_times, ctrl_values):
    emissions = obs_values[:, None] if obs_values.ndim == 1 else obs_values
    t_len = emissions.shape[0]
    if dynamics.control_dim == 0:
        inputs = jnp.zeros((t_len, 0))
    elif ctrl_values is None:
        inputs = jnp.zeros((t_len, dynamics.control_dim))
    elif ctrl_values.shape[0] > t_len:
        inds = jnp.searchsorted(ctrl_times, obs_times, side="left")
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
    key,
    emissions,
    inputs,
):
    """Call cd-dynamax's SLDS RBPF implementation for particle histories."""
    if filter_config.proposal == "prior":
        return rbpfilter(
            filter_config.n_particles,
            params,
            emissions,
            key,
            inputs=inputs,
            ess_threshold=filter_config.ess_threshold_ratio,
        )
    if filter_config.proposal == "optimal":
        return rbpfilter_optimal(
            filter_config.n_particles,
            params,
            emissions,
            key,
            inputs=inputs,
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
    rbpf_output,
    *,
    num_regimes: int,
) -> dict[str, jax.Array]:
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


def _prepare_inputs(dynamics, obs_values, obs_times, ctrl_times, ctrl_values):
    """Prepare emissions and inputs arrays for cd-dynamax discrete filters."""
    emissions = obs_values
    t1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
        inputs = jnp.zeros((t1, control_dim))
    elif ctrl_values.shape[0] > t1:
        inds = jnp.searchsorted(ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        inputs = ctrl_values
    return emissions, inputs


def compute_cd_dynamax_discrete_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key=None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cd-dynamax discrete filter computation (no numpyro side-effects)."""
    if isinstance(filter_config, RBPFConfig):
        if key is None:
            raise ValueError(
                "compute_cd_dynamax_discrete_filter requires a PRNG key for RBPFConfig."
            )
        params = _slds_to_dynamax_params(dynamics)
        rbpf_emissions, rbpf_inputs = _prepare_slds_rbpf_inputs(
            dynamics, obs_values, obs_times, ctrl_times, ctrl_values
        )
        rbpf_output = _call_slds_rbpfilter(
            params, filter_config, key, rbpf_emissions, rbpf_inputs
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
        "Expected KFConfig, EKFConfig, or UKFConfig."
    )


def _add_kf_sites(
    name: str, posterior: PosteriorGSSMFiltered | dict, record_kwargs: dict
) -> None:
    """Add requested cd-dynamax filter summaries as deterministic sites."""
    max_elems = record_kwargs["record_max_elems"]
    means = _filter_output_field(posterior, "filtered_means")
    covs = _filter_output_field(posterior, "filtered_covariances")
    particles = _filter_output_field(posterior, "particles")
    log_weights = _filter_output_field(posterior, "log_weights")
    regime_probs = _filter_output_field(posterior, "filtered_regime_probs")

    if means is not None and _should_record_field(
        record_kwargs["record_filtered_states_mean"], means.shape, max_elems
    ):
        numpyro.deterministic(f"{name}_filtered_states_mean", means)
    if covs is not None and _should_record_field(
        record_kwargs["record_filtered_states_cov"], covs.shape, max_elems
    ):
        numpyro.deterministic(f"{name}_filtered_states_cov", covs)
    if covs is not None and _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], covs.shape[:-1], max_elems
    ):
        diag_cov = jnp.diagonal(covs, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
    if particles is not None and _should_record_field(
        record_kwargs["record_filtered_particles"], particles.shape, max_elems
    ):
        numpyro.deterministic(f"{name}_filtered_particles", particles)
    if log_weights is not None and _should_record_field(
        record_kwargs["record_filtered_log_weights"], log_weights.shape, max_elems
    ):
        numpyro.deterministic(f"{name}_filtered_log_weights", log_weights)
    if regime_probs is not None and _should_record_field(
        record_kwargs.get("record_filtered_regime_probs"),
        regime_probs.shape,
        max_elems,
    ):
        numpyro.deterministic(f"{name}_filtered_regime_probs", regime_probs)


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key=None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[dist.Distribution]:
    """Run discrete-time filter via cd-dynamax (KF, EKF, UKF, RBPF)."""
    posterior = compute_cd_dynamax_discrete_filter(
        dynamics,
        filter_config,
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    record_kwargs = _config_to_record_kwargs(filter_config)
    marginal_loglik = _filter_output_field(posterior, "marginal_loglik")
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)
    _add_kf_sites(name, posterior, record_kwargs)

    return _posterior_sequence_to_dists(
        posterior,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=isinstance(filter_config, RBPFConfig),
        missing="empty",
    )


__all__ = [
    "compute_cd_dynamax_discrete_filter",
    "run_discrete_filter",
    "_lti_to_lgssm_params",
    "_prepare_inputs",
]
