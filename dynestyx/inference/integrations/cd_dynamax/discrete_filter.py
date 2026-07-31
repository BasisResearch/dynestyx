"""Discrete-time filters via cd-dynamax: KF, EKF, UKF, and SLDS RBPF."""

from typing import NamedTuple

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
    rbpfilter,
    rbpfilter_optimal,
)
from jaxtyping import Array, Float, Int, PRNGKeyArray

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
from dynestyx.inference.utils.distribution_utils import (
    _posterior_sequence_to_dists,
    _rbpf_sequence_to_dists,
)
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    MixedStateDistribution,
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)


class RBPFPosterior(NamedTuple):
    """Normalized dynestyx view of a cd-dynamax SLDS RBPF result.

    Attributes:
        marginal_loglik: Particle estimate of the total marginal log
            likelihood.
        weights: Normalized particle weights with shape
            ``(time, num_particles)``.
        regimes: Discrete regime particles with shape
            ``(time, num_particles)``.
        means: Conditional continuous-state means with shape
            ``(time, num_particles, state_dim)``.
        covariances: Conditional continuous-state covariances with shape
            ``(time, num_particles, state_dim, state_dim)``.
        filtered_means: Mixture mean of the continuous state.
        filtered_covariances: Mixture covariance of the continuous state,
            including both within-particle and between-particle uncertainty.
        filtered_regime_probs: Filtered categorical probabilities over
            regimes.
    """

    marginal_loglik: Float[Array, ""]
    weights: Float[Array, "time num_particles"]
    regimes: Int[Array, "time num_particles"]
    means: Float[Array, "time num_particles state_dim"]
    covariances: Float[Array, "time num_particles state_dim state_dim"]
    filtered_means: Float[Array, "time state_dim"]
    filtered_covariances: Float[Array, "time state_dim state_dim"]
    filtered_regime_probs: Float[Array, "time num_regimes"]

    @property
    def particles(self) -> Float[Array, "time num_particles joint_state_dim"]:
        """Joint particles encoded as ``[regime, *conditional_mean]``."""
        return jnp.concatenate(
            (self.regimes[..., None].astype(self.means.dtype), self.means),
            axis=-1,
        )

    @property
    def log_weights(self) -> Float[Array, "time num_particles"]:
        """Logarithms of normalized particle weights."""
        return jnp.log(self.weights)


def _slds_to_params(dynamics: DynamicalModel) -> ParamsSLDS:
    """Translate a structured dynestyx SLDS to cd-dynamax parameters."""
    if not (
        isinstance(dynamics.state_evolution, SwitchingLinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, SwitchingLinearGaussianObservation)
        and isinstance(dynamics.initial_condition, MixedStateDistribution)
    ):
        raise TypeError(
            "RBPFConfig requires a DynamicalModel with a "
            "MixedStateDistribution initial condition, "
            "SwitchingLinearGaussianStateEvolution, and "
            "SwitchingLinearGaussianObservation."
        )

    evolution = dynamics.state_evolution
    observation = dynamics.observation_model
    initial = dynamics.initial_condition
    num_regimes = evolution.num_regimes
    state_dim = evolution.continuous_state_dim
    observation_dim = dynamics.observation_dim
    control_dim = dynamics.control_dim

    if observation.num_regimes != num_regimes:
        raise ValueError(
            "State evolution and observation model disagree on num_regimes: "
            f"{num_regimes} != {observation.num_regimes}."
        )
    if initial.num_regimes != num_regimes:
        raise ValueError(
            "Initial condition and state evolution disagree on num_regimes: "
            f"{initial.num_regimes} != {num_regimes}."
        )
    if initial.continuous_state_dim != state_dim:
        raise ValueError(
            "Initial condition and state evolution disagree on continuous "
            f"state dimension: {initial.continuous_state_dim} != {state_dim}."
        )

    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=initial.categorical_probs,
            transition_matrix=evolution.transition_matrix,
            # Dynestyx intentionally exposes only the bootstrap proposal here.
            # An arbitrary proposal requires an importance-ratio correction
            # that is not part of the current cd-dynamax RBPF contract.
            proposal_transition_matrix=evolution.transition_matrix,
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=initial.continuous_locs,
            initial_cov=initial.continuous_covariances,
            dynamics_weights=evolution.A,
            dynamics_cov=evolution.cov,
            dynamics_bias=(
                jnp.zeros((num_regimes, state_dim))
                if evolution.bias is None
                else evolution.bias
            ),
            dynamics_input_weights=(
                jnp.zeros((num_regimes, state_dim, control_dim))
                if evolution.B is None
                else evolution.B
            ),
            emission_weights=observation.H,
            emission_cov=observation.R,
            emission_bias=(
                jnp.zeros((num_regimes, observation_dim))
                if observation.bias is None
                else observation.bias
            ),
            emission_input_weights=(
                jnp.zeros((num_regimes, observation_dim, control_dim))
                if observation.D is None
                else observation.D
            ),
            initialized=True,
        ),
    )


def _prepare_slds_inputs(
    dynamics: DynamicalModel,
    obs_values: jax.Array,
    obs_times: jax.Array,
    ctrl_times: jax.Array | None,
    ctrl_values: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Prepare observation and control arrays for the cd-dynamax SLDS API."""
    observations = obs_values[:, None] if obs_values.ndim == 1 else obs_values
    _, inputs = _prepare_inputs(
        dynamics,
        observations,
        obs_times,
        ctrl_times,
        ctrl_values,
    )
    return observations, inputs


def _normalize_rbpf_output(
    output,
    *,
    num_regimes: int,
) -> RBPFPosterior:
    """Validate and normalize the cd-dynamax RBPF output contract."""
    marginal_loglik = getattr(output, "marginal_loglik", None)
    if marginal_loglik is None:
        raise RuntimeError(
            "The installed cd-dynamax version does not expose an SLDS RBPF "
            "`marginal_loglik`. Install the commit pinned by dynestyx while "
            "the upstream RBPF likelihood release is pending."
        )
    weights = output.weights
    regimes = output.states
    means = output.means
    covariances = output.covariances
    if any(value is None for value in (weights, regimes, means, covariances)):
        raise RuntimeError("cd-dynamax returned an incomplete SLDS RBPF posterior.")

    filtered_means = jnp.einsum("...tp,...tpd->...td", weights, means)
    centered = means - filtered_means[..., :, None, :]
    filtered_covariances = jnp.einsum(
        "...tp,...tpij->...tij",
        weights,
        covariances + centered[..., :, :, None] * centered[..., :, None, :],
    )
    filtered_regime_probs = jnp.einsum(
        "...tp,...tpk->...tk",
        weights,
        jax.nn.one_hot(regimes, num_regimes),
    )
    return RBPFPosterior(
        marginal_loglik=marginal_loglik,
        weights=weights,
        regimes=regimes,
        means=means,
        covariances=covariances,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        filtered_regime_probs=filtered_regime_probs,
    )


def _compute_slds_rbpf(
    dynamics: DynamicalModel,
    filter_config: RBPFConfig,
    key: PRNGKeyArray | None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times: jax.Array | None,
    ctrl_values: jax.Array | None,
) -> RBPFPosterior:
    """Run cd-dynamax's SLDS RBPF and normalize its posterior."""
    if key is None:
        raise ValueError(
            "RBPFConfig requires a PRNG key. Set `crn_seed` or run the model "
            "inside a NumPyro seed handler."
        )
    params = _slds_to_params(dynamics)
    observations, inputs = _prepare_slds_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )
    if filter_config.proposal == "prior":
        output = rbpfilter(
            filter_config.n_particles,
            params,
            observations,
            key,
            inputs=inputs,
            ess_threshold=filter_config.ess_threshold_ratio,
        )
    elif filter_config.proposal == "optimal":
        output = rbpfilter_optimal(
            filter_config.n_particles,
            params,
            observations,
            key,
            inputs=inputs,
        )
    else:
        raise ValueError(f"Unknown RBPF proposal: {filter_config.proposal!r}.")
    return _normalize_rbpf_output(
        output,
        num_regimes=int(params.discrete.transition_matrix.shape[-1]),
    )


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
    key: PRNGKeyArray | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cd-dynamax discrete filter computation (no numpyro side-effects)."""
    if isinstance(filter_config, RBPFConfig):
        return _compute_slds_rbpf(
            dynamics,
            filter_config,
            key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
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
        "Expected KFConfig, EKFConfig, UKFConfig, or RBPFConfig."
    )


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> tuple[jax.Array, object, list[dist.Distribution]]:
    """Run a discrete-time cd-dynamax filter (KF, EKF, UKF, or SLDS RBPF).

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
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    if isinstance(filter_config, RBPFConfig):
        filtered_dists = _rbpf_sequence_to_dists(posterior)
    else:
        filtered_dists = _posterior_sequence_to_dists(
            posterior,
            means_attr="filtered_means",
            covariances_attr="filtered_covariances",
            particle_mode=False,
            missing="empty",
        )
    return posterior.marginal_loglik, posterior, filtered_dists


__all__ = [
    "compute_cd_dynamax_discrete_filter",
    "run_discrete_filter",
    "_lti_to_lgssm_params",
    "_prepare_inputs",
    "RBPFPosterior",
]
