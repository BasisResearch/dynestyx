"""SLDS-specific cd-dynamax parameter and posterior adaptation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    RBPFiltered,
    rbpfilter,
    rbpfilter_optimal,
)
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Real

from dynestyx.inference.configs.filter import RBPFConfig
from dynestyx.inference.integrations.cd_dynamax.utils import _prepare_inputs
from dynestyx.models import (
    DynamicalModel,
    MixedStateDistribution,
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)
from dynestyx.observation_missingness import prepare_observation_views


def _slds_to_dynamax_params(dynamics: DynamicalModel) -> ParamsSLDS:
    """Build cd-dynamax SLDS params from a structured dynestyx SLDS model."""
    state_dim = dynamics.state_dim - 1
    observation_dim = dynamics.observation_dim
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
        observation_input_weights = (
            jnp.zeros((num_regimes, observation_dim, control_dim))
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
                emission_bias=jnp.zeros((num_regimes, observation_dim))
                if obs.bias is None
                else obs.bias,
                emission_input_weights=observation_input_weights,
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
    observations: Real[Array, "obs_time observation_dim"],
    inputs: Real[Array, "obs_time control_dim"],
    observation_mask: Bool[Array, "obs_time observation_dim"],
) -> RBPFiltered:
    """Call cd-dynamax's SLDS RBPF implementation for particle histories."""
    if filter_config.proposal == "prior":
        return rbpfilter(
            filter_config.n_particles,
            params,
            observations,
            key,
            inputs=inputs,
            ess_threshold=filter_config.ess_threshold_ratio,
            emission_mask=observation_mask,
        )
    if filter_config.proposal == "optimal":
        return rbpfilter_optimal(
            filter_config.n_particles,
            params,
            observations,
            key,
            inputs=inputs,
            emission_mask=observation_mask,
        )
    raise ValueError(f"Unknown RBPF proposal: {filter_config.proposal!r}")


class SLDSFilterPosterior(NamedTuple):
    """Dynestyx summaries adapted from cd-dynamax's ``RBPFiltered``.

    Means and covariances describe the continuous state. ``particles`` packs
    each discrete state with its conditional continuous mean, not a Gaussian
    draw. Leading batch axes are added by plate vmap. This internal result is
    stored in ``ConditionedResult.states``; it is not a simulated trajectory.
    """

    marginal_loglik: Float[Array, "*batch"]
    filtered_means: Float[Array, "*batch time continuous_state_dim"]
    filtered_covariances: Float[
        Array, "*batch time continuous_state_dim continuous_state_dim"
    ]
    filtered_regime_probs: Float[Array, "*batch time num_categories"]
    particles: Float[Array, "*batch time n_particles mixed_state_dim"]
    log_weights: Float[Array, "*batch time n_particles"]


def _slds_rbpfilter_output_to_filter_output(
    rbpf_output: RBPFiltered,
    *,
    num_regimes: int,
) -> SLDSFilterPosterior:
    """Convert cd-dynamax RBPF output to dynestyx's generic filter fields."""
    weights = rbpf_output.weights
    means = rbpf_output.means
    covs = rbpf_output.covariances
    states = rbpf_output.states
    if weights is None or means is None or covs is None or states is None:
        raise ValueError(
            "cd-dynamax SLDS RBPF output must include weights, means, "
            "covariances, and states."
        )
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
    return SLDSFilterPosterior(
        marginal_loglik=rbpf_output.marginal_loglik,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        filtered_regime_probs=regime_probs,
        particles=particles,
        log_weights=log_weights,
    )


def compute_slds_rbpf(
    dynamics: DynamicalModel,
    filter_config: RBPFConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    _obs_values_filled: Array | None = None,
    _obs_mask: Array | None = None,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
) -> SLDSFilterPosterior:
    """Run SLDS RBPF and adapt backend output to Dynestyx posterior fields."""
    if key is None:
        raise ValueError("compute_slds_rbpf requires a PRNG key for RBPFConfig.")
    params = _slds_to_dynamax_params(dynamics)
    if _obs_values_filled is None or _obs_mask is None:
        _obs_values_filled, _obs_mask, _ = prepare_observation_views(
            dynamics, obs_values
        )
    if _obs_values_filled is None or _obs_mask is None:
        raise ValueError("RBPF filtering requires observed values and a mask.")
    rbpf_observations, rbpf_inputs = _prepare_inputs(
        dynamics,
        _obs_values_filled,
        obs_times,
        ctrl_times,
        ctrl_values,
    )
    rbpf_output = _call_slds_rbpfilter(
        params,
        filter_config,
        key,
        rbpf_observations,
        rbpf_inputs,
        _obs_mask,
    )
    return _slds_rbpfilter_output_to_filter_output(
        rbpf_output,
        num_regimes=params.discrete.transition_matrix.shape[0],
    )
