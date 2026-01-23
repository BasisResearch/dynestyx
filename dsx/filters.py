import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional
import dataclasses

from dsx.ops import Context
from dsx.handlers import BaseCDDynamaxLogFactorAdder
from dsx.dynamical_models import DynamicalModel
from dsx.utils import dsx_to_cd_dynamax, _get_controls, _validate_control_dim
from dsx.hmm_filter import hmm_log_components, hmm_filter
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
import numpyro


@dataclasses.dataclass
class FilterBasedMarginalLogLikelihood(BaseCDDynamaxLogFactorAdder):
    """Log factor adder that computes marginal log likelihood via CD-Dynamax filtering."""

    key: Optional[jax.Array] = None
    filter_type: str = "EnKF"
    filter_state_order: str = "first"
    filter_emission_order: str = "first"
    filter_num_iter: int = 1
    filter_state_cov_rescaling: float = 1.0
    filter_dt_average: float = 0.1
    enkf_N_particles: int = 25
    enkf_inflation_delta: float = 0.0
    diffeqsolve_max_steps: int = 1_000
    diffeqsolve_dt0: float = 0.01
    output_fields = None
    diffeqsolve_kwargs: dict = dataclasses.field(default_factory=dict)
    extra_filter_kwargs: dict = dataclasses.field(default_factory=dict)
    warn: bool = True

    def add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Pull observed trajectory from context
        obs_traj = context.observations
        if obs_traj.times is None or obs_traj.values is None:
            # No observations → nothing to factor
            return

        obs_times = obs_traj.times[:, None]  # shape (T, 1)
        if isinstance(obs_traj.values, dict):
            raise ValueError("obs_traj.values must be an Array, not a dict")
        obs_values = obs_traj.values  # shape (T, emission_dim)

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_traj.times)

        # Validate that control_dim is set when controls are present
        _validate_control_dim(dynamics, ctrl_values)

        # Generate a CD-Dynamax-compatible parameter dict
        params = dsx_to_cd_dynamax(dynamics)

        # Instantiate the CD-Dynamax model
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            input_dim=dynamics.control_dim,
        )

        # Choose a key
        key = self.key if self.key is not None else jr.PRNGKey(0)

        # Compute the marginal log likelihood via filtering
        filtered = cd_dynamax_model.filter(
            params=params,
            emissions=obs_values,
            t_emissions=obs_times,
            inputs=ctrl_values,
            key=key,
            filter_type=self.filter_type,
            filter_state_order=self.filter_state_order,
            filter_emission_order=self.filter_emission_order,
            filter_num_iter=self.filter_num_iter,
            filter_state_cov_rescaling=self.filter_state_cov_rescaling,
            filter_dt_average=self.filter_dt_average,
            enkf_N_particles=self.enkf_N_particles,
            enkf_inflation_delta=self.enkf_inflation_delta,
            diffeqsolve_max_steps=self.diffeqsolve_max_steps,
            diffeqsolve_dt0=self.diffeqsolve_dt0,
            diffeqsolve_kwargs=self.diffeqsolve_kwargs,
            extra_filter_kwargs=self.extra_filter_kwargs,
            output_fields=self.output_fields,
            warn=self.warn,
        )

        # Add the marginal log likelihood as a numpyro factor
        numpyro.factor(f"{name}_marginal_log_likelihood", filtered.marginal_loglik)

        # numpyro.deterministic(f"{name}_filtered_states_mean", filtered.filtered_means)
        # numpyro.deterministic(f"{name}_filtered_states_cov", filtered.filtered_covariances)
        # numpyro.deterministic(f"{name}_predicted_states_mean", filtered.predicted_means)
        # numpyro.deterministic(f"{name}_predicted_states_cov", filtered.predicted_covariances)


@dataclasses.dataclass
class FilterBasedHMMMarginalLogLikelihood(BaseCDDynamaxLogFactorAdder):
    """
    Exact HMM marginal log-likelihood via forward filtering.

    Optionally, (log-)filtered states are recorded if `record_(log_)filtered == True`.
    """

    record_filtered: bool = False
    record_log_filtered: bool = False

    def add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        obs = context.observations
        if obs.times is None or obs.values is None:
            return

        if isinstance(obs.values, dict):
            raise ValueError("obs.values must be an Array, not a dict")
        obs_values = obs.values

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs.times)

        log_pi, log_A_seq, log_emit_seq = hmm_log_components(
            dynamics,
            obs.times,
            obs_values,
            ctrl_values=ctrl_values,
        )

        loglik, log_filt_seq = hmm_filter(
            log_pi,
            log_A_seq,
            log_emit_seq,
        )

        numpyro.factor(
            f"{name}_marginal_loglik",
            loglik,
        )

        if self.record_log_filtered:
            numpyro.deterministic(
                f"{name}_log_filtered_states",
                log_filt_seq,  # (T, K)
            )

        if self.record_filtered:
            numpyro.deterministic(
                f"{name}_filtered_states",
                jnp.exp(log_filt_seq),  # (T, K)
            )
