import jax
import jax.random as jr
from typing import Optional
import dataclasses

from dsx.ops import Context
from dsx.handlers import BaseCDDynamaxLogFactorAdder
from dsx.dynamical_models import DynamicalModel
from dsx.utils import dsx_to_cd_dynamax
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
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
    dpf_num_particles: int = 100
    dpf_resampling_type: str = "soft"
    enkf_N_particles: int = 25
    enkf_inflation_delta: float = 0.0
    diffeqsolve_max_steps: int = 1000
    diffeqsolve_dt0: float = 0.01
    output_fields = None
    diffeqsolve_kwargs: dict = dataclasses.field(default_factory=dict)
    extra_filter_kwargs: dict = dataclasses.field(default_factory=dict)
    warn: bool = True

    def add_log_factors(
        self,
        dynamics: DynamicalModel,
        context: Context,
        name: Optional[str] = "EnKF",
    ):
        # Pull observed trajectory from context
        obs_traj = context.observations
        if obs_traj.times is None or obs_traj.values is None:
            # No observations → nothing to factor
            return

        obs_times = obs_traj.times[:, None]  # shape (T, 1)
        obs_values = obs_traj.values  # shape (T, emission_dim)

        if self.filter_type.lower() == "dpf":
            cd_dynamax_model = ContDiscreteNonlinearSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
            )
        else:
            # Instantiate the CD-Dynamax model
            cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
            )

        # Generate a CD-Dynamax-compatible parameter dict using the chosen model
        params = dsx_to_cd_dynamax(dynamics, cd_model=cd_dynamax_model)

        # Choose a key
        key = self.key if self.key is not None else jr.PRNGKey(0)

        # Compute the marginal log likelihood via filtering
        if self.filter_type.lower() == "dpf":
            filter_kwargs = {
                "params": params,
                "emissions": obs_values,
                "t_emissions": obs_times,
                "key": key,
                "N_particles": self.dpf_num_particles,
                "extra_filter_kwargs": {"resampling_type": self.dpf_resampling_type},
                "diffeqsolve_max_steps": self.diffeqsolve_max_steps,
                "diffeqsolve_dt0": self.diffeqsolve_dt0,
                "diffeqsolve_kwargs": self.diffeqsolve_kwargs,
                "output_fields": self.output_fields,
                "warn": self.warn,
            }
            if self.extra_filter_kwargs:
                filter_kwargs.update(self.extra_filter_kwargs)
        else:
            filter_kwargs = {
                "params": params,
                "emissions": obs_values,
                "t_emissions": obs_times,
                "key": key,
                "filter_type": self.filter_type,
                "filter_state_order": self.filter_state_order,
                "filter_emission_order": self.filter_emission_order,
                "filter_num_iter": self.filter_num_iter,
                "filter_state_cov_rescaling": self.filter_state_cov_rescaling,
                "filter_dt_average": self.filter_dt_average,
                "enkf_N_particles": self.enkf_N_particles,
                "enkf_inflation_delta": self.enkf_inflation_delta,
                "diffeqsolve_max_steps": self.diffeqsolve_max_steps,
                "diffeqsolve_dt0": self.diffeqsolve_dt0,
                "diffeqsolve_kwargs": self.diffeqsolve_kwargs,
                "extra_filter_kwargs": self.extra_filter_kwargs,
                "output_fields": self.output_fields,
                "warn": self.warn,
            }

        filtered = cd_dynamax_model.filter(**filter_kwargs)

        # Add the marginal log likelihood as a numpyro factor
        numpyro.factor(f"{name}_marginal_log_likelihood", filtered.marginal_loglik)

        # numpyro.deterministic(f"{name}_filtered_states_mean", filtered.filtered_means)
        # numpyro.deterministic(f"{name}_filtered_states_cov", filtered.filtered_covariances)
        # numpyro.deterministic(f"{name}_predicted_states_mean", filtered.predicted_means)
        # numpyro.deterministic(f"{name}_predicted_states_cov", filtered.predicted_covariances)
