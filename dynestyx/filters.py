import dataclasses

import jax
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM

from dynestyx.dynamical_models import DynamicalModel
from dynestyx.handlers import BaseCDDynamaxLogFactorAdder
from dynestyx.hmm_filter import hmm_filter, hmm_log_components
from dynestyx.inference.cd_dynamax.continuous_time_filters import (
    _CONTINUOUS_FILTER_TYPES,
    _filter_continuous_time,
)
from dynestyx.inference.cuthbert.discrete_time_filters import (
    _DISCRETE_FILTER_TYPES,
    _filter_discrete_time,
)
from dynestyx.ops import Context
from dynestyx.utils import _get_controls

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


@dataclasses.dataclass
class FilterBasedMarginalLogLikelihood(BaseCDDynamaxLogFactorAdder):
    """
    Object for filtering a dynamical model, and adding the resulting marginal log likelihood as a numpyro factor.

    There are several different options for the filter method, depending on the type of dynamical model.
    For discrete-time models, we interface with cuthbert; by default, we use the Taylor linearized Kalman filter (filter_type="taylor_kf"),
    but the particle filter is also available (filter_type="pf").
    For continuous-time models, we interface with CD-Dynamax; by default, we use the ensemble Kalman filter (filter_type="enkf"),
    but the differentiable particle filter is also available (filter_type="dpf").

    The filter_kwargs dictionary can be used to pass additional keyword arguments to the filter.
    TODO: Document choices available for each filter type.
    """

    key: jax.Array | None = None
    filter_type: str = "default"
    filter_kwargs: dict = dataclasses.field(default_factory=dict)
    record_filtered_states_mean: bool = True
    record_filtered_states_cov: bool = True
    record_filtered_states_cov_diag: bool = True
    record_filtered_particles: bool = True
    record_filtered_log_weights: bool = True
    record_filtered_states_chol_cov: bool = True

    def __init__(self, filter_type="default", **filter_kwargs):
        super().__init__()
        self.filter_type = filter_type
        self.filter_kwargs = filter_kwargs if filter_kwargs is not None else {}
        self.record_kwargs = {
            "record_filtered_states_mean": self.record_filtered_states_mean,
            "record_filtered_states_cov": self.record_filtered_states_cov,
            "record_filtered_states_cov_diag": self.record_filtered_states_cov_diag,
            "record_filtered_particles": self.record_filtered_particles,
            "record_filtered_log_weights": self.record_filtered_log_weights,
            "record_filtered_states_chol_cov": self.record_filtered_states_chol_cov,
        }

    def add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        """
        Add the marginal log likelihood as a numpyro factor.

        Args:
            name: Name of the factor.
            dynamics: Dynamical model to filter.
            context: Context containing the observations and controls.
        """
        if dynamics.continuous_time:
            if self.filter_type.lower() not in _CONTINUOUS_FILTER_TYPES:
                raise ValueError(
                    f"Invalid filter type: {self.filter_type}. Valid types: {_CONTINUOUS_FILTER_TYPES}"
                )
            _filter_continuous_time(
                name,
                self.filter_type,
                dynamics,
                context,
                self.key,
                self.filter_kwargs,
                self.record_kwargs,
            )
        else:
            if self.filter_type.lower() not in _DISCRETE_FILTER_TYPES:
                raise ValueError(
                    f"Invalid filter type: {self.filter_type}. Valid types: {_DISCRETE_FILTER_TYPES}"
                )
            _filter_discrete_time(
                name,
                self.filter_type,
                dynamics,
                context,
                self.key,
                self.filter_kwargs,
                self.record_kwargs,
            )


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
            f"{name}_marginal_log_likelihood",
            loglik,
        )

        # For use in predictive sampling
        numpyro.deterministic(
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
