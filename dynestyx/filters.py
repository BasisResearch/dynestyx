import dataclasses
from typing import NamedTuple, Optional, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from cuthbert import filter as cuthbert_filter
from cuthbert.gaussian import taylor
from cuthbert.smc import particle_filter

from dynestyx.cuthbert_patches import systematic_resampling
from dynestyx.dynamical_models import DynamicalModel
from dynestyx.handlers import BaseCDDynamaxLogFactorAdder
from dynestyx.hmm_filter import hmm_filter, hmm_log_components
from dynestyx.ops import Context
from dynestyx.utils import _get_controls, _validate_control_dim, dsx_to_cd_dynamax

SSMType: TypeAlias = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

_DISCRETE_FILTER_TYPES: list[str] = ["default", "taylor_kf", "pf"]
_CONTINUOUS_FILTER_TYPES: list[str] = ["default", "enkf", "dpf"]


class _CuthbertInputs(NamedTuple):
    """Model inputs pytree for cuthbert; leading time dim must be T+1."""

    y: jax.Array  # (T+1, emission_dim)
    u: jax.Array  # (T+1, control_dim) or (T+1, 0)
    u_prev: jax.Array  # (T+1, control_dim) or (T+1, 0)
    time: jax.Array  # (T+1,)
    time_prev: jax.Array  # (T+1,)


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

    key: Optional[jax.Array] = None
    filter_type: str = "default"
    output_fields = None
    filter_kwargs: dict = dataclasses.field(default_factory=dict)

    def __init__(self, filter_type="default", output_fields=None, **filter_kwargs):
        super().__init__()
        self.filter_type = filter_type
        self.output_fields = output_fields
        self.filter_kwargs = filter_kwargs

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
            self._filter_continuous_time(name, dynamics, context)
        else:
            if self.filter_type.lower() not in _DISCRETE_FILTER_TYPES:
                raise ValueError(
                    f"Invalid filter type: {self.filter_type}. Valid types: {_DISCRETE_FILTER_TYPES}"
                )
            self._filter_discrete_time(name, dynamics, context)

    def _filter_discrete_time(
        self, name: str, dynamics: DynamicalModel, context: Context
    ):
        """
        Discrete-time marginal likelihood via cuthbert.
        """

        obs_traj = context.observations
        if obs_traj.values is None:
            return
        if isinstance(obs_traj.values, dict):
            raise ValueError("obs_traj.values must be an Array, not a dict")

        ys = obs_traj.values
        T1 = int(ys.shape[0])  # this is T+1 in cuthbert's convention
        if T1 == 0:
            return

        # Time axis (scalar at each step after slicing by cuthbert.filter)
        if obs_traj.times is None:
            times = jnp.arange(T1, dtype=jnp.float32)
        else:
            times = jnp.asarray(obs_traj.times)

        # Align controls (if any) to observation times
        _, ctrl_values = _get_controls(context, times)
        _validate_control_dim(dynamics, ctrl_values)

        if ctrl_values is None:
            ctrl_values = jnp.zeros((T1, 0), dtype=ys.dtype)

        dt0 = times[1] - times[0]

        time_prev = jnp.concatenate([times[:1] - dt0, times[:-1]], axis=0)

        u_prev = jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0)

        key = self.key if self.key is not None else numpyro.prng_key()

        cuthbert_inputs = _CuthbertInputs(
            y=ys, u=ctrl_values, u_prev=u_prev, time=times, time_prev=time_prev
        )

        if self.filter_type.lower() in ["taylor_kf", "default"]:
            filter_obj = self._cuthbert_filter_taylor_kf(dynamics)
        elif self.filter_type.lower() == "pf":
            filter_obj = self._cuthbert_filter_pf(dynamics)
        else:
            raise ValueError(
                f"Invalid filter type: {self.filter_type}. Valid types: {_DISCRETE_FILTER_TYPES}"
            )

        states = cuthbert_filter(filter_obj, cuthbert_inputs, parallel=False, key=key)

        marginal_loglik = states.log_normalizing_constant[-1]

        numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)

    def _filter_continuous_time(
        self, name: str, dynamics: DynamicalModel, context: Context
    ):
        """Continuous-time marginal likelihood via CD-Dynamax.

        Args:
            name: Name of the factor.
            dynamics: Dynamical model to filter.
            context: Context containing the observations and controls.
        """

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

        if self.filter_type.lower() in ["enkf", "default"]:
            cd_dynamax_model: SSMType = ContDiscreteNonlinearGaussianSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
                input_dim=dynamics.control_dim,
            )
        elif self.filter_type.lower() == "dpf":
            cd_dynamax_model = ContDiscreteNonlinearSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
                input_dim=dynamics.control_dim,
            )
        else:
            raise ValueError(
                f"Invalid filter type: {self.filter_type}. Valid types: {_CONTINUOUS_FILTER_TYPES}"
            )

        # Generate a CD-Dynamax-compatible parameter dict using the chosen model
        params, _ = dsx_to_cd_dynamax(dynamics, cd_model=cd_dynamax_model)

        # Choose a key
        key = self.key if self.key is not None else jr.PRNGKey(0)

        # Compute the marginal log likelihood via filtering
        if self.filter_type.lower() in ["enkf", "default"]:
            filter_kwargs = {
                "params": params,
                "emissions": obs_values,
                "t_emissions": obs_times,
                "key": key,
                "filter_type": self.filter_kwargs.get("filter_type", "EnKF"),
                "filter_state_order": self.filter_kwargs.get(
                    "filter_state_order", "first"
                ),
                "filter_emission_order": self.filter_kwargs.get(
                    "filter_emission_order", "first"
                ),
                "filter_num_iter": self.filter_kwargs.get("filter_num_iter", 1),
                "filter_state_cov_rescaling": self.filter_kwargs.get(
                    "filter_state_cov_rescaling", 1.0
                ),
                "filter_dt_average": self.filter_kwargs.get("filter_dt_average", 0.1),
                "enkf_N_particles": self.filter_kwargs.get("enkf_N_particles", 25),
                "enkf_inflation_delta": self.filter_kwargs.get(
                    "enkf_inflation_delta", 0.0
                ),
                "diffeqsolve_max_steps": self.filter_kwargs.get(
                    "diffeqsolve_max_steps", 1_000
                ),
                "diffeqsolve_dt0": self.filter_kwargs.get("diffeqsolve_dt0", 0.01),
                "diffeqsolve_kwargs": self.filter_kwargs.get("diffeqsolve_kwargs", {}),
                "extra_filter_kwargs": self.filter_kwargs.get(
                    "extra_filter_kwargs", {}
                ),
                "output_fields": self.output_fields,
                "warn": self.filter_kwargs.get("warn", True),
                "inputs": ctrl_values,
            }
        elif self.filter_type.lower() == "dpf":
            filter_kwargs = {
                "params": params,
                "emissions": obs_values,
                "t_emissions": obs_times,
                "key": key,
                "N_particles": self.filter_kwargs.get("N_particles", 1_000),
                "extra_filter_kwargs": {
                    "resampling_type": self.filter_kwargs.get(
                        "resampling_type", "stop_gradient"
                    )
                },
                "diffeqsolve_max_steps": self.filter_kwargs.get(
                    "diffeqsolve_max_steps", 1_000
                ),
                "diffeqsolve_dt0": self.filter_kwargs.get("diffeqsolve_dt0", 0.01),
                "diffeqsolve_kwargs": self.filter_kwargs.get("diffeqsolve_kwargs", {}),
                "output_fields": self.output_fields,
                "warn": self.filter_kwargs.get("warn", True),
                "inputs": ctrl_values,
            }
        else:
            raise ValueError(
                f"Invalid filter type: {self.filter_type}. Valid types: {_CONTINUOUS_FILTER_TYPES}"
            )

        filtered = cd_dynamax_model.filter(**filter_kwargs)  # type: ignore

        # Add the marginal log likelihood as a numpyro factor
        numpyro.factor(f"{name}_marginal_log_likelihood", filtered.marginal_loglik)

        # numpyro.deterministic(f"{name}_filtered_states_mean", filtered.filtered_means)
        # numpyro.deterministic(f"{name}_filtered_states_cov", filtered.filtered_covariances)
        # numpyro.deterministic(f"{name}_predicted_states_mean", filtered.predicted_means)
        # numpyro.deterministic(f"{name}_predicted_states_cov", filtered.predicted_covariances)

    def _cuthbert_filter_pf(self, dynamics: DynamicalModel):
        def init_sample(key, mi: _CuthbertInputs):
            return dynamics.initial_condition.sample(key)

        def propagate_sample(key, x_prev, mi: _CuthbertInputs):
            # TODO: Resolve these types later.
            dist = dynamics.state_evolution(x_prev, mi.u_prev, mi.time_prev, mi.time)  # type: ignore
            return dist.sample(key)  # type: ignore

        def log_potential(x_prev, x, mi: _CuthbertInputs):
            edist = dynamics.observation_model(x, mi.u, mi.time)
            return jnp.asarray(edist.log_prob(mi.y)).sum()

        ess_threshold = float(self.filter_kwargs.get("ess_threshold", 0.7))

        pf = particle_filter.build_filter(
            init_sample=init_sample,  # type: ignore
            propagate_sample=propagate_sample,  # type: ignore
            log_potential=log_potential,  # type: ignore
            n_filter_particles=int(self.filter_kwargs.get("n_filter_particles", 1_000)),
            resampling_fn=systematic_resampling.resampling,  # type: ignore
            ess_threshold=ess_threshold,
        )
        return pf

    def _cuthbert_filter_taylor_kf(self, dynamics: DynamicalModel):
        rtol = self.filter_kwargs.get("rtol", None)

        def get_init_log_density(mi: _CuthbertInputs):
            dist0 = dynamics.initial_condition

            def init_log_density(x):
                return jnp.asarray(dist0.log_prob(x)).sum()

            x0_lin = dist0.mean
            return init_log_density, x0_lin

        def get_dynamics_log_density(
            state: taylor.LinearizedKalmanFilterState, mi: _CuthbertInputs
        ):
            # log p(x_t | x_{t-1})
            def dynamics_log_density(x_prev, x):
                dist = dynamics.state_evolution(
                    x_prev, mi.u_prev, mi.time_prev, mi.time
                )
                return jnp.asarray(dist.log_prob(x)).sum()

            # Linearize around previous filtered mean.
            x_prev_lin = state.mean

            # A decent guess for the x_t linearization point is the conditional mean at x_prev_lin (if available).
            dist_at_lin = dynamics.state_evolution(  # type: ignore
                x_prev_lin, mi.u_prev, mi.time_prev, mi.time
            )
            try:
                x_lin = dist_at_lin.mean  # type: ignore
            except Exception:
                raise ValueError(
                    "dist_at_lin.mean is not available. Linearized Kalman filter requires a mean-able distribution."
                )

            return dynamics_log_density, x_prev_lin, x_lin

        def get_observation_func(
            state: taylor.LinearizedKalmanFilterState, mi: _CuthbertInputs
        ):
            def log_potential(x):
                edist = dynamics.observation_model(x, mi.u, mi.time)
                return jnp.asarray(edist.log_prob(mi.y)).sum()

            return log_potential, state.mean

        kf = taylor.build_filter(
            get_init_log_density,  # type: ignore
            get_dynamics_log_density,  # type: ignore
            get_observation_func,  # type: ignore
            associative=False,
            rtol=rtol,
            ignore_nan_dims=True,
        )

        return kf


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
