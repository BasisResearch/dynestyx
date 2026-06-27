"""NumPyro-facing latent-path builder for joint state and parameter inference."""

from __future__ import annotations

import dataclasses
from typing import cast

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements
from jax import Array
from jax.errors import TracerBoolConversionError
from jaxtyping import Real

from dynestyx.handlers import _condition_intp
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
)
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.observation_missingness import prepare_observation_views
from dynestyx.path_log_prob import (
    discrete_path_score_terms,
    observation_log_prob_terms,
)
from dynestyx.simulation_utils import _ensure_trailing_dim, _tile_times
from dynestyx.simulators import BaseSimulator
from dynestyx.solvers import solve_ode
from dynestyx.types import as_scalar_time_array
from dynestyx.utils import (
    _build_control_path,
    _get_val_or_None,
    _raise_now_or_error_if,
)


@dataclasses.dataclass
class LatentStateBuilder(BaseSimulator):
    """Build latent NumPyro sites, then score observations against them.

    This interpretation is the explicit latent-path counterpart to filtering. It is
    intended for joint state + parameter inference, especially for discrete-time or
    discretized models where users want direct control over the latent path.
    """

    chunk_size: int | None = None
    n_simulations: int = 1

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Array | None = None,
        _obs_mask: Array | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        filtered_times=None,
        filtered_dists=None,
        smoothed_times=None,
        smoothed_dists=None,
        **kwargs,
    ):
        posterior_rollout_final_only = kwargs.pop(
            "_posterior_rollout_final_only", False
        )
        effective_predict_times = (
            predict_times if predict_times is not None else obs_times
        )
        if obs_values is not None and (_obs_values_filled is None or _obs_mask is None):
            _obs_values_filled, _obs_mask, _obs_has_missing = prepare_observation_views(
                dynamics, obs_values
            )

        if plate_shapes:
            results = self._run_plated_simulation(
                name,
                dynamics,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                _obs_has_missing=_obs_has_missing,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=effective_predict_times,
                filtered_times=filtered_times,
                filtered_dists=filtered_dists,
                smoothed_times=smoothed_times,
                smoothed_dists=smoothed_dists,
                _posterior_rollout_final_only=posterior_rollout_final_only,
                **kwargs,
            )
        else:
            results = self._run_single_member_simulation(
                name,
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                _obs_has_missing=_obs_has_missing,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=effective_predict_times,
                filtered_times=filtered_times,
                filtered_dists=filtered_dists,
                smoothed_times=smoothed_times,
                smoothed_dists=smoothed_dists,
                _posterior_rollout_final_only=posterior_rollout_final_only,
                **kwargs,
            )

        if results is not None:
            for site_name, trajectory in results.items():
                numpyro.deterministic(f"{name}_{site_name}", trajectory)

        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            _obs_values_filled=_obs_values_filled,
            _obs_mask=_obs_mask,
            _obs_has_missing=_obs_has_missing,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            **kwargs,
        )

    def _register_observation_log_probs(self, name: str, observation_log_probs):
        for t_idx in range(len(observation_log_probs)):
            numpyro.deterministic(
                f"{name}_y_{t_idx}_lp",
                observation_log_probs[t_idx],
            )
        numpyro.factor(f"{name}_observation_lp", jnp.sum(observation_log_probs))

    def _register_observation_terms(self, name: str, obs_values, observation_log_probs):
        for t_idx in range(len(observation_log_probs)):
            numpyro.deterministic(f"{name}_y_{t_idx}", obs_values[t_idx])
        self._register_observation_log_probs(name, observation_log_probs)

    def _build_discrete_latent_path(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        times,
        ctrl_values,
    ):
        with numpyro.plate(f"{name}_n_simulations", 1):
            x_prev_site = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        x0 = x_prev_site[0]
        if len(times) == 1:
            return jnp.expand_dims(x0, axis=0), jnp.array(
                0.0, dtype=jnp.asarray(x0).dtype
            )

        latent_shape = (len(times) - 1, *jnp.shape(x0))
        dummy_dist = (
            dist.Normal(0.0, 1.0).expand(latent_shape).to_event(len(latent_shape))
        )
        with numpyro.plate(f"{name}_n_simulations", 1):
            x_latents_site = numpyro.sample(f"{name}_x_latents", dummy_dist)
        x_latents = x_latents_site[0]
        states = jnp.concatenate([jnp.expand_dims(x0, axis=0), x_latents], axis=0)
        dummy_log_prob = dummy_dist.log_prob(x_latents)
        return states, dummy_log_prob

    def _simulate_dirac_fully_observed_path(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        times,
        ctrl_times,
        ctrl_values,
        obs_values,
    ):
        state_dim = 1 if obs_values.ndim == 1 else obs_values.shape[-1]
        obs_mask_2d = jnp.ones((len(times), state_dim), dtype=bool)

        with numpyro.plate(f"{name}_n_simulations", 1):
            numpyro.sample(
                f"{name}_x_0",
                dynamics.initial_condition,
                obs=jnp.expand_dims(obs_values[0], axis=0),
            )
        state_array = jnp.asarray(obs_values)
        terms = discrete_path_score_terms(
            dynamics,
            state_array,
            obs_times=times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            precomputed_filled_obs=obs_values
            if obs_values.ndim > 1
            else obs_values[:, None],
            precomputed_obs_mask=obs_mask_2d,
        )
        for t_idx in range(len(times)):
            numpyro.deterministic(f"{name}_y_{t_idx}", obs_values[t_idx])
        numpyro.factor(f"{name}_latent_lp", jnp.sum(terms.transition_log_probs))
        self._register_observation_log_probs(name, terms.observation_log_probs)
        numpyro.deterministic(f"{name}_init_lp", terms.init_log_prob)
        numpyro.deterministic(f"{name}_transition_lp", terms.transition_log_probs)
        numpyro.deterministic(
            f"{name}_observation_total_lp",
            terms.observation_log_prob,
        )
        numpyro.deterministic(f"{name}_path_log_prob", terms.total_log_prob)
        return {
            "times": _tile_times(times, 1),
            "states": _ensure_trailing_dim(jnp.expand_dims(state_array, axis=0)),
            "observations": _ensure_trailing_dim(jnp.expand_dims(obs_values, axis=0)),
        }

    def _simulate_dirac_missing_path(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        times,
        ctrl_times,
        ctrl_values,
        obs_values,
        _obs_mask,
    ):
        if _obs_mask is None:
            raise ValueError("Dirac latent-state building expects a precomputed mask.")

        obs_mask_2d = _obs_mask[:, None] if obs_values.ndim == 1 else _obs_mask
        row_has_all_observed = jnp.all(obs_mask_2d, axis=-1)
        row_has_any_observed = jnp.any(obs_mask_2d, axis=-1)
        _raise_now_or_error_if(
            obs_values,
            jnp.any(row_has_any_observed & ~row_has_all_observed),
            "LatentStateBuilder currently requires DiracIdentityObservation rows "
            "to be either fully observed or fully missing.",
        )

        state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)
        states = []

        if bool(row_has_all_observed[0]):
            with numpyro.plate(f"{name}_n_simulations", 1):
                x_prev_site = numpyro.sample(
                    f"{name}_x_0",
                    dynamics.initial_condition,
                    obs=jnp.expand_dims(obs_values[0], axis=0),
                )
            x_prev = x_prev_site[0]
        else:
            with numpyro.plate(f"{name}_n_simulations", 1):
                x_prev_site = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
            x_prev = x_prev_site[0]
        states.append(x_prev)
        numpyro.deterministic(f"{name}_y_0", obs_values[0])

        for t_idx in range(len(times) - 1):
            t_now = times[t_idx]
            t_next = times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            trans_dist = state_transition(x=x_prev, u=u_now, t_now=t_now, t_next=t_next)
            if bool(row_has_all_observed[t_idx + 1]):
                with numpyro.plate(f"{name}_n_simulations", 1):
                    x_site = numpyro.sample(
                        f"{name}_x_{t_idx + 1}",
                        trans_dist,
                        obs=jnp.expand_dims(obs_values[t_idx + 1], axis=0),
                    )
                x_prev = x_site[0]
            else:
                with numpyro.plate(f"{name}_n_simulations", 1):
                    x_site = numpyro.sample(f"{name}_x_{t_idx + 1}", trans_dist)
                x_prev = x_site[0]
            numpyro.deterministic(f"{name}_y_{t_idx + 1}", obs_values[t_idx + 1])
            states.append(x_prev)

        state_array = jnp.stack(states, axis=0)
        terms = discrete_path_score_terms(
            dynamics,
            state_array,
            obs_times=times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            precomputed_filled_obs=obs_values
            if obs_values.ndim > 1
            else obs_values[:, None],
            precomputed_obs_mask=obs_mask_2d,
        )
        self._register_observation_log_probs(name, terms.observation_log_probs)
        numpyro.deterministic(f"{name}_init_lp", terms.init_log_prob)
        numpyro.deterministic(f"{name}_transition_lp", terms.transition_log_probs)
        numpyro.deterministic(
            f"{name}_observation_total_lp",
            terms.observation_log_prob,
        )
        numpyro.deterministic(f"{name}_path_log_prob", terms.total_log_prob)
        return {
            "times": _tile_times(times, 1),
            "states": _ensure_trailing_dim(jnp.expand_dims(state_array, axis=0)),
            "observations": _ensure_trailing_dim(jnp.expand_dims(obs_values, axis=0)),
        }

    def _simulate_ode_conditioned(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        times,
        obs_values,
        _obs_values_filled=None,
        _obs_mask=None,
        ctrl_times=None,
        ctrl_values=None,
    ):
        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval = lambda t: control_path.evaluate(t, left=False)
        else:
            control_path_eval = lambda t: None

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]
        dt0_arr = as_scalar_time_array(1e-3, name="dt0", dtype=times.dtype)
        diffeqsolve_settings = {
            "solver": dfx.Tsit5(),
            "stepsize_controller": dfx.ConstantStepSize(),
            "adjoint": dfx.RecursiveCheckpointAdjoint(),
            "dt0": dt0_arr,
            "max_steps": 100_000,
        }
        with numpyro.plate(f"{name}_n_simulations", 1):
            x0 = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        states = solve_ode(
            dynamics,
            t0,
            times,
            jnp.asarray(x0)[0],
            control_path_eval,
            diffeqsolve_settings,
        )
        ctrl_values_aligned = None
        if ctrl_times is not None and ctrl_values is not None:
            inds = jnp.searchsorted(ctrl_times, times, side="left")
            ctrl_values_aligned = ctrl_values[inds]
        obs_lp_terms = observation_log_prob_terms(
            dynamics,
            states,
            obs_times=times,
            obs_values=obs_values,
            ctrl_values=ctrl_values_aligned,
            precomputed_filled_obs=(
                None
                if _obs_values_filled is None
                else (
                    _obs_values_filled[:, None]
                    if obs_values.ndim == 1
                    else _obs_values_filled
                )
            ),
            precomputed_obs_mask=(
                None
                if _obs_mask is None
                else (_obs_mask[:, None] if obs_values.ndim == 1 else _obs_mask)
            ),
        )
        self._register_observation_terms(name, obs_values, obs_lp_terms)
        init_lp = dynamics.initial_condition.log_prob(jnp.asarray(x0)[0])
        numpyro.deterministic(f"{name}_init_lp", init_lp)
        numpyro.deterministic(
            f"{name}_transition_lp",
            jnp.zeros((0,), dtype=init_lp.dtype),
        )
        numpyro.deterministic(
            f"{name}_observation_total_lp",
            jnp.sum(obs_lp_terms),
        )
        numpyro.deterministic(
            f"{name}_path_log_prob",
            init_lp + jnp.sum(obs_lp_terms),
        )
        observations = obs_values
        return {
            "times": _tile_times(times, 1),
            "states": jnp.expand_dims(states, axis=0),
            "observations": jnp.expand_dims(observations, axis=0),
        }

    def _simulate(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        _obs_values_filled=None,
        _obs_mask=None,
        _obs_has_missing=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ):
        del predict_times, kwargs
        if obs_times is None or obs_values is None:
            raise ValueError(
                "LatentStateBuilder requires obs_times and obs_values for path construction."
            )
        if self.n_simulations != 1:
            raise ValueError(
                "LatentStateBuilder currently supports n_simulations=1 only."
            )

        has_missing = _obs_has_missing
        if has_missing is None and _obs_mask is not None:
            try:
                has_missing = bool(jnp.any(~_obs_mask))
            except TracerBoolConversionError:
                has_missing = None
        if has_missing is None:
            try:
                has_missing = bool(np.isnan(np.asarray(obs_values)).any())
            except Exception:
                has_missing = None

        if not dynamics.continuous_time:
            if isinstance(dynamics.observation_model, DiracIdentityObservation):
                if has_missing is False:
                    return self._simulate_dirac_fully_observed_path(
                        name,
                        dynamics,
                        times=obs_times,
                        ctrl_times=ctrl_times,
                        ctrl_values=ctrl_values,
                        obs_values=obs_values,
                    )
                if has_missing is True:
                    return self._simulate_dirac_missing_path(
                        name,
                        dynamics,
                        times=obs_times,
                        ctrl_times=ctrl_times,
                        ctrl_values=ctrl_values,
                        obs_values=obs_values,
                        _obs_mask=_obs_mask,
                    )
            states, dummy_log_prob = self._build_discrete_latent_path(
                name,
                dynamics,
                times=obs_times,
                ctrl_values=ctrl_values,
            )
            terms = discrete_path_score_terms(
                dynamics,
                states,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                precomputed_filled_obs=(
                    None
                    if _obs_values_filled is None
                    else (
                        _obs_values_filled[:, None]
                        if obs_values.ndim == 1
                        else _obs_values_filled
                    )
                ),
                precomputed_obs_mask=(
                    None
                    if _obs_mask is None
                    else (_obs_mask[:, None] if obs_values.ndim == 1 else _obs_mask)
                ),
            )
            self._register_observation_terms(
                name,
                obs_values,
                terms.observation_log_probs,
            )
            numpyro.factor(
                f"{name}_latent_lp",
                jnp.sum(terms.transition_log_probs) - dummy_log_prob,
            )
            numpyro.deterministic(f"{name}_init_lp", terms.init_log_prob)
            numpyro.deterministic(f"{name}_transition_lp", terms.transition_log_probs)
            numpyro.deterministic(f"{name}_latent_adjustment_lp", -dummy_log_prob)
            numpyro.deterministic(
                f"{name}_observation_total_lp",
                terms.observation_log_prob,
            )
            numpyro.deterministic(f"{name}_path_log_prob", terms.total_log_prob)
            return {
                "times": _tile_times(obs_times, 1),
                "states": _ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
                "observations": _ensure_trailing_dim(
                    jnp.expand_dims(obs_values, axis=0)
                ),
            }

        if isinstance(
            dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
        ):
            return self._simulate_ode_conditioned(
                name,
                dynamics,
                times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

        raise NotImplementedError(
            "LatentStateBuilder currently supports discrete-time or deterministic "
            "continuous-time models. For SDEs, discretize first or use filtering."
        )


__all__ = ["LatentStateBuilder"]
