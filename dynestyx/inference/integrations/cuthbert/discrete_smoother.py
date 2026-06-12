"""Discrete-time smoothers via cuthbert: Kalman, Taylor-KF, and PF backward sampling."""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from cuthbert import smoother as cuthbert_smoother
from cuthbert.gaussian import kalman, taylor
from cuthbert.smc import backward_sampler
from cuthbertlib.resampling import multinomial, stop_gradient_decorator, systematic
from cuthbertlib.smc.smoothing import exact_sampling, mcmc, tracing

from dynestyx.inference.distribution_utils import _cholesky_state_sequence_to_dists
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    CuthbertInputs,
    _config_to_filter_kwargs,
    _drop_cuthbert_dummy_step,
    _kalman_dynamics_params_builder,
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.utils import (
    squeeze_leading_singletons,
)
from dynestyx.inference.smoother_configs import (
    EKFSmootherConfig,
    KFSmootherConfig,
    PFSmootherConfig,
)
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)

CuthbertSmootherConfig = KFSmootherConfig | EKFSmootherConfig | PFSmootherConfig


def _kalman_get_dynamics_params(dynamics: DynamicalModel):
    if not (
        isinstance(dynamics.state_evolution, LinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, LinearGaussianObservation)
        and isinstance(dynamics.initial_condition, dist.MultivariateNormal)
    ):
        raise TypeError(
            "cuthbert Kalman smoother expects a DynamicalModel with "
            "LinearGaussianStateEvolution and LinearGaussianObservation, and "
            "initial_condition as MultivariateNormal."
        )

    evo = dynamics.state_evolution
    ic = dynamics.initial_condition
    state_dim = dynamics.state_dim

    m0 = jnp.reshape(
        jnp.atleast_1d(squeeze_leading_singletons(ic.loc, 1)), (state_dim,)
    )

    return _kalman_dynamics_params_builder(evo, state_dim=state_dim, dtype=m0.dtype)


def _taylor_get_dynamics_log_density(dynamics: DynamicalModel):
    transition = cast(
        Callable[
            [jax.Array, jax.Array | None, jax.Array, jax.Array], dist.Distribution
        ],
        dynamics.state_evolution,
    )

    def get_dynamics_log_density(
        state: taylor.LinearizedKalmanFilterState, mi: CuthbertInputs
    ):
        def dynamics_log_density(x_prev, x):
            normal_logp = jnp.asarray(
                transition(x_prev, mi.u_prev, mi.time_prev, mi.time).log_prob(x)
            ).sum()
            noop_logp = -1e10 * jnp.sum((x - x_prev) ** 2)
            return jnp.where(mi.is_first_step, noop_logp, normal_logp)

        x_prev_lin = jnp.atleast_1d(jnp.asarray(state.mean))
        dist_at_lin = transition(x_prev_lin, mi.u_prev, mi.time_prev, mi.time)
        try:
            x_lin = jnp.atleast_1d(jnp.asarray(dist_at_lin.mean))
        except Exception as exc:
            raise ValueError(
                "dist_at_lin.mean is not available. Linearized Kalman smoother requires a mean-able distribution."
            ) from exc

        x_lin = jnp.where(mi.is_first_step, x_prev_lin, x_lin)
        return dynamics_log_density, x_prev_lin, x_lin

    return get_dynamics_log_density


def _pf_log_potential(dynamics: DynamicalModel):
    def log_potential(x_prev, x, mi: CuthbertInputs):
        edist = dynamics.observation_model(x, mi.u, mi.time)
        return jnp.asarray(edist.log_prob(mi.y)).sum()

    return log_potential


def _pf_resampling_fn(filter_kwargs: dict):
    base_method = filter_kwargs.get("resampling_base_method", "systematic")
    if base_method == "systematic":
        base_resampling_fn = systematic.resampling
    elif base_method == "multinomial":
        base_resampling_fn = multinomial.resampling
    else:
        raise ValueError(
            f"Unsupported cuthbert PF base resampling method: {base_method!r}. "
            "Expected one of: 'systematic', 'multinomial'."
        )

    differential_method = filter_kwargs.get(
        "resampling_differential_method", "stop_gradient"
    )
    if differential_method == "stop_gradient":
        base_resampling_fn = stop_gradient_decorator(base_resampling_fn)
    elif differential_method == "straight_through":
        pass
    else:
        raise ValueError(
            "Unsupported cuthbert PF differential resampling method: "
            f"{differential_method!r}. Expected one of: "
            "'stop_gradient', 'straight_through'."
        )

    return base_resampling_fn


def _pf_backward_sampling_fn(config: PFSmootherConfig):
    method = config.pf_backward_sampling_method
    if method == "tracing":
        return tracing.simulate
    if method == "exact":
        return exact_sampling.simulate
    if method == "mcmc":
        return partial(mcmc.simulate, n_steps=config.pf_mcmc_n_steps)
    raise ValueError(
        "Unsupported PF smoother backward sampling method: "
        f"{method!r}. Expected one of: 'tracing', 'exact', 'mcmc'."
    )


def compute_cuthbert_smoother(
    dynamics: DynamicalModel,
    smoother_config: CuthbertSmootherConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cuthbert smoother computation (no numpyro side-effects)."""
    obs_len = int(obs_values.shape[0])
    marginal_loglik, filtered_states = compute_cuthbert_filter(
        dynamics,
        smoother_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        align_to_observations=False,
    )

    filter_kwargs = _config_to_filter_kwargs(smoother_config)

    if isinstance(smoother_config, KFSmootherConfig):
        smoother_obj = kalman.build_smoother(
            get_dynamics_params=_kalman_get_dynamics_params(dynamics)
        )
        smoothed_states = cuthbert_smoother(
            smoother_obj, filtered_states, model_inputs=None, parallel=False, key=key
        )
    elif isinstance(smoother_config, EKFSmootherConfig):
        smoother_obj = taylor.build_smoother(
            _taylor_get_dynamics_log_density(dynamics),
            rtol=filter_kwargs.get("rtol", None),
            ignore_nan_dims=True,
        )
        smoothed_states = cuthbert_smoother(
            smoother_obj, filtered_states, model_inputs=None, parallel=False, key=key
        )
    elif isinstance(smoother_config, PFSmootherConfig):
        if key is None:
            raise ValueError(
                "Particle smoother requires a PRNG key: set 'crn_seed' in the filter config, "
                "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
            )
        n_smoother_particles = (
            smoother_config.pf_n_smoother_particles
            if smoother_config.pf_n_smoother_particles is not None
            else int(smoother_config.n_particles)
        )
        smoother_obj = backward_sampler.build_smoother(
            log_potential=_pf_log_potential(dynamics),
            backward_sampling_fn=_pf_backward_sampling_fn(smoother_config),
            resampling_fn=_pf_resampling_fn(filter_kwargs),
            n_smoother_particles=n_smoother_particles,
        )
        smoothed_states = cuthbert_smoother(
            smoother_obj,
            filtered_states,
            model_inputs=None,
            parallel=False,
            key=key,
        )
    else:
        raise ValueError(
            f"Unsupported cuthbert smoother config: {type(smoother_config).__name__}. "
            "Expected KFSmootherConfig, EKFSmootherConfig, PFSmootherConfig."
        )

    smoothed_states = _drop_cuthbert_dummy_step(smoothed_states, obs_len=obs_len)
    return marginal_loglik, smoothed_states


def run_discrete_smoother(
    name: str,
    dynamics: DynamicalModel,
    smoother_config: CuthbertSmootherConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> tuple[jax.Array | None, object | None, list[dist.Distribution]]:
    """Run discrete-time smoother via cuthbert.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}),
              or None if obs_values is empty.
            - raw_states: cuthbert smoother state object, or None if obs_values
              is empty.
            - smoothed_dists: list of distributions p(x_t | y_{1:T}) at each
              obs time, for posterior rollout.
    """
    t1 = int(obs_values.shape[0])
    if t1 == 0:
        return None, None, []

    marginal_loglik, states = compute_cuthbert_smoother(
        dynamics,
        smoother_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    smoothed_dists = _cholesky_state_sequence_to_dists(
        states,
        particle_mode=isinstance(smoother_config, PFSmootherConfig),
    )
    return marginal_loglik, states, smoothed_dists


__all__ = ["compute_cuthbert_smoother", "run_discrete_smoother"]
