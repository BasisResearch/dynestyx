"""Discrete-time smoothers via cuthbert: Kalman, Taylor-KF, and PF backward sampling."""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cuthbert import smoother as cuthbert_smoother
from cuthbert.gaussian import kalman, taylor
from cuthbert.smc import backward_sampler
from cuthbertlib.resampling import multinomial, stop_gradient_decorator, systematic
from cuthbertlib.smc.smoothing import exact_sampling, mcmc, tracing

from dynestyx.inference.filter_configs import _config_to_smoother_record_kwargs
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    CuthbertInputs,
    _config_to_filter_kwargs,
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.utils import particles_to_delta_mixtures
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
from dynestyx.utils import _should_record_field

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

    m0 = jnp.reshape(jnp.atleast_1d(jnp.asarray(ic.loc)), (state_dim,))

    A = jnp.asarray(evo.A)
    chol_Q = jnp.linalg.cholesky(jnp.asarray(evo.cov))

    evo_bias = (
        jnp.zeros((state_dim,), dtype=m0.dtype)
        if evo.bias is None
        else jnp.reshape(jnp.atleast_1d(jnp.asarray(evo.bias)), (state_dim,))
    )

    B = None if evo.B is None else jnp.asarray(evo.B)

    def get_dynamics_params(mi: CuthbertInputs):
        def _noop(mi):
            return (
                jnp.eye(state_dim, dtype=m0.dtype),
                jnp.zeros((state_dim,), dtype=m0.dtype),
                jnp.zeros((state_dim, state_dim), dtype=m0.dtype),
            )

        def _evolve(mi):
            c = evo_bias
            if B is not None:
                c = c + B @ jnp.atleast_1d(jnp.asarray(mi.u_prev))
            return A, c, chol_Q

        return jax.lax.cond(mi.is_first_step, _noop, _evolve, mi)

    return get_dynamics_params


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
    marginal_loglik, filtered_states = compute_cuthbert_filter(
        dynamics,
        smoother_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
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

    return marginal_loglik, smoothed_states


def _add_sites_pf(name: str, states, record_kwargs: dict):
    log_weights = states.log_weights[1:]
    particles = states.particles[1:]
    if particles.ndim == 2:
        particles = particles[..., None]
    max_elems = record_kwargs["record_max_elems"]
    t1, _, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs["record_smoothed_particles"], particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs["record_smoothed_log_weights"], log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], (t1, state_dim), max_elems
    )
    add_smoothed_states_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_smoothed_states_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"], (t1, state_dim), max_elems
    )

    need_means = add_mean or add_smoothed_states_cov or add_smoothed_states_cov_diag
    if need_means:
        w = jax.nn.softmax(log_weights, axis=1)[..., None]
        smoothed_means = jnp.sum(particles * w, axis=1)

    if add_smoothed_states_cov or add_smoothed_states_cov_diag:
        second_mom = jnp.einsum(
            "...tnj,...tnk,...tn->...tjk", particles, particles, w.squeeze(-1)
        )
        smoothed_cov = second_mom - jnp.einsum(
            "...tj,...tk->...tjk", smoothed_means, smoothed_means
        )

    if add_particles:
        numpyro.deterministic(f"{name}_smoothed_particles", particles)
    if add_log_weights:
        numpyro.deterministic(f"{name}_smoothed_log_weights", log_weights)
    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", smoothed_means)
    if add_smoothed_states_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", smoothed_cov)
    if add_smoothed_states_cov_diag:
        diag_cov = jnp.diagonal(smoothed_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)


def _add_sites_taylor_kf(name: str, states, record_kwargs: dict):
    max_elems = record_kwargs["record_max_elems"]
    mean = states.mean[1:]
    chol_cov = states.chol_cov[1:]
    t1, state_dim, _ = chol_cov.shape

    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], mean.shape, max_elems
    )
    add_chol_cov = _should_record_field(
        record_kwargs["record_smoothed_states_chol_cov"],
        chol_cov.shape,
        max_elems,
    )
    add_smoothed_states_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_smoothed_states_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"], (t1, state_dim), max_elems
    )

    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", mean)
    if add_chol_cov:
        numpyro.deterministic(f"{name}_smoothed_states_chol_cov", chol_cov)

    if add_smoothed_states_cov or add_smoothed_states_cov_diag:
        chol_t = jnp.transpose(chol_cov, (0, 2, 1))
        smoothed_cov = jnp.matmul(chol_cov, chol_t)

    if add_smoothed_states_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", smoothed_cov)
    if add_smoothed_states_cov_diag:
        diag_cov = jnp.diagonal(smoothed_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)


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
) -> list[dist.Distribution]:
    """Run discrete-time smoother via cuthbert."""
    t1 = int(obs_values.shape[0])
    if t1 == 0:
        return []

    marginal_loglik, states = compute_cuthbert_smoother(
        dynamics,
        smoother_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    record_kwargs = _config_to_smoother_record_kwargs(smoother_config)

    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    if isinstance(smoother_config, PFSmootherConfig):
        _add_sites_pf(name, states, record_kwargs)
        particles = states.particles[1:]
        log_weights = states.log_weights[1:]
        if particles.ndim == 2:
            particles = particles[..., None]
        return particles_to_delta_mixtures(particles, log_weights)

    _add_sites_taylor_kf(name, states, record_kwargs)
    mean = states.mean[1:]
    chol_cov = states.chol_cov[1:]
    chol_t = jnp.transpose(chol_cov, (0, 2, 1))
    cov = jnp.matmul(chol_cov, chol_t)
    return [
        dist.MultivariateNormal(mean[i], covariance_matrix=cov[i])
        for i in range(mean.shape[0])
    ]


__all__ = ["compute_cuthbert_smoother", "run_discrete_smoother"]
