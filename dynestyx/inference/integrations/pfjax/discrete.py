import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
from jax import lax, random
from jax.scipy.special import logsumexp

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    MarginalPFConfig,
    PFConfig,
    PFResamplingConfig,
    _config_to_record_kwargs,
)
from dynestyx.models import DynamicalModel
from dynestyx.utils import (
    _should_record_field,
    _validate_control_dim,
    _validate_controls,
)
from pfjax.particle_filter import particle_filter as pfjax_particle_filter
from pfjax.particle_filter import particle_filter_rb as pfjax_particle_filter_rb


def _config_to_filter_kwargs(config: BaseFilterConfig) -> dict:
    """Build filter kwargs dict from config dataclass."""
    kwargs = dict(config.extra_filter_kwargs)
    if isinstance(config, PFConfig):
        kwargs["n_filter_particles"] = config.n_particles
        kwargs["ess_threshold"] = config.ess_threshold_ratio
    if isinstance(config, MarginalPFConfig):
        kwargs["stop_proposal_gradient"] = config.stop_proposal_gradient
    return kwargs


def _tree_take(tree: Any, idx: jax.Array) -> Any:
    """Index a pytree along axis 0."""
    return jax.tree.map(lambda x: x[idx], tree)


# TODO: Move these to a separate file, and reuse the multinomial resampling from PFJax.
def _resample_multinomial(
    key: jax.Array, x_particles_prev: Any, logw: jax.Array
) -> dict[str, Any]:
    n_particles = int(logw.shape[0])
    logw_norm = logw - logsumexp(logw)
    prob = jnp.exp(logw_norm)
    ancestors = random.choice(
        key, a=jnp.arange(n_particles), shape=(n_particles,), p=prob
    )
    return {
        "x_particles": _tree_take(x_particles_prev, ancestors),
        "ancestors": ancestors,
    }


def _resample_systematic(
    key: jax.Array, x_particles_prev: Any, logw: jax.Array
) -> dict[str, Any]:
    n_particles = int(logw.shape[0])
    logw_norm = logw - logsumexp(logw)
    prob = jnp.exp(logw_norm)
    cdf = jnp.cumsum(prob)
    u = (random.uniform(key, ()) + jnp.arange(n_particles)) / n_particles
    ancestors = jnp.searchsorted(cdf, u, side="right")
    ancestors = jnp.minimum(ancestors, n_particles - 1)
    return {
        "x_particles": _tree_take(x_particles_prev, ancestors),
        "ancestors": ancestors,
    }


def _resample_stratified(
    key: jax.Array, x_particles_prev: Any, logw: jax.Array
) -> dict[str, Any]:
    n_particles = int(logw.shape[0])
    logw_norm = logw - logsumexp(logw)
    prob = jnp.exp(logw_norm)
    cdf = jnp.cumsum(prob)
    u = (jnp.arange(n_particles) + random.uniform(key, (n_particles,))) / n_particles
    ancestors = jnp.searchsorted(cdf, u, side="right")
    ancestors = jnp.minimum(ancestors, n_particles - 1)
    return {
        "x_particles": _tree_take(x_particles_prev, ancestors),
        "ancestors": ancestors,
    }


def _make_resampler(
    resampling_config: PFResamplingConfig,
    ess_threshold: float,
):
    base_method = resampling_config.base_method
    if base_method == "multinomial":
        base_resampler = _resample_multinomial
    elif base_method == "systematic":
        base_resampler = _resample_systematic
    elif base_method == "stratified":
        base_resampler = _resample_stratified
    else:
        raise ValueError(f"Unsupported PF resampling base method: {base_method}")

    def resampler(
        key: jax.Array, x_particles_prev: Any, logw: jax.Array
    ) -> dict[str, Any]:
        n_particles = int(logw.shape[0])
        logw_norm = logw - logsumexp(logw)
        prob = jnp.exp(logw_norm)
        ess = 1.0 / jnp.sum(jnp.square(prob))
        threshold = ess_threshold * n_particles

        def _do_resample(_: None) -> jax.Array:
            return base_resampler(
                key=key, x_particles_prev=x_particles_prev, logw=logw
            )["ancestors"]

        def _skip_resample(_: None) -> jax.Array:
            return jnp.arange(n_particles, dtype=jnp.int32)

        ancestors = lax.cond(
            ess < threshold, _do_resample, _skip_resample, operand=None
        )
        return {
            "x_particles": _tree_take(x_particles_prev, ancestors),
            "ancestors": ancestors,
        }

    return resampler


def _resolve_stop_gradient(filter_config: PFConfig) -> bool:
    method = filter_config.resampling_method.differential_method
    if method == "stop_gradient":
        return True
    if method == "straight_through":
        return False
    raise ValueError(
        "PFJax integration does not support "
        "resampling_method.differential_method='soft'."
    )


class _PFJaxDynestyxModel:
    """PFJax model adapter for dynestyx discrete-time dynamical models."""

    def __init__(self, dynamics: DynamicalModel):
        self.dynamics = dynamics

    def pf_init(self, key, y_init, theta):
        del theta
        x_init = self.dynamics.initial_condition.sample(key)
        edist = self.dynamics.observation_model(x_init, y_init["u"], y_init["time"])
        logw = jnp.asarray(edist.log_prob(y_init["y"])).sum()
        return x_init, logw

    def pf_step(self, key, x_prev, y_curr, theta):
        del theta
        pdist = self.dynamics.state_evolution(
            x_prev, y_curr["u_prev"], y_curr["time_prev"], y_curr["time"]
        )
        x_curr = pdist.sample(key)
        edist = self.dynamics.observation_model(x_curr, y_curr["u"], y_curr["time"])
        logw = jnp.asarray(edist.log_prob(y_curr["y"])).sum()
        return x_curr, logw


class _PFJaxMarginalDynestyxModel:
    """PFJax marginal PF model adapter for dynestyx discrete-time models."""

    def __init__(self, dynamics: DynamicalModel):
        self.dynamics = dynamics

    def _pack_state(
        self, x: jax.Array, y_step: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        return {
            "x": x,
            "u": y_step["u"],
            "u_prev": y_step["u_prev"],
            "time": y_step["time"],
            "time_prev": y_step["time_prev"],
        }

    def pf_init(self, key, y_init, theta):
        del theta
        x_raw = self.dynamics.initial_condition.sample(key)
        x_init = self._pack_state(x_raw, y_init)
        logw = self.meas_lpdf(y_curr=y_init, x_curr=x_init, theta=jnp.zeros((0,)))
        return x_init, logw

    def step_sample(self, key, x_prev, y_curr, theta):
        del theta
        pdist = self.dynamics.state_evolution(
            x_prev["x"], y_curr["u_prev"], y_curr["time_prev"], y_curr["time"]
        )
        x_raw = pdist.sample(key)
        return self._pack_state(x_raw, y_curr)

    def step_lpdf(self, x_curr, x_prev, y_curr, theta):
        del theta, y_curr
        return self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=jnp.zeros((0,)))

    def state_lpdf(self, x_curr, x_prev, theta):
        del theta
        pdist = self.dynamics.state_evolution(
            x_prev["x"], x_curr["u_prev"], x_curr["time_prev"], x_curr["time"]
        )
        return jnp.asarray(pdist.log_prob(x_curr["x"])).sum()

    def meas_lpdf(self, y_curr, x_curr, theta):
        del theta
        edist = self.dynamics.observation_model(
            x_curr["x"], y_curr["u"], y_curr["time"]
        )
        return jnp.asarray(edist.log_prob(y_curr["y"])).sum()


def _normalize_marginal_pf_out(pf_out: dict[str, Any]) -> dict[str, Any]:
    out = dict(pf_out)
    out["x_particles"] = pf_out["x_particles"]["x"]
    out["logw"] = pf_out["logw_bar"]
    return out


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> None:
    """Run discrete-time particle filter via PFJax."""
    del kwargs

    if not isinstance(filter_config, (PFConfig, MarginalPFConfig)):
        raise ValueError(
            f"Unsupported pfjax config: {type(filter_config).__name__}. "
            "Expected PFConfig or MarginalPFConfig."
        )
    if key is None:
        raise ValueError(
            "Particle filter requires a PRNG key: set 'crn_seed' in the filter config, "
            "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
        )

    filter_kwargs = _config_to_filter_kwargs(filter_config)
    record_kwargs = _config_to_record_kwargs(filter_config)

    ys = obs_values
    t1 = int(ys.shape[0])
    if t1 == 0:
        return

    times = obs_times
    _validate_controls(times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    if ctrl_values is None:
        control_dim = dynamics.control_dim
        ctrl_values = jnp.zeros((t1, control_dim), dtype=ys.dtype)

    dt0 = times[1] - times[0] if t1 > 1 else jnp.asarray(1.0, dtype=times.dtype)
    time_prev = jnp.concatenate([times[:1] - dt0, times[:-1]], axis=0)
    u_prev = jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0)

    y_meas = {
        "y": ys,
        "u": ctrl_values,
        "u_prev": u_prev,
        "time": times,
        "time_prev": time_prev,
    }

    n_particles = int(filter_kwargs.pop("n_filter_particles", 1_000))
    ess_threshold = float(filter_kwargs.pop("ess_threshold", 0.7))

    stop_gradient = _resolve_stop_gradient(filter_config)
    if stop_gradient and ess_threshold < 1.0:
        if filter_config.warn:
            warnings.warn(
                "PFJax stop-gradient mode requires resampling at every step for "
                "its correction term to be valid. Overriding ess_threshold_ratio "
                f"from {ess_threshold} to 1.0.",
                stacklevel=2,
            )
        ess_threshold = 1.0
    resampler = _make_resampler(filter_config.resampling_method, ess_threshold)

    theta = filter_kwargs.pop("theta", jnp.zeros((0,), dtype=ys.dtype))
    for reserved in (
        "model",
        "key",
        "y_meas",
        "n_particles",
        "resampler",
        "history",
        "stop_gradient",
    ):
        filter_kwargs.pop(reserved, None)

    if isinstance(filter_config, MarginalPFConfig):
        stop_proposal_gradient = bool(filter_kwargs.pop("stop_proposal_gradient", True))
        model = _PFJaxMarginalDynestyxModel(dynamics)
        pf_out = pfjax_particle_filter_rb(
            model=model,
            key=key,
            y_meas=y_meas,
            theta=theta,
            n_particles=n_particles,
            resampler=resampler,  # type: ignore[arg-type]
            history=True,
            stop_gradient=stop_gradient,
            stop_proposal_gradient=stop_proposal_gradient,
            **filter_kwargs,
        )
        pf_out = _normalize_marginal_pf_out(pf_out)
    else:
        model = _PFJaxDynestyxModel(dynamics)  # type: ignore
        pf_out = pfjax_particle_filter(
            model=model,
            key=key,
            y_meas=y_meas,
            theta=theta,
            n_particles=n_particles,
            resampler=resampler,  # type: ignore[arg-type]
            history=True,
            stop_gradient=stop_gradient,
            **filter_kwargs,
        )

    marginal_loglik = pf_out["loglik"]
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)
    _add_sites_pf(name, pf_out, record_kwargs)


def _add_sites_pf(name: str, pf_out: dict[str, Any], record_kwargs: dict):
    particles = pf_out["x_particles"]
    log_weights = pf_out["logw"]
    if particles.ndim == 2:
        particles = particles[..., None]

    max_elems = record_kwargs["record_max_elems"]
    t1, n_particles, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs["record_filtered_particles"], particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs["record_filtered_log_weights"], log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], (t1, state_dim), max_elems
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (t1, state_dim), max_elems
    )

    need_filtered_moments = (
        add_mean or add_filtered_states_cov or add_filtered_states_cov_diag
    )
    if need_filtered_moments:
        log_weights_norm = log_weights - logsumexp(log_weights, axis=1, keepdims=True)
        w = jnp.exp(log_weights_norm)[..., None]
        filtered_means = jnp.sum(particles * w, axis=1)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        second_mom = jnp.einsum(
            "...tnj,...tnk,...tn->...tjk", particles, particles, w.squeeze(-1)
        )
        filtered_covariances = second_mom - jnp.einsum(
            "...tj,...tk->...tjk", filtered_means, filtered_means
        )

    if add_particles:
        numpyro.deterministic(f"{name}_filtered_particles", particles)
    if add_log_weights:
        numpyro.deterministic(f"{name}_filtered_log_weights", log_weights)
    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", filtered_means)
    if add_filtered_states_cov:
        numpyro.deterministic(f"{name}_filtered_states_cov", filtered_covariances)
    if add_filtered_states_cov_diag:
        diag_cov = jnp.diagonal(filtered_covariances, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
