from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx.dynamical_models import (
    Context,
    ContinuousTimeStateEvolution,
    DynamicalModel,
    ObservationModel,
    Trajectory,
)
from dynestyx.observations import DiracIdentityObservation, LinearGaussianObservation


class LinearDrift(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray | None
    use_controls: bool

    def __call__(self, x, u, t):
        x_vec = _as_vector(x)
        out = self.A @ x_vec
        if self.use_controls and u is not None and self.B is not None:
            out = out + self.B @ _as_vector(u)
        return out


class ZeroDrift(eqx.Module):
    def __call__(self, x, u, t):
        return jnp.zeros_like(_as_vector(x))


class GaussianTransition(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray | None
    cov: jnp.ndarray
    nonlinear: bool
    use_controls: bool

    def __call__(self, x, u, t_now, t_next):
        x_vec = _as_vector(x)
        loc = _linear_map(self.A, x_vec)
        if self.nonlinear:
            loc = jnp.tanh(loc)
        if self.use_controls and u is not None and self.B is not None:
            u_vec = _as_vector(u)
            loc = loc + _linear_map(self.B, u_vec)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


class CategoricalTransition(eqx.Module):
    probs: jnp.ndarray
    use_controls: bool

    def __call__(self, x, u, t_now, t_next):
        probs = self.probs[x]
        if self.use_controls and u is not None:
            u_effect = jnp.sum(_as_vector(u))
            probs = probs + u_effect * (1.0 / self.probs.shape[0] - probs)
        probs = jnp.clip(probs, 1e-6, 1.0)
        probs = probs / jnp.sum(probs)
        return dist.Categorical(probs=probs)


class PoissonObservation(ObservationModel):
    bias: jnp.ndarray
    rate_scale: float = eqx.field(static=True, default=1.0)

    def __call__(self, x, u, t):
        x_vec = _as_vector(x)
        if x_vec.size == self.bias.size:
            x_obs = x_vec
        elif x_vec.size == 1:
            x_obs = jnp.broadcast_to(x_vec, self.bias.shape)
        else:
            x_obs = x_vec[: self.bias.size]
        rate = self.rate_scale * jnp.exp(x_obs + self.bias)
        return dist.Poisson(rate=rate).to_event(1)


def _as_vector(x) -> jnp.ndarray:
    x_arr = jnp.asarray(x)
    if x_arr.ndim == 0:
        return x_arr[None]
    return x_arr


def _linear_map(A: jnp.ndarray, x_vec: jnp.ndarray) -> jnp.ndarray:
    if x_vec.ndim == 1:
        return A @ x_vec
    if x_vec.ndim == 2:
        if x_vec.shape[0] == A.shape[0]:
            return (A @ x_vec).T
        if x_vec.shape[1] == A.shape[0]:
            return x_vec @ A.T
    return A @ x_vec


def _make_obs_values(observation_kind: str, timesteps: int, dim: int) -> jnp.ndarray:
    dtype = jnp.int32 if observation_kind == "poisson" else jnp.float32
    return jnp.zeros((timesteps, dim), dtype=dtype)


def _make_controls(timesteps: int, control_dim: int) -> jnp.ndarray:
    return jnp.ones((timesteps, control_dim), dtype=jnp.float32)


def _build_initial_condition(init_kind: str, dim: int):
    if init_kind == "mvn":
        return dist.MultivariateNormal(
            loc=jnp.zeros(dim), covariance_matrix=jnp.eye(dim)
        )
    if init_kind == "uniform":
        return dist.Uniform(low=-1.0 * jnp.ones(dim), high=1.0 * jnp.ones(dim))
    if init_kind == "categorical":
        probs = jnp.ones(dim) / dim
        return dist.Categorical(probs=probs)
    raise ValueError(f"Unknown init_kind: {init_kind}")


def _build_observation_model(observation_kind: str, dim: int):
    if observation_kind == "linear_gaussian":
        H = jnp.eye(dim)
        R = 0.1 * jnp.eye(dim)
        return LinearGaussianObservation(H=H, R=R)
    if observation_kind == "perfect":
        return DiracIdentityObservation()
    if observation_kind == "poisson":
        return PoissonObservation(bias=jnp.zeros(dim), rate_scale=0.5)
    raise ValueError(f"Unknown observation_kind: {observation_kind}")


def _build_drift(
    drift_kind: str,
    state_evolution_kind: str,
    dim: int,
    control_dim: int,
    use_controls: bool,
):
    if drift_kind == "linear":
        A = 0.1 * jnp.eye(dim)
        B = 0.05 * jnp.ones((dim, control_dim)) if use_controls else None
        if state_evolution_kind == "module":
            return LinearDrift(A=A, B=B, use_controls=use_controls)

        def drift(x, u, t):
            x_vec = _as_vector(x)
            out = A @ x_vec
            if use_controls and u is not None and B is not None:
                out = out + B @ _as_vector(u)
            return out

        return drift
    if drift_kind == "zero":
        if state_evolution_kind == "module":
            return ZeroDrift()

        def drift(x, u, t):
            return jnp.zeros_like(_as_vector(x))

        return drift
    raise ValueError(f"Unknown drift_kind: {drift_kind}")


def _build_continuous_state_evolution(
    drift_kind: str,
    diffusion_coeff: str,
    diffusion_cov: str,
    state_evolution_kind: str,
    dim: int,
    control_dim: int,
    use_controls: bool,
):
    drift = _build_drift(
        drift_kind, state_evolution_kind, dim, control_dim, use_controls
    )
    diffusion_coeff_fn = (
        (lambda x, u, t: jnp.eye(dim)) if diffusion_coeff == "eye" else None
    )
    diffusion_cov_fn = (
        (lambda x, u, t: jnp.eye(dim)) if diffusion_cov == "eye" else None
    )
    return ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=diffusion_coeff_fn,
        diffusion_covariance=diffusion_cov_fn,
    )


def _build_discrete_state_evolution(
    transition_kind: str,
    state_evolution_kind: str,
    dim: int,
    control_dim: int,
    use_controls: bool,
):
    cov = 0.1 * jnp.eye(dim)
    A = 0.5 * jnp.eye(dim)
    B = 0.25 * jnp.ones((dim, control_dim)) if use_controls else None

    if transition_kind in ("mvn_linear", "mvn_nonlinear"):
        nonlinear = transition_kind == "mvn_nonlinear"
        if state_evolution_kind == "module":
            return GaussianTransition(
                A=A, B=B, cov=cov, nonlinear=nonlinear, use_controls=use_controls
            )

        def transition(x, u, t_now, t_next):
            x_vec = _as_vector(x)
            loc = _linear_map(A, x_vec)
            if nonlinear:
                loc = jnp.tanh(loc)
            if use_controls and u is not None and B is not None:
                u_vec = _as_vector(u)
                loc = loc + _linear_map(B, u_vec)
            return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

        return transition

    if transition_kind == "categorical":
        probs = jnp.ones((dim, dim)) / dim
        if state_evolution_kind == "module":
            return CategoricalTransition(probs=probs, use_controls=use_controls)

        def transition(x, u, t_now, t_next):
            row = probs[x]
            if use_controls and u is not None:
                u_effect = jnp.sum(_as_vector(u))
                row = row + u_effect * (1.0 / dim - row)
            row = jnp.clip(row, 1e-6, 1.0)
            row = row / jnp.sum(row)
            return dist.Categorical(probs=row)

        return transition

    raise ValueError(f"Unknown transition_kind: {transition_kind}")


def _build_model(
    model_kind: str,
    dim: int,
    control_dim: int,
    init_kind: str,
    state_evolution_kind: str,
    drift_kind: str | None,
    diffusion_coeff: str | None,
    diffusion_cov: str | None,
    transition_kind: str | None,
    observation_kind: str,
):
    initial_condition = _build_initial_condition(init_kind, dim)
    observation_model = _build_observation_model(observation_kind, dim)

    if model_kind == "continuous":
        state_evolution = _build_continuous_state_evolution(
            drift_kind=drift_kind or "linear",
            diffusion_coeff=diffusion_coeff or "eye",
            diffusion_cov=diffusion_cov or "eye",
            state_evolution_kind=state_evolution_kind,
            dim=dim,
            control_dim=control_dim,
            use_controls=control_dim > 0,
        )
    else:
        state_evolution = _build_discrete_state_evolution(
            transition_kind=transition_kind or "mvn_linear",
            state_evolution_kind=state_evolution_kind,
            dim=dim,
            control_dim=control_dim,
            use_controls=control_dim > 0,
        )

    dynamics = DynamicalModel(
        state_dim=dim,
        observation_dim=dim,
        control_dim=control_dim,
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    def model():
        dsx.sample("f", dynamics)

    return model


def _build_context(
    observation_kind: str,
    timesteps: int,
    dim: int,
    use_controls: bool,
    control_dim: int,
) -> Context:
    obs_times = jnp.arange(timesteps, dtype=jnp.float32)
    obs_values = _make_obs_values(observation_kind, timesteps, dim)
    controls = Trajectory()
    if use_controls:
        ctrl_values = _make_controls(timesteps, control_dim)
        controls = Trajectory(times=obs_times, values=ctrl_values)
    return Context(
        observations=Trajectory(times=obs_times, values=obs_values),
        controls=controls,
    )


def _build_predictive_context(
    observation_kind: str,
    timesteps: int,
    dim: int,
    use_controls: bool,
    control_dim: int,
    context_variant: str,
) -> Context:
    obs_times = jnp.arange(timesteps, dtype=jnp.float32)
    obs_values = None
    if context_variant != "obs_times":
        obs_values = _make_obs_values(observation_kind, timesteps, dim)

    controls = Trajectory()
    if context_variant == "obs_times_values_controls" and use_controls:
        ctrl_values = _make_controls(timesteps, control_dim)
        controls = Trajectory(times=obs_times, values=ctrl_values)
    return Context(
        observations=Trajectory(times=obs_times, values=obs_values),
        controls=controls,
    )
