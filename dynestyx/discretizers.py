import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.dynamical_models import (
    ContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
)


def _as_array(x):
    return x if isinstance(x, jnp.ndarray) else jnp.asarray(x)


class _EulerMaruyamaDiscreteEvolution(DiscreteTimeStateEvolution):
    """x_{t+1} ~ N(x + drift*dt, (L@Q@L.T)*dt).
    Batch-last convention: x is (state_dim,batch_dim (T-1))."""

    def __init__(self, cte: ContinuousTimeStateEvolution):
        self.cte = cte
        L_fn = getattr(cte, "diffusion_coefficient", None)
        if L_fn is None:
            raise AttributeError(
                "ContinuousTimeStateEvolution must define diffusion_coefficient."
            )
        self._L_fn = L_fn
        self._Q_fn = getattr(cte, "diffusion_covariance", None)
        if self._Q_fn is None:
            raise AttributeError(
                "ContinuousTimeStateEvolution must define diffusion_covariance."
            )

    def __call__(self, x, u, t_now, t_next):
        x = _as_array(x)
        t_now = _as_array(t_now)
        t_next = _as_array(t_next)

        # Canonicalize to batched "batch last": x -> (D,B), dt -> (B,)
        # comments use D for state_dim, B for batch_dim
        squeezed = False
        if x.ndim == 1:
            squeezed = True
            x = x[:, None]  # (D,1)
            if u is not None:
                u = _as_array(u)
                # Optional: if you ever pass unbatched controls with shape (U,)
                if u.ndim == 1:
                    u = u[:, None]  # (U,1)
            if t_now.ndim == 0:
                t_now = t_now[None]  # (1,)
            if t_next.ndim == 0:
                t_next = t_next[None]  # (1,)

        dt = t_next - t_now  # (B,)

        drift = self.cte.drift(x, u, t_now)  # (D,B) under your convention

        # L: (D,D) or (B,D,D)
        L = self._L_fn(x, u, t_now) if callable(self._L_fn) else self._L_fn
        L = _as_array(L)

        # Q: (D,D) or (B,D,D), default I
        Q = self._Q_fn(x, u, t_now) if callable(self._Q_fn) else self._Q_fn
        Q = _as_array(Q)

        mean = x + drift * dt[None, :]  # (D,B)

        cov0 = jnp.einsum("...ik,...kl,...jl->...ij", L, Q, L)  # (...,D,D)
        cov = cov0 * dt[..., None, None]  # (B,D,D) if batched; (D,D) if not

        loc = jnp.swapaxes(mean, 0, 1)  # (B,D)

        # If we lifted from unbatched, return unbatched dist shapes
        if squeezed:
            return dist.MultivariateNormal(loc=loc[0], covariance_matrix=cov[0])

        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)


def euler_maruyama(cte: ContinuousTimeStateEvolution) -> DiscreteTimeStateEvolution:
    """Discretize continuous-time state evolution via Euler-Maruyama. (CTSE) -> DTSE.

    Args:
        cte: ContinuousTimeStateEvolution to discretize.
    Returns:
        DiscreteTimeStateEvolution: The discretized state evolution.

    Note:
        No dt is passed; it is set to t_next - t_now in the __call__ method.

    How it works:
        x_{t+1} ~ N(x_t + drift * delta_t, (L@Q@L.T)*delta_t)
        where:
            x_t is the current state
            drift is the drift function
            L is the diffusion coefficient
            Q is the diffusion covariance
            delta_t is the time step between timepoints (t_next - t_now)
    """
    return _EulerMaruyamaDiscreteEvolution(cte)
