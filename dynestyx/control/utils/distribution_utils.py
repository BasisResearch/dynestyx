"""Noise distributions for MPPI's candidate control perturbations.

Each is a NumPyro distribution over `(horizon, control_dim)` perturbations with
unit marginal variance per step, built from the planning grid `times`, i.e.
`MPPI.horizon`, `[0, t_1, ..., t_H]`: perturbation `k` is applied over
`[times[k], times[k + 1])`. Sampling is reparameterized (a deterministic
transform of standard normals), so `rsample` works too.
"""

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
from jax import Array
from jaxtyping import PRNGKeyArray, Real
from numpyro.distributions import constraints


class WhiteNoise(dist.Distribution):
    """i.i.d. standard Gaussian perturbations, uncorrelated across the horizon.
    Only the number of planning steps matters, not their times.

    Args:
        times: The planning grid `[0, t_1, ..., t_H]` (`MPPI.horizon`).
        control_dim: The control dimension.
    """

    support = constraints.independent(constraints.real, 2)
    pytree_data_fields = ("times",)
    pytree_aux_fields = ("control_dim",)

    def __init__(
        self,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
        *,
        validate_args: bool | None = None,
    ):
        self.times = times
        self.control_dim = control_dim
        super().__init__(
            event_shape=(times.shape[0] - 1, control_dim), validate_args=validate_args
        )

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Real[Array, "*sample horizon control_dim"]:
        return jr.normal(key, sample_shape + self.event_shape)


class AR1Noise(dist.Distribution):
    r"""Ornstein-Uhlenbeck perturbations observed at the planning times,
    correlated as `Cov(eps_h, eps_h') = rho ** |t_h - t_h'|`, where `t_h` is
    the time perturbation `h` starts. Smoother than `WhiteNoise`; `rho=0` is
    equivalent to `WhiteNoise`.

    Args:
        times: The planning grid `[0, t_1, ..., t_H]` (`MPPI.horizon`).
        control_dim: The control dimension.
        rho: Correlation between perturbations one time unit apart, in
            `[0, 1]`. Defaults to `0.5`.
    """

    support = constraints.independent(constraints.real, 2)
    pytree_data_fields = ("times", "rho")
    pytree_aux_fields = ("control_dim",)

    def __init__(
        self,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
        rho: float = 0.5,
        *,
        validate_args: bool | None = None,
    ):
        if isinstance(rho, (int, float)) and not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}.")
        self.times = times
        self.control_dim = control_dim
        self.rho = rho
        super().__init__(
            event_shape=(times.shape[0] - 1, control_dim), validate_args=validate_args
        )

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Real[Array, "*sample horizon control_dim"]:
        # eps_h = rho_h * eps_{h-1} + sqrt(1 - rho_h**2) * xi_h with
        # rho_h = rho ** (t_h - t_{h-1}).
        horizon, control_dim = self.event_shape
        xi = jr.normal(key, (horizon, *sample_shape, control_dim))

        rhos = self.rho ** jnp.diff(self.times[:-1])  # (horizon - 1,)

        def step(eps_prev, inputs):
            xi_h, rho_h, var_h = inputs
            eps_h = rho_h * eps_prev + jnp.sqrt(var_h) * xi_h
            return eps_h, eps_h

        _, rest = jax.lax.scan(step, xi[0], (xi[1:], rhos, 1.0 - rhos**2))
        eps = jnp.concatenate([xi[:1], rest], axis=0)  # horizon axis first
        return jnp.moveaxis(eps, 0, -2)


class ColoredNoise(dist.Distribution):
    r"""Power-law (`1/f**beta`) perturbations generated in the frequency
    domain. Smoother, low-frequency-dominated perturbations for larger
    `beta`. `beta=0` = `WhiteNoise`.

    The FFT assumes equally spaced planning times. On an uneven grid the
    spectrum is over the step index rather than time, and a warning is raised.

    Args:
        times: The planning grid `[0, t_1, ..., t_H]` (`MPPI.horizon`).
        control_dim: The control dimension.
        beta: Power-law exponent. `0` is white, `1` is "pink", `2` is
            Brownian-like. Defaults to `2.0`.
    """

    support = constraints.independent(constraints.real, 2)
    pytree_data_fields = ("times", "beta")
    pytree_aux_fields = ("control_dim",)

    def __init__(
        self,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
        beta: float = 2.0,
        *,
        validate_args: bool | None = None,
    ):
        # Tolerant, so float round-off in the times doesn't trigger it.
        steps = np.diff(times)
        if not np.allclose(steps, steps.mean(), rtol=1e-3, atol=0.0):
            warnings.warn(
                "It seems that your planning time steps are not equally "
                "spaced (steps "
                f"{np.array2string(steps, precision=4, separator=', ')}). "
                "ColoredNoise shapes its 1/f**beta "
                "spectrum over the step index, so the resulting noise process "
                "is power-law in steps, not in time: long and short steps get "
                "the same correlation. Use AR1Noise for noise that adapts to "
                "the actual times.",
                UserWarning,
                stacklevel=2,
            )
        self.times = times
        self.control_dim = control_dim
        self.beta = beta
        super().__init__(
            event_shape=(times.shape[0] - 1, control_dim), validate_args=validate_args
        )

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Real[Array, "*sample horizon control_dim"]:
        # Power-law (1/f**beta) noise: scale the rfft of white noise by
        # freq**(-beta/2) along the horizon axis.
        horizon, control_dim = self.event_shape
        white = jr.normal(key, (*sample_shape, horizon, control_dim))
        freqs = jnp.fft.rfftfreq(horizon)
        freqs = jnp.maximum(freqs, 1.0 / horizon)
        scale = freqs ** (-self.beta / 2.0)
        n_freqs = scale.shape[0]
        is_edge = (jnp.arange(n_freqs) == 0) | (
            (horizon % 2 == 0) & (jnp.arange(n_freqs) == n_freqs - 1)
        )
        mult = jnp.where(is_edge, 1.0, 2.0)
        sigma = jnp.sqrt(jnp.sum(scale**2 * mult) / horizon)
        scale = scale / sigma

        spectrum = jnp.fft.rfft(white, axis=-2) * scale[:, None]
        return jnp.fft.irfft(spectrum, n=horizon, axis=-2)


__all__ = ["WhiteNoise", "AR1Noise", "ColoredNoise"]
