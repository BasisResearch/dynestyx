"""Basic Model Predictive Path Integral (MPPI) controller.

Deliberately simple: samples candidate control sequences as Gaussian
perturbations (white, AR(1), or power-law/colored across the horizon --
see `MPPI.noise_config` and `WhiteNoise`/`AR1Noise`/`ColoredNoise`) around a
nominal sequence, scores each with a user-supplied loss, and returns the
softmax-weighted mean -- the standard MPPI-style control law. No adaptive
covariance or other refinements; the goal is a plain example that plugs into
`DiscreteControlLoopSimulator`'s `control_policy=` slot (see
`dynestyx.control.discrete_controller_simulators.PolicyCallable`), not a
state-of-the-art implementation.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jax import Array
from jaxtyping import PRNGKeyArray, Real
from numpyro.distributions import Distribution

import dynestyx as dsx
from dynestyx.control.discrete_controller_simulators import ControlledSimulatedResult
from dynestyx.models import DynamicalModel

# (result: ControlledSimulatedResult) -> scalar, called once per sampled rollout
# (vmapped across all n_samples candidates) on that candidate's full rollout result.
# See MPPI.loss_fn for the full shape contract.
type MPPILossFn = Callable[[ControlledSimulatedResult], Real[Array, ""]]


class NoiseConfig(eqx.Module):
    """Base class for `MPPI.noise_config` variants (see `WhiteNoise`,
    `AR1Noise`, `ColoredNoise`). Not instantiated directly."""


class WhiteNoise(NoiseConfig):
    """i.i.d. Gaussian perturbations, uncorrelated across the horizon:
    `Cov(eps_h, eps_h') = 0` for `h != h'`. The original, uncorrelated MPPI
    noise -- no hyperparameters."""


class AR1Noise(NoiseConfig):
    r"""AR(1)/Ornstein-Uhlenbeck-style perturbations, correlated across the
    horizon as `Cov(eps_h, eps_h') = rho ** |h - h'|`. Smoother than
    `WhiteNoise`; `rho=0` is equivalent to `WhiteNoise`.

    Attributes:
        rho: Correlation coefficient in `[0, 1)`. Defaults to `0.5`.
    """

    rho: float = 0.5


class ColoredNoise(NoiseConfig):
    r"""Power-law (`1/f**beta`) perturbations generated in the frequency
    domain -- smoother, low-frequency-dominated perturbations for larger
    `beta`. `beta=0` is equivalent to `WhiteNoise`.

    Attributes:
        beta: Power-law exponent. `0` is white, `1` is "pink", `2` is
            Brownian-like. Defaults to `2.0`.
    """

    beta: float = 2.0


class MPPI(eqx.Module):
    r"""Model Predictive Path Integral (MPPI) controller.

    At each call: sample `n_samples` candidate control sequences of length
    `horizon` as Gaussian perturbations around a nominal sequence (the policy
    state `s`, warm-started from the previous call), roll each one forward
    `horizon` steps through `dynamics.state_evolution` (built internally --
    the caller only ever supplies the *one-step* dynamics, never a
    hand-written rollout), score the resulting trajectories with `loss_fn`,
    and combine them via the standard MPPI weighting

    $$w_i \\propto \\exp(-\\mathrm{loss}_i / \\lambda), \\qquad
      u_{0:H-1} = \\sum_i w_i\\, u^{(i)}_{0:H-1}$$

    i.e. a softmax over the (negated, temperature-scaled) per-sample losses.
    Only the first control of that weighted-mean sequence is applied this
    step (receding horizon); the remainder becomes next step's nominal
    sequence, shifted left by one with the last entry repeated.

    Attributes:
        dynamics: a `DynamicalModel` (the same model used for the real simulation
            or some approximate). Each candidate rollout is computed by calling `dsx.simulate`.
            If `dynamics` holds trainable parameters you're also
            fitting via the outer simulation, they remain in the differentiable
            pytree so gradients through planning are tracked too.
        loss_fn: `MPPILossFn`, i.e. `(result: ControlledSimulatedResult) -> scalar`,
            called once per sample (vmapped) on that candidate's full rollout. Every
            field carries a leading `n_simulations=1` axis -- e.g.
            `result.states.shape == (1, horizon + 1, state_dim)` -- matching how
            `dsx.simulate` never drops that axis, even for one trajectory;
            `jnp.sum(result.states**2)`-style reductions don't need to care, but
            explicit indexing does (`result.controls[0, 0]` is the whole first control
            vector, not a scalar). `times`/`states`/`observations` have length
            `horizon + 1` (including the starting state) and `controls` has length
            `horizon`, matching `ControlledSimulatedResult`'s own
            `control_time = time - 1` convention.
        horizon: Planning horizon length `H` -- the number of internal
            one-step `dynamics` calls per rollout. Defaults to `10`.
        noise_std: Standard deviation of the Gaussian perturbations added to
            the nominal sequence, scalar or shape `(control_dim,)`. Marginal
            (per-timestep) standard deviation regardless of `noise_config`
            -- every `NoiseConfig` variant has unit marginal variance per
            timestep before this scaling is applied. Defaults to `1.0`.
        noise_config: A `NoiseConfig` selecting how the perturbations are
            correlated across the horizon: `WhiteNoise()` (i.i.d.),
            `AR1Noise(rho=...)` (default, `rho=0.5`), or
            `ColoredNoise(beta=...)` (power-law). See each class's
            docstring.
        n_samples: Number of sampled control sequences per call. Defaults to
            `20`.
        dt: Fixed planning step size. Defaults to `1.0`.
        temperature: MPPI's $\\lambda$; higher values flatten the weights
            toward a uniform average, lower values concentrate weight on the
            lowest-loss samples.
        batched: Whether the `n_samples` candidate rollouts are computed with
            `jax.vmap` (default, fast, requires `dynamics.state_evolution` to
            be vmap-compatible) or `jax.lax.map` (a sequential loop -- slower,
            but works for a `dynamics.state_evolution` that isn't
            vmap-compatible, e.g. wraps an external simulator via
            `jax.pure_callback`).
        seed: Seeds MPPI's own PRNG key, carried inside the policy state `s`
            (as `(nominal_sequence, key)`) and split internally on every
            call.
    """

    dynamics: DynamicalModel
    loss_fn: MPPILossFn = eqx.field(static=True)
    horizon: int = eqx.field(static=True, default=10)
    noise_std: Real[Array, ""] | Real[Array, " control_dim"] = eqx.field(
        default_factory=lambda: jnp.array(1.0)
    )
    noise_config: NoiseConfig = eqx.field(default_factory=AR1Noise)
    n_samples: int = eqx.field(static=True, default=20)
    dt: float = eqx.field(static=True, default=1.0)
    temperature: float = 1.0
    batched: bool = eqx.field(static=True, default=True)
    seed: int = eqx.field(static=True, default=0)

    def __check_init__(self) -> None:
        if not isinstance(self.noise_config, NoiseConfig):
            raise TypeError(
                "noise_config must be a NoiseConfig instance (WhiteNoise(), "
                f"AR1Noise(rho=...), or ColoredNoise(beta=...)), got "
                f"{self.noise_config!r}"
            )

    def initial_state(
        self,
    ) -> tuple[Real[Array, "horizon control_dim"], PRNGKeyArray]:
        """Zero nominal control sequence plus MPPI's own seeded PRNG key.
        Pass this call's result as `initial_policy_state` to `simulate`/
        `dsx.simulate` -- `DiscreteControlLoopSimulator` never calls this
        automatically, so it must be supplied explicitly."""
        return (
            jnp.zeros((self.horizon, self.dynamics.control_dim)),
            jr.PRNGKey(self.seed),
        )

    def _sample_noise(
        self, key: PRNGKeyArray, control_dim: int
    ) -> Real[Array, "n_samples horizon control_dim"]:
        """Draw `(n_samples, horizon, control_dim)` perturbations with unit
        marginal variance per timestep and the horizon-correlation structure
        selected by `noise_config`, then scale by `noise_std`. The concrete
        `NoiseConfig` subclass is part of the pytree structure (not a leaf),
        so this `isinstance` dispatch is resolved at trace time -- each
        compiled instance only ever contains one mode's ops."""
        shape = (self.n_samples, self.horizon, control_dim)

        if isinstance(self.noise_config, WhiteNoise):
            eps = jr.normal(key, shape)

        elif isinstance(self.noise_config, AR1Noise):
            # eps_h = rho * eps_{h-1} + sqrt(1 - rho**2) * xi_h, xi_h ~ N(0, I),
            # eps_0 = xi_0 -- a stationary AR(1) process with unit marginal
            # variance and Cov(eps_h, eps_h') = rho**|h-h'|.
            xi = jr.normal(key, (self.horizon, self.n_samples, control_dim))
            rho = self.noise_config.rho

            def step(eps_prev, xi_h):
                eps_h = rho * eps_prev + jnp.sqrt(1.0 - rho**2) * xi_h
                return eps_h, eps_h

            _, rest = jax.lax.scan(step, xi[0], xi[1:])
            eps = jnp.concatenate([xi[:1], rest], axis=0).transpose(1, 0, 2)

        else:  # ColoredNoise
            # Power-law (1/f**beta) noise: scale the rfft of white noise by
            # freq**(-beta/2) along the horizon axis, then invert. The f=0
            # bin is clamped to the fundamental frequency 1/horizon (rather
            # than zeroed or left to blow up) following the standard
            # Timmer-Koenig cutoff, and the scale is renormalized (via
            # Parseval's theorem) so each timestep keeps unit variance,
            # matching WhiteNoise/AR1Noise for a fair noise_std comparison.
            assert isinstance(self.noise_config, ColoredNoise)
            white = jr.normal(key, shape)
            freqs = jnp.fft.rfftfreq(self.horizon)
            freqs = jnp.maximum(freqs, 1.0 / self.horizon)
            scale = freqs ** (-self.noise_config.beta / 2.0)
            n_freqs = scale.shape[0]
            is_edge = (jnp.arange(n_freqs) == 0) | (
                (self.horizon % 2 == 0) & (jnp.arange(n_freqs) == n_freqs - 1)
            )
            mult = jnp.where(is_edge, 1.0, 2.0)
            sigma = jnp.sqrt(jnp.sum(scale**2 * mult) / self.horizon)
            scale = scale / sigma

            spectrum = jnp.fft.rfft(white, axis=1) * scale[None, :, None]
            eps = jnp.fft.irfft(spectrum, n=self.horizon, axis=1)

        return self.noise_std * eps

    def _rollout_and_score_one(
        self,
        x0: Real[Array, " state_dim"],
        u_seq: Real[Array, "horizon control_dim"],
        key: PRNGKeyArray,
        t_now: Real[Array, ""],
    ) -> tuple[
        Real[Array, ""],
        Real[Array, "horizon+1 state_dim"],
        Real[Array, "horizon+1 observation_dim"],
    ]:
        """Roll out one candidate control sequence by calling `dsx.simulate`
        on a copy of `dynamics` pinned to start at `x0`, then score it with
        `loss_fn`. Returns `(loss, states, observations)` -- plain arrays
        only, since `ControlledSimulatedResult` isn't JAX-pytree-registered
        and so can never itself cross a `vmap` boundary; it's built and fully
        consumed here, inside the per-candidate function that gets vmapped."""
        times = t_now + jnp.arange(self.horizon + 1) * self.dt  # (horizon+1,)

        pinned_dynamics = eqx.tree_at(
            lambda m: m.initial_condition,
            self.dynamics,
            dist.Delta(x0, event_dim=1),
        )
        # dsx.simulate's same-index convention pairs ctrl_values[t] with both
        # the transition from t and the observation at t, so it needs
        # horizon+1 entries; u_seq only has horizon (one per transition).
        # Pad with a repeat of the last control, used only to drive the final
        # (never-transitioned-from) observation -- purely internal plumbing,
        # never seen by loss_fn (which gets the real, unpadded u_seq below).
        ctrl_padded = jnp.concatenate([u_seq, u_seq[-1:]], axis=0)

        # Relies on dsx.simulate's internals (Simulator/DiscreteTimeSimulator)
        # staying plain JAX array ops with no data-dependent Python branching,
        # so this whole function is safe to vmap over candidates.
        res = dsx.simulate(
            pinned_dynamics,
            rng_key=key,
            predict_times=times,
            ctrl_times=times,
            ctrl_values=ctrl_padded,
        )
        assert res.states is not None
        assert res.observations is not None
        states = res.states[0]  # squeezed, for plan_step's own batching below
        observations = res.observations[0]

        # ControlledSimulatedResult's fields all require a leading
        # n_simulations axis (matching how dsx.simulate never drops it, even
        # for one trajectory) -- so loss_fn sees the same n_simulations=1
        # shape a real single dsx.simulate() call would produce, not a
        # squeezed one.
        result = ControlledSimulatedResult(
            times=times[None],
            x_0=x0[None],
            states=states[None],
            observations=observations[None],
            controls=u_seq[None],
        )
        return self.loss_fn(result), states, observations

    def plan_step(
        self,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        s: tuple[Real[Array, "horizon control_dim"], PRNGKeyArray],
    ) -> tuple[
        Real[Array, " control_dim"],
        tuple[Real[Array, "horizon control_dim"], PRNGKeyArray],
        ControlledSimulatedResult,
    ]:
        """Do MPPI's full planning step and also return the batch of every
        candidate rollout considered (`n_samples`-wide `ControlledSimulatedResult`)
        -- useful for debugging/plotting what MPPI weighed, or diagnosing a
        `loss_fn`. `__call__` (used by `DiscreteControlLoopSimulator`) is a
        thin wrapper around this that drops the rollout batch, since
        `PolicyCallable`'s return signature can't carry a third value.

        Note: the returned result's leading axis indexes *candidates*, not
        independent draws from the true generative process -- it's not a
        real simulated trajectory. `filtered_states_mean`/`policy_states`/
        `predicted_*` are always `None` (not meaningful for a planning
        rollout).
        """
        x0 = x_hat.mean
        nominal, key = s
        key, noise_key, rollout_key = jr.split(key, 3)
        control_dim = nominal.shape[-1]

        noise = self._sample_noise(noise_key, control_dim)
        control_candidates = (
            nominal[None, :, :] + noise
        )  # (n_samples, horizon, control_dim)
        rollout_keys = jr.split(rollout_key, self.n_samples)

        if self.batched:
            losses, states_batch, obs_batch = jax.vmap(
                self._rollout_and_score_one, in_axes=(None, 0, 0, None)
            )(x0, control_candidates, rollout_keys, t_now)
        else:
            losses, states_batch, obs_batch = jax.lax.map(
                lambda args: self._rollout_and_score_one(x0, args[0], args[1], t_now),
                (control_candidates, rollout_keys),
            )
        # A candidate whose rollout numerically diverges can produce a
        # nan loss. Clamping to the largest finite value keeps that candidate's
        # weight ~0 without corrupting the others
        losses = jnp.where(jnp.isfinite(losses), losses, jnp.finfo(losses.dtype).max)

        weights = jax.nn.softmax(-losses / self.temperature)
        weighted_seq = jnp.einsum("k,khc->hc", weights, control_candidates)

        u0 = weighted_seq[0]
        next_nominal = jnp.concatenate([weighted_seq[1:], weighted_seq[-1:]], axis=0)

        times = t_now + jnp.arange(self.horizon + 1) * self.dt
        result = ControlledSimulatedResult(
            times=jnp.broadcast_to(times, (self.n_samples, self.horizon + 1)),
            x_0=jnp.broadcast_to(x0, (self.n_samples,) + x0.shape),
            states=states_batch,
            observations=obs_batch,
            controls=control_candidates,
        )
        return u0, (next_nominal, key), result

    def __call__(
        self,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        t_next: Real[Array, ""],
        s: tuple[Real[Array, "horizon control_dim"], PRNGKeyArray],
    ) -> tuple[
        Real[Array, " control_dim"],
        tuple[Real[Array, "horizon control_dim"], PRNGKeyArray],
    ]:
        # t_next (the real simulation's next observation time) is unused --
        # MPPI plans its own horizon-step lookahead from t_now using its own dt.
        del t_next
        u0, next_s, _ = self.plan_step(x_hat, t_now, s)
        return u0, next_s


__all__ = [
    "MPPI",
    "MPPILossFn",
    "NoiseConfig",
    "WhiteNoise",
    "AR1Noise",
    "ColoredNoise",
]
