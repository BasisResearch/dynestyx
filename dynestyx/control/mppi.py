"""Basic Model Predictive Path Integral (MPPI) controller.

Samples candidate control sequences as Gaussian
perturbations (white, AR(1), or power-law/colored across the horizon --
see `MPPI.noise_config` and `WhiteNoise`/`AR1Noise`/`ColoredNoise`) around a
nominal sequence, scores each with a user-supplied loss, and returns the
softmax-weighted mean (the standard MPPI control law). The proposal, the
weighting, and where rollouts start can be overridden by subclassing `MPPI`.
"""

import abc
import warnings
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
from jax import Array
from jaxtyping import PRNGKeyArray, Real
from numpyro.distributions import Distribution

import dynestyx as dsx
from dynestyx.models import DynamicalModel, ObservationControlAlignment
from dynestyx.types import SimulatedResult

type MPPILossFn = Callable[[SimulatedResult], Real[Array, ""]]


class MPPIStepInfo(NamedTuple):
    """What one MPPI planning step computed, handed to `MPPI.update_state`.

    Attributes:
        x_hat: The belief handed to the policy.
        t_now: The current time.
        candidates: The candidate control sequences,
            `(n_samples, horizon, control_dim)`.
        losses: Their (always finite) losses, `(n_samples,)`.
        results: Their rollouts, a `SimulatedResult` batched over candidates.
    """

    x_hat: Distribution
    t_now: Real[Array, ""]
    candidates: Real[Array, "n_samples horizon control_dim"]
    losses: Real[Array, " n_samples"]
    results: SimulatedResult


class NoiseConfig(eqx.Module):
    """Base class for `MPPI.noise_config` variants (see `WhiteNoise`,
    `AR1Noise`, `ColoredNoise`).
    """

    @abc.abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
    ) -> Real[Array, "n_samples horizon control_dim"]:
        """Draw `(n_samples, horizon, control_dim)` perturbations with unit
        marginal variance per timestep, correlated across the horizon (axis 1)
        as the variant defines. `MPPI` scales them by `noise_std`.

        `times` is the rollout grid relative to the current time,
        `[0, t_1, ..., t_H]` (`MPPI.horizon`, so `H = times.shape[0] - 1`):
        perturbation `k` is applied over `[times[k], times[k + 1])`."""
        raise NotImplementedError()


class WhiteNoise(NoiseConfig):
    """i.i.d. Gaussian perturbations, uncorrelated across the horizon. Only the
    number of planning steps matters, not their times."""

    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
    ) -> Real[Array, "n_samples horizon control_dim"]:
        return jr.normal(key, (n_samples, times.shape[0] - 1, control_dim))


class AR1Noise(NoiseConfig):
    r"""Ornstein-Uhlenbeck perturbations observed at the planning times,
    correlated as `Cov(eps_h, eps_h') = rho ** |t_h - t_h'|`. Smoother than
    `WhiteNoise`; `rho=0` is equivalent to `WhiteNoise`.

    Attributes:
        rho: Correlation between perturbations one time unit apart, in
            `[0, 1]`. Defaults to `0.5`.
    """

    rho: float = 0.5

    def __check_init__(self) -> None:
        # Only a plain number can be checked here; a traced rho can't.
        if isinstance(self.rho, (int, float)) and not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {self.rho}.")

    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
    ) -> Real[Array, "n_samples horizon control_dim"]:
        # eps_h = rho_h * eps_{h-1} + sqrt(1 - rho_h**2) * xi_h with
        # rho_h = rho ** (t_h - t_{h-1}).
        horizon = times.shape[0] - 1
        xi = jr.normal(key, (horizon, n_samples, control_dim))

        rhos = self.rho ** jnp.diff(times[:-1])  # (horizon - 1,)

        def step(eps_prev, inputs):
            xi_h, rho_h, var_h = inputs
            eps_h = rho_h * eps_prev + jnp.sqrt(var_h) * xi_h
            return eps_h, eps_h

        _, rest = jax.lax.scan(step, xi[0], (xi[1:], rhos, 1.0 - rhos**2))
        return jnp.concatenate([xi[:1], rest], axis=0).transpose(1, 0, 2)


class ColoredNoise(NoiseConfig):
    r"""Power-law (`1/f**beta`) perturbations generated in the frequency
    domain. Smoother, low-frequency-dominated perturbations for larger
    `beta`. `beta=0` = `WhiteNoise`.

    The FFT assumes equally spaced planning times. On an uneven grid the
    spectrum is over the step index rather than time, and a warning is raised.

    Attributes:
        beta: Power-law exponent. `0` is white, `1` is "pink", `2` is
            Brownian-like. Defaults to `2.0`.
    """

    beta: float = 2.0

    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int,
        times: Real[Array, " horizon_plus_one"],
        control_dim: int,
    ) -> Real[Array, "n_samples horizon control_dim"]:
        # NumPy, not jnp: a Python `if` can't branch on jnp results while the
        # planning step is being compiled. Tolerant, so float round-off in the
        # times doesn't trigger it.
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
        horizon = times.shape[0] - 1
        # Power-law (1/f**beta) noise: scale the rfft of white noise by
        # freq**(-beta/2) along the horizon axis.
        white = jr.normal(key, (n_samples, horizon, control_dim))
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

        spectrum = jnp.fft.rfft(white, axis=1) * scale[None, :, None]
        return jnp.fft.irfft(spectrum, n=horizon, axis=1)


class MPPI(eqx.Module):
    r"""Model Predictive Path Integral (MPPI) controller.

    At each call: sample `n_samples` candidate control sequences of length
    `horizon` as Gaussian perturbations around a nominal sequence (carried in
    the policy state `s`, warm-started from the previous call), roll each one
    forward `horizon` steps through the dynamics, score the resulting trajectories with `loss_fn`,
    and combine them via the standard MPPI weighting

    $$w_i \propto \exp(-\mathrm{loss}_i / \lambda), \qquad
      u_{0:H-1} = \sum_i w_i\, u^{(i)}_{0:H-1}$$

    i.e. a softmax over the (negated, temperature-scaled) per-sample losses.
    Only the first control of that weighted-mean sequence is applied this
    step. The remainder becomes next step's nominal
    sequence, shifted left by one with the last entry repeated.

    By default all candidates are rolled out with the same PRNG key (common
    random numbers), so they face the same process and observation noise. The `n_simulations` rollouts
    of one candidate draw different noise. Set `common_randomness=False` to
    give each candidate its own noise instead (increases variance).

    Each rollout is run under the `"previous_transition"` observation/
    control convention, so a candidate's $u_k$ influences $x_{k+1}$ and
    $y_{k+1}$.

    **Customizing.** Subclass `MPPI` and override any of the hooks below;
    `plan_step` calls them in this order, and the defaults implement the
    standard MPPI above. Each hook takes its own inputs followed by the same
    context `(x_hat, t_now, s)`: the belief handed to the policy, the current
    time, and the policy state `s`, a dict with entries `"nominal_sequence"`
    (the sequence the candidates are sampled around) and `"key"` (MPPI's own
    PRNG key). A subclass may add other entries (any pytrees) in
    `initial_state`; after that the entries are fixed.

    1. `rollout_initial_condition(x_hat, t_now, s)`: the distribution each
       rollout's $x_0$ is drawn from. Default: a `Delta` at `x_hat.mean`.
    2. `sample_controls(key, x_hat, t_now, s)`: the candidate control
       sequences (the proposal). Default: `s["nominal_sequence"]` plus
       `noise_std`-scaled `noise_config` noise.
    3. Rollouts and `loss_fn` (not overridable). Non-finite losses are clamped
       to the largest finite value.
    4. `combine_sequences(losses, candidates, x_hat, t_now, s)`: chooses the
       weights and returns the plan, the combined sequence. Default: the
       softmax weighting above.
    5. `update_state(plan, info, s)`: returns the next policy state, the only
       hook that does. `info` is an `MPPIStepInfo` with the rest of the step:
       `x_hat`, `t_now`, `candidates`, `losses` and `results` (the rollouts).
       Default: `s` with `"nominal_sequence"` set to the plan shifted left by
       one (last entry repeated), other entries unchanged.

    `plan_step` then applies `plan[0]` and advances the next state's `"key"`
    entry.


    Hooks run inside `jax.lax.scan` (and possibly `jax.grad`), so they must be
    pure and JAX-traceable and must not draw randomness from `s["key"]`
    (`sample_controls` gets its own key). The state `update_state` returns
    must keep the structure, shapes and dtypes of the one it received. New fields on an `MPPI` subclass need a default (or
    `eqx.field(kw_only=True)`).

    Attributes:
        dynamics: a `DynamicalModel` (the same model used for the real simulation
            or some approximate). Each candidate rollout is computed by calling `dsx.simulate`.
            If `dynamics` holds trainable parameters you're also
            fitting via the outer simulation, they remain in the differentiable
            pytree so gradients through planning are tracked too.
        loss_fn: `MPPILossFn`, i.e. `(result: SimulatedResult) -> scalar`,
            called once per sample (vmapped) on that candidate's full rollout. Every
            field carries a leading `n_simulations` axis -- e.g.
            `result.states.shape == (n_simulations, horizon, state_dim)`, so
            `(1, horizon, state_dim)` by default. `times`/`states`/`observations`/`controls` all
            have length `horizon` and are index-aligned: at index `k`,
            `states[k]` is $x_{k+1}$, `observations[k]` is $y_{k+1}$, and
            `controls[k]` is $u_k$ -- the control that produced that state. The
            starting state $x_0$ is not in `states` (no control produced it); it
            is available separately as `result.x_0`, shape
            `(n_simulations, state_dim)`.
        horizon: The planning grid relative to `t_now`, `[0, t_1, ..., t_H]`
            (starting at 0, strictly increasing), with the rollout run on
            `t_now + horizon` -- e.g. `jnp.linspace(0.0, 1.0, 11)` for 10
            steps of 0.1. `H` is the number of internal one-step `dynamics`
            calls per rollout (see `horizon_length`). `noise_config` receives
            the grid: `AR1Noise` adapts to uneven steps, `ColoredNoise` warns
            about them. Required.
        noise_std: Standard deviation of the Gaussian perturbations added to
            the nominal sequence, scalar or shape `(control_dim,)`. Marginal
            (per-timestep) standard deviation regardless of `noise_config`
            -- every `NoiseConfig` variant has unit marginal variance per
            timestep before this scaling is applied. Defaults to `1.0`.
        noise_config: A `NoiseConfig` selecting how the perturbations are
            correlated across the horizon: `WhiteNoise()` (i.i.d.),
            `AR1Noise(rho=...)` (default, `rho=0.5`), or
            `ColoredNoise(beta=...)` (power-law). See each class's
            docstring; subclass `NoiseConfig` to add a noise type.
        n_samples: Number of sampled control sequences per call. Defaults to
            `20`.
        n_simulations: Number of rollouts drawn per candidate control
            sequence, forwarded to `dsx.simulate`. Each draws different noise,
            shared across candidates when `common_randomness` is `True`.
            Defaults to `1`.
        common_randomness: Whether all candidates are rolled out with the
            same PRNG key, so their losses differ only through their controls
            (lower-variance comparisons). With `False`, each candidate draws
            independent process and observation noise. Defaults to `True`.
        temperature: MPPI's $\lambda \ge 0$; higher values flatten the weights
            toward a uniform average, lower values concentrate weight on the
            lowest-loss samples, and `0` applies the lowest-loss candidate
            alone. Defaults to `1.0`.
        batched: Whether the `n_samples` candidate rollouts are computed with
            `jax.vmap` (default, fast, requires `dynamics.state_evolution` to
            be vmap-compatible) or `jax.lax.map` (a sequential loop -- slower,
            but works for a `dynamics.state_evolution` that isn't
            vmap-compatible, e.g. wraps an external simulator via
            `jax.pure_callback`).
        seed: Seeds MPPI's own PRNG key, carried inside the policy state `s`
            (as `s["key"]`) and split internally on every call.
    """

    dynamics: DynamicalModel
    loss_fn: MPPILossFn = eqx.field(static=True)

    horizon: Real[Array, " horizon_plus_one"]
    noise_std: Real[Array, ""] | Real[Array, " control_dim"] = eqx.field(
        default_factory=lambda: jnp.array(1.0)
    )
    noise_config: NoiseConfig = eqx.field(default_factory=AR1Noise)
    n_samples: int = eqx.field(static=True, default=10)
    n_simulations: int = eqx.field(static=True, default=1)
    common_randomness: bool = eqx.field(static=True, default=True)
    temperature: float = 1.0
    batched: bool = eqx.field(static=True, default=True)
    seed: int = eqx.field(static=True, default=0)

    def __check_init__(self) -> None:
        alignment = self.dynamics.observation_control_alignment
        if alignment not in (None, ObservationControlAlignment.PREVIOUS_TRANSITION):
            warnings.warn(
                f"dynamics.observation_control_alignment is '{alignment}', but "
                "MPPI plans its rollouts under 'previous_transition'.",
                UserWarning,
                stacklevel=3,
            )

        if not isinstance(self.noise_config, NoiseConfig):
            raise TypeError(
                "noise_config must be a NoiseConfig instance (WhiteNoise(), "
                f"AR1Noise(rho=...), or ColoredNoise(beta=...)), got "
                f"{self.noise_config!r}"
            )

        # Only a plain number can be checked here; a traced temperature can't.
        if isinstance(self.temperature, (int, float)) and self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}.")

        h = self.horizon
        if jnp.ndim(h) != 1 or h.shape[0] < 2 or h[0] != 0 or jnp.any(jnp.diff(h) <= 0):
            raise ValueError(
                "horizon must be a 1-D array of planning times relative to the "
                "current time, [0, t_1, ..., t_H]: starting at 0, strictly "
                "increasing, with at least one step (an int horizon is not "
                "supported; for H uniform steps of size dt use "
                f"jnp.arange(H + 1) * dt). Got {h}."
            )

    @property
    def horizon_length(self) -> int:
        """Number of planning steps `H` in the grid `horizon`."""
        return self.horizon.shape[0] - 1

    def initial_state(self) -> dict:
        """Zero nominal control sequence plus MPPI's own seeded PRNG key, as
        `{"nominal_sequence": ..., "key": ...}`. `DiscreteControlLoopSimulator`
        never calls this automatically, so it must be supplied explicitly.
        Override it to add extra entries to carry across steps."""
        return {
            "nominal_sequence": jnp.zeros(
                (self.horizon_length, self.dynamics.control_dim)
            ),
            "key": jr.PRNGKey(self.seed),
        }

    def rollout_initial_condition(
        self, x_hat: Distribution, t_now: Real[Array, ""], s: dict
    ) -> Distribution:
        """Distribution each candidate rollout's $x_0$ is drawn from.

        Default: a `Delta` at `x_hat.mean`, i.e. plan from the point estimate.
        Return `x_hat` itself to plan over the whole belief instead: each of
        the `n_simulations` rollouts then draws its own $x_0$."""
        return dist.Delta(x_hat.mean, event_dim=1)

    def sample_controls(
        self,
        key: PRNGKeyArray,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        s: dict,
    ) -> Real[Array, "n_samples horizon control_dim"]:
        """Candidate control sequences to roll out and score (the proposal).

        Default: `s["nominal_sequence"]` plus `noise_std`-scaled
        `noise_config` perturbations."""
        nominal = s["nominal_sequence"]
        noise = self.noise_config.sample(
            key, self.n_samples, self.horizon, nominal.shape[-1]
        )
        return nominal[None, :, :] + self.noise_std * noise

    def combine_sequences(
        self,
        losses: Real[Array, " n_samples"],
        candidates: Real[Array, "n_samples horizon control_dim"],
        x_hat: Distribution,
        t_now: Real[Array, ""],
        s: dict,
    ) -> Real[Array, "horizon control_dim"]:
        """Weight the candidates and combine them into this step's plan, whose
        first entry is applied. `losses` are always finite.

        Default: the softmax weights `softmax(-losses / temperature)`
        and the weighted mean of the candidates. With
        `temperature=0` all the weight goes to the lowest-loss candidate."""

        positive = self.temperature > 0
        safe_temperature = jnp.where(positive, self.temperature, 1.0)
        weights = jnp.where(
            positive,
            jax.nn.softmax(-losses / safe_temperature),
            jax.nn.one_hot(jnp.argmin(losses), losses.shape[0], dtype=losses.dtype),
        )
        return jnp.einsum("k,khc->hc", weights, candidates)

    def update_state(
        self,
        plan: Real[Array, "horizon control_dim"],
        info: MPPIStepInfo,
        s: dict,
    ) -> dict:
        """Next policy state, from this step's plan and `info` (an
        `MPPIStepInfo`: the belief, time, candidates, their finite losses and
        rollouts). Must return the same entries as `s`.

        Default: `s` with `"nominal_sequence"` set to the plan shifted left by
        one with the last entry repeated (the receding-horizon warm start);
        other entries unchanged."""
        next_nominal = jnp.concatenate([plan[1:], plan[-1:]], axis=0)
        return {**s, "nominal_sequence": next_nominal}

    def _rollout_and_score_one(
        self,
        initial_condition: Distribution,
        u_seq: Real[Array, "horizon control_dim"],
        key: PRNGKeyArray,
        t_now: Real[Array, ""],
    ) -> tuple[Real[Array, ""], SimulatedResult]:
        """Roll out one candidate control sequence by calling `dsx.simulate`
        on a copy of `dynamics` starting from `initial_condition`, then score
        it with `loss_fn`.

        Returns `(loss, result)`."""
        times = t_now + self.horizon  # (horizon+1,)

        # Start the rollout from initial_condition, and plan under the
        # "previous_transition" convention so y_{k+1} and x_{k+1} both produced by u_k.
        pinned_dynamics = eqx.tree_at(
            lambda m: (m.initial_condition, m.observation_control_alignment),
            self.dynamics,
            (initial_condition, "previous_transition"),
            is_leaf=lambda x: x is None,
        )

        # Relies on dsx.simulate's internals (Simulator/DiscreteTimeSimulator)
        # staying plain JAX array ops with no data-dependent Python branching,
        # so this whole function is safe to vmap over candidates.
        res = dsx.simulate(
            pinned_dynamics,
            rng_key=key,
            predict_times=times,
            ctrl_times=times[:-1],
            ctrl_values=u_seq,
            n_simulations=self.n_simulations,
        )
        assert res.times is not None
        assert res.states is not None

        # We drop t_0/x_0 to avoid accidentally using the initial condition (which is independent of the candidate control sequence.
        # Still accessible through res.x_0.
        result = eqx.tree_at(
            lambda r: (r.times, r.states),
            res,
            (res.times[:, 1:], res.states[:, 1:]),  # drop t_0 / x_0
        )
        return self.loss_fn(result), result

    def plan_step(
        self,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        s: dict,
    ) -> tuple[Real[Array, " control_dim"], dict, MPPIStepInfo]:
        """Do MPPI's full planning step and also return what it computed, as
        the `MPPIStepInfo`, useful for debugging.

        `__call__` (used by `DiscreteControlLoopSimulator`) is a
        thin wrapper around this that drops the info, since
        `PolicyCallable`'s return signature can't carry a third value.

        Every field of `info.results` is shaped
        `(n_samples, n_simulations, horizon, ...)`. `predicted_*` are always
        `None` (not meaningful for a planning rollout).
        """
        if not isinstance(s, dict) or "key" not in s:
            got = f"entries {sorted(s)}" if isinstance(s, dict) else type(s).__name__
            raise ValueError(
                "MPPI's policy state must be a dict with a 'key' entry (MPPI's "
                f"PRNG key), as returned by initial_state(); got {got}."
            )
        key, sample_key, rollout_key = jr.split(s["key"], 3)

        initial_condition = self.rollout_initial_condition(x_hat, t_now, s)
        candidates = self.sample_controls(sample_key, x_hat, t_now, s)

        # With common random numbers every candidate shares rollout_key, so the
        # candidates face the same noise and their losses differ only through
        # their controls; otherwise each candidate gets its own key.
        n_candidates = candidates.shape[0]
        if self.common_randomness:
            rollout_keys = jnp.broadcast_to(
                rollout_key, (n_candidates, *rollout_key.shape)
            )
        else:
            rollout_keys = jr.split(rollout_key, n_candidates)

        def rollout_and_score(u_seq, key):
            return self._rollout_and_score_one(initial_condition, u_seq, key, t_now)

        if self.batched:
            losses, results = jax.vmap(rollout_and_score)(candidates, rollout_keys)
        else:
            losses, results = jax.lax.map(
                lambda args: rollout_and_score(*args), (candidates, rollout_keys)
            )
        # A candidate whose rollout numerically diverges can produce a
        # nan loss. Clamping to the largest finite value keeps that candidate's
        # weight ~0 without corrupting the others
        losses = jnp.where(jnp.isfinite(losses), losses, jnp.finfo(losses.dtype).max)

        plan = self.combine_sequences(losses, candidates, x_hat, t_now, s)
        info = MPPIStepInfo(
            x_hat=x_hat,
            t_now=t_now,
            candidates=candidates,
            losses=losses,
            results=results,
        )
        next_s = self.update_state(plan, info, s)
        if not isinstance(next_s, dict) or set(next_s) != set(s):
            got = set(next_s) if isinstance(next_s, dict) else set()
            raise ValueError(
                "update_state must return a dict with the same entries as the "
                "state it received (those created by initial_state()); "
                f"added {sorted(got - set(s))}, removed {sorted(set(s) - got)}."
            )
        # MPPI owns its randomness: always advance the key.
        next_s = {**next_s, "key": key}

        # Also return the step's info, for debugging/analysis.
        return plan[0], next_s, info

    def __call__(
        self,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        t_next: Real[Array, ""],
        s: dict,
    ) -> tuple[Real[Array, " control_dim"], dict]:
        del t_next
        u0, next_s, _ = self.plan_step(x_hat, t_now, s)
        return u0, next_s


__all__ = [
    "MPPI",
    "MPPILossFn",
    "MPPIStepInfo",
    "NoiseConfig",
    "WhiteNoise",
    "AR1Noise",
    "ColoredNoise",
]
