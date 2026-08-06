"""Basic Model Predictive Path Integral (MPPI) controller.

Deliberately simple: samples candidate control sequences as Gaussian
perturbations around a nominal sequence, scores each with a user-supplied
loss, and returns the softmax-weighted mean -- the standard MPPI control law.
No colored noise, adaptive covariance, or other refinements; the goal is a
plain example that plugs into `DiscreteControlLoopSimulator`'s
`control_policy=` slot (see `dynestyx.control.discrete_controller_simulators.
PolicyCallable`), not a state-of-the-art implementation.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import PRNGKeyArray, Real
from numpyro.distributions import Distribution

from dynestyx.models import DynamicalModel


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
        dynamics: The *same* `DynamicalModel` used for the real simulation
            (or a distinct approximate model for planning) -- MPPI calls
            `dynamics.state_evolution(x, u, t_now, t_next)` once per rollout
            step, `horizon` times per call, internally via `jax.lax.scan`.
            Not `eqx.field(static=True)`: if `dynamics` holds trainable
            parameters you're also fitting via the outer simulation, they
            must stay in the differentiable pytree for gradients through
            planning to be tracked too (see module notes on this in the
            project's design discussion).
            Uses the transition's `.mean` when available (the standard MPPI
            simplification -- a deterministic prediction is enough for
            planning); falls back to `.sample()`, using MPPI's own
            internally-carried key (see `seed`), for a genuine black-box
            transition that only exposes `.sample()`.
        loss_fn: `(x_seq, u_seq) -> scalar`, called once per sample (vmapped)
            over the rolled-out state and control trajectories.
        horizon: Planning horizon length `H` -- the number of internal
            one-step `dynamics` calls per rollout, and the length of each
            candidate control sequence. Defaults to `10`.
        noise_std: Standard deviation of the Gaussian perturbations added to
            the nominal sequence, scalar or shape `(control_dim,)`. Defaults
            to `1.0`.
        n_samples: Number of sampled control sequences per call. Defaults to
            `20`.
        dt: Fixed planning step size. Rollout step $i$ (of `horizon`) calls
            `dynamics.state_evolution(x, u, t_now + i*dt, t_now + (i+1)*dt)`,
            where `t_now` is the real current simulation time passed into
            `__call__` -- so a genuinely time-varying `dynamics.state_evolution`
            plans from the correct absolute time, even though MPPI re-plans
            a fresh `horizon`-step lookahead from scratch on every call.
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
            call -- `DiscreteControlLoopSimulator` never passes a key to
            `control_policy`, so MPPI owns and advances all of its own
            randomness itself (exploration noise, and `.sample()` calls when
            `dynamics` is a black box). Two `MPPI` instances with the same
            `seed` explore identically regardless of the simulation's own
            `rng_key`; use a different `seed` to get a different exploration
            sequence.
    """

    dynamics: DynamicalModel
    loss_fn: Callable = eqx.field(static=True)
    horizon: int = eqx.field(static=True, default=10)
    noise_std: Real[Array, ""] | Real[Array, " control_dim"] = eqx.field(
        default_factory=lambda: jnp.array(1.0)
    )
    n_samples: int = eqx.field(static=True, default=20)
    dt: float = eqx.field(static=True, default=1.0)
    temperature: float = 1.0
    batched: bool = eqx.field(static=True, default=True)
    seed: int = eqx.field(static=True, default=0)

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

    def _rollout_one(
        self,
        x0: Real[Array, " state_dim"],
        u_seq: Real[Array, "horizon control_dim"],
        key: PRNGKeyArray,
        t_now: Real[Array, ""],
    ) -> Real[Array, "horizon state_dim"]:
        """Unroll `horizon` one-step `dynamics` calls for one candidate
        control sequence, starting from real time `t_now` -- the internal
        replacement for what used to be a hand-written rollout function
        supplied by the caller."""

        def step(carry, u_and_idx):
            x, step_key = carry
            u, idx = u_and_idx
            step_t_now = t_now + idx * self.dt
            step_t_next = step_t_now + self.dt
            transition = self.dynamics.state_evolution(x, u, step_t_now, step_t_next)
            if hasattr(transition, "mean"):
                x_next = transition.mean
            else:
                step_key, sample_key = jr.split(step_key)
                x_next = transition.sample(sample_key)
            return (x_next, step_key), x_next

        idxs = jnp.arange(self.horizon)
        (_, _), xs = jax.lax.scan(step, (x0, key), (u_seq, idxs))
        return xs

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
        x0 = x_hat.mean
        nominal, key = s
        key, noise_key, rollout_key = jr.split(key, 3)
        control_dim = nominal.shape[-1]

        noise = self.noise_std * jr.normal(
            noise_key, (self.n_samples, self.horizon, control_dim)
        )
        candidates = nominal[None, :, :] + noise  # (n_samples, horizon, control_dim)
        rollout_keys = jr.split(rollout_key, self.n_samples)

        if self.batched:
            x_trajectories = jax.vmap(self._rollout_one, in_axes=(None, 0, 0, None))(
                x0, candidates, rollout_keys, t_now
            )
        else:
            x_trajectories = jax.lax.map(
                lambda args: self._rollout_one(x0, args[0], args[1], t_now),
                (candidates, rollout_keys),
            )
        losses = jax.vmap(self.loss_fn)(x_trajectories, candidates)
        # A candidate whose rollout numerically diverges (e.g. an unstable
        # system explored too far by an unlucky noise draw) can produce a
        # +-inf or nan loss. A lone +-inf is harmless under softmax (it gets
        # weight ~0), but a lone nan poisons every weight (softmax subtracts
        # max(losses); nan - anything is nan). Clamping to the largest
        # finite value keeps that candidate's weight ~0 without corrupting
        # the others -- and keeps softmax well-defined even if every
        # candidate this happens to.
        losses = jnp.where(jnp.isfinite(losses), losses, jnp.finfo(losses.dtype).max)

        weights = jax.nn.softmax(-losses / self.temperature)
        weighted_seq = jnp.einsum("k,khc->hc", weights, candidates)

        u0 = weighted_seq[0]
        next_nominal = jnp.concatenate([weighted_seq[1:], weighted_seq[-1:]], axis=0)
        return u0, (next_nominal, key)


__all__ = ["MPPI"]
