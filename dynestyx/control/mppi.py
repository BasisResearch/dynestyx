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
from jaxtyping import PRNGKeyArray, PyTree, Real

from dynestyx.control.discrete_controller_simulators import filter_state_mean


class MPPI(eqx.Module):
    r"""Model Predictive Path Integral (MPPI) controller.

    At each call: sample `n_samples` candidate control sequences of length
    `horizon` as Gaussian perturbations around a nominal sequence (the policy
    state `s`, warm-started from the previous call), roll each through
    `dynamics_model`, score the resulting trajectories with `loss_fn`, and
    combine them via the standard MPPI weighting

    $$w_i \\propto \\exp(-\\mathrm{loss}_i / \\lambda), \\qquad
      u_{0:H-1} = \\sum_i w_i\\, u^{(i)}_{0:H-1}$$

    i.e. a softmax over the (negated, temperature-scaled) per-sample losses.
    Only the first control of that weighted-mean sequence is applied this
    step (receding horizon); the remainder becomes next step's nominal
    sequence, shifted left by one with the last entry repeated.

    Attributes:
        dynamics_model: Any callable `(x0, u_seq) -> x_seq`, deliberately not
            tied to dynestyx's own `DynamicalModel`/filtering machinery -- a
            bare JAX-compatible rollout function (e.g. wrapping a
            `DynamicalModel`'s `state_evolution` with a `jax.lax.scan`, or an
            external simulator). If `batched=True` (default), it must accept
            `u_seq` shaped `(n_samples, horizon, control_dim)` and return
            `(n_samples, horizon, state_dim)` in one call (e.g. internally
            vmapped). If `batched=False`, it only supports a single
            `(horizon, control_dim) -> (horizon, state_dim)` call at a time;
            `MPPI` then drives it with `jax.lax.map`, JAX's for-loop
            construct that calls it once per sample without requiring the
            model itself to support batching.
        loss_fn: `(x_seq, u_seq) -> scalar`, called once per sample (vmapped)
            over the rolled-out state and control trajectories.
        horizon: Planning horizon length `H`.
        control_dim: Control dimension, used by `initial_state()` to build the
            zero nominal sequence `DiscreteControlLoopSimulator` seeds the
            policy state with (via `initial_state()`) -- no need to pass an
            initial policy state in yourself.
        n_samples: Number of sampled control sequences per call.
        noise_std: Standard deviation of the Gaussian perturbations added to
            the nominal sequence, scalar or shape `(control_dim,)`.
        temperature: MPPI's $\\lambda$; higher values flatten the weights
            toward a uniform average, lower values concentrate weight on the
            lowest-loss samples.
        batched: Whether `dynamics_model` accepts a batch of control
            sequences in one call (see above).
        seed: Seeds MPPI's own exploration-noise PRNG key, carried inside the
            policy state `s` (as `(nominal_sequence, key)`) and split
            internally on every call -- `DiscreteControlLoopSimulator` never
            passes a key to `control_policy`, so MPPI owns and advances its
            randomness entirely by itself. Two `MPPI` instances with the same
            `seed` explore identically regardless of the simulation's own
            `rng_key`; use a different `seed` to get a different exploration
            sequence.
    """

    dynamics_model: Callable = eqx.field(static=True)
    loss_fn: Callable = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    noise_std: Real[Array, ""] | Real[Array, " control_dim"]
    n_samples: int = eqx.field(static=True, default=10)
    temperature: float = 1.0
    batched: bool = eqx.field(static=True, default=True)
    seed: int = eqx.field(static=True, default=0)

    def initial_state(
        self,
    ) -> tuple[Real[Array, "horizon control_dim"], PRNGKeyArray]:
        """Zero nominal control sequence plus MPPI's own seeded PRNG key --
        `DiscreteControlLoopSimulator` calls this automatically to seed the
        policy state; no need to build or pass one in yourself."""
        return jnp.zeros((self.horizon, self.control_dim)), jr.PRNGKey(self.seed)

    def __call__(
        self,
        x_hat: PyTree,
        s: tuple[Real[Array, "horizon control_dim"], PRNGKeyArray],
    ) -> tuple[
        Real[Array, " control_dim"],
        tuple[Real[Array, "horizon control_dim"], PRNGKeyArray],
    ]:
        x0 = filter_state_mean(x_hat)
        nominal, key = s
        key, noise_key = jr.split(key)
        control_dim = nominal.shape[-1]

        noise = self.noise_std * jr.normal(
            noise_key, (self.n_samples, self.horizon, control_dim)
        )
        candidates = nominal[None, :, :] + noise  # (n_samples, horizon, control_dim)

        if self.batched:
            x_trajectories = self.dynamics_model(x0, candidates)
        else:
            x_trajectories = jax.lax.map(
                lambda u_seq: self.dynamics_model(x0, u_seq), candidates
            )
        losses = jax.vmap(self.loss_fn)(x_trajectories, candidates)

        weights = jax.nn.softmax(-losses / self.temperature)
        weighted_seq = jnp.einsum("k,khc->hc", weights, candidates)

        u0 = weighted_seq[0]
        next_nominal = jnp.concatenate([weighted_seq[1:], weighted_seq[-1:]], axis=0)
        return u0, (next_nominal, key)


__all__ = ["MPPI"]
