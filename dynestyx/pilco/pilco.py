"""Main PILCO algorithm implementation.

Orchestrates the GP dynamics model learning, moment-matching trajectory
prediction, and gradient-based policy optimization. Integrates with dynestyx
via effectful handlers and ``DynamicalModel``.

The central handler is ``MomentMatchingPropagator``, an effectful interpretation
of ``dsx.sample`` that replaces stochastic sampling with analytic Gaussian
moment matching through the learned GP dynamics model. This follows the same
pattern as dynestyx's ``Filter`` and ``Discretizer`` handlers.

References:
    Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and
    Data-Efficient Approach to Policy Search. ICML, Algorithm 1.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import optax
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import Array

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.models import DynamicalModel
from dynestyx.pilco.controllers import squash_sin
from dynestyx.pilco.mgpr import MGPR
from dynestyx.pilco.rewards import ExponentialReward
from dynestyx.types import FunctionOfTime


@dataclasses.dataclass
class MomentMatchingPropagator(ObjectInterpretation, HandlesSelf):
    """Effectful handler that interprets ``dsx.sample`` via moment matching.

    Instead of drawing samples from the dynamics (as ``Simulator`` does) or
    computing a marginal likelihood (as ``Filter`` does), this handler propagates
    a Gaussian state belief forward through the GP dynamics model using the
    analytic moment matching equations from PILCO (Eqs. 10-23).

    This is PILCO's core innovation: rather than sampling trajectories, we
    analytically compute the predictive distribution at each time step.

    The propagated trajectory (means, covariances) and cumulative expected
    reward are recorded as ``numpyro.deterministic`` sites.

    Usage::

        with MomentMatchingPropagator(pilco=pilco):
            model(obs_times=obs_times)

    Attributes:
        pilco: The ``PILCO`` instance containing the GP model, controller,
            reward function, and propagation settings.
    """

    pilco: "PILCO"

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        if obs_times is not None:
            self._propagate_moments(name, obs_times)

        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    def _propagate_moments(self, name: str, obs_times: Array):
        """Propagate Gaussian belief via moment matching and record results.

        Records:
            - ``{name}_mm_means``: Predicted state means, shape ``(H+1, D)``.
            - ``{name}_mm_covs_diag``: Diagonal of predicted covariances, shape ``(H+1, D)``.
            - ``{name}_mm_reward``: Cumulative expected reward (scalar).
        """
        p = self.pilco
        m_x, s_x = p.m_init, p.s_init

        means = [m_x]
        covs_diag = [jnp.diag(s_x)]
        total_reward = jnp.array(0.0)

        T = min(len(obs_times), p.horizon)
        for _ in range(T):
            m_x, s_x = p.propagate(m_x, s_x)
            r, _ = p.reward(m_x, s_x)
            total_reward = total_reward + r
            means.append(m_x)
            covs_diag.append(jnp.diag(s_x))

        numpyro.deterministic(f"{name}_mm_means", jnp.stack(means))
        numpyro.deterministic(f"{name}_mm_covs_diag", jnp.stack(covs_diag))
        numpyro.deterministic(f"{name}_mm_reward", total_reward)


class PILCO(eqx.Module):
    """PILCO: Probabilistic Inference for Learning COntrol.

    Implements the full PILCO algorithm (Algorithm 1) with deep integration
    into the dynestyx framework:

    - The GP dynamics model can be converted to a dynestyx
      ``DiscreteTimeStateEvolution`` via ``self.mgpr.to_state_evolution()``,
      enabling use with ``DiscreteTimeSimulator`` and ``Filter``.
    - The ``MomentMatchingPropagator`` effectful handler implements
      ``dsx.sample`` interpretation for PILCO's trajectory prediction.
    - Data collection uses dynestyx ``DynamicalModel`` + ``Simulator`` handlers.

    Attributes:
        mgpr: Multi-output GP dynamics model.
        controller: Policy (``LinearController`` or ``RBFController``).
        reward: Reward function.
        horizon: Planning horizon (number of prediction steps).
        m_init: Initial state mean, shape ``(state_dim,)``.
        s_init: Initial state covariance, shape ``(state_dim, state_dim)``.
        max_action: Maximum action for squashing (``None`` = no squashing).
    """

    mgpr: MGPR
    controller: eqx.Module
    reward: ExponentialReward
    horizon: int
    m_init: Array
    s_init: Array
    max_action: Array | None

    def __init__(
        self,
        X: Array,
        Y: Array,
        controller: eqx.Module,
        reward: ExponentialReward,
        horizon: int = 25,
        m_init: Array | None = None,
        s_init: Array | None = None,
        max_action: Array | None = None,
    ):
        state_dim = Y.shape[1]
        self.mgpr = MGPR(X, Y)
        self.controller = controller
        self.reward = reward
        self.horizon = horizon
        self.m_init = m_init if m_init is not None else X[0, :state_dim]
        self.s_init = (
            s_init if s_init is not None else 0.1 * jnp.eye(state_dim)
        )
        self.max_action = max_action

    def set_data(self, X: Array, Y: Array) -> "PILCO":
        """Return new PILCO with updated training data."""
        new_mgpr = self.mgpr.set_data(X, Y)
        return eqx.tree_at(lambda p: p.mgpr, self, new_mgpr)

    def to_dynamical_model(
        self,
        initial_condition=None,
        observation_model=None,
    ) -> DynamicalModel:
        """Create a dynestyx ``DynamicalModel`` using the GP as state evolution.

        This allows the learned GP dynamics to be used with any dynestyx handler:
        ``DiscreteTimeSimulator`` for trajectory sampling, ``Filter`` for state
        estimation, etc.

        Args:
            initial_condition: NumPyro distribution over initial state.
                Defaults to ``N(m_init, s_init)``.
            observation_model: Dynestyx observation model.
                Defaults to ``DiracIdentityObservation``.

        Returns:
            A ``DynamicalModel`` with GP-based discrete-time state evolution.
        """
        import numpyro.distributions as dist

        from dynestyx.models.observations import DiracIdentityObservation

        state_dim = self.mgpr.state_dim
        control_dim = self.mgpr.X.shape[1] - state_dim

        if initial_condition is None:
            initial_condition = dist.MultivariateNormal(
                loc=self.m_init,
                covariance_matrix=self.s_init,
            )
        if observation_model is None:
            observation_model = DiracIdentityObservation()

        return DynamicalModel(
            initial_condition=initial_condition,
            state_evolution=self.mgpr.to_state_evolution(),
            observation_model=observation_model,
            control_dim=control_dim if control_dim > 0 else None,
        )

    def propagate(
        self, m_x: Array, s_x: Array
    ) -> tuple[Array, Array]:
        """Single-step state propagation via moment matching (Eqs. 10-12).

        1. Compute action distribution from controller
        2. Optionally squash through sin()
        3. Form joint state-action distribution
        4. Predict state delta through GP moment matching
        5. Compute next state distribution
        """
        state_dim = m_x.shape[0]

        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        if self.max_action is not None:
            m_u, s_u, c_squash = squash_sin(m_u, s_u, self.max_action)
            c_xu = c_xu @ c_squash

        m_joint = jnp.concatenate([m_x, m_u])
        s_joint = jnp.block([
            [s_x, c_xu],
            [c_xu.T, s_u],
        ])

        M_delta, S_delta, V_delta = self.mgpr.predict_given_factorizations(
            m_joint, s_joint
        )

        s1 = s_joint[:state_dim, :]
        m_next = m_x + M_delta
        s_next = s_x + S_delta + s1 @ V_delta + (s1 @ V_delta).T
        s_next = (s_next + s_next.T) / 2.0

        return m_next, s_next

    def predict(
        self, m_x: Array | None = None, s_x: Array | None = None
    ) -> tuple[Array, list[Array], list[Array]]:
        """Predict trajectory and compute cumulative reward."""
        if m_x is None:
            m_x = self.m_init
        if s_x is None:
            s_x = self.s_init

        total_reward = jnp.array(0.0)
        means = [m_x]
        covs = [s_x]

        for t in range(self.horizon):
            m_x, s_x = self.propagate(m_x, s_x)
            reward_t, _ = self.reward(m_x, s_x)
            total_reward = total_reward + reward_t
            means.append(m_x)
            covs.append(s_x)

        return total_reward, means, covs

    def predict_jit(
        self, m_x: Array, s_x: Array
    ) -> Array:
        """JIT-friendly version returning only the total reward."""

        def step_fn(carry, _):
            m, s, reward = carry
            m_new, s_new = self.propagate(m, s)
            r, _ = self.reward(m_new, s_new)
            return (m_new, s_new, reward + r), None

        (_, _, total_reward), _ = jax.lax.scan(
            step_fn, (m_x, s_x, jnp.array(0.0)), None, length=self.horizon
        )
        return total_reward

    def optimize_models(
        self,
        num_restarts: int = 3,
        max_iters: int = 200,
        learning_rate: float = 0.01,
    ) -> "PILCO":
        """Optimize GP hyperparameters by maximizing log marginal likelihood."""
        best_mgpr = self.mgpr
        best_lml = -jnp.inf

        hp_filter = jax.tree.map(lambda _: False, self.mgpr)
        hp_filter = eqx.tree_at(
            lambda m: (m.log_lengthscales, m.log_signal_variance, m.log_noise_variance),
            hp_filter,
            (True, True, True),
        )

        @eqx.filter_jit
        def train_step(mgpr, opt_state):
            loss, grads = eqx.filter_value_and_grad(
                lambda m: -m.log_marginal_likelihood()
            )(mgpr)
            updates, new_opt_state = optimizer.update(
                eqx.filter(grads, hp_filter), opt_state, eqx.filter(mgpr, hp_filter)
            )
            mgpr = eqx.apply_updates(mgpr, updates)
            return mgpr, new_opt_state, loss

        for restart in range(num_restarts):
            mgpr = self.mgpr if restart == 0 else _randomize_hyperparams(self.mgpr, restart)
            optimizer = optax.adam(learning_rate)
            opt_state = optimizer.init(eqx.filter(mgpr, hp_filter))

            for _ in range(max_iters):
                mgpr, opt_state, loss = train_step(mgpr, opt_state)

            lml = mgpr.log_marginal_likelihood()
            if lml > best_lml:
                best_lml = lml
                best_mgpr = mgpr

        return eqx.tree_at(lambda p: p.mgpr, self, best_mgpr)

    def optimize_policy(
        self,
        max_iters: int = 100,
        learning_rate: float = 0.01,
    ) -> "PILCO":
        """Optimize policy by maximizing expected cumulative reward.

        Uses Adam with analytic gradients through the full moment matching graph.
        """
        ctrl_filter = jax.tree.map(lambda _: False, self)
        ctrl_filter = eqx.tree_at(
            lambda p: p.controller, ctrl_filter,
            jax.tree.map(lambda _: True, self.controller),
        )

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(self, ctrl_filter))

        @eqx.filter_jit
        def train_step(pilco, opt_state):
            def neg_reward(pilco):
                return -pilco.predict_jit(pilco.m_init, pilco.s_init)

            loss, grads = eqx.filter_value_and_grad(neg_reward)(pilco)
            updates, new_opt_state = optimizer.update(
                eqx.filter(grads, ctrl_filter), opt_state,
                eqx.filter(pilco, ctrl_filter),
            )
            pilco = eqx.apply_updates(pilco, updates)
            return pilco, new_opt_state, loss

        pilco = self
        for _ in range(max_iters):
            pilco, opt_state, loss = train_step(pilco, opt_state)

        return pilco


def _randomize_hyperparams(mgpr: MGPR, seed: int) -> MGPR:
    """Randomize GP hyperparameters for restart."""
    key = jax.random.PRNGKey(seed * 42)
    k1, k2, k3 = jax.random.split(key, 3)
    new_ls = mgpr.log_lengthscales + 0.5 * jax.random.normal(k1, mgpr.log_lengthscales.shape)
    new_sv = mgpr.log_signal_variance + 0.5 * jax.random.normal(k2, mgpr.log_signal_variance.shape)
    new_nv = mgpr.log_noise_variance + 0.5 * jax.random.normal(k3, mgpr.log_noise_variance.shape)
    return eqx.tree_at(
        lambda m: (m.log_lengthscales, m.log_signal_variance, m.log_noise_variance),
        mgpr,
        (new_ls, new_sv, new_nv),
    )
