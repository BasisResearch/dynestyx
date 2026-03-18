"""Main PILCO algorithm implementation.

Orchestrates the GP dynamics model learning, moment-matching trajectory
prediction, and gradient-based policy optimization.

References:
    Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and
    Data-Efficient Approach to Policy Search. ICML, Algorithm 1.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from dynestyx.pilco.controllers import squash_sin
from dynestyx.pilco.mgpr import MGPR
from dynestyx.pilco.rewards import ExponentialReward


class PILCO(eqx.Module):
    """PILCO: Probabilistic Inference for Learning COntrol.

    Implements the full PILCO algorithm loop:
    1. Learn GP dynamics model from data
    2. Predict trajectory via moment matching
    3. Optimize policy by maximizing expected cumulative reward

    Attributes:
        mgpr: Multi-output GP dynamics model.
        controller: Policy (LinearController or RBFController).
        reward: Reward function.
        horizon: Planning horizon (number of prediction steps).
        m_init: Initial state mean, shape (state_dim,).
        s_init: Initial state covariance, shape (state_dim, state_dim).
        max_action: Maximum action for squashing (None = no squashing).
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
        """Initialize PILCO.

        Args:
            X: Training inputs [states, actions], shape (n, state_dim + control_dim).
            Y: Training targets (state deltas), shape (n, state_dim).
            controller: Policy module with compute_action method.
            reward: Reward function.
            horizon: Planning horizon.
            m_init: Initial state mean. Defaults to first state in data.
            s_init: Initial state covariance. Defaults to 0.1 * I.
            max_action: Maximum action for sin-squashing. None disables squashing.
        """
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

    def propagate(
        self, m_x: Array, s_x: Array
    ) -> tuple[Array, Array]:
        """Single-step state propagation via moment matching.

        1. Compute action distribution from controller
        2. Optionally squash through sin()
        3. Form joint state-action distribution
        4. Predict state delta through GP
        5. Compute next state distribution

        Implements Eqs. 10-12 of Deisenroth & Rasmussen (2011).

        Args:
            m_x: Current state mean, shape (state_dim,).
            s_x: Current state covariance, shape (state_dim, state_dim).

        Returns:
            m_next: Next state mean, shape (state_dim,).
            s_next: Next state covariance, shape (state_dim, state_dim).
        """
        state_dim = m_x.shape[0]

        # 1. Get action distribution
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        # 2. Optional squashing
        if self.max_action is not None:
            m_u, s_u, c_squash = squash_sin(m_u, s_u, self.max_action)
            c_xu = c_xu @ c_squash

        # 3. Form joint [x, u] distribution
        control_dim = m_u.shape[0]
        m_joint = jnp.concatenate([m_x, m_u])
        s_joint = jnp.block([
            [s_x, c_xu],
            [c_xu.T, s_u],
        ])

        # 4. Predict delta through GP
        M_delta, S_delta, V_delta = self.mgpr.predict_given_factorizations(
            m_joint, s_joint
        )

        # 5. Next state = current + delta (Eq. 10-11)
        # Extract state portion of cross-covariance
        # V_delta is (state_dim + control_dim, state_dim)
        # We need cov[x, delta] which involves V_delta[:state_dim, :]
        # plus the contribution through the action correlation (Eq. 12)

        # Full input-output cross-covariance from the joint input
        # s1 = cov[x, joint_input] = [s_x, c_xu] = s_joint[:state_dim, :]
        s1 = s_joint[:state_dim, :]  # (state_dim, state_dim + control_dim)

        m_next = m_x + M_delta  # Eq. 10
        s_next = (
            s_x
            + S_delta
            + s1 @ V_delta
            + (s1 @ V_delta).T
        )  # Eq. 11

        # Symmetrize for numerical stability
        s_next = (s_next + s_next.T) / 2.0

        return m_next, s_next

    def predict(
        self, m_x: Array | None = None, s_x: Array | None = None
    ) -> tuple[Array, list[Array], list[Array]]:
        """Predict trajectory and compute cumulative reward.

        Unrolls the moment matching for `horizon` steps and accumulates
        the expected reward at each step.

        Args:
            m_x: Initial state mean. Defaults to self.m_init.
            s_x: Initial state covariance. Defaults to self.s_init.

        Returns:
            total_reward: Cumulative expected reward (scalar).
            means: List of state means at each step.
            covs: List of state covariances at each step.
        """
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
        """JIT-friendly version that returns only the total reward.

        Uses jax.lax.fori_loop for efficient compilation.
        """

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
        """Optimize GP hyperparameters by maximizing log marginal likelihood.

        Args:
            num_restarts: Number of random restarts.
            max_iters: Maximum optimization iterations per restart.
            learning_rate: Adam learning rate.

        Returns:
            New PILCO with optimized GP hyperparameters.
        """
        best_mgpr = self.mgpr
        best_lml = -jnp.inf

        # Get the trainable parameters (hyperparameters only, not data)
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
        """Optimize policy parameters by maximizing expected cumulative reward.

        Uses Adam optimizer with analytic gradients through the moment matching
        computation graph.

        Args:
            max_iters: Maximum optimization iterations.
            learning_rate: Adam learning rate.

        Returns:
            New PILCO with optimized controller parameters.
        """
        # Filter: only optimize controller parameters
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
    new_ls = mgpr.log_lengthscales + 0.5 * jax.random.normal(
        k1, mgpr.log_lengthscales.shape
    )
    new_sv = mgpr.log_signal_variance + 0.5 * jax.random.normal(
        k2, mgpr.log_signal_variance.shape
    )
    new_nv = mgpr.log_noise_variance + 0.5 * jax.random.normal(
        k3, mgpr.log_noise_variance.shape
    )
    return eqx.tree_at(
        lambda m: (m.log_lengthscales, m.log_signal_variance, m.log_noise_variance),
        mgpr,
        (new_ls, new_sv, new_nv),
    )
