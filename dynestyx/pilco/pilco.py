"""PILCO algorithm with dynestyx effectful handlers and NumPyro integration."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as npdist
import optax
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import Array
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator
from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.models import DynamicalModel
from dynestyx.models.observations import DiracIdentityObservation
from dynestyx.pilco.controllers import LinearController, RBFController
from dynestyx.pilco.mgpr import MGPR
from dynestyx.pilco.rewards import ExponentialReward
from dynestyx.types import FunctionOfTime

Controller = LinearController | RBFController


@dataclasses.dataclass
class MomentMatchingPropagator(ObjectInterpretation, HandlesSelf):
    """
    Effectful handler interpreting ``dsx.sample`` via analytic moment matching.

    Instead of sampling trajectories (``Simulator``) or computing marginal
    likelihoods (``Filter``), propagates a Gaussian state belief forward through
    the GP dynamics using Eqs. 10-23 of Deisenroth & Rasmussen (2011).

    Records ``{name}_mm_means``, ``{name}_mm_covs_diag``, and
    ``{name}_mm_reward`` as ``numpyro.deterministic`` sites.
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


def collect_rollout(
    env_dynamics: DynamicalModel,
    obs_times: Array,
    key: Array,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
) -> dict:
    """Collect a trajectory using ``dsx.sample()`` + ``DiscreteTimeSimulator``."""

    def _model(obs_times, ctrl_times=None, ctrl_values=None):
        return dsx.sample(
            "env",
            env_dynamics,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )

    with DiscreteTimeSimulator():
        predictive = Predictive(_model, num_samples=1, exclude_deterministic=False)
        result = predictive(
            key,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )

    return {k: v.squeeze(0) for k, v in result.items()}


def collect_random_rollout(
    env_dynamics: DynamicalModel,
    obs_times: Array,
    key: Array,
    max_action: float,
    control_dim: int = 1,
) -> dict:
    """Collect a trajectory with uniform random controls via ``dsx.sample()``."""
    k1, k2 = jax.random.split(key)
    T = len(obs_times)
    ctrl_values = jax.random.uniform(
        k1, (T, control_dim), minval=-max_action, maxval=max_action
    )
    return collect_rollout(
        env_dynamics,
        obs_times=obs_times,
        key=k2,
        ctrl_times=obs_times,
        ctrl_values=ctrl_values,
    )


class PILCO(eqx.Module):
    """
    PILCO: Probabilistic Inference for Learning COntrol.

    The GP dynamics model wraps as ``GPStateEvolution``
    (a ``DiscreteTimeStateEvolution``), usable with all dynestyx handlers.
    Data collection uses ``collect_rollout()`` via ``dsx.sample()`` +
    ``DiscreteTimeSimulator``. Trajectory prediction uses the
    ``MomentMatchingPropagator`` effectful handler.
    """

    mgpr: MGPR
    controller: Controller
    reward: ExponentialReward
    horizon: int
    m_init: Array
    s_init: Array
    max_action: Array | None

    def __init__(
        self,
        X: Array,
        Y: Array,
        controller: Controller,
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
        self.s_init = s_init if s_init is not None else 0.1 * jnp.eye(state_dim)
        self.max_action = max_action

    def set_data(self, X: Array, Y: Array) -> "PILCO":
        return eqx.tree_at(lambda p: p.mgpr, self, self.mgpr.set_data(X, Y))

    def to_dynamical_model(
        self,
        initial_condition=None,
        observation_model=None,
    ) -> DynamicalModel:
        """Create a ``DynamicalModel`` using the GP as state evolution.

        Works with ``DiscreteTimeSimulator``, ``Filter``,
        ``MomentMatchingPropagator``, etc.
        """
        state_dim = self.mgpr.state_dim
        control_dim = self.mgpr.X.shape[1] - state_dim

        if initial_condition is None:
            initial_condition = npdist.MultivariateNormal(
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

    def make_numpyro_model(self):
        """Return a NumPyro model calling ``dsx.sample()`` with GP dynamics.

        Usable with any handler context::

            with DiscreteTimeSimulator():
                Predictive(pilco.make_numpyro_model(), ...)(key, obs_times=...)

            with MomentMatchingPropagator(pilco=pilco):
                Predictive(pilco.make_numpyro_model(), ...)(key, obs_times=...)
        """
        gp_dynamics = self.to_dynamical_model()

        def _model(obs_times, obs_values=None, ctrl_times=None, ctrl_values=None):
            return dsx.sample(
                "gp",
                gp_dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

        return _model

    def propagate(self, m_x: Array, s_x: Array) -> tuple[Array, Array]:
        """Single-step moment matching propagation (Eqs. 10-12)."""
        state_dim = m_x.shape[0]

        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        if self.max_action is not None:
            # Squash through sin() for bounded controls: u_sat = max_action * sin(u)
            s_diag = jnp.diag(s_u)
            e = jnp.exp(-0.5 * s_diag)
            m_sin = self.max_action * e * jnp.sin(m_u)
            lq = -(s_diag[:, None] + s_diag[None, :]) / 2.0
            s_sin = (
                jnp.exp(lq + s_u) * jnp.cos(m_u[:, None] - m_u[None, :])
                - jnp.exp(lq - s_u) * jnp.cos(m_u[:, None] + m_u[None, :])
            ) / 2.0
            s_sin = self.max_action[:, None] * self.max_action[None, :] * s_sin
            s_sin = s_sin - jnp.outer(m_sin, m_sin)
            c_squash = jnp.diag(self.max_action * jnp.cos(m_u) * e)
            m_u, s_u = m_sin, s_sin
            c_xu = c_xu @ c_squash

        s_u = (s_u + s_u.T) / 2.0 + 1e-6 * jnp.eye(m_u.shape[0])

        m_joint = jnp.concatenate([m_x, m_u])
        s_joint = jnp.block(
            [
                [s_x, c_xu],
                [c_xu.T, s_u],
            ]
        )
        s_joint = (s_joint + s_joint.T) / 2.0

        M_delta, S_delta, V_delta = self.mgpr.predict_given_factorizations(
            m_joint, s_joint
        )

        s1 = s_joint[:state_dim, :]
        m_next = m_x + M_delta
        s_next = s_x + S_delta + s1 @ V_delta + (s1 @ V_delta).T
        s_next = (s_next + s_next.T) / 2.0
        eigvals, eigvecs = jnp.linalg.eigh(s_next)
        eigvals = jnp.maximum(eigvals, 1e-6)
        s_next = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        return m_next, s_next

    def predict(
        self, m_x: Array | None = None, s_x: Array | None = None
    ) -> tuple[Array, list[Array], list[Array]]:
        if m_x is None:
            m_x = self.m_init
        if s_x is None:
            s_x = self.s_init

        total_reward = jnp.array(0.0)
        means = [m_x]
        covs = [s_x]

        for _ in range(self.horizon):
            m_x, s_x = self.propagate(m_x, s_x)
            reward_t, _ = self.reward(m_x, s_x)
            total_reward = total_reward + reward_t
            means.append(m_x)
            covs.append(s_x)

        return total_reward, means, covs

    def predict_jit(self, m_x: Array, s_x: Array) -> Array:
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
        best_lml: Array = jnp.array(-jnp.inf)

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
                eqx.filter(grads, hp_filter),
                opt_state,
                eqx.filter(mgpr, hp_filter),
            )
            mgpr = eqx.apply_updates(mgpr, updates)
            return mgpr, new_opt_state, loss

        for restart in range(num_restarts):
            mgpr = (
                self.mgpr
                if restart == 0
                else _randomize_hyperparams(self.mgpr, restart)
            )
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
        """Optimize policy by maximizing expected cumulative reward."""
        ctrl_filter = jax.tree.map(lambda _: False, self)
        ctrl_filter = eqx.tree_at(
            lambda p: p.controller,
            ctrl_filter,
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
                eqx.filter(grads, ctrl_filter),
                opt_state,
                eqx.filter(pilco, ctrl_filter),
            )
            pilco = eqx.apply_updates(pilco, updates)
            return pilco, new_opt_state, loss

        pilco = self
        for _ in range(max_iters):
            pilco, opt_state, loss = train_step(pilco, opt_state)

        return pilco


def _randomize_hyperparams(mgpr: MGPR, seed: int) -> MGPR:
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
