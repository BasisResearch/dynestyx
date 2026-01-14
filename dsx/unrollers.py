import jax
import jax.numpy as jnp
import jax.random as jr
import dataclasses
from typing import Optional, Callable, Any

from dsx.ops import Context
from dsx.handlers import BaseUnroller
from dsx.dynamical_models import DynamicalModel
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan as nscan
import diffrax as dfx
import rodeo
from dsx import probabilistic_numerics as pn


@dataclasses.dataclass
class DiscreteTimeUnroller(BaseUnroller):
    """Assume we have ic, transition, and observation distributions,
    as well as (time_index, observation) pairs in the context.

    Simply unroll the model and add obs=data as you would in numpyro.

    This does not explicitly add logfactors; it let's numpyro do it automatically.
    Instead, it just unrolls the model and adds observed sites
    (which numpyro uses to compute logfactors).
    """

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        if obs_times is None:
            return
        if isinstance(obs_traj.values, dict):
            raise ValueError("obs_traj.values must be an Array or None, not a dict")
        obs_values = obs_traj.values

        T = len(obs_times)

        # Sample initial state
        x_prev = numpyro.sample("x_0", dynamics.initial_condition)

        # sample initial observation
        y_0 = numpyro.sample(
            "y_0",
            dynamics.observation_model(x=x_prev, u=None, t=obs_times[0]),
            obs=obs_values[0] if obs_values is not None else None,
        )

        def _step(x_prev, t_idx):
            t = obs_times[t_idx]
            # Sample next state
            x_t = numpyro.sample(
                f"x_{t_idx + 1}", dynamics.state_evolution(x=x_prev, u=None, t=t)
            )

            # Sample observation
            y_t = numpyro.sample(
                f"y_{t_idx + 1}",
                dynamics.observation_model(x=x_t, u=None, t=t),
                obs=obs_values[t_idx + 1] if obs_values is not None else None,
            )
            return x_t, (x_t, y_t)

        # Run scan and collect states and observations
        # scan_outputs will be (scan_states, scan_observations) where each is shape (T-1, ...)
        _, scan_outputs = nscan(_step, x_prev, jnp.arange(T - 1))
        scan_states, scan_observations = scan_outputs

        # Stack initial state/observation with scanned results
        # x_prev is shape (state_dim,) or scalar, scan_states is (T-1, state_dim)
        # y_0 is shape (obs_dim,) or scalar, scan_observations is (T-1, obs_dim)
        # Use expand_dims to ensure proper shape for concatenation
        x_0_expanded = jnp.expand_dims(x_prev, axis=0)  # shape (1, state_dim) or (1,)
        y_0_expanded = jnp.expand_dims(y_0, axis=0)  # shape (1, obs_dim) or (1,)
        states = jnp.concatenate(
            [x_0_expanded, scan_states], axis=0
        )  # shape (T, state_dim)
        observations = jnp.concatenate(
            [y_0_expanded, scan_observations], axis=0
        )  # shape (T, obs_dim)

        # Add deterministic sites
        numpyro.deterministic("times", obs_times)
        numpyro.deterministic("states", states)
        numpyro.deterministic("observations", observations)


@dataclasses.dataclass
class ODEUnroller(BaseUnroller):
    """Assume we have ic, transition, and observation distributions,
    as well as (time_index, observation) pairs in the context.

    Simply unroll the model and add obs=data as you would in numpyro.

    This does not add logfactors, just unrolls the model and adds observed sites.
    """

    solver: dfx.AbstractSolver = dfx.Tsit5()
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize()
    dt0: float = 0.01
    max_steps: int = 10_000

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        obs_values = obs_traj.values
        if obs_times is None:
            return
        T = len(obs_times)

        # Sample initial state
        x_prev = numpyro.sample("x_0", dynamics.initial_condition)

        def f(t, y, args):
            return dynamics.state_evolution.drift(x=y, u=None, t=t)

        # Solve ODE at all observation times using diffrax
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(f),
            solver=self.solver,
            t0=obs_times[0],
            t1=obs_times[-1],
            dt0=self.dt0,
            y0=x_prev,
            saveat=dfx.SaveAt(ts=obs_times),
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )
        x_sol = sol.ys  # shape (T, state_dim) # includes initial state at t0

        # use scan to sample observations and collect them
        def _step(carry, t_idx):
            x_t = x_sol[t_idx]
            t = obs_times[t_idx]
            # Sample observation
            y_t = numpyro.sample(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=None, t=t),
                obs=obs_values[t_idx] if obs_values is not None else None,
            )
            return carry, y_t

        # Run scan and collect observations
        # scan_outputs will be observations of shape (T, obs_dim)
        _, scan_observations = nscan(_step, None, jnp.arange(T))
        observations = scan_observations  # shape (T, obs_dim)

        # Add deterministic sites
        numpyro.deterministic("times", obs_times)
        numpyro.deterministic("states", x_sol)  # x_sol already has shape (T, state_dim)
        numpyro.deterministic("observations", observations)


@dataclasses.dataclass
class ProbabilisticODEUnroller(BaseUnroller):
    """Probabilistic ODE unroller using rodeo for probabilistic numerics.
    
    Uses rodeo to solve the ODE probabilistically, representing the solution
    as a joint Gaussian distribution. This joint Gaussian is then related to
    data (if present) via the observation_model.
    
    This does not add logfactors, just unrolls the model and adds observed sites.
    
    Requires that observation times are on a regular grid.
    """

    key: Optional[jax.Array] = None
    ode_weight: Optional[Any] = None
    interrogate: Callable = rodeo.interrogate.interrogate_kramer
    prior_pars: Optional[Any] = None
    kalman_type: str = "standard"
    sigma: float = 1.0  # Prior variance parameter

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        obs_values = obs_traj.values
        if obs_times is None:
            return
        T = len(obs_times)
        state_dim = dynamics.state_dim

        # Choose a key
        key = self.key if self.key is not None else jr.PRNGKey(0)

        # Sample initial state (known, no variance in the solution)
        x0 = numpyro.sample("x_0", dynamics.initial_condition)

        # Prepare all inputs for rodeo
        rodeo_inputs = pn.prepare_rodeo_inputs(
            dynamics=dynamics,
            obs_times=obs_times,
            x0=x0,
            ode_weight=self.ode_weight,
            interrogate=self.interrogate,
            prior_pars=self.prior_pars,
            sigma=self.sigma,
        )

        # Solve ODE probabilistically with rodeo
        mean_state_smooth, var_state_smooth = rodeo.solve_mv(
            key=key,
            **rodeo_inputs,
        )
        
        # Reshape rodeo output to (T, state_dim) format
        # rodeo returns (n_steps+1, ...) = (T, ...) including initial condition at index 0
        states_mean, states_cov = pn.reshape_rodeo_output(
            mean_state_smooth, var_state_smooth, T, state_dim
        )
        
        # Exclude x0 (index 0) from the probabilistic solution since it's known
        # Build joint distribution for times 1..T-1 (excluding initial condition)
        states_mean_future = states_mean[1:]  # (T-1, state_dim) - exclude x0
        states_cov_future = states_cov[1:]    # (T-1, state_dim, state_dim)
        
        # Build joint covariance matrix for future states (excluding x0)
        mean_flat = states_mean_future.flatten()  # ((T-1) * state_dim,)
        covariance_flat = pn.build_joint_covariance(states_cov_future, T - 1, state_dim)
        
        # Sample future states from the joint Gaussian (x0 is excluded, has no variance)
        solution_dist = dist.MultivariateNormal(loc=mean_flat, covariance_matrix=covariance_flat)
        states_future_flat = numpyro.sample("states_future_flat", solution_dist)
        states_future = states_future_flat.reshape(T - 1, state_dim)  # (T-1, state_dim)
        
        # Stitch x0 with future states to get full trajectory
        x0_expanded = jnp.expand_dims(x0, axis=0)  # (1, state_dim)
        states = jnp.concatenate([x0_expanded, states_future], axis=0)  # (T, state_dim)

        # Apply observation_model explicitly to relate states to observations
        def _step(carry, t_idx):
            x_t = states[t_idx]
            t = obs_times[t_idx]
            # Sample observation from observation_model
            y_t = numpyro.sample(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=None, t=t),
                obs=obs_values[t_idx] if obs_values is not None else None,
            )
            return carry, y_t

        # Run scan and collect observations
        # scan_outputs will be observations of shape (T, obs_dim)
        _, scan_observations = nscan(_step, None, jnp.arange(T))
        observations = scan_observations  # shape (T, obs_dim)

        # Add deterministic sites
        numpyro.deterministic("times", obs_times)
        numpyro.deterministic("states", states)  # states has shape (T, state_dim)
        numpyro.deterministic("observations", observations)
