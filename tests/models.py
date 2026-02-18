import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx.dynamical_models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    LinearGaussianStateEvolution,
)
from dynestyx.observations import DiracIdentityObservation, LinearGaussianObservation


def hmm_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    K = 3  # number of discrete states

    # -------------------------------------------------
    # Transition matrix
    # -------------------------------------------------
    A = numpyro.sample(
        "A", dist.Dirichlet(jnp.ones(K)).expand([K]).to_event(1)
    )  # shape (K, K)

    # -------------------------------------------------
    # Emission parameters
    # -------------------------------------------------
    mu = numpyro.sample(
        "mu", dist.Normal(0.0, 10.0).expand([K]).to_event(1)
    )  # shape (K,)

    sigma = numpyro.sample("sigma", dist.Uniform(0.1, 2.0))

    # -------------------------------------------------
    # Initial condition
    # -------------------------------------------------
    initial_condition = dist.Categorical(probs=jnp.ones(K) / K)

    # -------------------------------------------------
    # State evolution
    # -------------------------------------------------
    def state_evolution(x, u, t_now, t_next):
        # x is an integer in {0, ..., K-1}
        # Add light dependence on u: shift transition probabilities slightly
        u_effect = jnp.sum(u) if u is not None else 0.0
        probs = A[x] + u_effect * (1.0 / K - A[x])  # Small perturbation toward uniform
        probs = jnp.clip(probs, 1e-6, 1.0)  # Ensure valid probabilities
        probs = probs / jnp.sum(probs)  # Renormalize
        return dist.Categorical(probs=probs)

    # -------------------------------------------------
    # Observation model
    # -------------------------------------------------
    def observation_model(x, u, t):
        # y_t | x_t ~ N(mu[x_t] + u_effect, sigma^2)
        u_effect = jnp.sum(u) if u is not None else 0.0
        return dist.Normal(mu[x] + u_effect, sigma)

    dynamics = DynamicalModel(
        state_dim=K,
        observation_dim=1,
        control_dim=1,  # Model uses controls, and are ignored when u=None
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def discrete_time_l63_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Model that samples drift parameter rho and uses it in dynamics."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    def drift(x):
        return jnp.array(
            [
                10.0 * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - (8.0 / 3.0) * x[2],
            ]
        )

    def state_evolution(x, u, t_now, t_next):
        # Add light dependence on u: add small control term to drift
        drift_term = drift(x)
        if u is None or u.shape == (0,):
            u_effect = jnp.zeros_like(drift_term)
        else:
            u_effect = 10 * u
        loc = x + 0.01 * (drift_term + u_effect)
        cov = 0.01 * jnp.eye(3)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
        control_dim=1,  # Model uses controls, and are ignored when u=None
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=20.0**2 * jnp.eye(3)
        ),
        state_evolution=state_evolution,
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[1.0**2]])
        ),
    )

    # TODO: observation_model should simply be dist.MultivariateNormal(...) here,
    # but for now we wrap it in LinearGaussianObservation for so that we can extract
    # H and R later for CD-Dynamax conversion (structure exploiting algorithms).
    # In the future, we will build internal logic to identify linear-gaussian observation models
    # and extract H, R automatically.

    # TODO: Functions for drift, diffusion_coefficient, diffusion_covariance should not
    # require (x, u, t) arguments if they are not used. We can wrap them internally.
    # e.g. diffusion_coefficient=jnp.eye(3)
    # e.g. drift = lambda x: F(x, rho)

    # Return a sampled dynamical model, named "f".
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def continuous_time_stochastic_l63_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Model that samples drift parameter rho and uses it in dynamics."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
        control_dim=1,  # Model uses controls, and are ignored when u=None
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=20.0**2 * jnp.eye(3)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: (
                jnp.array(
                    [
                        10.0 * (x[1] - x[0]),
                        x[0] * (rho - x[2]) - x[1],
                        x[0] * x[1] - (8.0 / 3.0) * x[2],
                    ]
                )
                + (10 * u if u is not None else jnp.zeros(3))
            ),
            diffusion_coefficient=lambda x, u, t: jnp.eye(3),
            diffusion_covariance=lambda x, u, t: jnp.eye(3),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[1.0**2]])
        ),
    )

    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def continuous_time_stochastic_l63_model_dirac_obs(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """L63 SDE with full-state Dirac observations (observation_dim=state_dim=3)."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=3,
        control_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=20.0**2 * jnp.eye(3)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: (
                jnp.array(
                    [
                        10.0 * (x[1] - x[0]),
                        x[0] * (rho - x[2]) - x[1],
                        x[0] * x[1] - (8.0 / 3.0) * x[2],
                    ]
                )
                + (10 * u if u is not None else jnp.zeros_like(x))
            ),
            diffusion_coefficient=lambda x, u, t: jnp.eye(3),
            diffusion_covariance=lambda x, u, t: jnp.eye(3),
        ),
        observation_model=DiracIdentityObservation(),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def continuous_time_LTI_gaussian(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """2D linear SDE with a sampled coupling."""
    rho = numpyro.sample("rho", dist.Uniform(0.0, 5.0))

    A = jnp.array([[-1.0, 0.0], [rho, -1.0]])

    dynamics = DynamicalModel(
        state_dim=2,
        observation_dim=1,
        control_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=1.0**2 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: A @ x + (10 * u if u is not None else jnp.zeros(2)),
            diffusion_coefficient=lambda x, u, t: jnp.eye(2),
            diffusion_covariance=lambda x, u, t: jnp.eye(2),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[0.0, 1.0]]), R=jnp.array([[1.0**2]])
        ),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def stochastic_volatility(
    identity_observation: bool = False,
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Discrete-time stochastic volatility: log-variance follows AR(1).
    One unknown parameter: phi (persistence). No controls.
    If identity_observation=True, y_t = x_t (DiracIdentityObservation); else noisily observed."""
    phi = numpyro.sample("phi", dist.Uniform(0.0, 1.0))  # type: ignore[arg-type]
    sigma_eta = 0.5  # fixed vol-of-vol

    initial_condition = dist.Normal(0.0, 1.0)

    def state_evolution(x, u, t_now, t_next):
        return dist.Normal(phi * x, sigma_eta)

    if identity_observation:
        observation_model = DiracIdentityObservation()
    else:

        def observation_model(x, u, t):  # type: ignore[misc]
            return dist.Normal(0.0, jnp.exp(x / 2.0))

    dynamics = DynamicalModel(
        state_dim=1,
        observation_dim=1,
        control_dim=0,
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def continuous_time_deterministic_l63_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Model that samples drift parameter rho and uses it in dynamics (ODE, no diffusion)."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
        control_dim=1,  # Model uses controls, and are ignored when u=None
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=2.0**2 * jnp.eye(3)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: (
                jnp.array(
                    [
                        10.0 * (x[1] - x[0]),
                        x[0] * (rho - x[2]) - x[1],
                        x[0] * x[1] - (8.0 / 3.0) * x[2],
                    ]
                )
                + (10 * u if u is not None else jnp.zeros(3))
            ),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[1.0**2]])
        ),
    )

    # Return a sampled dynamical model, named "f".
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def discrete_time_lti_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Discrete-time LTI with one sampled parameter alpha (F[0,0]); for use with filter_type='kf'.
    Supports controls: when control_trajectory is provided in context, B and D are used (state_dim=2, control_dim=1).
    """
    alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7))
    state_dim = 2
    emission_dim = 1
    control_dim = 1
    A = jnp.array([[alpha, 0.0], [0.0, 0.8]])
    Q = 0.1 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.5**2]])
    initial_mean = jnp.zeros(state_dim)
    initial_cov = jnp.eye(state_dim)
    B = jnp.array([[0.1], [0.0]])
    b = jnp.zeros(state_dim)
    D = jnp.array([[0.01]])
    d = jnp.zeros(emission_dim)
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(initial_mean, initial_cov),
        state_evolution=LinearGaussianStateEvolution(A=A, B=B, bias=b, cov=Q),
        observation_model=LinearGaussianObservation(H=H, D=D, bias=d, R=R),
        state_dim=state_dim,
        observation_dim=emission_dim,
        control_dim=control_dim,
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def jumpy_controls_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    dynamics = DynamicalModel(
        state_dim=1,
        observation_dim=1,
        control_dim=1,
        initial_condition=dist.MultivariateNormal(0.0, 1.0 * jnp.eye(1)),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            x + u, 0.01 * jnp.eye(1)
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]), R=jnp.array([[0.1**2]])
        ),
    )

    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
