import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete


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

    # TODO: Functions for drift, diffusion_coefficient should not
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
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[1.0**2]])
        ),
    )

    # TODO: observation_model should simply be dist.MultivariateNormal(...) here,
    # but for now we wrap it in LinearGaussianObservation for so that we can extract
    # H and R later for CD-Dynamax conversion (structure exploiting algorithms).
    # In the future, we will build internal logic to identify linear-gaussian observation models
    # and extract H, R automatically.

    # TODO: Functions for drift, diffusion_coefficient should not
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


def continuous_time_stochastic_l63_model_dirac_obs(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """L63 SDE with full-state Dirac observations (observation_dim=state_dim=3)."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    dynamics = DynamicalModel(
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


def continuous_time_lti_simplified_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Continuous-time LTI using LTI_continuous factory: only rho = A[1,0] is sampled."""
    rho = numpyro.sample("rho", dist.Uniform(0.0, 5.0))
    state_dim = 2
    A = jnp.array([[-1.0, 0.0], [rho, -1.0]])
    L = jnp.eye(state_dim)
    H = jnp.array([[0.0, 1.0]])
    R = jnp.array([[1.0**2]])
    B = jnp.array([[0.0], [10.0]])
    dynamics = LTI_continuous(A=A, L=L, H=H, R=R, B=B)
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
        control_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=1.0**2 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: A @ x + (10 * u if u is not None else jnp.zeros(2)),
            diffusion_coefficient=lambda x, u, t: jnp.eye(2),
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


def continuous_time_potential_dynamics_model(
    mode: str = "both",
    obs_times=None,
    obs_values=None,
):
    """1D continuous-time model supporting drift-only, grad-only, or both."""
    if mode not in {"drift_only", "grad_only", "both"}:
        raise ValueError(f"Unsupported mode: {mode}")

    alpha = jnp.asarray(numpyro.sample("alpha", dist.Uniform(0.1, 2.0)))  # type: ignore[arg-type]
    beta = jnp.asarray(numpyro.sample("beta", dist.Uniform(0.1, 2.0)))  # type: ignore[arg-type]

    drift = None
    potential = None
    if mode in {"drift_only", "both"}:
        drift = lambda x, u, t: jnp.asarray(-alpha * x)
    if mode in {"grad_only", "both"}:
        potential = lambda x, u, t: jnp.asarray(0.5 * beta * jnp.sum(x**2))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1), covariance_matrix=0.5**2 * jnp.eye(1)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=drift,
            potential=potential,
            use_negative_gradient=True,
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


def discrete_time_lti_simplified_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    """Discrete-time LTI using LTI_discrete factory: only alpha = A[0,0] is sampled."""
    alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7))
    state_dim = 2
    A = jnp.array([[alpha, 0.1], [0.1, 0.8]])
    Q = 0.1 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.5**2]])
    B = jnp.array([[0.1], [0.0]])
    D = jnp.array([[0.01]])
    dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R, B=B, D=D)
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
    dynamics = LTI_discrete(
        A=jnp.array([[1.0]]),
        Q=0.01 * jnp.eye(1),
        H=jnp.array([[1.0]]),
        R=jnp.array([[0.1**2]]),
        B=jnp.array([[1.0]]),
    )

    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )


def jumpy_controls_model_sde(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x + u,
        diffusion_coefficient=lambda x, u, t: 0.01 * jnp.eye(1),
    )
    dynamics = DynamicalModel(
        control_dim=1,
        initial_condition=dist.MultivariateNormal(0.0, 1.0 * jnp.eye(1)),
        state_evolution=state_evolution,
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


def jumpy_controls_model_ode(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
):
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x + u,
    )
    dynamics = DynamicalModel(
        control_dim=1,
        initial_condition=dist.MultivariateNormal(0.0, 1.0 * jnp.eye(1)),
        state_evolution=state_evolution,
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]), R=jnp.array([[0.01**2]])
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


def interacting_particles_gaussian_kernel_model(
    N=4,
    obs_times=None,
    obs_values=None,
    sigma: float = 0.2,
    bg_centers=None,
    bg_strengths=None,
):
    """N particles in 1D with pairwise Gaussian interaction kernel + known background potential.

    Drift for particle i:
        interaction: coefficient * sum_j K(x_j - x_i; scale) * (x_j - x_i) / N
        background:  -grad V(x_i),  V = -sum_k bg_strengths[k] * exp(-0.5*||x_i - bg_centers[k]||^2)

    where K(r; scale) = exp(-0.5 * (r / scale)^2)  (kernel centred at r=0).

    Learnable parameters:
      coefficient – interaction amplitude; negative = repulsion (particles spread within wells)
      scale       – kernel width (range of the interaction)
    Background potential parameters (bg_centers, bg_strengths) are KNOWN / fixed.
    Observations are noise-free (DiracIdentityObservation); partial NaN rows
    trigger the two-mask scan which samples latent states for unobserved particles.
    """
    if bg_centers is None:
        bg_centers = jnp.array([[-2.0], [2.0]])  # (K, 1)
    if bg_strengths is None:
        bg_strengths = jnp.array([1.0, 1.0])  # (K,)

    K_bg = bg_centers.shape[0]

    coefficient = numpyro.sample("coefficient", dist.Normal(0.0, 2.0))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 0.5))

    def state_evolution(x, u, t_now, t_next):
        dt = t_next - t_now
        # Pairwise Gaussian interaction drift (kernel centred at r=0)
        r = x[None, :] - x[:, None]  # (N, N) r_ij = x_j - x_i
        K_pair = jnp.exp(-0.5 * (r / scale) ** 2)
        interaction_drift = coefficient * jnp.sum(K_pair * r, axis=1) / N  # (N,)

        # Background Gaussian-mixture drift: -grad V per particle
        def bg_drift_single(x_i):
            g = jnp.zeros(())
            for k in range(K_bg):
                diff = x_i - bg_centers[k, 0]
                g = g - bg_strengths[k] * (-diff) * jnp.exp(-0.5 * diff**2)
            return -g

        bg_drift = jax.vmap(bg_drift_single)(x)  # (N,)
        drift = interaction_drift + bg_drift
        mean = x + dt * drift  # (N,) in scan, (N, T-1) in plate
        std = jnp.sqrt(sigma**2 * dt) * jnp.ones_like(mean)
        # Transpose so event_shape=(N,) and batch_shape=(T-1,) in plate;
        # .T is a no-op on 1D arrays in scan.
        return dist.Independent(dist.Normal(mean.T, std.T), 1)

    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.Independent(
            dist.Normal(jnp.zeros(N), jnp.full(N, jnp.sqrt(8.0))), 1
        ),
        state_evolution=state_evolution,
        observation_model=DiracIdentityObservation(),
    )

    dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)


def particle_sde_gaussian_potential_model(
    N=3,
    D=2,
    K=2,
    sigma=0.5,
    obs_times=None,
    obs_values=None,
):
    """N particles in D dimensions with drift = -grad(V), V = sum of weighted Gaussians.

    Learnable parameters: centers (K, D) and strengths (K,) of the Gaussian components.
    Diffusion is diagonal with known sigma.
    """
    centers = numpyro.sample(
        "centers", dist.Normal(0.0, 3.0).expand([K, D]).to_event(2)
    )
    strengths = numpyro.sample(
        "strengths", dist.LogNormal(0.0, 1.0).expand([K]).to_event(1)
    )

    def potential(x, u, t):
        particles = x.reshape(N, D)
        V = 0.0
        for k in range(K):
            diff = particles - centers[k]
            V = V - strengths[k] * jnp.sum(jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1)))
        return V

    state_dim = N * D
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(state_dim),
            covariance_matrix=2.0**2 * jnp.eye(state_dim),
        ),
        state_evolution=ContinuousTimeStateEvolution(
            potential=potential,
            use_negative_gradient=True,
            diffusion_coefficient=lambda x, u, t: sigma * jnp.eye(state_dim),
        ),
        observation_model=DiracIdentityObservation(),
    )

    dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)
