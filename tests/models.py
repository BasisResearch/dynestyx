import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution
from dsx.observations import LinearGaussianObservation
from dsx.ops import sample_ds


def hmm_model():
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
    def state_evolution(x, u, t):
        # x is an integer in {0, ..., K-1}
        return dist.Categorical(probs=A[x])

    # -------------------------------------------------
    # Observation model
    # -------------------------------------------------
    def observation_model(x, u, t):
        # y_t | x_t ~ N(mu[x_t], sigma^2)
        return dist.Normal(mu[x], sigma)

    dynamics = DynamicalModel(
        state_dim=K,
        observation_dim=1,
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    return sample_ds("f", dynamics)


def discrete_time_l63_model():
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

    def state_evolution(x, u, t):
        loc = x + 0.01 * drift(x)
        cov = 0.01 * jnp.eye(3)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
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
    return sample_ds("f", dynamics)


def continuous_time_stochastic_l63_model():
    """Model that samples drift parameter rho and uses it in dynamics."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=20.0**2 * jnp.eye(3)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.array(
                [
                    10.0 * (x[1] - x[0]),
                    x[0] * (rho - x[2]) - x[1],
                    x[0] * x[1] - (8.0 / 3.0) * x[2],
                ]
            ),
            diffusion_coefficient=lambda x, u, t: jnp.eye(3),
            diffusion_covariance=lambda x, u, t: jnp.eye(3),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[5.0**2]])
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
    return sample_ds("f", dynamics)


def continuous_time_lingam_model():
    """2D linear SDE with a sampled coupling."""
    rho = numpyro.sample("rho", dist.Uniform(0.0, 5.0))

    A = jnp.array([[-1.0, 0.0], [rho, -1.0]])

    dynamics = DynamicalModel(
        state_dim=2,
        observation_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=1.0**2 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: A @ x,
            diffusion_coefficient=lambda x, u, t: jnp.eye(2),
            diffusion_covariance=lambda x, u, t: jnp.eye(2),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[0.0, 1.0]]), R=jnp.array([[1.0**2]])
        ),
    )
    return sample_ds("f", dynamics)


def continuous_time_deterministic_l63_model():
    """Model that samples drift parameter rho and uses it in dynamics (ODE, no diffusion)."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=2.0**2 * jnp.eye(3)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.array(
                [
                    10.0 * (x[1] - x[0]),
                    x[0] * (rho - x[2]) - x[1],
                    x[0] * x[1] - (8.0 / 3.0) * x[2],
                ]
            ),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[1.0**2]])
        ),
    )

    # Return a sampled dynamical model, named "f".
    return sample_ds("f", dynamics)
