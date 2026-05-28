import equinox as eqx
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianObservation,
    GaussianStateEvolution,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from dynestyx.models.state_evolution import AffineDrift

DISCRETE_A = jnp.array([[0.72, 0.08], [0.0, 0.65]])
DISCRETE_Q = jnp.array([[0.05, 0.01], [0.01, 0.04]])
GAUSSIAN_R = jnp.array([[0.45, 0.12], [0.12, 0.35]])
INDEPENDENT_SCALE = jnp.array([0.45, 0.65])
ODE_A = jnp.array([[-0.4, 0.2], [-0.1, -0.55]])


def _nonlinear_observation_mean(x, u, t):
    return jnp.array(
        [
            x[0] + 0.15 * x[1] ** 2,
            x[1] - 0.1 * jnp.sin(t),
        ]
    )


def _independent_observation_mean(x, u, t):
    return jnp.array([x[0] + 0.25 * x[1], x[1] - 0.1 * t])


def discrete_linear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def discrete_nonlinear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=GaussianObservation(
            h=_nonlinear_observation_mean, R=GAUSSIAN_R
        ),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def discrete_independent_normal_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=lambda x, u, t: dist.Independent(
            dist.Normal(_independent_observation_mean(x, u, t), INDEPENDENT_SCALE), 1
        ),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def sampled_discrete_linear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(0.72, 0.05))
    A = DISCRETE_A.at[0, 0].set(alpha)
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=A, cov=DISCRETE_Q),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def ode_linear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: ODE_A @ x),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def ode_nonlinear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: ODE_A @ x),
        observation_model=GaussianObservation(
            h=_nonlinear_observation_mean, R=GAUSSIAN_R
        ),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def ode_independent_normal_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: ODE_A @ x),
        observation_model=lambda x, u, t: dist.Independent(
            dist.Normal(_independent_observation_mean(x, u, t), INDEPENDENT_SCALE), 1
        ),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def sampled_ode_linear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(0.2, 0.05))
    A = ODE_A.at[0, 1].set(alpha)
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: A @ x),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


class _PlateLinearTransition(eqx.Module):
    A: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        return x @ jnp.swapaxes(self.A, -1, -2)


def plated_discrete_linear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    with dsx.plate("trajectories", M):
        alpha = jnp.linspace(0.65, 0.78, M)
        A = jnp.broadcast_to(DISCRETE_A, (M, 2, 2)).copy().at[:, 0, 0].set(alpha)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
            ),
            state_evolution=GaussianStateEvolution(
                F=_PlateLinearTransition(A=A), cov=DISCRETE_Q
            ),
            observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def plated_ode_linear_gaussian_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    with dsx.plate("trajectories", M):
        alpha = jnp.linspace(0.15, 0.3, M)
        A = jnp.broadcast_to(ODE_A, (M, 2, 2)).copy().at[:, 0, 1].set(alpha)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
            ),
            state_evolution=ContinuousTimeStateEvolution(drift=AffineDrift(A=A)),
            observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )
