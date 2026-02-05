import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.handlers import Condition, Discretizer
from dynestyx.observations import DiracIdentityObservation
from dynestyx.ops import Context, Trajectory
from dynestyx.simulators import DiscreteTimeSimulator, SDESimulator


def continuous_time_l63_dirac():
    """
    Simple continuous-time L63 model with DiracIdentityObservation (fully observed state).

    This is intentionally written in the natural continuous-time style; the combination
    of Discretizer + DiscreteTimeSimulator + DiracIdentityObservation in the test below
    is expected to expose current broadcasting issues in the discretizer/simulator
    interaction (it is okay for the test to fail for now).
    """
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=3,
        control_dim=0,
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
        observation_model=DiracIdentityObservation(),
    )

    dsx.sample_ds("f", dynamics)


@pytest.mark.skipif(
    True,
    reason=(
        "Expected to fail currently: exposes broadcasting interaction between "
        "Discretizer, DiracIdentityObservation, and DiscreteTimeSimulator."
    ),
)
def test_discretizer_dirac_broadcast_mcmc_smoke():
    """
    Minimal MCMC smoke test that combines:

    - Continuous-time L63 with DiracIdentityObservation (fully observed state),
    - SDESimulator to generate synthetic data,
    - Discretizer + DiscreteTimeSimulator + Condition for inference,
    - NUTS MCMC over rho.

    The current implementation is expected to hit a broadcasting error inside the
    discretizer / DiscreteTimeSimulator Dirac branch. This test documents that
    failure mode (and can be un-skipped once the underlying issue is fixed).
    """
    # Generate synthetic data
    true_rho = 28.0
    obs_times = jnp.arange(start=0.0, stop=2.0, step=0.01)
    rng_key = jr.PRNGKey(0)
    data_init_key, data_solver_key, mcmc_key = jr.split(rng_key, 3)

    def model_dirac():
        return continuous_time_l63_dirac()

    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        model_dirac,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(observations=Trajectory(times=obs_times))
    with SDESimulator(key=data_solver_key):
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # (T, 3)
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    # Data-conditioned model using Discretizer + DiscreteTimeSimulator
    def data_conditioned_model():
        ctx = Context(observations=observation_trajectory)
        with DiscreteTimeSimulator():
            with Discretizer():
                with Condition(ctx):
                    return continuous_time_l63_dirac()

    # Small MCMC run to trigger the model / simulator logic
    mcmc = MCMC(NUTS(data_conditioned_model), num_samples=10, num_warmup=10)
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples
