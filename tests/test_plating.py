"""Tests for plate functionality and batching."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist

from dynestyx.handlers import Condition, plate
from dynestyx.ops import Context, Trajectory
from dynestyx.simulators import SDESimulator
from dynestyx.utils import infer_batch_shape


class LinearDrift(eqx.Module):
    """Example eqx.Module drift that can be vmapped over its parameters."""

    A: jnp.ndarray

    def __call__(self, x, u, t):
        # Intentionally written in the "natural" unbatched style.
        # If `A` is batched, then `jax.vmap` over this module (passed as an argument)
        # will slice `A` per batch element automatically.
        return self.A @ x


def test_batch_shape_inference():
    """Test that batch shapes are correctly inferred from plate stack."""
    # numpyro.clear_param_store()

    def model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        assert infer_batch_shape() is None

        with plate("plate", 10, dim=-2):
            batch_shape = infer_batch_shape()
            assert batch_shape == (10, 1)

            with plate("plate_2", 100, dim=-1):
                batch_shape = infer_batch_shape()
                assert batch_shape == (10, 100)

                y = numpyro.sample("y", dist.Normal(x, 1))

        assert infer_batch_shape() is None
        return y

    numpyro.handlers.seed(model, 0)()


def test_eqx_module_drift_can_vmap_over_parameters():
    """
    Demonstrates the pattern needed for "automatic batching over parameters":
    store parameters as `eqx.Module` leaves and pass the module as an explicit argument.
    """

    def model():
        with plate("batch", 10, dim=-1):
            rho = numpyro.sample("rho", dist.Normal(0.0, 1.0))

        A0 = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        Ar = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        A = A0 + rho[..., None, None] * Ar  # (10, 2, 2)

        drift = LinearDrift(A=A)  # PyTree with batched leaf

        x = jnp.zeros((10, 2))
        dx = jax.vmap(lambda d, x_i: d(x_i, None, 0.0))(drift, x)

        assert dx.shape == (10, 2)
        return dx

    numpyro.handlers.seed(model, 0)()

    # Now: run a real dsx simulation where the *model is constructed under the plate*
    # and the drift parameters live inside an `eqx.Module`.
    from numpyro.infer import Predictive

    import dynestyx as dsx
    from dynestyx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel
    from dynestyx.observations import LinearGaussianObservation

    # Use strictly increasing times to satisfy diffrax.
    times = jnp.linspace(0.0, 0.2, 5)
    context = Context(observations=Trajectory(times=times, values=None))

    def simulate_model():
        with plate("batch", 10, dim=-1):
            rho = numpyro.sample("rho", dist.Normal(0.0, 1.0))

            A0 = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
            Ar = jnp.array([[0.0, 0.0], [1.0, 0.0]])
            A = A0 + rho[..., None, None] * Ar  # (10, 2, 2)

            dynamics = DynamicalModel(
                state_dim=2,
                observation_dim=1,
                initial_condition=dist.MultivariateNormal(
                    loc=jnp.zeros(2), covariance_matrix=1.0**2 * jnp.eye(2)
                ),
                state_evolution=ContinuousTimeStateEvolution(
                    drift=LinearDrift(A=A),
                    diffusion_coefficient=lambda x, u, t: jnp.eye(2),
                    diffusion_covariance=lambda x, u, t: jnp.eye(2),
                ),
                observation_model=LinearGaussianObservation(
                    H=jnp.array([[1.0, 0.0]]), R=jnp.array([[0.1**2]])
                ),
            )

            return dsx.sample_ds("f", dynamics)

    predictive_model = Predictive(
        simulate_model, num_samples=1, exclude_deterministic=False
    )
    key = jr.PRNGKey(0)
    sim_key, pred_key = jr.split(key)
    with SDESimulator(key=sim_key):
        with Condition(context=context):
            out = predictive_model(pred_key)

    assert "states" in out and "observations" in out and "times" in out

    # import matplotlib.pyplot as plt

    # for i in range(out["states"].shape[2]):
    #     plt.plot(out["times"][0], out["states"][0, :, i, 0], label="x[0]")
    #     plt.plot(out["times"][0], out["states"][0, :, i, 1], label="x[1]")
    # plt.legend()
    # plt.show()


def test_vmap_context_trajectory():
    """Test vmap with Context and Trajectory."""

    def f(ctx: Context):
        return ctx.observations.values.sum()  # type: ignore

    batched_f = jax.vmap(
        f,
        in_axes=(
            Context(
                solve=None,
                observations=Trajectory(times=0, values=0),
                controls=None,
                extras=None,
            ),
        ),
    )

    ctx = Context(
        observations=Trajectory(times=jnp.zeros((2, 10)), values=jnp.ones((2, 10)))
    )

    out = batched_f(ctx)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([10.0, 10.0]))


def test_vmapped_add_solved_sites():
    """Test vmapped add_solved_sites."""
    from dynestyx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel
    from dynestyx.observations import LinearGaussianObservation

    simulator = SDESimulator(key=jr.PRNGKey(0))

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
                    x[0] * (28.0 - x[2]) - x[1],
                    x[0] * x[1] - (8.0 / 3.0) * x[2],
                ]
            )
            + (10 * u if u is not None else jnp.zeros(3)),
            diffusion_coefficient=lambda x, u, t: jnp.eye(3),
            diffusion_covariance=lambda x, u, t: jnp.eye(3),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[5.0**2]])
        ),
    )

    vmapped_add_solved_sites = jax.vmap(
        simulator.add_solved_sites,
        in_axes=(
            None,
            None,
            Context(
                solve=None,
                observations=Trajectory(times=0, values=None),
                controls=None,
                extras=None,
            ),
        ),
    )

    batched_times = jnp.array(
        [
            jnp.arange(start=0.0, stop=10.0, step=0.05),
            jnp.arange(start=0.0, stop=10.0, step=0.05),
        ]
    )
    batched_ctx = Context(observations=Trajectory(times=batched_times))
    vmapped_add_solved_sites("f", dynamics, batched_ctx)


def make_lti_gaussian_model(rho=None):
    """Helper to create LTI Gaussian model with optional rho parameter."""
    A0 = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    Ar = jnp.array([[0.0, 0.0], [1.0, 0.0]])

    A = A0 + rho[..., None, None] * Ar

    from dynestyx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel
    from dynestyx.observations import LinearGaussianObservation

    dynamics = DynamicalModel(
        state_dim=2,
        observation_dim=1,
        control_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=1.0**2 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            # Make drift explicitly batch-aware:
            # - if rho is scalar -> A is (2,2) and x is (2,) => returns (2,)
            # - if rho is batched -> A is (...,2,2) and x is (...,2) => returns (...,2)
            drift=lambda x, u, t: jnp.einsum("...ij,...j->...i", A, x),
            diffusion_coefficient=lambda x, u, t: jnp.eye(2),
            diffusion_covariance=lambda x, u, t: jnp.eye(2),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[0.0, 1.0]]), R=jnp.array([[0.15**2]])
        ),
    )

    return dynamics


def test_hierarchical_lti_gaussian_model():
    """Test hierarchical LTI Gaussian model with plates."""
    import dynestyx as dsx

    def hierarchical_model():
        global_mean = numpyro.sample("global_mean", dist.Normal(0.0, 1.0))
        global_scale = numpyro.sample("global_scale", dist.Uniform(0.5, 2.0))

        with plate("batch", 10, dim=-1):
            rho = numpyro.sample("rho", dist.Normal(global_mean, global_scale))

            dynamics = make_lti_gaussian_model(rho)
            out = dsx.sample_ds("f", dynamics)

        return out

    # Test that the model can be called (even if it may fail during simulation)
    # numpyro.clear_param_store()
    try:
        numpyro.handlers.seed(hierarchical_model, 0)()
    except ValueError as e:
        # Expected to fail with broadcasting error when plates interact with SDE solver
        # This documents the current limitation
        assert "broadcast" in str(e).lower() or "compatible" in str(e).lower()


def test_hierarchical_lti_gaussian_predictive():
    """Test hierarchical model with Predictive (currently expected to fail)."""
    from numpyro.infer import Predictive

    import dynestyx as dsx

    def hierarchical_model():
        global_mean = numpyro.sample("global_mean", dist.Normal(0.0, 1.0))
        global_scale = numpyro.sample("global_scale", dist.Uniform(0.5, 2.0))

        with plate("batch", 10, dim=-1):
            rho = numpyro.sample("rho", dist.Normal(global_mean, global_scale))

            dynamics = make_lti_gaussian_model(rho)
            out = dsx.sample_ds("f", dynamics)

        return out

    obs_times = jnp.array(
        [[jnp.arange(start=0.0, stop=20.0, step=0.01) for _ in range(10)]]
    )
    context = Context(observations=Trajectory(times=obs_times))

    prng_key = jr.PRNGKey(0)
    sde_solver_key, predictive_key = jr.split(prng_key, 2)

    predictive_model = Predictive(hierarchical_model, num_samples=1)

    with SDESimulator(key=sde_solver_key):
        with Condition(context=context):
            predictive_model(predictive_key)


# if __name__ == "__main__":
#     test_batch_shape_inference()
#     test_vmap_context_trajectory()
#     test_vmapped_add_solved_sites()
#     test_hierarchical_lti_gaussian_model()
#     test_hierarchical_lti_gaussian_predictive()
