"""Minimal reproduction of bm_dim not being inferred inside plate context."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel


def test_bm_dim_inferred_outside_plate():
    """bm_dim is correctly inferred when not in a plate context."""
    state_dim = 2
    bm_dim = 1

    state_evo = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: -x,
        diffusion_coefficient=lambda x, u, t: jnp.ones((state_dim, bm_dim)),
    )
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.zeros(state_dim), jnp.eye(state_dim)
        ),
        state_evolution=state_evo,
        observation_model=lambda x, u, t: dist.MultivariateNormal(
            x, 0.1 * jnp.eye(state_dim)
        ),
    )
    assert dynamics.state_evolution.bm_dim == bm_dim, (
        f"Expected bm_dim={bm_dim}, got {dynamics.state_evolution.bm_dim}"
    )


def test_bm_dim_inferred_inside_plate():
    """bm_dim should be inferred when model is constructed inside a plate context."""
    state_dim = 2
    bm_dim = 1
    M = 3

    def model():
        with numpyro.plate("trajectories", M):
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

            state_evo = ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -x,
                diffusion_coefficient=lambda x, u, t: (
                    sigma[..., None, None] * jnp.ones((state_dim, bm_dim))
                ),
            )
            dynamics = DynamicalModel(
                initial_condition=dist.MultivariateNormal(
                    jnp.zeros(state_dim), jnp.eye(state_dim)
                ),
                state_evolution=state_evo,
                observation_model=lambda x, u, t: dist.MultivariateNormal(
                    x, 0.1 * jnp.eye(state_dim)
                ),
            )
            # This is the bug: bm_dim stays None inside plate context
            assert dynamics.state_evolution.bm_dim is not None, (
                "bm_dim should not be None after DynamicalModel construction in plate"
            )
            assert dynamics.state_evolution.bm_dim == bm_dim, (
                f"Expected bm_dim={bm_dim}, got {dynamics.state_evolution.bm_dim}"
            )

    # Run the model with seed to trigger numpyro.sample
    with numpyro.handlers.seed(rng_seed=0):
        model()


def test_bm_dim_inferred_inside_plate_unbatched_diffusion():
    """bm_dim should be inferred even when diffusion doesn't use batched params."""
    state_dim = 2
    bm_dim = 1
    M = 3

    def model():
        with numpyro.plate("trajectories", M):
            _ = numpyro.sample("sigma", dist.HalfNormal(1.0))

            state_evo = ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -x,
                diffusion_coefficient=lambda x, u, t: jnp.ones((state_dim, bm_dim)),
            )
            dynamics = DynamicalModel(
                initial_condition=dist.MultivariateNormal(
                    jnp.zeros(state_dim), jnp.eye(state_dim)
                ),
                state_evolution=state_evo,
                observation_model=lambda x, u, t: dist.MultivariateNormal(
                    x, 0.1 * jnp.eye(state_dim)
                ),
            )
            assert dynamics.state_evolution.bm_dim is not None, (
                "bm_dim should not be None after DynamicalModel construction in plate"
            )
            assert dynamics.state_evolution.bm_dim == bm_dim, (
                f"Expected bm_dim={bm_dim}, got {dynamics.state_evolution.bm_dim}"
            )

    with numpyro.handlers.seed(rng_seed=0):
        model()


if __name__ == "__main__":
    print("Test 1: bm_dim outside plate...")
    test_bm_dim_inferred_outside_plate()
    print("  PASSED")

    print("Test 2: bm_dim inside plate (unbatched diffusion)...")
    try:
        test_bm_dim_inferred_inside_plate_unbatched_diffusion()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("Test 3: bm_dim inside plate (batched diffusion)...")
    try:
        test_bm_dim_inferred_inside_plate()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
