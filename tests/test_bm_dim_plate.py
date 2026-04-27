"""Minimal reproduction of bm_dim not being inferred inside plate context."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel


def _make_diffusion_spec(
    diffusion_form: str,
    *,
    state_dim: int,
    sigma=None,
):
    if diffusion_form == "full":
        return jnp.ones((state_dim, 1)), None, None
    if diffusion_form == "diag":
        return jnp.ones((state_dim,)), "diag", state_dim
    if diffusion_form == "scalar":
        return jnp.array(1.0), "scalar", state_dim
    if diffusion_form == "callable_full":
        return (
            (lambda x, u, t: sigma[..., None, None] * jnp.ones((state_dim, 1)))
            if sigma is not None
            else (lambda x, u, t: jnp.ones((state_dim, 1))),
            None,
            None,
        )
    if diffusion_form == "callable_diag":
        return (
            (lambda x, u, t: sigma[..., None] * jnp.ones((state_dim,)))
            if sigma is not None
            else (lambda x, u, t: jnp.ones((state_dim,))),
            "diag",
            state_dim,
        )
    if diffusion_form == "callable_scalar":
        return (
            (lambda x, u, t: sigma[..., None])
            if sigma is not None
            else (lambda x, u, t: jnp.array([1.0])),
            "scalar",
            state_dim,
        )
    raise ValueError(f"Unknown diffusion form: {diffusion_form}")


@pytest.mark.parametrize(
    "diffusion_form",
    ["full", "diag", "scalar", "callable_full", "callable_diag", "callable_scalar"],
)
def test_bm_dim_resolved_outside_plate(diffusion_form):
    """bm_dim is resolved correctly when not in a plate context."""
    state_dim = 2
    expected_bm_dim = 1 if "full" in diffusion_form else state_dim

    diffusion_coefficient, diffusion_type, bm_dim = _make_diffusion_spec(
        diffusion_form,
        state_dim=state_dim,
    )

    state_evo = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: -x,
        diffusion_coefficient=diffusion_coefficient,
        diffusion_type=diffusion_type,
        bm_dim=bm_dim,
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
    assert dynamics.state_evolution.bm_dim == expected_bm_dim, (
        f"Expected bm_dim={expected_bm_dim}, got {dynamics.state_evolution.bm_dim}"
    )


@pytest.mark.parametrize(
    "diffusion_form",
    ["full", "diag", "scalar", "callable_full", "callable_diag", "callable_scalar"],
)
def test_bm_dim_resolved_inside_plate(diffusion_form):
    """bm_dim should be resolved when model is constructed inside a plate context."""
    state_dim = 2
    expected_bm_dim = 1 if "full" in diffusion_form else state_dim
    M = 3

    def model():
        with numpyro.plate("trajectories", M):
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            diffusion_coefficient, diffusion_type, bm_dim = _make_diffusion_spec(
                diffusion_form,
                state_dim=state_dim,
                sigma=sigma,
            )

            state_evo = ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -x,
                diffusion_coefficient=diffusion_coefficient,
                diffusion_type=diffusion_type,
                bm_dim=bm_dim,
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
            assert dynamics.state_evolution.bm_dim == expected_bm_dim, (
                f"Expected bm_dim={expected_bm_dim}, got {dynamics.state_evolution.bm_dim}"
            )

    with numpyro.handlers.seed(rng_seed=0):
        model()


if __name__ == "__main__":
    for form in [
        "full",
        "diag",
        "scalar",
        "callable_full",
        "callable_diag",
        "callable_scalar",
    ]:
        print(f"Testing {form} outside plate...")
        test_bm_dim_resolved_outside_plate(form)
        print("  PASSED")
        print(f"Testing {form} inside plate...")
        test_bm_dim_resolved_inside_plate(form)
        print("  PASSED")
