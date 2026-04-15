import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

from dynestyx.models.lti_dynamics import LTI_continuous
from dynestyx.solvers import solve_sde


@pytest.mark.parametrize(
    ("source", "diffeqsolve_settings"),
    [
        (
            "diffrax",
            {
                "solver": dfx.Heun(),
                "stepsize_controller": dfx.ConstantStepSize(),
                "adjoint": dfx.RecursiveCheckpointAdjoint(),
                "dt0": 0.1,
                "max_steps": None,
            },
        ),
        ("em_scan", {"dt0": 0.1}),
    ],
)
def test_sde_solver_early_return_with_key(source, diffeqsolve_settings):
    """No-op horizons should return repeated x0."""
    dynamics = LTI_continuous(
        A=jnp.array([[0.0, 0.1], [0.0, -0.2]]),
        L=jnp.array([[0.2], [0.1]]),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.1]]),
    )
    x0 = jnp.array([1.25, -0.5])
    t0 = 1.0
    saveat_times = jnp.array([0.3, 0.6, 1.0])

    states = solve_sde(
        source=source,
        dynamics=dynamics,
        t0=t0,
        saveat_times=saveat_times,
        x0=x0,
        control_path_eval=lambda t: None,
        diffeqsolve_settings=diffeqsolve_settings,
        key=jr.PRNGKey(0),
    )

    expected = jnp.broadcast_to(x0, (len(saveat_times), x0.shape[0]))
    assert states.shape == expected.shape
    assert jnp.allclose(states, expected)


def test_sde_solver_em_scan_accepts_jax_scalar_dt0():
    """em_scan should accept scalar JAX numeric values for dt0."""
    dynamics = LTI_continuous(
        A=jnp.array([[0.0, 0.1], [0.0, -0.2]]),
        L=jnp.array([[0.2], [0.1]]),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.1]]),
    )
    states = solve_sde(
        source="em_scan",
        dynamics=dynamics,
        t0=0.0,
        saveat_times=jnp.array([0.1, 0.2, 0.3]),
        x0=jnp.array([0.0, 0.0]),
        control_path_eval=lambda t: None,
        diffeqsolve_settings={"dt0": jnp.asarray(0.05, dtype=jnp.float32)},
        key=jr.PRNGKey(0),
    )
    assert states.shape == (3, 2)
