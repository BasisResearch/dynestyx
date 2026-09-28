"""Local affine drift semantics and JAX transformations."""

import jax
import jax.numpy as jnp

import dynestyx as dsx


def test_linearize_drift_matches_tangent_at_fixed_control_and_time():
    def drift(x, u, t):
        return jnp.array([u[0] * x[0] ** 2 + jnp.sin(x[1]) + t, x[0] * x[1]])

    state = jnp.array([0.3, -0.6])
    control = jnp.array([0.4])
    time = 0.2
    local = dsx.linearize_drift(drift, x=state, u=control, t=time)
    jacobian = jnp.array(
        [[2 * control[0] * state[0], jnp.cos(state[1])], [state[1], state[0]]]
    )
    displacement = jnp.array([0.5, -0.2])

    assert isinstance(local, dsx.AffineDrift)
    assert local.B is None
    assert jnp.allclose(local.A, jacobian)
    assert jnp.allclose(local(state, None, 3.0), drift(state, control, time))
    assert jnp.allclose(
        local(state + displacement, jnp.array([9.0]), 7.0),
        drift(state, control, time) + jacobian @ displacement,
    )


def test_linearize_drift_supports_jit_vmap_and_grad():
    def local(state, control):
        return dsx.linearize_drift(
            lambda x, u, t: u[0] * x**3, x=state, u=control, t=0.0
        )

    states = jnp.array([[0.3, -0.6], [-0.2, 0.7]])
    controls = jnp.array([[0.4], [-0.1]])
    drifts = jax.jit(jax.vmap(local))(states, controls)

    assert drifts.B is None
    assert jnp.allclose(drifts.A, jax.vmap(jnp.diag)(3 * controls * states**2))
    assert jnp.allclose(drifts.b, -2 * controls * states**3)

    def objective(state, control):
        drift = local(state, control)
        return jnp.sum(drift.A) + jnp.sum(drift.b)

    state_gradient, control_gradient = jax.grad(objective, argnums=(0, 1))(
        states[0], controls[0]
    )
    assert jnp.allclose(state_gradient, 6 * controls[0] * (states[0] - states[0] ** 2))
    assert jnp.allclose(
        control_gradient, jnp.sum(3 * states[0] ** 2 - 2 * states[0] ** 3)
    )
