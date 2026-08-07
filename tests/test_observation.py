"""Tests for GridInterpolator and interpolation_observation_model."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from dynestyx.models import DynamicalModel
from dynestyx.observation import GridInterpolator, interpolation_observation_model

# --- 1D fixtures -----------------------------------------------------------

NUM_POINTS = 8
DOMAIN_EXTENT = 8.0  # unit spacing, dx = 1.0, for easy hand-checked arithmetic
X_GRID = jnp.linspace(0, DOMAIN_EXTENT, NUM_POINTS, endpoint=False)


def _values_1d():
    # a smooth-ish, non-constant field so interpolation is a nontrivial check
    return jnp.sin(2 * jnp.pi * X_GRID / DOMAIN_EXTENT) + 0.5 * X_GRID


def _periodic_linear_interp_reference(x_query, u_values):
    """Independent reference implementation (mirrors the notebook's hand-rolled version)."""
    dx = X_GRID[1] - X_GRID[0]
    idx_float = (x_query % DOMAIN_EXTENT) / dx
    idx0 = jnp.floor(idx_float).astype(jnp.int32) % NUM_POINTS
    idx1 = (idx0 + 1) % NUM_POINTS
    frac = idx_float - jnp.floor(idx_float)
    return (1 - frac) * u_values[idx0] + frac * u_values[idx1]


# --- basic correctness -------------------------------------------------


@pytest.mark.parametrize("boundary", ["periodic", "edge", "constant"])
def test_linear_exact_at_grid_points_1d(boundary):
    values = _values_1d()
    query = X_GRID[:, None]  # exactly at every grid point
    interp = GridInterpolator((X_GRID,), query, method="linear", boundary=boundary)
    assert jnp.allclose(interp(values), values, atol=1e-6)


def test_nearest_exact_at_grid_points_1d():
    values = _values_1d()
    query = X_GRID[:, None]
    interp = GridInterpolator((X_GRID,), query, method="nearest", boundary="periodic")
    assert jnp.allclose(interp(values), values, atol=1e-6)


def test_nearest_picks_closest_point():
    values = jnp.arange(NUM_POINTS, dtype=jnp.float32)
    # query points offset by 0.3*dx from grid point 3 -> should round down to 3
    query = jnp.array([[X_GRID[3] + 0.3]])
    interp = GridInterpolator((X_GRID,), query, method="nearest", boundary="periodic")
    assert jnp.allclose(interp(values), jnp.array([3.0]))
    # offset by 0.7*dx -> should round up to 4
    query = jnp.array([[X_GRID[3] + 0.7]])
    interp = GridInterpolator((X_GRID,), query, method="nearest", boundary="periodic")
    assert jnp.allclose(interp(values), jnp.array([4.0]))


def test_linear_matches_manual_periodic_reference():
    values = _values_1d()
    key_points = jnp.array([0.0, 1.75, 3.4, 6.999, 7.999, -0.5, 8.5])[:, None]
    interp = GridInterpolator(
        (X_GRID,), key_points, method="linear", boundary="periodic"
    )
    expected = _periodic_linear_interp_reference(key_points[:, 0], values)
    assert jnp.allclose(interp(values), expected, atol=1e-6)


def test_linear_partition_of_unity_for_bias_free_boundaries():
    query = jnp.array([0.3, 2.6, 5.1, 7.9, -1.2, 9.4])[:, None]
    for boundary in ("periodic", "edge"):
        interp = GridInterpolator((X_GRID,), query, method="linear", boundary=boundary)
        H = interp.as_matrix()
        assert jnp.allclose(jnp.sum(H, axis=1), 1.0, atol=1e-6)
        assert jnp.allclose(interp.bias(), 0.0)


# --- boundary modes ------------------------------------------------------


def test_edge_boundary_clamps_beyond_domain():
    values = _values_1d()
    # far outside the domain in both directions
    query = jnp.array([[-5.0], [50.0]])
    interp = GridInterpolator((X_GRID,), query, method="linear", boundary="edge")
    expected = jnp.array([values[0], values[-1]])
    assert jnp.allclose(interp(values), expected, atol=1e-6)


def test_constant_boundary_far_outside_returns_fill_value():
    values = _values_1d()
    query = jnp.array([[-50.0], [500.0]])
    interp = GridInterpolator(
        (X_GRID,), query, method="linear", boundary="constant", fill_value=3.25
    )
    assert jnp.allclose(interp(values), jnp.array([3.25, 3.25]))
    # every corner is invalid this far out, so as_matrix()'s row is all zero and bias
    # alone carries the value
    assert jnp.allclose(interp.as_matrix(), 0.0)
    assert jnp.allclose(interp.bias(), jnp.array([3.25, 3.25]))


def test_constant_boundary_partial_straddle_mixes_grid_value_and_fill():
    values = _values_1d()
    dx = X_GRID[1] - X_GRID[0]
    # query point half a grid cell past the last grid point: one bracketing corner
    # (the last grid point) is valid, the other (one past it) is not.
    query = jnp.array([[X_GRID[-1] + 0.5 * dx]])
    fill_value = 10.0
    interp = GridInterpolator(
        (X_GRID,), query, method="linear", boundary="constant", fill_value=fill_value
    )
    expected = 0.5 * values[-1] + 0.5 * fill_value
    assert jnp.allclose(interp(values), expected, atol=1e-6)


def test_call_matches_matrix_and_bias_for_every_boundary():
    values = _values_1d()
    query = jnp.array([0.3, 2.6, 5.1, 7.9, -1.2, 9.4])[:, None]
    for boundary in ("periodic", "edge", "constant"):
        interp = GridInterpolator(
            (X_GRID,), query, method="linear", boundary=boundary, fill_value=-1.0
        )
        reconstructed = interp.as_matrix() @ values + interp.bias()
        assert jnp.allclose(interp(values), reconstructed, atol=1e-6)


# --- sparse export ---------------------------------------------------------


def test_as_sparse_matches_as_matrix():
    query = jnp.array([0.3, 2.6, 5.1, 7.9])[:, None]
    interp = GridInterpolator((X_GRID,), query, method="linear", boundary="periodic")
    assert jnp.allclose(interp.as_sparse().todense(), interp.as_matrix(), atol=1e-6)


# --- N-D generality ---------------------------------------------------------


def test_2d_bilinear_exact_on_affine_function():
    nx, ny = 6, 5
    x_axis = jnp.linspace(0.0, 5.0, nx)
    y_axis = jnp.linspace(0.0, 4.0, ny)

    def f(x, y):
        return 2.0 * x - 3.0 * y + 1.5

    xs, ys = jnp.meshgrid(x_axis, y_axis, indexing="ij")
    values = f(xs, ys).reshape(-1)

    query = jnp.array([[0.5, 0.5], [1.3, 2.7], [4.9, 3.9], [0.0, 0.0]])
    # non-periodic domain (no wraparound for an affine test function): use "edge" so
    # in-range queries are unaffected by boundary handling.
    interp = GridInterpolator((x_axis, y_axis), query, method="linear", boundary="edge")
    expected = f(query[:, 0], query[:, 1])
    assert jnp.allclose(interp(values), expected, atol=1e-5)


def test_2d_state_dim_and_matrix_shape():
    nx, ny = 4, 3
    x_axis = jnp.linspace(0.0, 3.0, nx)
    y_axis = jnp.linspace(0.0, 2.0, ny)
    query = jnp.array([[0.5, 0.5], [1.5, 1.0]])
    interp = GridInterpolator((x_axis, y_axis), query, method="linear", boundary="edge")
    assert interp.state_dim == nx * ny
    assert interp.as_matrix().shape == (2, nx * ny)


# --- validation --------------------------------------------------------


def test_invalid_method_raises():
    # dynestyx runs jaxtyping+typeguard checks package-wide (see pyproject.toml), so an
    # out-of-Literal value is rejected as a jaxtyping.TypeCheckError before __init__'s own
    # `if method not in _VALID_METHODS` check would run; either way it must raise.
    with pytest.raises(Exception):
        GridInterpolator((X_GRID,), X_GRID[:1, None], method="cubic")  # type: ignore[arg-type]


def test_invalid_boundary_raises():
    with pytest.raises(Exception):
        GridInterpolator((X_GRID,), X_GRID[:1, None], boundary="reflect")  # type: ignore[arg-type]


def test_wrong_query_points_shape_raises():
    with pytest.raises(Exception):
        GridInterpolator((X_GRID,), X_GRID)  # missing the trailing ndim axis


# --- integration with LinearGaussianObservation / DynamicalModel -----------


def test_interpolation_observation_model_builds_valid_dynamical_model():
    values = _values_1d()
    query = jnp.array([0.3, 2.6, 5.1])[:, None]
    obs_model = interpolation_observation_model(
        (X_GRID,), query, R=0.01 * jnp.eye(3), method="linear", boundary="periodic"
    )

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(NUM_POINTS), covariance_matrix=jnp.eye(NUM_POINTS)
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=x, covariance_matrix=jnp.eye(NUM_POINTS)
        ),
        observation_model=obs_model,
        control_dim=0,
    )

    assert dynamics.state_dim == NUM_POINTS
    assert dynamics.observation_dim == 3

    obs_dist = obs_model(values, None, jnp.array(0.0))
    interp = GridInterpolator((X_GRID,), query, method="linear", boundary="periodic")
    assert jnp.allclose(obs_dist.mean, interp(values), atol=1e-6)


def test_interpolation_observation_model_with_constant_boundary_carries_bias():
    values = _values_1d()
    query = jnp.array([[-50.0]])
    obs_model = interpolation_observation_model(
        (X_GRID,),
        query,
        R=jnp.array([[0.01]]),
        method="linear",
        boundary="constant",
        fill_value=7.0,
    )
    obs_dist = obs_model(values, None, jnp.array(0.0))
    assert jnp.allclose(obs_dist.mean, jnp.array([7.0]))
