"""Grid-interpolation utilities for building observation operators.

`GridInterpolator` observes a state that lives on a regular N-D grid at arbitrary query
points -- not necessarily grid-aligned -- via nearest-neighbor or (multilinear)
piecewise-linear interpolation. Both interpolation kernels are linear in the grid values,
so the whole operation reduces to a fixed matrix applied to the flattened state; this module
computes that matrix (and, for the ``"constant"`` boundary mode, an additive bias) directly
from the interpolation weights, rather than recovering it after the fact via autodiff.

The intended pairing is `dynestyx.models.observations.LinearGaussianObservation`:

```python
interp = GridInterpolator((x_grid,), query_points, method="linear", boundary="periodic")
observation_model = LinearGaussianObservation(H=interp.as_matrix(), R=obs_noise_cov)
```

or, more directly, `interpolation_observation_model(...)` below builds that
`LinearGaussianObservation` in one call.
"""

from __future__ import annotations

import itertools
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse as jax_sparse
from jaxtyping import Bool, Float, Int, Real

from dynestyx.models.observations import LinearGaussianObservation

BoundaryMode = Literal["periodic", "constant", "edge"]
InterpolationMethod = Literal["nearest", "linear"]

_VALID_METHODS = ("nearest", "linear")
_VALID_BOUNDARIES = ("periodic", "constant", "edge")


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """C-order (row-major) strides for flattening a `shape`-shaped array."""
    strides = []
    acc = 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= size
    return tuple(reversed(strides))


def _axis_candidates(
    q: Float[Array, " n_query"],
    x_axis: Real[Array, " _"],
    method: InterpolationMethod,
    boundary: BoundaryMode,
) -> tuple[
    Int[Array, "n_query k"], Float[Array, "n_query k"], Bool[Array, "n_query k"]
]:
    """Per-axis candidate grid indices, weights, and validity for one query coordinate.

    `k` is `1` for `method="nearest"` and `2` for `method="linear"` (the two grid points
    bracketing the query coordinate), kept uniform across axes -- including degenerate
    single-point axes, where the second `"linear"` candidate is a zero-weight duplicate --
    so that axes can be combined by a plain outer product later.
    """
    n = x_axis.shape[0]
    k = 1 if method == "nearest" else 2

    if n == 1:
        idx0 = jnp.zeros_like(q, dtype=jnp.int32)
        idxs = jnp.stack([idx0] * k, axis=-1)
        if method == "nearest":
            ws = jnp.ones_like(q)[:, None]
        else:
            ws = jnp.stack([jnp.ones_like(q), jnp.zeros_like(q)], axis=-1)
        valid = jnp.ones((q.shape[0], k), dtype=bool)
        return idxs, ws, valid

    x0 = x_axis[0]
    dx = x_axis[1] - x_axis[0]
    extent = n * dx  # assumes x_axis is uniform and, for boundary="periodic", does not
    # repeat the periodic image of the first point (e.g. jnp.linspace(..., endpoint=False)).

    if boundary == "periodic":
        q_adj = (q - x0) % extent + x0
    elif boundary == "edge":
        q_adj = jnp.clip(q, x0, x_axis[-1])
    else:  # "constant"
        q_adj = q

    idx_float = (q_adj - x0) / dx

    if method == "nearest":
        idxs = jnp.round(idx_float).astype(jnp.int32)[:, None]
        ws = jnp.ones_like(idx_float)[:, None]
    else:
        idx0 = jnp.floor(idx_float).astype(jnp.int32)
        frac = idx_float - idx0
        idxs = jnp.stack([idx0, idx0 + 1], axis=-1)
        ws = jnp.stack([1.0 - frac, frac], axis=-1)

    if boundary == "periodic":
        idxs_final = idxs % n
        valid = jnp.ones_like(idxs, dtype=bool)
    elif boundary == "edge":
        idxs_final = jnp.clip(idxs, 0, n - 1)
        valid = jnp.ones_like(idxs, dtype=bool)
    else:  # "constant"
        valid = (idxs >= 0) & (idxs < n)
        idxs_final = jnp.clip(idxs, 0, n - 1)  # safe for gather; masked out via `valid`

    return idxs_final, ws, valid


class GridInterpolator(eqx.Module):
    r"""Linear observation operator for a regular N-D grid, observed at arbitrary points.

    Precomputes, once at construction, the sparse "corner" structure of either
    nearest-neighbor or (multilinear) piecewise-linear interpolation: for each query point,
    the up-to-$2^d$ bracketing grid points and their weights. Both kernels are linear in the
    grid values, so `as_matrix()`/`as_sparse()` expose that structure directly as an
    observation matrix $H$, and `__call__` evaluates the same structure directly rather than
    via a matrix-vector product.

    Attributes:
        grid_shape: Shape of the regular grid, e.g. `(64,)` for 1D or `(64, 64)` for 2D.
            The flattened state this operates on has length `prod(grid_shape)`, in C
            (row-major) order.
        method: `"nearest"` or `"linear"` (multilinear in N-D).
        boundary: How out-of-range query points are handled -- a single, global setting
            for now (not per-axis):

            - `"periodic"`: wrap around (query points are taken modulo the domain extent
              per axis). No bias term; every row of the resulting $H$ sums to 1.
            - `"edge"`: clamp query points to the grid's extent (flat/Neumann-like
              extrapolation). Also bias-free.
            - `"constant"`: query points (or interpolation corners) outside the grid
              contribute `fill_value` instead of a grid value. This introduces an additive
              **bias** (see `bias()`) alongside $H$ -- pair with
              `LinearGaussianObservation(H=..., bias=...)`, not `H` alone.
        fill_value: Constant used for out-of-range contributions when `boundary="constant"`.
            Unused otherwise.

    Note:
        Grid axes are assumed uniformly spaced. For `boundary="periodic"`, an axis is
        assumed *not* to repeat its periodic image as an explicit grid point (matching
        `jnp.linspace(..., endpoint=False)`), consistent with how periodic PDE grids are
        built elsewhere in this codebase.
    """

    grid_shape: tuple[int, ...] = eqx.field(static=True)
    n_query: int = eqx.field(static=True)
    method: InterpolationMethod = eqx.field(static=True)
    boundary: BoundaryMode = eqx.field(static=True)
    fill_value: float = eqx.field(static=True)
    corner_indices: Int[Array, "n_query n_corners"]
    corner_weights: Float[Array, "n_query n_corners"]
    corner_valid: Bool[Array, "n_query n_corners"]

    def __init__(
        self,
        grid_axes: tuple[Real[Array, " _"], ...],
        query_points: Real[Array, "n_query ndim"],
        *,
        method: InterpolationMethod = "linear",
        boundary: BoundaryMode = "periodic",
        fill_value: float = 0.0,
    ):
        """
        Args:
            grid_axes: Per-axis 1D grid coordinates, e.g. `(x_grid,)` for a 1D grid or
                `(x_grid, y_grid)` for a 2D grid. Any number of axes is supported.
            query_points: Observation locations, shape `(n_query, len(grid_axes))` --
                including for 1D grids, where this is `(n_query, 1)`, not `(n_query,)`.
            method: `"nearest"` or `"linear"`.
            boundary: `"periodic"`, `"constant"`, or `"edge"`. See class docstring.
            fill_value: Constant used for out-of-range contributions when
                `boundary="constant"`.
        """
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}.")
        if boundary not in _VALID_BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {_VALID_BOUNDARIES}, got {boundary!r}."
            )
        query_points = jnp.asarray(query_points)
        if query_points.ndim != 2 or query_points.shape[-1] != len(grid_axes):
            raise ValueError(
                "query_points must have shape (n_query, len(grid_axes)); got "
                f"{query_points.shape} for {len(grid_axes)} grid axes."
            )

        self.grid_shape = tuple(int(axis.shape[0]) for axis in grid_axes)
        self.n_query = int(query_points.shape[0])
        self.method = method
        self.boundary = boundary
        self.fill_value = float(fill_value)

        ndim = len(grid_axes)
        strides = _row_major_strides(self.grid_shape)

        axis_results = [
            _axis_candidates(query_points[:, i], grid_axes[i], method, boundary)
            for i in range(ndim)
        ]
        k = axis_results[0][0].shape[-1]  # uniform across axes by construction

        flat_indices = []
        weights = []
        valid = []
        for combo in itertools.product(range(k), repeat=ndim):
            corner_idx = jnp.zeros((self.n_query,), dtype=jnp.int32)
            corner_w = jnp.ones((self.n_query,))
            corner_valid = jnp.ones((self.n_query,), dtype=bool)
            for axis, sel in enumerate(combo):
                idxs_i, ws_i, valid_i = axis_results[axis]
                corner_idx = corner_idx + idxs_i[:, sel] * strides[axis]
                corner_w = corner_w * ws_i[:, sel]
                corner_valid = corner_valid & valid_i[:, sel]
            flat_indices.append(corner_idx)
            weights.append(corner_w)
            valid.append(corner_valid)

        self.corner_indices = jnp.stack(flat_indices, axis=-1)
        self.corner_weights = jnp.stack(weights, axis=-1)
        self.corner_valid = jnp.stack(valid, axis=-1)

    @property
    def state_dim(self) -> int:
        state_dim = 1
        for size in self.grid_shape:
            state_dim *= size
        return state_dim

    def __call__(self, values: Real[Array, " state_dim"]) -> Float[Array, " n_query"]:
        """Interpolate a flattened grid state at the query points."""
        gathered = values.reshape(-1)[self.corner_indices]
        grid_contribution = (
            jnp.where(self.corner_valid, gathered, 0.0) * self.corner_weights
        )
        fill_contribution = (
            jnp.where(self.corner_valid, 0.0, self.fill_value) * self.corner_weights
        )
        return jnp.sum(grid_contribution + fill_contribution, axis=-1)

    def as_matrix(self) -> Float[Array, "n_query state_dim"]:
        """Dense observation matrix $H$ such that `H @ values == self(values) - self.bias()`."""
        n_corners = self.corner_indices.shape[-1]
        query_idx = jnp.repeat(jnp.arange(self.n_query), n_corners)
        col_idx = self.corner_indices.reshape(-1)
        valid_weights = jnp.where(self.corner_valid, self.corner_weights, 0.0).reshape(
            -1
        )
        H = jnp.zeros((self.n_query, self.state_dim))
        return H.at[query_idx, col_idx].add(valid_weights)

    def as_sparse(self) -> jax_sparse.BCOO:
        """Sparse (`jax.experimental.sparse.BCOO`) observation matrix $H$.

        Note:
            `dynestyx.models.observations.LinearGaussianObservation.__call__` currently
            calls `jnp.dot(H, x)`, which does not accept a `BCOO` operand (`H @ x` does,
            but `jnp.dot(H, x)` raises `TypeError`). Use `as_matrix()` when constructing a
            `LinearGaussianObservation` until that is changed; `as_sparse()` is for
            evaluation/inspection (`interp.as_sparse() @ values`) or for downstream code
            that calls `H @ x` directly.
        """
        n_corners = self.corner_indices.shape[-1]
        query_idx = jnp.repeat(jnp.arange(self.n_query), n_corners)
        col_idx = self.corner_indices.reshape(-1)
        valid_weights = jnp.where(self.corner_valid, self.corner_weights, 0.0).reshape(
            -1
        )
        indices = jnp.stack([query_idx, col_idx], axis=-1)
        return jax_sparse.BCOO(
            (valid_weights, indices), shape=(self.n_query, self.state_dim)
        )

    def bias(self) -> Float[Array, " n_query"]:
        """Additive constant from `boundary="constant"` fill-value contributions.

        Zero for `boundary in ("periodic", "edge")`, where every corner is always valid.
        """
        invalid_weights = jnp.where(self.corner_valid, 0.0, self.corner_weights)
        return jnp.sum(invalid_weights, axis=-1) * self.fill_value


def interpolation_observation_model(
    grid_axes: tuple[Real[Array, " _"], ...],
    query_points: Real[Array, "n_query ndim"],
    R: Float[Array, "n_query n_query"],
    *,
    method: InterpolationMethod = "linear",
    boundary: BoundaryMode = "periodic",
    fill_value: float = 0.0,
) -> LinearGaussianObservation:
    """Build a `LinearGaussianObservation` that observes a regular-grid state at arbitrary,
    non-grid-aligned points via nearest-neighbor or piecewise-linear interpolation.

    A thin convenience wrapper around `GridInterpolator`: constructs the interpolator, then
    returns `LinearGaussianObservation(H=interp.as_matrix(), R=R, bias=interp.bias())`
    (the `bias` argument is only nonzero for `boundary="constant"`).
    """
    interp = GridInterpolator(
        grid_axes, query_points, method=method, boundary=boundary, fill_value=fill_value
    )
    bias = interp.bias()
    return LinearGaussianObservation(
        H=interp.as_matrix(),
        R=R,
        bias=bias if boundary == "constant" else None,
    )
