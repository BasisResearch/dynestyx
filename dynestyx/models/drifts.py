"""Drift building blocks for continuous-time state evolution."""

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Real, Shaped


class Drift(Protocol):
    """
    Drift vector field for continuous-time state evolution.

    Mathematically, the drift is a mapping
    $\\mu: \\mathbb{R}^{d_x} \\times \\mathbb{R}^{d_u} \\times \\mathbb{R}
    \\to \\mathbb{R}^{d_x}$, i.e., $(x, u, t) \\mapsto \\mu(x, u, t)$.
    In the SDE formulation used by `ContinuousTimeStateEvolution`,
    $dx_t = \\mu(x_t, u_t, t) \\, dt + \\sigma(x_t, u_t, t) \\, dW_t$, this
    mapping forms the $\\mu$ term.

    Implementations should be compatible with JAX transformations (e.g., `jax.jit`,
    `jax.vmap`, and `jax.grad` when differentiable).

    Args:
        x (State): Current state $x \\in \\mathbb{R}^{d_x}$.
        u (Control | None): Current control input $u \\in \\mathbb{R}^{d_u}$ or None.
        t (Time): Current time (scalar or array).

    Returns:
        dState: Drift vector $\\mu(x, u, t) \\in \\mathbb{R}^{d_x}$.

    Note:
        This is a protocol interface; implement this callable signature; do not instantiate.
        We recommend simply using a plain Python function that matches this signature, e.g.:

        ```python
        def drift(x, u, t):
            return - x + u
        ```
        or `lambda x, u, t: - x + u`
    """

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError()


class Potential(Protocol):
    """
    Scalar potential energy for gradient-based drift
    
    $$dx_t = \\mu(x_t, u_t, t)dt -\\nabla V(x_t, u_t, t)dt + L(x_t, u_t, t)dW_t.$$

    A potential $V(x, u, t)$ maps state, control, and time to a scalar. Its
    gradient contributes to the drift via $\\pm \\nabla_x V(x, u, t)$, enabling
    Langevin-type dynamics. It is used in `ContinuousTimeStateEvolution` when
    `potential` is set; the sign is controlled by `use_negative_gradient`.

    Args:
        x (State): Current state $x \\in \\mathbb{R}^{d_x}$.
        u (Control | None): Current control input $u \\in \\mathbb{R}^{d_u}$ or None.
        t (Time): Current time.

    Returns:
        jax.Array: Scalar potential value $V(x, u, t) \\in \\mathbb{R}$.

    Note:
        This is a protocol interface; implement this callable signature; do not instantiate.
        We recommend simply using a plain Python function that matches this signature, e.g.:

        ```python
        def potential(x, u, t):
            return x[0]**2 + x[1]**2 + x[2]**2
        ```
        or `lambda x, u, t: x[0]**2 + x[1]**2 + x[2]**2`
    """

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Shaped[Array, ""]:
        raise NotImplementedError()


class AffineDrift(eqx.Module):
    """
    Affine drift function for continuous-time models.

    This implements an affine map of the form

    $$f(x, u, t) = A x + B u + b,$$

    where $A \\in \\mathbb{R}^{d_x \\times d_x}$, $B \\in \\mathbb{R}^{d_x \\times d_u}$
    (optional), and $b \\in \\mathbb{R}^{d_x}$ (optional). The time argument $t$
    is accepted for compatibility with the `Drift` protocol but is not used.

    This is commonly used as the drift term $\\mu(x_t, u_t, t)$ inside
    `ContinuousTimeStateEvolution`, and is a building block for LTI models such as
    `LTI_continuous`.

    Attributes:
        A (jax.Array): Drift matrix with shape $(d_x, d_x)$.
        B (jax.Array | None): Optional control matrix with shape $(d_x, d_u)$.
        b (jax.Array | None): Optional additive bias with shape $(d_x,)$.
    """

    A: Float[Array, "*a_plate state_dim state_dim"]
    B: Float[Array, "*b_matrix_plate state_dim control_dim"] | None = None
    b: Float[Array, "*bias_plate state_dim"] | None = None

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"]:
        out = jnp.dot(self.A, x)
        if self.B is not None:
            u_vec = u if u is not None else jnp.zeros(self.B.shape[1])
            out = out + jnp.dot(self.B, u_vec)
        if self.b is not None:
            out = out + self.b
        return out


class ImExDrift(eqx.Module):
    """
    Split explicit/implicit drift for IMEX (implicit-explicit) solvers:

    $$x_{n+1} = x_n + \\Delta t \\big( f_{ex}(x_n, u_n, t_n) + f_{im}(x_{n+1}, u_{n+1}, t_{n+1})\\big)$$

    Wraps two drift terms so a single object serves two roles:

    1. As a drop-in `drift` for `ContinuousTimeStateEvolution` used with an
       ordinary explicit solver (e.g. `diffrax.Tsit5`) or a fully-implicit
       solver (e.g. `diffrax.Kvaerno4`) -- `__call__` returns the combined
       vector field $f(x, u, t) = f_{ex}(x, u, t) + f_{im}(x, u, t)$.
    2. As the source of the explicit/implicit split consumed by diffrax's
       IMEX solvers (e.g. `diffrax.KenCarp3/4/5`, `diffrax.Sil3`, which
       require diffrax's `MultiTerm`), via `make_imex_tuple`.

    Construct with keyword-only arguments to avoid silently swapping the two
    terms: `ImExDrift(explicit_term=..., implicit_term=...)`.

    Wherever a plain `Drift`-shaped callable `(x, u, t) -> R^{d_x}` is
    expected, an `ImExDrift` instance may be used directly. When used with a
    solver that doesn't request diffrax's `MultiTerm` (i.e. not an IMEX
    solver), `dynestyx.solvers.odes.solve_ode_state_path` emits a warning
    naming the solver, since the explicit/implicit split goes unused.

    Attributes:
        explicit_term (Drift): Non-stiff component of the vector field,
            integrated explicitly by IMEX solvers.
        implicit_term (Drift): Stiff component of the vector field,
            integrated implicitly by IMEX solvers.

    Note:
        Cannot be combined with `potential` on `ContinuousTimeStateEvolution`
        -- doing so raises a `ValueError` when the `DynamicalModel` is
        constructed. Fold any potential-gradient contribution into
        `explicit_term` or `implicit_term` directly instead.
    """

    explicit_term: Drift
    implicit_term: Drift

    def __init__(self, *, explicit_term: Drift, implicit_term: Drift):
        self.explicit_term = explicit_term
        self.implicit_term = implicit_term

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        return self.explicit_term(x, u, t) + self.implicit_term(x, u, t)

    def make_imex_tuple(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> tuple[
        Real[Array, " state_dim"] | Real[Array, ""],
        Real[Array, " state_dim"] | Real[Array, ""],
    ]:
        """Return `(explicit_term(x, u, t), implicit_term(x, u, t))`.

        Order matches diffrax's `MultiTerm(ODETerm(explicit), ODETerm(implicit))`
        convention used by `solve_ode_state_path`.
        """
        return self.explicit_term(x, u, t), self.implicit_term(x, u, t)
