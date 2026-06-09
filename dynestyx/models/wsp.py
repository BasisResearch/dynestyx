r"""Weighted Sums Parameterization (WSP) for SDEs on compact box domains.

This module implements the box / unit-cube specialization of the *Weighted Sums
Parameterization* (WSP) of Lu, Liu, Nock & Yacoby, *Neural Stochastic
Differential Equations on Compact State Spaces* (ProbML 2026). WSP wraps an
"unconstrained" :class:`~dynestyx.models.core.ContinuousTimeStateEvolution`
(SDE) and returns a new state evolution whose solution provably stays inside an
axis-aligned box :math:`K = \prod_d [a_d, b_d]`.

The construction blends the original drift / diffusion with a boundary-respecting
correction, weighted by a function :math:`w(x) \in [0, 1]^{d_x}` that approaches
:math:`0` at the box faces and :math:`1` in the interior. Concretely, for each
state coordinate :math:`d` with distances to its two faces
:math:`d_{\text{lo}} = x_d - a_d` and :math:`d_{\text{hi}} = b_d - x_d`,

.. math::

    w_d(x_d) = \tanh\!\Bigl(\beta \prod_{s\in\{\text{lo},\text{hi}\}}
        \mathrm{softmax}_s \cdot \tanh(\alpha\, d_s)\Bigr),

so :math:`w_d \to 0` at either face (because :math:`\tanh(\alpha\cdot 0) = 0`) and
:math:`w_d \to 1` deep in the interior for large :math:`\beta`. The WSP dynamics are

.. math::

    h_d(x, t) &= w_d(x_d)\, \tilde{h}_d(x, t) + (1 - w_d(x_d))\, c_{h,d}(x), \\
    g(x, t)   &= \operatorname{diag}(w(x))\, \tilde{L}(x, t),

where :math:`\tilde{h}, \tilde{L}` are the inner (unconstrained) drift and
diffusion and :math:`c_h(x) = \gamma\,(z^\* - x)/(\lVert z^\* - x\rVert + \epsilon)`
is a pull toward the box center :math:`z^\* = (a + b)/2`. Because the diffusion
vanishes and the drift points inward at the boundary, the process is *viable* in
:math:`K` (Milian 1995; Theorems 3 & 6 of the paper).

Fixed-step SDE solvers can still step slightly outside :math:`K`; following the
paper, we clip the state into the box before evaluating the drift / diffusion
(``clip=True``, the default), which keeps the dynamics well-defined everywhere.

This is a thin, composable wrapper: :func:`WSP` returns an ordinary
:class:`~dynestyx.models.core.ContinuousTimeStateEvolution`, so the result works
unchanged with simulators, the Euler-Maruyama discretizer, and the
filtering/smoothing backends.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from dynestyx.models.core import ContinuousTimeStateEvolution, Drift
from dynestyx.models.diffusions import Diffusion, FullDiffusion


class Box(eqx.Module):
    r"""An axis-aligned box (hyper-rectangle) domain :math:`\prod_d [a_d, b_d]`.

    Args:
        lower: Lower corner :math:`a`, a length-``state_dim`` vector (a scalar is
            promoted to a length-1 vector for 1-D domains).
        upper: Upper corner :math:`b`, same shape as ``lower``.

    The box supplies the geometric quantities WSP needs: the Chebyshev center
    :attr:`center` (the box midpoint), the per-dimension weight :meth:`weight`,
    the center pull :meth:`center_pull`, and the input :meth:`clip` safeguard.
    """

    lower: Float[Array, " state_dim"]
    upper: Float[Array, " state_dim"]

    def __init__(
        self,
        lower: Array | float,
        upper: Array | float,
    ):
        # Loose annotations on purpose: validate shapes manually below so the
        # error messages are clear (rather than a generic shape-check failure).
        lower_arr = jnp.atleast_1d(jnp.asarray(lower))
        upper_arr = jnp.atleast_1d(jnp.asarray(upper))
        if lower_arr.ndim != 1 or upper_arr.ndim != 1:
            raise ValueError(
                "Box bounds must be scalars or 1-D vectors of shape (state_dim,). "
                f"Got lower.ndim={lower_arr.ndim}, upper.ndim={upper_arr.ndim}."
            )
        if lower_arr.shape != upper_arr.shape:
            raise ValueError(
                "Box lower and upper bounds must have matching shapes. "
                f"Got {lower_arr.shape} and {upper_arr.shape}."
            )
        self.lower = lower_arr
        self.upper = upper_arr

    @property
    def dim(self) -> int:
        """The dimension :math:`d_x` of the box (number of coordinates)."""
        return self.lower.shape[-1]

    @property
    def center(self) -> Float[Array, " state_dim"]:
        r"""The box center (Chebyshev center of an axis-aligned box), :math:`(a+b)/2`."""
        return 0.5 * (self.lower + self.upper)

    def clip(
        self, x: Real[Array, "*batch state_dim"]
    ) -> Real[Array, "*batch state_dim"]:
        """Clip ``x`` element-wise into ``[lower, upper]``."""
        return jnp.clip(x, self.lower, self.upper)

    def weight(
        self,
        x: Real[Array, "*batch state_dim"],
        alpha: Float[Array, ""] | float,
        beta: Float[Array, ""] | float,
    ) -> Float[Array, "*batch state_dim"]:
        r"""The per-dimension WSP weight :math:`w(x) \in [0, 1]^{d_x}`.

        Element-wise across coordinates, using the two faces of each dimension.
        ``alpha`` controls how sharply :math:`\tanh(\alpha d)` rises from the
        boundary; ``beta`` controls how quickly the weight saturates to 1 in the
        interior. Larger values make a stiffer boundary.
        """
        d_lo = x - self.lower
        d_hi = self.upper - x
        # softmax over the two boundary distances, written via the sigmoid for
        # numerical stability: exp(-d_lo) / (exp(-d_lo) + exp(-d_hi)).
        s_lo = jax.nn.sigmoid(d_hi - d_lo)
        s_hi = jax.nn.sigmoid(d_lo - d_hi)
        prod = (s_lo * jnp.tanh(alpha * d_lo)) * (s_hi * jnp.tanh(alpha * d_hi))
        w = jnp.tanh(beta * prod)
        # The construction guarantees w in [0, 1] inside the box; clip defends
        # against tiny float error and against inputs that fall outside the box.
        return jnp.clip(w, 0.0, 1.0)

    def center_pull(
        self,
        x: Real[Array, "*batch state_dim"],
        gamma: Float[Array, ""] | float,
        epsilon: Float[Array, ""] | float,
    ) -> Float[Array, "*batch state_dim"]:
        r"""The boundary drift correction :math:`\gamma (z^\* - x)/(\lVert z^\* - x\rVert + \epsilon)`.

        Points toward the box center :math:`z^\*` with magnitude controlled by
        ``gamma``; ``epsilon`` keeps it well-defined at the center.
        """
        diff = self.center - x
        norm = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        return gamma * diff / (norm + epsilon)


class WSPDrift(eqx.Module):
    r"""WSP-corrected drift :math:`h = w \cdot \tilde{h} + (1 - w) \cdot c_h`.

    Implements the :class:`~dynestyx.models.core.Drift` protocol. ``base`` is the
    inner state evolution (with diffusion stripped) whose ``total_drift`` provides
    :math:`\tilde{h}` (including any potential term). This is constructed by
    :func:`WSP`; users do not normally instantiate it directly.
    """

    base: ContinuousTimeStateEvolution
    domain: Box
    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    gamma: Float[Array, ""]
    epsilon: Float[Array, ""]
    clip: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        base: ContinuousTimeStateEvolution,
        domain: Box,
        alpha: Float[Array, ""] | float,
        beta: Float[Array, ""] | float,
        gamma: Float[Array, ""] | float,
        epsilon: Float[Array, ""] | float,
        clip: bool = True,
    ):
        self.base = base
        self.domain = domain
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.gamma = jnp.asarray(gamma)
        self.epsilon = jnp.asarray(epsilon)
        self.clip = clip

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"]:
        x_eval = self.domain.clip(x) if self.clip else x
        h_tilde = self.base.total_drift(x_eval, u, t)
        w = self.domain.weight(x_eval, self.alpha, self.beta)
        c = self.domain.center_pull(x_eval, self.gamma, self.epsilon)
        return w * h_tilde + (1.0 - w) * c


class WSPDiffusionCoefficient(eqx.Module):
    r"""Callable coefficient for the WSP diffusion, :math:`\operatorname{diag}(w(x))\,\tilde{L}(x,t)`.

    Scales row :math:`d` of the inner diffusion matrix by :math:`w_d(x_d)`, so the
    diffusion vanishes at the box faces. Wrapped in a
    :class:`~dynestyx.models.diffusions.FullDiffusion` by :func:`WSP`; the
    resulting Brownian dimension equals the inner diffusion's ``bm_dim``.
    """

    inner: Diffusion
    domain: Box
    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    clip: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        inner: Diffusion,
        domain: Box,
        alpha: Float[Array, ""] | float,
        beta: Float[Array, ""] | float,
        clip: bool = True,
    ):
        self.inner = inner
        self.domain = domain
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.clip = clip

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, "state_dim bm_dim"]:
        x_eval = self.domain.clip(x) if self.clip else x
        w = self.domain.weight(x_eval, self.alpha, self.beta)
        loading = self.inner.as_matrix(x=x_eval, u=u, t=t, state_dim=self.domain.dim)
        return w[..., :, None] * loading


def WSP(
    state_evolution: ContinuousTimeStateEvolution,
    domain: Box,
    *,
    alpha: Float[Array, ""] | float = 5.0,
    beta: Float[Array, ""] | float = 1000.0,
    gamma: Float[Array, ""] | float = 2.0,
    epsilon: Float[Array, ""] | float = 0.1,
    clip: bool = True,
) -> ContinuousTimeStateEvolution:
    r"""Wrap a continuous-time SDE so its solution stays inside a box ``domain``.

    Applies the Weighted Sums Parameterization (Lu et al., 2026): near the box
    faces the diffusion vanishes and the drift points inward toward the box
    center, while in the interior the original dynamics are recovered (for large
    ``beta``).

    Args:
        state_evolution: The inner, unconstrained
            :class:`~dynestyx.models.core.ContinuousTimeStateEvolution`. Its drift
            (and optional potential) and diffusion are reused; a continuous-time
            ODE (``diffusion=None``) is also supported, in which case only the
            drift is corrected.
        domain: The :class:`Box` the solution must remain inside.
        alpha: Boundary sharpness of :math:`\tanh(\alpha d)` (``> 0``).
        beta: Interior saturation rate of the weight (``> 0``). The paper uses
            ``beta=1000`` for a stiff unit-square boundary; smaller values give a
            gentler, more visible transition.
        gamma: Magnitude of the inward center pull (``> 0``).
        epsilon: Softening constant for the center pull (``> 0``).
        clip: If ``True`` (default), clip the state into ``domain`` before
            evaluating drift / diffusion, a numerical safeguard against
            fixed-step solver overshoot.

    Returns:
        A new :class:`~dynestyx.models.core.ContinuousTimeStateEvolution` carrying
        the WSP drift and (if applicable) the WSP diffusion. Pass it to
        :class:`~dynestyx.models.core.DynamicalModel` as usual; it is refined into
        a stochastic (or deterministic) continuous-time evolution and works with
        all simulators and inference backends.
    """
    if not isinstance(state_evolution, ContinuousTimeStateEvolution):
        raise TypeError(
            "WSP wraps a ContinuousTimeStateEvolution (continuous-time SDE/ODE). "
            f"Got {type(state_evolution).__name__}."
        )

    # Strip the diffusion off a copy of the inner evolution so that
    # `base.total_drift` yields the unconstrained drift (plus any potential term).
    base = ContinuousTimeStateEvolution(
        drift=state_evolution.drift,
        potential=state_evolution.potential,
        use_negative_gradient=state_evolution.use_negative_gradient,
    )
    wsp_drift: Drift = WSPDrift(
        base=base,
        domain=domain,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        epsilon=epsilon,
        clip=clip,
    )

    if state_evolution.diffusion is None:
        return ContinuousTimeStateEvolution(drift=wsp_drift)

    wsp_coefficient = WSPDiffusionCoefficient(
        inner=state_evolution.diffusion,
        domain=domain,
        alpha=alpha,
        beta=beta,
        clip=clip,
    )
    return ContinuousTimeStateEvolution(
        drift=wsp_drift,
        diffusion=FullDiffusion(wsp_coefficient),
    )
