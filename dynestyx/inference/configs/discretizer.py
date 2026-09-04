"""Configuration objects for continuous-to-discrete approximations."""

from __future__ import annotations

import abc
import dataclasses

import diffrax as dfx

from dynestyx.inference.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
)
from dynestyx.utils import _validate_nonnegative_float


def _validate_covariance_jitter(covariance_jitter: float) -> None:
    _validate_nonnegative_float("covariance_jitter", covariance_jitter)


def _validate_jitter_scale(jitter_scale: float) -> None:
    _validate_nonnegative_float("jitter_scale", jitter_scale)


def _default_diffrax_sde_solver() -> SDESimulatorConfig:
    return SDESimulatorConfig(source="diffrax", solver=dfx.Euler())


@dataclasses.dataclass
class BaseDiscretizerConfig(abc.ABC):
    r"""Base class for continuous-to-discrete configuration objects.

    Do not instantiate this marker class directly. Pass one of its concrete
    subclasses to `Discretizer`. For default settings, pass None
    as the config to `Discretizer`.
    """


@dataclasses.dataclass
class ODEFlowConfig(BaseDiscretizerConfig):
    r"""Numerical flow transition for a deterministic ODE.

    For the controlled ODE

    $$
    \dot X_s=f(X_s,u_k,s),
    $$

    this config numerically integrates from ``t_now`` to ``t_next`` while
    holding ``u_k`` fixed. With endpoint
    $\phi(x,u_k,t_{\mathrm{now}},t_{\mathrm{next}})$, the returned
    transition is a Delta distribution when ``jitter_scale == 0``. For
    positive ``jitter_scale`` it is an independent Normal distribution with
    mean $\phi$, component scale ``jitter_scale``, and covariance
    ``jitter_scale**2 * I``.

    All Diffrax settings come from the existing ``ODESimulatorConfig``.

    Attributes:
        simulator_config (ODESimulatorConfig): Diffrax ODE solver settings.
            Defaults to ``ODESimulatorConfig()``.
        jitter_scale (float): Finite, nonnegative component scale added
            after integrating the flow. Defaults to ``0.0``.
    """

    simulator_config: ODESimulatorConfig = dataclasses.field(
        default_factory=ODESimulatorConfig
    )
    jitter_scale: float = 0.0

    def __post_init__(self) -> None:
        _validate_jitter_scale(self.jitter_scale)


@dataclasses.dataclass
class EulerMaruyamaConfig(BaseDiscretizerConfig):
    r"""Euler--Maruyama Gaussian transition for a general Itô SDE.

    This is the inexpensive, general-purpose discretization. Consider

    $$
    dX_s=f(X_s,u_k,s)\,ds+L(X_s,u_k,s)\,dW_s,
    \qquad
    a(x,u,s)=L(x,u,s)L(x,u,s)^\top ,
    $$

    on \(s\in[t_k,t_{k+1}]\), where \(W_s\) is standard Brownian motion,
    \(h=t_{k+1}-t_k>0\), and the control \(u_k\) is held fixed over the
    interval. Euler--Maruyama freezes both coefficients at its left endpoint:

    $$
    X_{k+1}\mid X_k=x,u_k
    \;\approx\;
    \mathcal N\!\left(
        x+h f(x,u_k,t_k),
        h\,a(x,u_k,t_k)+\epsilon I
    \right),
    $$

    where \(\epsilon\) is `covariance_jitter`.

    The method applies to nonlinear drift and state-, time-, or
    control-dependent diffusion. Under the usual global Lipschitz and growth
    assumptions it has strong order \(1/2\) and weak order \(1\); for additive
    noise its strong order improves to \(1\). See Kloeden and Platen (1992)
    for more details. It requires one drift and one diffusion evaluation per
    transition, but a single large step can poorly represent nonlinear, stiff,
    or rapidly varying dynamics.

    Attributes:
        covariance_jitter (float): Nonnegative \(\epsilon\) added to the
            transition covariance as \(\epsilon I\). Defaults to `0.0`.
            Positive jitter changes the stochastic model and should be chosen
            explicitly rather than treated as an invisible numerical fix.

    ??? note "Algorithm Reference"
        - Särkkä, S., & Solin, A. (2019). *Applied Stochastic Differential
          Equations*. Cambridge University Press, Algorithm 8.1 and
          Equations (8.28)--(8.30).
          [Online book](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
        - Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of
          Stochastic Differential Equations*. Springer, Chapter 10.
          [https://doi.org/10.1007/978-3-662-12616-5](https://doi.org/10.1007/978-3-662-12616-5).
    """

    covariance_jitter: float = 0.0

    def __post_init__(self) -> None:
        _validate_covariance_jitter(self.covariance_jitter)


@dataclasses.dataclass
class ExactAffineConfig(BaseDiscretizerConfig):
    r"""Exact discrete transition for an affine SDE with additive noise.

    Use this config only when the continuous transition has an `AffineDrift`,
    constant diffusion, and no potential. For the Itô SDE

    $$
    dX_s=(F X_s+B u_k+b)\,ds+L\,dW_s
    $$

    over an interval of length \(h=t_{k+1}-t_k>0\), with standard Brownian
    motion and a held control \(u_k\), the exact conditional transition is

    $$
    X_{k+1}\mid X_k=x,u_k
    \sim
    \mathcal N(A_hx+B_hu_k+b_h,\;Q_h+\epsilon I),
    $$

    with

    $$
    A_h=e^{Fh},\qquad
    B_h=\int_0^h e^{Fs}B\,ds,\qquad
    b_h=\int_0^h e^{Fs}b\,ds,
    $$

    and

    $$
    Q_h=\int_0^h e^{Fs}LL^\top e^{F^\top s}\,ds.
    $$

    The implementation uses an augmented matrix exponential for the forcing
    terms and evaluates the Van Loan covariance on a safely scaled interval
    before composing it to the requested interval. This avoids overflowing the
    growing block associated with a stable, stiff drift. It never forms
    \(F^{-1}\), so singular drift matrices are supported. With
    `covariance_jitter=0`, the transition is exact up to floating-point error
    for every positive interval length, including nonuniform observation grids.
    It preserves a linear Gaussian state-evolution representation and therefore
    enables exact Kalman filtering when the rest of the model is also linear
    Gaussian.

    Matrix exponentials cost cubic time in the augmented state dimension and
    can be expensive when many distinct interval lengths are used. Drifts
    not explicitly marked as affine/constant, state-dependent diffusion, and
    potentials are rejected. Singular \(Q_h\) requires positive jitter.

    Attributes:
        covariance_jitter (float): Nonnegative \(\epsilon\) added to \(Q_h\)
            as \(\epsilon I\). Defaults to `0.0`.

    ??? note "Algorithm Reference"
        - Van Loan, C. F. (1978). Computing integrals involving the matrix
          exponential. *IEEE Transactions on Automatic Control*, 23(3),
          395--404.
          [https://doi.org/10.1109/TAC.1978.1101743](https://doi.org/10.1109/TAC.1978.1101743).
        - Särkkä, S., & Svensson, L. (2023). *Bayesian Filtering and
          Smoothing*, 2nd ed. Cambridge University Press, Theorem 4.3 and
          Lemma A.9.
          [Online book](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
        - Axelsson, P., & Gustafsson, F. (2015). Discrete-time solutions to
          the continuous-time differential Lyapunov equation. *IEEE
          Transactions on Automatic Control*, 60(3), 632--643.
          [https://doi.org/10.1109/TAC.2014.2353112](https://doi.org/10.1109/TAC.2014.2353112).
    """

    covariance_jitter: float = 0.0

    def __post_init__(self) -> None:
        _validate_covariance_jitter(self.covariance_jitter)


@dataclasses.dataclass
class LocalLinearizationConfig(BaseDiscretizerConfig):
    r"""Locally linearized Gaussian transition with frozen additive noise.

    This method is intended for nonlinear drift with structurally constant,
    additive diffusion. For the Itô SDE

    $$
    dX_s=f(X_s,u_k,s)\,ds+L\,dW_s,
    $$

    define, at the left endpoint \(x=X_{t_k}\),

    $$
    f_0=f(x,u_k,t_k),\qquad
    J=\left.\frac{\partial f}{\partial x}\right|_{(x,u_k,t_k)} .
    $$

    With standard Brownian motion, \(h=t_{k+1}-t_k>0\), and \(u_k\) held
    fixed, approximate the original process by

    $$
    dZ_s=\left[f_0+J(Z_s-x)\right]ds+L\,dW_s,\qquad Z_0=x.
    $$

    Its exact Gaussian transition has

    $$
    m_h=x+\int_0^h e^{Js}f_0\,ds,
    \qquad
    P_h=\int_0^h e^{Js}LL^\top e^{J^\top s}\,ds,
    $$

    and this config returns
    \(\mathcal N(m_h,P_h+\epsilon I)\), where \(\epsilon\) is
    `covariance_jitter`.

    The method is exact for affine drift with additive diffusion. With some amount
    of nonlinearity, it can be more accurate than Euler--Maruyama, but at higher
    cost, since Jacobians must be computed.

    State-, time-, or control-dependent diffusion are rejected.
    The returned object is a Gaussian transition distribution, compatible
    with any method using Gaussian transitions. Degenerate transition covariance
    requires explicit positive jitter.

    Attributes:
        covariance_jitter (float): Nonnegative \(\epsilon\) added to \(P_h\)
            as \(\epsilon I\). Defaults to `0.0`.

    ??? note "Algorithm Reference"
        - Särkkä, S., & Svensson, L. (2023). *Bayesian Filtering and
          Smoothing*, 2nd ed. Cambridge University Press, Theorem 4.18.
          [Online book](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
        - Ozaki, T. (1992). A bridge between nonlinear time series models and
          nonlinear stochastic dynamical systems: A local linearization
          approach. *Statistica Sinica*, 2(1), 113--135.
          [Article](https://www3.stat.sinica.edu.tw/statistica/j2n1/j2n16/j2n16.htm).
    """

    covariance_jitter: float = 0.0

    def __post_init__(self) -> None:
        _validate_covariance_jitter(self.covariance_jitter)


@dataclasses.dataclass
class MeanTrajectoryLinearizationConfig(BaseDiscretizerConfig):
    r"""Gaussian moment transition linearized along the conditional mean.

    This config integrates a first-order Gaussian assumed-density
    approximation for the Itô SDE

    $$
    dX_s=f(X_s,u_k,s)\,ds+L(X_s,u_k,s)\,dW_s,\qquad
    a(x,u,s)=L(x,u,s)L(x,u,s)^\top .
    $$

    On \(s\in[t_k,t_{k+1}]\), standard Brownian motion is assumed and \(u_k\)
    is held fixed. Starting from the deterministic conditional state

    $$
    m(t_k)=x,\qquad P(t_k)=0,
    $$

    integrate

    $$
    \dot m_s=f(m_s,u_k,s),
    $$

    $$
    \dot P_s=J_sP_s+P_sJ_s^\top+a(m_s,u_k,s),
    \qquad
    J_s=\frac{\partial f}{\partial x}(m_s,u_k,s),
    $$

    and return
    \(\mathcal N(m(t_{k+1}),P(t_{k+1})+\epsilon I)\).

    This approximation supports nonlinear drift and state-, time-, or
    control-dependent diffusion. It is exact for affine drift with additive
    diffusion up to the numerical ODE solver tolerance. Relative to a
    left-endpoint local linearization, it updates the Jacobian along the mean
    trajectory.

    Every transition performs a joint mean/covariance ODE solve and evaluates a
    state Jacobian at each right-hand-side call, so it is considerably more
    expensive than Euler--Maruyama. A singular final covariance requires explicit
    positive jitter.

    Attributes:
        ode_solver (ODESimulatorConfig): Existing Diffrax ODE configuration
            used unchanged for the joint mean/covariance integration.
            Defaults to `ODESimulatorConfig()`.
        covariance_jitter (float): Nonnegative \(\epsilon\) added to the final
            covariance as \(\epsilon I\). Defaults to `0.0`.

    ??? note "Algorithm Reference"
        - Särkkä, S., & Solin, A. (2019). *Applied Stochastic Differential
          Equations*. Cambridge University Press, Algorithm 9.4,
          Equation (9.15), and Algorithm 9.8 with Equation (9.28).
          [Online book](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
        - Särkkä, S., & Svensson, L. (2023). *Bayesian Filtering and
          Smoothing*, 2nd ed. Cambridge University Press, Theorem 4.13 and
          Algorithm 4.14.
          [Online book](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
        - Särkkä, S., & Sarmavuori, J. (2013). Gaussian filtering and
          smoothing for continuous-discrete dynamic systems. *Signal
          Processing*, 93(2), 500--510.
          [https://doi.org/10.1016/j.sigpro.2012.09.002](https://doi.org/10.1016/j.sigpro.2012.09.002).
    """

    ode_solver: ODESimulatorConfig = dataclasses.field(
        default_factory=ODESimulatorConfig
    )
    covariance_jitter: float = 0.0

    def __post_init__(self) -> None:
        _validate_covariance_jitter(self.covariance_jitter)


@dataclasses.dataclass
class DiffraxSampleConfig(BaseDiscretizerConfig):
    r"""Sample-only transition obtained by solving each interval with Diffrax.

    This config is for simulation, ensemble Kalman filtering, bootstrap
    particle filtering, and genealogy-tracing particle smoothing when a
    numerical SDE path solver is preferable to an analytic transition
    approximation. For

    $$
    dX_s=f(X_s,u_k,s)\,ds+L(X_s,u_k,s)\,dW_s,
    $$

    over \(s\in[t_k,t_{k+1}]\), with \(u_k\) held fixed, it defines the
    transition operationally as

    $$
    X_{k+1}
    =
    \Psi_{\mathrm{Diffrax}}
    (x,u_k,t_k,t_{k+1};\omega),
    $$

    where \(\Psi_{\mathrm{Diffrax}}\) is the numerical solution selected by
    `sde_solver` and \(\omega\) is generated from the key passed to
    `sample` or `rsample`. A fresh Brownian tree is used for each Markov
    interval. The transition stores no PRNG key and performs no solve when it
    is constructed; solving is deferred until sampling so model shape probes
    remain cheap.

    This transition has no tractable density in general. It supports
    `sample_shape`, reparameterized `rsample`, JIT compilation, external `vmap`,
    deterministic replay with the same key, `DiscreteTimeSimulator`, cuthbert
    EnKF, and cuthbert bootstrap PF. A cuthbert `PFSmootherConfig` using
    genealogy tracing is also compatible because it follows stored ancestors
    without evaluating a transition density. `log_prob`, `mean`, and `variance`
    raise targeted errors at the point where they are requested. Kalman,
    extended/unscented Kalman,
    density-based particle smoothers, latent-path scoring, and proposals that
    require transition densities or analytic moments are not supported.

    The implementation supports any Diffrax solver that operates from ordinary
    Brownian increments, including `diffrax.Euler()` and `diffrax.Heun()`.
    The default inner config selects `diffrax.Euler()`, the Euler--Maruyama
    solver. Cost scales with the number of internal solver steps and requested
    samples. `source="em_scan"` is rejected. Reverse-mode differentiation
    through the default `diffrax.RecursiveCheckpointAdjoint()` requires a finite
    `sde_solver.max_steps`; set that bound explicitly for gradient-based
    inference.

    Attributes:
        sde_solver (SDESimulatorConfig): Existing SDE simulator configuration
            used for each interval. It must have `source="diffrax"`. Defaults
            to `SDESimulatorConfig(source="diffrax",
            solver=diffrax.Euler())`.

    ??? note "Algorithm Reference"
        - Kidger, P. Diffrax documentation:
          [SDE solvers](https://docs.kidger.site/diffrax/api/solvers/sde_solvers/)
          and the
          [SDE solver/order table](https://docs.kidger.site/diffrax/devdocs/SDE_solver_table/).
    """

    sde_solver: SDESimulatorConfig = dataclasses.field(
        default_factory=_default_diffrax_sde_solver
    )

    def __post_init__(self) -> None:
        if self.sde_solver.source != "diffrax":
            raise ValueError(
                "DiffraxSampleConfig requires sde_solver.source='diffrax', "
                f"got source={self.sde_solver.source!r}."
            )


type DiscretizerConfig = (
    ODEFlowConfig
    | EulerMaruyamaConfig
    | ExactAffineConfig
    | LocalLinearizationConfig
    | MeanTrajectoryLinearizationConfig
    | DiffraxSampleConfig
)
