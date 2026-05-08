"""Smoother configuration dataclasses. Shared by dispatchers and integration backends."""

import abc
import dataclasses
from typing import Literal

from dynestyx.inference.filter_configs import (
    ContinuousTimeEKFConfig,
    ContinuousTimeKFConfig,
    EKFConfig,
    KFConfig,
    PFConfig,
    UKFConfig,
)

PFBackwardSamplingMethod = Literal["tracing", "exact", "mcmc"]
CDKFSmootherType = Literal["cd_smoother_1", "cd_smoother_2"]


@dataclasses.dataclass
class SmootherConfig(abc.ABC):
    """Shared base class for all smoother configs.

    You do not instantiate this class directly; use one of the concrete
    subclasses (e.g. `KFSmootherConfig`, `PFSmootherConfig`).

    Concrete smoother configs inherit from their corresponding filter configs,
    so backend selection, filter tuning, continuous-time solver options, and
    `record_smoothed_*` fields follow the same conventions as
    `dynestyx.inference.filter_configs`.
    """

    def __post_init__(self):
        if type(self) is SmootherConfig:
            raise TypeError("SmootherConfig is abstract and cannot be instantiated.")


@dataclasses.dataclass
class KFSmootherConfig(KFConfig, SmootherConfig):
    r"""Rauch-Tung-Striebel Kalman Smoother (RTS) for discrete-time models.

    The exact Bayesian smoother for discrete-time linear-Gaussian state-space
    models. Use this when your model was built with `LTI_discrete` or with
    `LinearGaussianStateEvolution` + `LinearGaussianObservation`.

    The smoother computes \(p(x_t \mid y_{1:T})\), while `KFConfig` computes
    \(p(x_t \mid y_{1:t})\). Use the inherited `record_smoothed_*` fields to
    save smoothed means, covariances, or covariance diagonals in the NumPyro
    trace.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.
        associative (bool | None): Whether to enable cuthbert's associative
            parallel-in-time scan. This is only supported when
            `filter_source="cuthbert"`. Defaults to `None`, which selects
            an associative scan if `filter_source="cuthbert"`, and a
            sequential scan otherwise.

    ??? note "Algorithm Reference"
        After a forward Kalman filtering pass, the RTS smoother runs a backward
        recursion over the filtered and one-step-ahead predicted Gaussian
        states. For a linear transition matrix \(A_t\), the smoothing gain is

        $$
            G_t = P_{t|t} A_t^\top P_{t+1|t}^{-1}.
        $$

        The smoothed mean and covariance are then

        $$
            m_{t|T} = m_{t|t} + G_t (m_{t+1|T} - m_{t+1|t}),
        $$

        and

        $$
            P_{t|T} = P_{t|t}
                + G_t (P_{t+1|T} - P_{t+1|t}) G_t^\top.
        $$

        References:

        - For the original smoother, see: Rauch, H. E., Tung, F., & Striebel, C. T. (1965).
            Maximum likelihood estimates of linear dynamic systems. AIAA Journal, 3(8), 1445-1450.
        - For a modern textbook reference, see: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """


@dataclasses.dataclass
class EKFSmootherConfig(EKFConfig, SmootherConfig):
    r"""Extended Kalman Smoother (EKS) for discrete-time models.

    The Gaussian smoother corresponding to `EKFConfig`. Use this for
    differentiable nonlinear dynamics when local linearization is a reasonable
    approximation to the filtering and smoothing distributions.

    This is exact (but wasteful) for linear-Gaussian models. For those models,
    prefer `KFSmootherConfig`.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cuthbert"`.

    ??? note "Algorithm Reference"
        The EKS applies the RTS backward recursion to the linear-Gaussian
        approximations produced by an EKF filtering pass. The transition matrix
        in the smoothing gain is the Jacobian of the transition function at the
        filtered mean:

        $$
            G_t = P_{t|t} F_t^\top P_{t+1|t}^{-1}.
        $$

        Smoothed means and covariances are then computed with the same
        backward recursion as `KFSmootherConfig`.

        References:

        - For a classical textbook treatment, see: Jazwinski, A. H. (1970).
            Stochastic Processes and Filtering Theory. Academic Press.
        - For a modern textbook reference, see: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """


@dataclasses.dataclass
class UKFSmootherConfig(UKFConfig, SmootherConfig):
    r"""Unscented Kalman Smoother (UKS) for discrete-time models.

    A derivative-free Gaussian smoother that handles nonlinearities by
    propagating deterministic sigma points rather than Jacobians. Use this when
    sigma-point propagation is a better local approximation than EKF
    linearization.

    This smoother is currently supported with `filter_source="cd_dynamax"` only.
    Cuthbert UKF smoothing is not implemented.

    Attributes:
        alpha (float): Spread of sigma points around the current mean.
            Smaller → tighter cluster; larger → sigma points reach further.
            Defaults to \(\sqrt{3}\).
        beta (int): Encodes prior knowledge about the distribution shape.
            `2` is optimal for Gaussians. Defaults to `2`.
        kappa (int): Secondary scaling parameter. Defaults to `1`.
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        The UKS uses a forward UKF pass followed by an RTS-style backward pass.
        Instead of using a Jacobian in the smoothing gain, it estimates the
        lag-one cross-covariance \(P_{t,t+1|t}\) by propagating sigma points:

        $$
            G_t = P_{t,t+1|t} P_{t+1|t}^{-1}.
        $$

        The smoothed mean and covariance then follow the same correction form
        as the Kalman and extended Kalman smoothers.

        References:

        - For the original unscented Kalman filter paper, see: Julier, S. J., & Uhlmann, J. K. (1997).
            New extension of the Kalman filter to nonlinear systems. SPIE Proceedings, 3068.
        - For the unscented RTS smoother, see: Särkkä, S. (2008).
            Unscented Rauch-Tung-Striebel smoother. IEEE Transactions on Automatic Control, 53(3), 845-849.
        - For a modern textbook reference, see: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """


@dataclasses.dataclass
class PFSmootherConfig(PFConfig, SmootherConfig):
    r"""Particle smoother for discrete-time models.

    Use this for nonlinear or non-Gaussian discrete-time models. Particle
    smoothing is more flexible than Gaussian smoothing, but accuracy scales
    with the number of particles and backward-sampled smoother trajectories.

    This smoother is currently supported with `filter_source="cuthbert"` only.

    Attributes:
        n_particles (int): Number of particles in the forward particle filter.
            More particles give a lower-variance log-likelihood and smoothing
            approximation at linear compute cost. Defaults to `1_000`.
        pf_backward_sampling_method (PFBackwardSamplingMethod): Backward
            simulation method. `"tracing"` is the default, `"exact"` performs
            exact backward sampling where available, and `"mcmc"` uses an MCMC
            backward kernel.
        pf_mcmc_n_steps (int): Number of MCMC steps used when
            `pf_backward_sampling_method="mcmc"`.
        pf_n_smoother_particles (int | None): Number of backward-sampled
            smoother particles. `None` inherits `n_particles` from `PFConfig`.
        filter_source (FilterSource): Backend. Defaults to `"cuthbert"`, which
            is currently the only available implementation.

    ??? note "Algorithm Reference"
        Particle smoothers augment a forward particle filter with a backward
        pass that reconstructs trajectories from the particle approximation to
        the filtering distributions.

        In backward simulation, a sampled state \(x_{t+1}^{(j)}\) chooses an
        ancestor at time \(t\) with probabilities proportional to

        $$
            w_t^{(i)} p(x_{t+1}^{(j)} \mid x_t^{(i)}).
        $$

        The `"tracing"` method follows stored particle ancestors. The `"exact"`
        and `"mcmc"` methods use backward kernels that can better approximate
        the full smoothing distribution at higher compute cost.

        References:

        - For backward simulation particle smoothing, see: Godsill, S. J., Doucet, A., & West, M. (2004).
            Monte Carlo smoothing for nonlinear time series. Journal of the American Statistical Association, 99(465), 156-168.
        - For a general SMC/particle filtering reference, see: Doucet, A., De Freitas, N., & Gordon, N. (2001).
            An Introduction to Sequential Monte Carlo Methods. In Sequential Monte Carlo Methods in Practice (pp. 3-14).
            New York, NY: Springer New York.
        - For a modern textbook reference, see: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """

    pf_backward_sampling_method: PFBackwardSamplingMethod = "tracing"
    pf_mcmc_n_steps: int = 10
    pf_n_smoother_particles: int | None = None


@dataclasses.dataclass
class ContinuousTimeKFSmootherConfig(ContinuousTimeKFConfig, SmootherConfig):
    r"""Continuous-discrete Kalman Smoother (CD-KS).

    The exact Bayesian smoother for continuous-time linear-Gaussian models with
    discrete observations. Use this when your model was built with
    `LTI_continuous`.

    This smoother is currently implemented through the `cd_dynamax` backend.
    Inherited differential-equation solver options control the continuous-time
    filtering pass, and inherited `record_smoothed_*` fields control trace
    recording.

    Attributes:
        cdlgssm_smoother_type (CDKFSmootherType): CD-Dynamax smoother variant.
            `"cd_smoother_1"` is the default; `"cd_smoother_2"` exposes the
            backend's alternate continuous-discrete RTS implementation.
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        Continuous-discrete Kalman smoothing runs a continuous-time Kalman
        filtering pass between observations, applies discrete Kalman updates at
        observation times, and then integrates the corresponding backward
        smoothing equations through the filtered trajectory.

        The result is the continuous-discrete analogue of the RTS smoother:
        each state is conditioned on all observations \(y_{1:T}\), not just the
        observations available up to that time.

        References:

        - For continuous-discrete smoothing equations, see: Särkkä, S. (2006).
            Recursive Bayesian Inference on Stochastic Differential Equations.
        - For a modern textbook reference, see Chapter 10.6 of:
            Särkkä, S., & Solin, A. (2019). Applied Stochastic Differential Equations.
            Cambridge University Press. [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
        - For general Gaussian smoothing background, see: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """

    cdlgssm_smoother_type: CDKFSmootherType = "cd_smoother_1"


@dataclasses.dataclass
class ContinuousTimeEKFSmootherConfig(ContinuousTimeEKFConfig, SmootherConfig):
    r"""Continuous-discrete Extended Kalman Smoother (CD-EKS).

    Gaussian smoother for mildly nonlinear continuous-time dynamics with
    discrete observations. Requires differentiable dynamics; JAX autodiff is
    used to construct the local linearizations.

    This smoother is currently implemented through the `cd_dynamax` backend.
    See `ContinuousTimeEKFConfig` for linearisation and solver options.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        The CD-EKS combines a continuous-discrete EKF filtering pass with a
        backward smoothing pass through the locally linear Gaussian
        approximations. It is the continuous-discrete analogue of
        `EKFSmootherConfig`.

        The inherited `filter_state_order` and `filter_emission_order` options
        control the Taylor approximation used during the continuous-time
        filtering and smoothing computation.

        References:

        - For continuous-discrete smoothing equations, see: Särkkä, S. (2006).
            Recursive Bayesian Inference on Stochastic Differential Equations.
        - For a modern textbook reference, see Chapter 10.9 of:
            Särkkä, S., & Solin, A. (2019). Applied Stochastic Differential Equations.
            Cambridge University Press. [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
        - For general Gaussian smoothing background, see: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """


DiscreteTimeSmootherConfigs: tuple[type, ...] = (
    KFSmootherConfig,
    EKFSmootherConfig,
    UKFSmootherConfig,
    PFSmootherConfig,
)

ContinuousTimeSmootherConfigs: tuple[type, ...] = (
    ContinuousTimeKFSmootherConfig,
    ContinuousTimeEKFSmootherConfig,
)

__all__ = [
    "CDKFSmootherType",
    "ContinuousTimeEKFSmootherConfig",
    "ContinuousTimeKFSmootherConfig",
    "ContinuousTimeSmootherConfigs",
    "DiscreteTimeSmootherConfigs",
    "EKFSmootherConfig",
    "KFSmootherConfig",
    "PFBackwardSamplingMethod",
    "PFSmootherConfig",
    "SmootherConfig",
    "UKFSmootherConfig",
]
