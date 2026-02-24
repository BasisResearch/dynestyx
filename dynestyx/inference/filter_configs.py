"""Filter configuration dataclasses. Shared by dispatchers and integration backends."""

import dataclasses
import math
from typing import Literal

import jax
import jax.random as jr

ResamplingBaseMethod = Literal["systematic", "multinomial", "stratified"]
ResamplingDifferentiableMethod = Literal["stop_gradient", "straight_through", "soft"]
FilterSource = Literal["cuthbert", "cd_dynamax", "dynestyx"]
FilterEmissionOrder = Literal["zeroth", "first", "second"]
FilterStateOrder = Literal["zeroth", "first", "second"]


@dataclasses.dataclass
class BaseFilterConfig:
    """Shared configuration options inherited by all filter configs.

    You do not instantiate this class directly; use one of the concrete
    subclasses (e.g. `KFConfig`, `PFConfig`).

    The `record_*` fields let you save intermediate filtering outputs into the
    NumPyro trace as `numpyro.deterministic` sites, making them accessible
    after inference (e.g. for plotting filtered trajectories).  `None` defers
    to the backend's default for that quantity.

    Attributes:
        record_filtered_states_mean (bool | None): Save the posterior mean
            \(\mathbb{E}[x_t \mid y_{1:t}]\) at each time step.
        record_filtered_states_cov (bool | None): Save the full posterior
            covariance at each step. Can be large — prefer
            `record_filtered_states_cov_diag` for high-dimensional states.
        record_filtered_states_cov_diag (bool | None): Save only the marginal
            variances (diagonal of the covariance) at each step.
        record_filtered_states_chol_cov (bool | None): Save the Cholesky
            factor of the posterior covariance (Gaussian filters only).
        record_filtered_particles (bool | None): Save the full particle array
            at each step (particle-based filters only).
        record_filtered_log_weights (bool | None): Save the log importance
            weights at each step (particle-based filters only).
        record_max_elems (int): Hard cap on total scalar elements saved across
            all `record_*` sites. Prevents accidentally filling device memory
            for long sequences or large state spaces. Defaults to `100_000`.
        cov_rescaling (float | None): Multiply all predicted covariances by
            this factor before the update. Values slightly above `1.0`
            implement covariance inflation, which can improve robustness when
            the model is misspecified. `None` disables rescaling.
        crn_seed (jax.Array | None): Fix the PRNG key for stochastic filters
            (EnKF, PF). Useful when differentiating through the filter:
            a fixed key makes the randomness a deterministic function of model
            parameters. `None` draws a fresh key each call.
        warn (bool): Whether or not to suppress warnings from filtering backends.
            Defaults to `True`.
        filter_source (FilterSource | None): Internal backend library. Set by
            each subclass; rarely needs to be changed manually.
        extra_filter_kwargs (dict): Extra keyword arguments passed directly to
            the backend. Useful for advanced backend-specific options.
    """

    extra_filter_kwargs: dict = dataclasses.field(default_factory=dict)
    warn: bool = True
    record_filtered_states_mean: bool | None = None
    record_filtered_states_cov: bool | None = None
    record_filtered_states_cov_diag: bool | None = None
    record_filtered_particles: bool | None = None
    record_filtered_log_weights: bool | None = None
    record_filtered_states_chol_cov: bool | None = None
    record_max_elems: int = 100_000
    filter_source: FilterSource | None = None
    cov_rescaling: float | None = None
    crn_seed: jax.Array | None = None


@dataclasses.dataclass
class EnKFConfig(BaseFilterConfig):
    r"""Ensemble Kalman Filter (EnKF) for discrete-time models.

    A good general-purpose filter for nonlinear models. Works with any
    differentiable or non-differentiable dynamics and scales well to moderate
    state dimensions. Cheaper per-step than the particle filter, but assumes
    observations are approximately Gaussian given the ensemble.

    The primary tuning knob is `n_particles`, with more particles providing
    more accurate results at the cost of higher compute.
    If the ensemble collapses over long trajectories, increase
    `inflation_delta` slightly (e.g. `0.05`–`0.2`).

    Attributes:
        n_particles (int): Number of ensemble members. More members give a
            better covariance estimate at higher compute cost. Defaults to
            `30`.
        crn_seed (jax.Array | None): Fixed PRNG key for the ensemble. Defaults
            to `jr.PRNGKey(0)`, i.e., common random numbers are used. This
            can reduce variance in gradient-based learning, but introduces
            further bias.
        perturb_measurements (bool | None): Add noise to observations before
            the ensemble update (stochastic EnKF). Set `False` for the
            square-root variant. `None` defers to the backend default.
        inflation_delta (float | None): Scale ensemble anomalies by
            \(\sqrt{1 + \delta}\) before the update to prevent collapse.
            `None` disables inflation.
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        The ensemble Kalman filter comprises ensemble members $x_t^{(i)}, i = 1, \ldots, N_{\text{particles}}$.
        There are many implementation tricks in the EnKF; we describe the basic version here.

        For each time step \(t\), the ensemble is propagated forward by the transition model:

        $$
            \hat{x}_t^{(i)} = f(x_t^{(i)}, u_t, t_t) + \epsilon_t^{(i)},
        $$

        where \(u_t\) is the control input at time \(t\) and \(t_t\) is the time of the transition,
        and \(\epsilon_t^{(i)} \sim \mathcal{N}(0, Q)\) is the process noise.

        Each ensemble member is then updated using observations:

        $$
            x_t^{(i)} = \hat{x}_t^{(i)} + \hat{K}_t^{(i)} \left(y_t - h(x_t^{(i)}, u_t, t_t)\right),
        $$

        where $\hat{K}_t^{(i)}$ is the Kalman gain for the \(i\)-th ensemble member, computed as

        $$
            \hat{K}_t^{(i)} = \hat{P}_t^{(i)} H^\top (H \hat{P}_t^{(i)} H^\top + R)^{-1},
        $$

        where $\hat{P}_t^{(i)}$ is the empirical covariance of the particles, and $R$ is the
        covariance of the observation model.

        The resulting estimator is known to be biased for non-linear observations, but is often rather
        robust in practice to moderate nonlinearities. It is particualrly effective for high-dimensional
        inverse problems, where other particle methods like particle filters often struggle.

        References:

            - The implementation details are due to: Sanz-Alonso, D., Stuart, A. M., & Taeb, A. (2018).
                Inverse problems and data assimilation. [arXiv:1810.06191](https://arxiv.org/abs/1810.06191).
            - For a classical reference to the ensemble Kalman filter, see: Evensen, G. (2003).
                The ensemble Kalman filter: Theoretical formulation and practical implementation. Ocean Dynamics, 53(4), 343-367.
            - The solution using automatic differentiation for nonlinear dynamics is due to: Chen, Y., Sanz-Alonso, D., & Willett, R. (2022).
                Autodifferentiable ensemble Kalman filters. SIAM Journal on Mathematics of Data Science, 4(2), 801-833.
                [Available Online](https://epubs.siam.org/doi/abs/10.1137/21M1434477).
    """

    n_particles: int = 30
    crn_seed: jax.Array | None = dataclasses.field(
        default_factory=lambda: jr.PRNGKey(0)
    )
    perturb_measurements: bool | None = None
    inflation_delta: float | None = None
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class PFResamplingConfig:
    """Resampling strategy for particle-based filters.

    The defaults (`"systematic"`, `"stop_gradient"`) are appropriate for most
    workflows. The resampling step includes both a non-differentiable base method,
    and a differentiable method for handling gradients.

    For most problems, especially with bootstrap particle filters, the `stop_gradient` method is preferred.
    The `soft` method can be useful when a non-bootstrap PF is used and gradients are required to flow through the resampling step.
    The `straight_through` approach is biased on the backwards pass, and is not recommended, but is okay to use for gradient-free inference.

    Attributes:
        base_method (ResamplingBaseMethod): Algorithm used to draw new
            particles from the weight distribution. `"systematic"` *(default)*
            has the lowest variance and is preferred in most cases.
            `"multinomial"` and `"stratified"` are also available.
        differential_method (ResamplingDifferentiableMethod): How gradients
            are handled across the discrete resampling step.
            `"stop_gradient"` *(default)* treats indices as constants.
            `"straight_through"` or `"soft"` allow gradients to pass through
            (needed for gradient-based training of model parameters).
        softness (float): Temperature for soft resampling. Only used when
            `differential_method="soft"`. Lower values are closer to non-differentiable resampling,
            and higher values are closer to the differentiable method. Defaults to `0.7`.

    ??? note "Algorithm Reference"
        The stop gradient method provides an unbiased score estimate for the marginal likelihood via the classical Fisher estimate.
        This is accomplished directly through automatic differentiation. For non-bootstrap filters, the proposal estimates will be biased.

        The soft resampling method provides biased score estimates, but propagates gradients through the reasmpling step. It is fast.

        References:

            - For the stop_gradient method, see: Ścibior, A., & Wood, F. (2021).
                Differentiable particle filtering without modifying the forward pass. [arXiv:2106.10314](https://arxiv.org/abs/2106.10314).
            - For the soft method, see: Karkus, P., Hsu, D., & Lee, W. S. (2018, October). Particle filter networks with application to visual localization.
                In Conference on Robot Learning (pp. 169-178). [Available Online](https://proceedings.mlr.press/v87/karkus18a.html).
            - For a recent review of differentiable particle filters, see: Brady, J. J., Cox, B., Li, Y., & Elvira, V. (2025).
                PyDPF: A Python Package for Differentiable Particle Filtering. [arXiv:2510.25693](https://arxiv.org/abs/2510.25693).
    """

    base_method: ResamplingBaseMethod = "systematic"
    differential_method: ResamplingDifferentiableMethod = "stop_gradient"
    softness: float = 0.7


@dataclasses.dataclass
class PFConfig(BaseFilterConfig):
    r"""Bootstrap Particle Filter (PF) for discrete-time models.

    The most flexible filter: works with any model, including non-Gaussian
    observations and highly nonlinear dynamics.
    The main cost is that accuracy scales with the number of particles, so
    large state dimensions can become expensive.

    The primary tuning knob is `n_particles`. Estimates will generally get better
    and less noisy with more particles, but introduces a linear computational cost.
    `ess_threshold_ratio` controls the frequency of resampling; sampling more frequently
    can help avoid particle degeneracy, but also increases variance.

    Attributes:
        n_particles (int): Number of particles. More particles give a lower-
            variance log-likelihood estimate at linear compute cost. Defaults
            to `1_000`.
        resampling_method (PFResamplingConfig): Controls the resampling
            algorithm and gradient behaviour. See `PFResamplingConfig`.
            Defaults to systematic resampling with stop-gradient.
        ess_threshold_ratio (float): Resampling fires when the effective
            sample size drops below `ess_threshold_ratio * n_particles`.
            `1.0` → always resample; `0.0` → never. Defaults to `0.7`.
        filter_source (FilterSource): Backend. Defaults to `"cuthbert"`, which is currently the only available implementation.

    ??? note "Algorithm Reference"
        At each step, particles are propagated through the transition and
        reweighted by the observation likelihood. The resulting empirical distribution
        is asymptotically exact to the true filtering distribution as the number of particles goes to infinity.
        The marginal log-likelihood without a resampling step is estimated as:

        \[
            \log p(y_{1:T}) \approx \sum_{t=1}^T \log \frac{1}{N}
            \sum_{i=1}^N \tilde{w}_t^{(i)}
        \]

        where $\tilde{w}_t^{(i)}$ is the are unnormalized weights of each particle.

        There are several different resampling algorithms available, which result in different
        approximations of the score function $\nabla_\theta \log p(y_{1:T} | \theta)$.
        For more information on these options, see `PFResamplingConfig`.

        References:

        - For a classical reference to particle filters, see: Doucet, A., De Freitas, N., & Gordon, N. (2001). An Introduction to Sequential Monte Carlo Methods.
            In Sequential Monte Carlo Methods in Practice (pp. 3-14). New York, NY: Springer New York.
        - For a more modern textbook, see Chapter 11.4 of: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
        - For a more recent review of differentiable particle filters, see: Brady, J. J., Cox, B., Li, Y., & Elvira, V. (2025).
            PyDPF: A Python Package for Differentiable Particle Filtering. [arXiv:2510.25693](https://arxiv.org/abs/2510.25693).
    """

    n_particles: int = 1_000
    resampling_method: PFResamplingConfig = dataclasses.field(
        default_factory=PFResamplingConfig
    )
    ess_threshold_ratio: float = 0.7
    filter_source: FilterSource = "cuthbert"


@dataclasses.dataclass
class EKFConfig(BaseFilterConfig):
    """Extended Kalman Filter (EKF) for discrete-time models.

    The EKF linearizes nonlinear dynamics at the current mean estimate
    via a first-order Taylor expansion. It is fast and simple, but may
    not work well for strongly nonlinear models. The Taylor series expansion
    is automatically performed via Jax autodiff.

    This is exact (but wasteful) for linear-Gaussian models.

    This is the **default discrete-time filter** when no `filter_config` is
    passed to `Filter`.

    Attributes:
        filter_emission_order (FilterEmissionOrder): Linearisation order for
            the observation function. `"first"` *(default)* is the standard
            Jacobian-based EKF. `"second"` reduces bias for strongly curved
            observation maps at the cost of Hessian computation. `"zeroth"`
            skips observation linearisation.
        filter_source (FilterSource): Backend. Defaults to `"cuthbert"`.

    ??? note "Algorithm Reference"
        The EKF propagates a Gaussian approximation
        \(\mathcal{N}(\hat x_{t|t}, P_{t|t})\) through Jacobian
        linearizations of \(f\) and \(h\):

        \[
            \hat x_{t|t-1} = f(\hat x_{t-1|t-1}),
            \quad P_{t|t-1} = F_t P_{t-1|t-1} F_t^\top + Q_t
        \]

        where $F_t$ is the Jacobian of $f$ at $\hat x_{t|t-1}$,
        and proceeds via the typical Kalman update.

        References:

        - The `cuthbert` implementation of the EKF is based on the `taylor_kf` module therein.
            See the [cuthbert documentation](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/taylor/) for more information.
        - For a more modern textbook reference, see Chapter 7 of: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """

    filter_source: FilterSource = "cuthbert"
    filter_emission_order: FilterEmissionOrder = "first"


@dataclasses.dataclass
class KFConfig(BaseFilterConfig):
    r"""Kalman Filter (KF) for discrete-time linear-Gaussian models.

    The exact Bayesian filter for linear-Gaussian state-space models; requires
    a model built with `LTI_discrete` or using
    `LinearGaussianStateEvolution` + `LinearGaussianObservation`. For
    nonlinear Gaussian models, use `EKFConfig`, `UKFConfig`, or `EnKFConfig` instead.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        When the dynamics and observation process of a dynamical system are both linear-Gaussian,
        the recursive updates can be computed in closed form.

        This proceeds via a "prediction" step, where the mean and covariance are propagated forward in time,
        and an "update" step, where the mean and covariance are updated with the observation.

        The prediction step is given by:

        $$
            \hat x_{t|t-1} = A \hat x_{t-1|t-1} + b,
            \quad P_{t|t-1} = A P_{t-1|t-1} A^\top + Q
        $$

        The update step is given by:

        $$
            \hat x_{t|t} = \hat x_{t|t-1} + K_t (y_t - H \hat x_{t|t-1}),
            \quad P_{t|t} = (I - K_t H) P_{t|t-1}
        $$

        where $K_t$ is the Kalman gain.

        The Kalman gain is given by:

        $$
            K_t = P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1}
        $$

        where $H$ is the Jacobian of $h$ at $\hat x_{t|t-1}$.

        There are variants to the particular algorithm; the `cuthbert` implementation is the so-called "square root" form.
        This provides a more numerically stable implementation of the Kalman filter.

        References:

        - For the classsical reference, see: Kalman, R. E. (1960).
            A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering, 82(1), 35-45.
        - For a more modern textbook reference, see Chapter 6 of: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
        - For more details on the `cuthbert` implementation, see the [cuthbert documentation](https://state-space-models.github.io/cuthbert/cuthbert_api/gaussian/kalman/).
    """

    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class UKFConfig(BaseFilterConfig):
    r"""Unscented Kalman Filter (UKF) for discrete-time models.

    A derivative-free Gaussian filter that handles stronger nonlinearities
    than the EKF by propagating a small, deterministic set of *sigma points*
    through the dynamics. No Jacobians are computed. Slightly more expensive
    than the EKF but often more accurate on curved manifolds.

    The default parameters (`alpha`, `beta`, `kappa`) work well for most
    problems; they rarely need to be changed.

    Attributes:
        alpha (float): Spread of sigma points around the current mean.
            Smaller → tighter cluster; larger → sigma points reach further.
            Defaults to \(\sqrt{3}\).
        beta (int): Encodes prior knowledge about the distribution shape.
            `2` is optimal for Gaussians. Defaults to `2`.
        kappa (int): Secondary scaling parameter. Defaults to `1`.
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        For a state of dimension \(n\), \(2n+1\) sigma points are placed as:

        $$
            \mathcal{X}_0 = \hat x, \quad
            \mathcal{X}_i = \hat x \pm \sqrt{(n + \lambda) P}_i, \quad
            \lambda = \alpha^2 (n + \kappa) - n
        $$

        Each sigma point is propagated through \(f\) and \(h\); the outputs
        are recombined with weights depending on \(\alpha, \beta, \kappa\) to
        recover the predicted mean and covariance.

        References:
        - For the original paper, see: Julier, S. J., & Uhlmann, J. K. (1997). New extension of the Kalman filter to nonlinear systems. SPIE Proceedings, 3068.
        - For a more modern textbook reference, see Section 8.8 of: Särkkä, S., & Svensson, L. (2023).
            Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press.
            [Available Online](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
    """

    alpha: float = math.sqrt(3)
    beta: int = 2
    kappa: int = 1
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeConfig:
    """Solver options shared by all continuous-discrete filter configs.

    Between observation times, the filter propagates a distribution (or
    ensemble/particles) forward in continuous time by solving an ODE/SDE
    numerically. These options control that solver.

    Attributes:
        filter_state_order (FilterStateOrder): Accuracy of the continuous-time
            propagation between observations. `"first"` *(default)* propagates
            both mean and covariance (sufficient for most problems). `"zeroth"`
            propagates only the mean (faster, less accurate). `"second"` adds
            a higher-order correction for strongly nonlinear drifts.
        diffeqsolve_max_steps (int): Maximum ODE solver steps between any two
            consecutive observations. Increase if the solver hits this limit
            (stiff dynamics or very long inter-observation gaps). Defaults to
            `1_000`.
        diffeqsolve_dt0 (float): Initial step-size hint for the solver.
            Adaptive solvers adjust this automatically; fixed-step solvers
            use it as the constant step. Defaults to `0.01`.
        diffeqsolve_kwargs (dict): Additional kwargs forwarded to
            `diffrax.diffeqsolve` (e.g. `solver`, `stepsize_controller`,
            `adjoint`). Defaults to `{}`.
    """

    filter_state_order: FilterStateOrder = "first"
    diffeqsolve_max_steps: int = 1_000
    diffeqsolve_dt0: float = 0.01
    diffeqsolve_kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ContinuousTimeKFConfig(BaseFilterConfig, ContinuousTimeConfig):
    r"""Continuous-discrete Kalman Filter (CD-KF).

    The exact Bayesian filter for continuous-time linear-Gaussian models.
    Use this when your model was built with `LTI_continuous`. For nonlinear
    SDEs, use `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, or
    `ContinuousTimeEnKFConfig`.

    Inherits solver options from `ContinuousTimeConfig` and recording
    options from `BaseFilterConfig`.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        Between observations the mean and covariance evolve via the
        Kalman–Bucy ODEs:

        $$
            \dot{\hat x} = A \hat x + B u,
            \quad \dot P = A P + P A^\top + L Q L^\top
        $$

        At each observation the standard Kalman update is applied.

        References:

        - For a modern textbook reference, see Chapter 10.6 of: Särkkä, S., & Solin, A. (2019).
            Applied Stochastic Differential Equations. Cambridge University Press.
            [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
    """

    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeEnKFConfig(EnKFConfig, ContinuousTimeConfig):
    r"""Continuous-discrete Ensemble Kalman Filter (CD-EnKF).

    The **default filter** for continuous-time models. Each ensemble member
    is propagated forward by solving the SDE between observations; the
    ensemble Kalman update is applied at observation times. Works with any
    SDE model without requiring gradients.

    See `EnKFConfig` for particle/ensemble tuning options and
    `ContinuousTimeConfig` for solver options.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        References:

        - The implementation details are due to: Sanz-Alonso, D., Stuart, A. M., & Taeb, A. (2018).
            Inverse problems and data assimilation. [arXiv:1810.06191](https://arxiv.org/abs/1810.06191).
        - For a classical reference to the ensemble Kalman filter, see: Evensen, G. (2003).
            The ensemble Kalman filter: Theoretical formulation and practical implementation. Ocean Dynamics, 53(4), 343-367.
        - The solution using automatic differentiation for nonlinear dynamics is due to: Chen, Y., Sanz-Alonso, D., & Willett, R. (2022).
            Autodifferentiable ensemble Kalman filters. SIAM Journal on Mathematics of Data Science, 4(2), 801-833.
            [Available Online](https://epubs.siam.org/doi/abs/10.1137/21M1434477).
    """

    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeDPFConfig(PFConfig, ContinuousTimeConfig):
    r"""Continuous-discrete Differentiable Particle Filter (CD-DPF).

    Particle filter for continuous-time SDEs. Particles are propagated by
    solving the SDE between observations; importance weights are updated at
    each observation time. Supports non-Gaussian observations and arbitrary
    nonlinear dynamics.

    Uses multinomial resampling by default (vs. systematic in `PFConfig`)
    for better compatibility with gradient-based training.

    See `PFConfig` for particle tuning options and `ContinuousTimeConfig`
    for solver options.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.
        resampling_method (PFResamplingConfig): Defaults to
            `PFResamplingConfig(base_method="multinomial")`.

    ??? note "Algorithm Reference"
        The bootstrap version of the continuous-discrete PF is the same as the discrete-time PF version.
        See `PFConfig` for more information.
    """

    filter_source: FilterSource = "cd_dynamax"
    resampling_method: PFResamplingConfig = dataclasses.field(
        default_factory=lambda: PFResamplingConfig(base_method="multinomial")
    )


@dataclasses.dataclass
class ContinuousTimeEKFConfig(EKFConfig, ContinuousTimeConfig):
    r"""Continuous-discrete Extended Kalman Filter (CD-EKF).

    Fast Gaussian filter for mildly nonlinear SDEs. Requires differentiable
    dynamics (JAX autodiff is used). The moment equations for the Gaussian
    approximation are solved between observations and a Kalman update is
    applied at each observation.

    See `EKFConfig` for linearisation options and `ContinuousTimeConfig`
    for solver options.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        References:

        - For a modern textbook reference, see Chapter 10.7 of: Särkkä, S., & Solin, A. (2019).
            Applied Stochastic Differential Equations. Cambridge University Press.
            [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
    """

    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeUKFConfig(UKFConfig, ContinuousTimeConfig):
    """Continuous-discrete Unscented Kalman Filter (CD-UKF).

    Derivative-free Gaussian filter for nonlinear SDEs. Sigma points are
    propagated through the SDE between observations; the unscented transform
    is applied at each observation update. More accurate than CD-EKF for
    strongly nonlinear drifts without requiring Jacobians.

    See `UKFConfig` for sigma-point tuning options and `ContinuousTimeConfig`
    for solver options.

    Attributes:
        filter_source (FilterSource): Backend. Defaults to `"cd_dynamax"`.

    ??? note "Algorithm Reference"
        References:

        - For a modern textbook reference, see Section 10.8 of: Särkkä, S., & Solin, A. (2019).
            Applied Stochastic Differential Equations. Cambridge University Press.
            [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
    """

    filter_source: FilterSource = "cd_dynamax"


DiscreteTimeConfigs: tuple[type, ...] = (
    EnKFConfig,
    PFConfig,
    EKFConfig,
    KFConfig,
    UKFConfig,
)

ContinuousTimeConfigs: tuple[type, ...] = (
    ContinuousTimeKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeUKFConfig,
)


@dataclasses.dataclass
class HMMConfig(BaseFilterConfig):
    """Exact filter for Hidden Markov Models (finite discrete state space).

    Use this when your latent state takes values in a finite set (e.g. a
    discrete regime model). The forward algorithm computes the exact marginal
    log-likelihood and filtered belief over states at each time step.

    For continuous latent-state models, use any other filter config.

    Attributes:
        record_filtered (bool | None): Save the filtered state probabilities
            \(p(z_t \mid y_{1:t})\) as a deterministic site. `None` defers to
            the backend default.
        record_log_filtered (bool | None): Save the log of the filtered state
            probabilities as a deterministic site. `None` defers to the
            backend default.
        filter_source (FilterSource): Backend. Defaults to `"dynestyx"`.

    ??? note "Algorithm Reference"
        At each step the belief vector \(\pi_t = p(z_t \mid y_{1:t})\) is
        updated exactly:

        \[
            \pi_t^{\text{pred}} = T^\top \pi_{t-1}, \quad
            \pi_t \propto \ell(y_t \mid z_t) \, \pi_t^{\text{pred}}
        \]

        The log-likelihood is the sum of the log normalisation constants.
    """

    record_filtered: bool | None = None
    record_log_filtered: bool | None = None
    filter_source: FilterSource = "dynestyx"


HMMConfigs: tuple[type, ...] = (HMMConfig,)


def _config_to_record_kwargs(config: BaseFilterConfig) -> dict:
    """Build record_kwargs dict from config. Config must have all record_* fields."""
    if isinstance(config, HMMConfig):
        return {
            "record_filtered": config.record_filtered,
            "record_log_filtered": config.record_log_filtered,
            "record_max_elems": config.record_max_elems,
        }
    else:
        return {
            "record_filtered_states_mean": config.record_filtered_states_mean,
            "record_filtered_states_cov": config.record_filtered_states_cov,
            "record_filtered_states_cov_diag": config.record_filtered_states_cov_diag,
            "record_filtered_particles": config.record_filtered_particles,
            "record_filtered_log_weights": config.record_filtered_log_weights,
            "record_filtered_states_chol_cov": config.record_filtered_states_chol_cov,
            "record_max_elems": config.record_max_elems,
        }
