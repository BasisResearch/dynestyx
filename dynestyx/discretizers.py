import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    GaussianStateEvolution,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.solvers import (
    euler_maruyama_loc_cov,
    frozen_jacobian_gaussian_loc_cov,
    simulated_likelihood_components,
    taylor_moment_loc_cov,
)
from dynestyx.types import FunctionOfTime


class EulerMaruyamaGaussianStateEvolution(GaussianStateEvolution):
    """`GaussianStateEvolution` backed by Euler-Maruyama moments."""

    cte: StochasticContinuousTimeStateEvolution

    def __init__(
        self,
        cte: StochasticContinuousTimeStateEvolution,
        F=None,
        cov=None,
    ):
        # Accept these for reconstruction paths, but derive both from `cte`.
        self.cte = cte

        def _loc(x, u, t_now, t_next):
            return euler_maruyama_loc_cov(cte, x, u, t_now, t_next)["loc"]

        def _cov(x, u, t_now, t_next):
            return euler_maruyama_loc_cov(cte, x, u, t_now, t_next)["cov"]

        super().__init__(
            F=_loc,
            cov=_cov,
        )

    def __call__(self, x, u, t_now, t_next):
        """Single-pass transition step (or batched time steps)."""
        em_result = euler_maruyama_loc_cov(self.cte, x, u, t_now, t_next)
        return dist.MultivariateNormal(
            loc=em_result["loc"], covariance_matrix=em_result["cov"]
        )


def euler_maruyama(
    cte: StochasticContinuousTimeStateEvolution,
) -> GaussianStateEvolution:
    """Discretize continuous-time state evolution via Euler-Maruyama.

    Euler-Maruyama is a first-order discrete approximation of a continuous-time
    SDE. The result is a `GaussianStateEvolution` with mean
    `x + drift(x,u,t)*dt` and covariance `(L@Q@L.T)*dt` (`Q = I`),
    where `dt = t_next - t_now`. The process covariance is **time-varying**
    (depends on `t_next - t_now`) and passed as a callable `cov`.

    Args:
        cte: `StochasticContinuousTimeStateEvolution` to discretize.
    Returns:
        GaussianStateEvolution: Discrete-time Gaussian transition with the
        same Euler–Maruyama semantics as before this refactor.

    Note:
        Each transition uses one Euler-Maruyama step with
        `dt = t_next - t_now`.

    ??? note "Algorithm Reference"
        The Euler Maruyama is a first order discretization.
        The resulting discrete-time state evolution is approximated as

        x_{t+1} ~ N(x_t + drift * delta_t, (L@Q@L.T)*delta_t)

        where:
            x_t is the current state
            drift is the drift function
            L is the diffusion coefficient
            Q is the diffusion covariance
            delta_t is the time step between timepoints (t_next - t_now)

        This is the first-order Ito-Taylor approximation.

        References:
            - This is the first-order Ito-Taylor approximation, discussed in Chapter 9.2 of: Särkkä, S., & Solin, A. (2019).
                Applied Stochastic Differential Equations. Cambridge University Press.
                [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
    """

    return EulerMaruyamaGaussianStateEvolution(cte)


class FrozenJacobianGaussianStateEvolution(GaussianStateEvolution):
    """`GaussianStateEvolution` backed by frozen-Jacobian affine moments."""

    cte: ContinuousTimeStateEvolution
    jitter: float

    def __init__(
        self,
        cte: ContinuousTimeStateEvolution,
        jitter: float = 1e-8,
        F=None,
        cov=None,
    ):
        del F, cov
        self.cte = cte
        self.jitter = float(jitter)
        super().__init__(
            F=lambda x, u, t_now, t_next: frozen_jacobian_gaussian_loc_cov(
                cte, x, u, t_now, t_next, jitter=self.jitter
            )["loc"],
            cov=lambda x, u, t_now, t_next: frozen_jacobian_gaussian_loc_cov(
                cte, x, u, t_now, t_next, jitter=self.jitter
            )["cov"],
        )

    def __call__(self, x, u, t_now, t_next):
        """Single-pass transition step (or batched time steps)."""
        result = frozen_jacobian_gaussian_loc_cov(
            self.cte, x, u, t_now, t_next, jitter=self.jitter
        )
        return dist.MultivariateNormal(
            loc=result["loc"], covariance_matrix=result["cov"]
        )


def frozen_jacobian_gaussian(
    cte: ContinuousTimeStateEvolution,
    *,
    jitter: float = 1e-8,
) -> GaussianStateEvolution:
    r"""Discretize an SDE by freezing its current Jacobian and diffusion.

    At state \(x\), control \(u\), and time \(t_k\), define
    \(f_0=f(x,u,t_k)\), \(F_0=\partial f/\partial x\vert_{(x,u,t_k)}\),
    \(L_0=L(x,u,t_k)\), and \(h=t_{k+1}-t_k\). Dynestyx replaces the
    nonlinear SDE over this step by the affine Itô SDE

    \[
        dZ_s = \{F_0 Z_s + b_0\}\,ds + L_0\,dW_s,\qquad
        Z_0=x,\qquad b_0=f_0-F_0x,
    \]

    equivalently \(d\delta_s=(f_0+F_0\delta_s)\,ds+L_0dW_s\) with
    \(\delta_0=0\). The returned transition is the exact Gaussian transition
    of this affine SDE under the Dynestyx convention that \(W_s\) is standard
    Brownian motion.

    Writing \(A=\exp(F_0h)\) and \(a=L_0L_0^\top\), the transition moments are

    \[
        m = x + \int_0^h \exp(F_0(h-\tau)) f_0\,d\tau,\qquad
        P = \int_0^h \exp(F_0(h-\tau)) a
            \exp(F_0(h-\tau))^\top d\tau.
    \]

    The mean integral is computed as the upper-right block of
    \[
        \exp\left[
            h\begin{pmatrix}F_0&f_0\\0&0\end{pmatrix}
        \right].
    \]
    The covariance uses the matrix-fraction exponential
    \[
        \exp\left[
            \begin{pmatrix}F_0&a\\0&-F_0^\top\end{pmatrix}h
        \right]
        =
        \begin{pmatrix}A&P A^{-\top}\\0&A^{-\top}\end{pmatrix},
    \]
    so \(P\) is recovered as the upper-right block times \(A^\top\).

    This discretizer is exact when the drift is affine and the diffusion is
    constant over the step. Otherwise it is a current-state first-order drift
    approximation with frozen diffusion. It is useful for drift fields with
    strong contraction, growth, or rotation, but still requires a step size
    small enough that the frozen coefficients are representative.

    Args:
        cte: Continuous-time state evolution to discretize.
        jitter: Minimum eigenvalue used when projecting the symmetrized
            process covariance onto the positive semidefinite cone.

    Returns:
        GaussianStateEvolution: Discrete-time Gaussian transition.

    ??? note "Algorithm Reference"
        The exact linear-Gaussian transition is Särkkä & Solin (2019),
        Chapter 6, Section 6.2, Equations (6.24)--(6.26). The affine
        constant-input form is Remark 6.7, Equations (6.51)--(6.53). The
        covariance matrix exponential is Chapter 6, Section 6.3,
        Equations (6.40)--(6.42), with \(Q=I\) because Dynestyx absorbs the
        diffusion scale into \(L_0\).

        Särkkä, S., & Solin, A. (2019). Applied Stochastic Differential
        Equations. Cambridge University Press.
        [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
    """

    return FrozenJacobianGaussianStateEvolution(cte, jitter=jitter)


class TaylorMomentGaussianStateEvolution(GaussianStateEvolution):
    """`GaussianStateEvolution` backed by second-order generator moments."""

    cte: ContinuousTimeStateEvolution
    jitter: float

    def __init__(
        self,
        cte: ContinuousTimeStateEvolution,
        jitter: float = 1e-8,
        F=None,
        cov=None,
    ):
        del F, cov
        self.cte = cte
        self.jitter = float(jitter)
        super().__init__(
            F=lambda x, u, t_now, t_next: taylor_moment_loc_cov(
                cte, x, u, t_now, t_next, jitter=self.jitter
            )["loc"],
            cov=lambda x, u, t_now, t_next: taylor_moment_loc_cov(
                cte, x, u, t_now, t_next, jitter=self.jitter
            )["cov"],
        )

    def __call__(self, x, u, t_now, t_next):
        """Single-pass transition step (or batched time steps)."""
        result = taylor_moment_loc_cov(
            self.cte, x, u, t_now, t_next, jitter=self.jitter
        )
        return dist.MultivariateNormal(
            loc=result["loc"], covariance_matrix=result["cov"]
        )


def taylor_moment_gaussian(
    cte: ContinuousTimeStateEvolution,
    *,
    jitter: float = 1e-8,
) -> GaussianStateEvolution:
    r"""Discretize an SDE by a second-order generator moment expansion.

    This is a weak transition-density approximation, not a strong pathwise
    Itô-Taylor simulator. It applies the frozen-time generator expansion to
    the first and second noncentral moments, then fits a Gaussian transition.
    It is useful when filtering, smoothing, or likelihood evaluation needs a
    better one-step Gaussian approximation than Euler-Maruyama without adding
    a new filtering backend.

    For a scalar test function \(\phi\), the frozen-time approximation is

    \[
        E[\phi(X_{t+h}) \mid X_t=x]
        \approx \phi(x) + A\phi(x)h + \tfrac12 A^2\phi(x)h^2,
    \]

    where \(A\phi = f^\top\nabla\phi +
    \tfrac12\operatorname{tr}(L L^\top\nabla^2\phi)\). Dynestyx applies this
    to \(\phi_i(x)=x_i\) and \(\phi_{ij}(x)=x_i x_j\), forms
    \(P=E[XX^\top]-mm^\top\), symmetrizes it, and adds covariance jitter.

    The approximation has better weak local accuracy than Euler-Maruyama for
    smooth coefficients and can improve approximate likelihoods and Gaussian
    filters. It uses automatic differentiation through drift and diffusion, so
    it is best suited to smooth, low-to-moderate-dimensional models. Time and
    controls are held fixed over the step, matching the existing discretizer
    convention; reduce the step size for strongly time-varying coefficients.

    Args:
        cte: Continuous-time state evolution to discretize.
        jitter: Minimum eigenvalue used when projecting the symmetrized
            process covariance onto the positive semidefinite cone.

    Returns:
        GaussianStateEvolution: Discrete-time Gaussian transition.

    ??? note "Algorithm Reference"
        This follows the Taylor expansion of conditional moments in
        Chapter 9.4, Algorithm 9.15 of:
        Särkkä, S., & Solin, A. (2019). Applied Stochastic Differential
        Equations. Cambridge University Press.
        [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
        The final Gaussian fit is the transition-density use described around
        Algorithms 9.8--9.9.
    """

    return TaylorMomentGaussianStateEvolution(cte, jitter=jitter)


class SimulatedLikelihoodStateEvolution(DiscreteTimeStateEvolution):
    """Discrete transition backed by Pedersen simulated likelihood."""

    cte: ContinuousTimeStateEvolution
    n_substeps: int
    n_simulations: int
    seed: int
    jitter: float
    standard_normals: Array

    def __init__(
        self,
        cte: ContinuousTimeStateEvolution,
        *,
        n_substeps: int = 4,
        n_simulations: int = 32,
        seed: int = 0,
        jitter: float = 1e-8,
        standard_normals: Array | None = None,
    ):
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1.")
        if n_simulations < 1:
            raise ValueError("n_simulations must be >= 1.")
        if cte.bm_dim is None:
            raise ValueError(
                "simulated_likelihood requires cte.bm_dim to be set or inferred."
            )
        self.cte = cte
        self.n_substeps = int(n_substeps)
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)
        self.jitter = float(jitter)
        n_em_steps = max(self.n_substeps - 1, 1)
        if standard_normals is None:
            standard_normals = jr.normal(
                jr.PRNGKey(self.seed),
                (self.n_simulations, n_em_steps, int(cte.bm_dim)),
            )
        self.standard_normals = standard_normals

    def __call__(self, x, u, t_now, t_next):
        result = simulated_likelihood_components(
            self.cte,
            x,
            u,
            t_now,
            t_next,
            n_substeps=self.n_substeps,
            standard_normals=self.standard_normals,
            jitter=self.jitter,
        )
        logits = jnp.zeros(result["loc"].shape[:-1], dtype=result["loc"].dtype)
        mixing = dist.Categorical(logits=logits)
        components = dist.MultivariateNormal(
            loc=result["loc"], covariance_matrix=result["cov"]
        )
        return dist.MixtureSameFamily(mixing, components)


def simulated_likelihood(
    cte: ContinuousTimeStateEvolution,
    *,
    n_substeps: int = 4,
    n_simulations: int = 32,
    seed: int = 0,
    jitter: float = 1e-8,
) -> DiscreteTimeStateEvolution:
    r"""Discretize an SDE with Pedersen's simulated-likelihood mixture.

    This method targets transition-density approximation for particle filters,
    simulators, and likelihood-based workflows where a non-Gaussian transition
    is acceptable. It divides \([t_k,t_{k+1}]\) into `n_substeps`, simulates
    `n_simulations` Euler-Maruyama paths only to the penultimate substep, and
    then averages the final Euler-Maruyama Gaussian kernel. The resulting
    transition is a uniform Gaussian mixture.

    With substep \(h=(t_{k+1}-t_k)/M\), simulated penultimate states
    \(\hat x^{(n)}_{M-1}\), and frozen controls, Dynestyx returns

    \[
        p(x_{k+1}\mid x_k) \approx \frac1N \sum_{n=1}^N
        N\left(x_{k+1}; \hat x^{(n)}_{M-1}
        + f(\hat x^{(n)}_{M-1})h,
        L(\hat x^{(n)}_{M-1})L(\hat x^{(n)}_{M-1})^\top h\right).
    \]

    The estimator converges to the true transition density as the number of
    simulations and substeps increase under the regularity assumptions of the
    simulated-likelihood method. It is more expressive than a single Gaussian
    but more expensive and stochastic; Dynestyx uses common random numbers
    from `seed` so repeated model evaluations are deterministic for inference.
    Because the transition is a mixture, prefer particle-filter/simulator
    workflows over Kalman-only backends.

    Args:
        cte: Continuous-time state evolution to discretize.
        n_substeps: Number of Euler-Maruyama subintervals over each transition.
        n_simulations: Number of simulated penultimate paths in the mixture.
        seed: Seed for common random numbers used by the simulated paths.
        jitter: Minimum eigenvalue used when projecting each symmetrized
            Gaussian component covariance onto the positive semidefinite cone.

    Returns:
        DiscreteTimeStateEvolution: Discrete transition returning
        `numpyro.distributions.MixtureSameFamily`.

    ??? note "Algorithm Reference"
        This implements the simulated likelihood methodology in Chapter 9.7,
        Algorithm 9.21 of:
        Särkkä, S., & Solin, A. (2019). Applied Stochastic Differential
        Equations. Cambridge University Press.
        [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
        The method is originally associated with Pedersen (1995) and related
        simulated-likelihood work by Brandt and Santa-Clara (2002).
    """

    return SimulatedLikelihoodStateEvolution(
        cte,
        n_substeps=n_substeps,
        n_simulations=n_simulations,
        seed=seed,
        jitter=jitter,
    )


class Discretizer(ObjectInterpretation, HandlesSelf):
    """
    Performs discretization of a continuous-time state evolution, converting it to a discrete-time state evolution.

    A `Discretizer` object should be used as a context manager around a call to a model with a `dsx.sample(...)`
    statement to discretize a continuous-time state evolution to a discrete-time state evolution. The `Discretizer`
    should be at a lower (i.e. inner) level in the current context stack than any inference (e.g., `Filter` or `Simulator`)
    objects.

    ??? example "Using a Euler Maruyama Discretizer"
        ```python
        import dynestyx as dsx
        from dynestyx.discretizers import Discretizer, euler_maruyama
        from dynestyx.inference.filters import Filter, EKFConfig
        from dynestyx.models import (
            ContinuousTimeStateEvolution,
            DiscreteTimeStateEvolution,
            DynamicalModel,
            FullDiffusion,
        )

        def model_with_ctse(obs_times=None, obs_values=None):
            dynamics = DynamicalModel(
                control_dim=0,
                initial_condition=dist.MultivariateNormal(
                    loc=jnp.zeros(state_dim),
                    covariance_matrix=jnp.eye(state_dim),
                ),
                state_evolution=ContinuousTimeStateEvolution(
                    drift=lambda x, u, t: x,
                    diffusion=FullDiffusion(
                        lambda x, u, t: jnp.eye(state_dim, bm_dim)
                    ),
                ),
                observation_model=lambda x, u, t: dist.MultivariateNormal(
                    x,
                    0.1**2 * jnp.eye(observation_dim),
                ),
            )
            return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

        def discretized_data_conditioned_model():
            # We use a discrete-time filter now
            with Filter(filter_config=EKFConfig()):
                with Discretizer(discretize=euler_maruyama):
                    return model_with_ctse(obs_times=obs_times, obs_values=obs_values)
        ```

    ??? note "Algorithm Reference"
        For an overview of discretization methods for SDEs, see Chapter 9 of: Särkkä, S., & Solin, A. (2019).
        Applied Stochastic Differential Equations. Cambridge University Press.
        [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).

    Attributes:
        discretize: A callable that converts a continuous-time state evolution to a discrete-time state evolution. Defaults to euler_maruyama.
    """

    def __init__(self, discretize=euler_maruyama):
        super().__init__()
        self.discretize = discretize

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> FunctionOfTime:
        if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
            discrete_evolution = self.discretize(dynamics.state_evolution)
            dynamics = DynamicalModel(
                initial_condition=dynamics.initial_condition,
                state_evolution=discrete_evolution,
                observation_model=dynamics.observation_model,
                control_model=dynamics.control_model,
                control_dim=dynamics.control_dim,
                t0=dynamics.t0,
            )
        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
