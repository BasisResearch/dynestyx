import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.models.observations import LinearGaussianObservation
from dynestyx.models.state_evolution import AffineDrift, LinearGaussianStateEvolution


def LTI_discrete(
    A: jax.Array,
    Q: jax.Array,
    H: jax.Array,
    R: jax.Array,
    B: jax.Array | None = None,
    b: jax.Array | None = None,
    D: jax.Array | None = None,
    d: jax.Array | None = None,
    initial_mean: jax.Array | None = None,
    initial_cov: jax.Array | None = None,
) -> DynamicalModel:
    """
    Build a discrete-time linear time-invariant (LTI) `DynamicalModel`.

    The model has transition and observation distributions

    $$
    \\begin{aligned}
    x_0 &\\sim \\mathcal{N}(m_0, C_0), \\\\
    x_{t_{k+1}} &\\sim \\mathcal{N}(A x_{t_k} + B u_{t_k} + b, Q), \\\\
    y_{t_k} &\\sim \\mathcal{N}(H x_{t_k} + D u_{t_k} + d, R).
    \\end{aligned}
    $$

    This factory composes `LinearGaussianStateEvolution` and
    `LinearGaussianObservation` into a core `DynamicalModel`.

    Args:
        A (jax.Array): State transition matrix with shape
            $(d_x, d_x)$.
        Q (jax.Array): Process-noise covariance with shape
            $(d_x, d_x)$.
        H (jax.Array): Observation matrix with shape
            $(d_y, d_x)$.
        R (jax.Array): Observation-noise covariance with shape
            $(d_y, d_y)$.
        B (jax.Array | None): Optional control matrix in the transition model
            with shape $(d_x, d_u)$. If None, no control term is used and
            `control_dim` is set to 0.
        b (jax.Array | None): Optional additive transition bias with shape
            $(d_x,)$.
        D (jax.Array | None): Optional control matrix in the observation model
            with shape $(d_y, d_u)$.
        d (jax.Array | None): Optional additive observation bias with shape
            $(d_y,)$.
        initial_mean (jax.Array | None): Optional initial-state mean $m_0$ with
            shape $(d_x,)$. Defaults to zeros.
        initial_cov (jax.Array | None): Optional initial-state covariance $C_0$
            with shape $(d_x, d_x)$. Defaults to identity.

    Returns:
        DynamicalModel: A discrete-time LTI state-space model.
    """
    state_dim = A.shape[-1]
    control_dim = B.shape[-1] if B is not None else 0

    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    elif initial_mean.shape[-1] != state_dim:
        raise ValueError(
            f"initial_mean must have last dim {state_dim}, got shape {initial_mean.shape}"
        )
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)
    elif initial_cov.shape[-1] != state_dim:
        raise ValueError(
            f"initial_cov must have last dim {state_dim}, got shape {initial_cov.shape}"
        )

    initial_condition = dist.MultivariateNormal(
        loc=initial_mean, covariance_matrix=initial_cov
    )
    state_evolution = LinearGaussianStateEvolution(A=A, cov=Q, B=B, bias=b)

    observation_model = LinearGaussianObservation(H=H, R=R, D=D, bias=d)

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_model=None,
        control_dim=control_dim,
    )


def LTI_continuous(
    A: jax.Array,
    L: jax.Array,
    H: jax.Array,
    R: jax.Array,
    B: jax.Array | None = None,
    b: jax.Array | None = None,
    D: jax.Array | None = None,
    d: jax.Array | None = None,
    initial_mean: jax.Array | None = None,
    initial_cov: jax.Array | None = None,
) -> DynamicalModel:
    """
    Build a continuous-time linear time-invariant (LTI) `DynamicalModel`.

    The state evolves according to the SDE and observation model

    $$
    \\begin{aligned}
    x_0 &\\sim \\mathcal{N}(m_0, C_0), \\\\
    dx_t &= (A x_t + B u_t + b) \\, dt + L \\, dW_t, \\\\
    y_t &\\sim \\mathcal{N}(H x_t + D u_t + d, R).
    \\end{aligned}
    $$

    Here, $L$ is a diffusion coefficient (not a covariance) with shape
    $(d_x, d_w)$. It multiplies a $d_w$-dimensional Brownian motion $W_t$ whose
    increments have identity covariance: $dW_t \\sim \\mathcal{N}(0, I_{d_w} \\, dt)$.
    The Brownian motion dimension $d_w$ is determined by the second dimension of
    $L$. Under this convention, the infinitesimal state covariance contributed by
    the noise term is $L L^\\top \\, dt$.

    Args:
        A (jax.Array): Drift matrix with shape $(d_x, d_x)$.
        L (jax.Array): Diffusion coefficient with shape $(d_x, d_w)$.
        H (jax.Array): Observation matrix with shape $(d_y, d_x)$.
        R (jax.Array): Observation-noise covariance with shape
            $(d_y, d_y)$.
        B (jax.Array | None): Optional control matrix in the drift with shape
            $(d_x, d_u)$. If None, no control term is used and `control_dim` is
            set to 0.
        b (jax.Array | None): Optional additive drift bias with shape
            $(d_x,)$.
        D (jax.Array | None): Optional control matrix in the observation model
            with shape $(d_y, d_u)$.
        d (jax.Array | None): Optional additive observation bias with shape
            $(d_y,)$.
        initial_mean (jax.Array | None): Optional initial-state mean $m_0$ with
            shape $(d_x,)$. Defaults to zeros.
        initial_cov (jax.Array | None): Optional initial-state covariance $C_0$
            with shape $(d_x, d_x)$. Defaults to identity.

    Returns:
        DynamicalModel: A continuous-time LTI state-space model.
    """
    state_dim = A.shape[-1]
    control_dim = B.shape[-1] if B is not None else 0

    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    elif initial_mean.shape[-1] != state_dim:
        raise ValueError(
            f"initial_mean must have last dim {state_dim}, got shape {initial_mean.shape}"
        )
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)
    elif initial_cov.shape[-1] != state_dim:
        raise ValueError(
            f"initial_cov must have last dim {state_dim}, got shape {initial_cov.shape}"
        )

    initial_condition = dist.MultivariateNormal(
        loc=initial_mean, covariance_matrix=initial_cov
    )

    drift = AffineDrift(A=A, B=B, b=b)

    state_evolution = ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=L,
        diffusion_type="full",
    )

    observation_model = LinearGaussianObservation(H=H, R=R, D=D, bias=d)

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_model=None,
        control_dim=control_dim,
    )
