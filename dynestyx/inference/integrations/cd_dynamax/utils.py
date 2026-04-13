from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
    ParamsCDLGSSM,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from cd_dynamax.dynamax.parameters import ParameterProperties

from dynestyx.models import (
    AffineDrift,
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianObservation,
    GaussianStateEvolution,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


def _normalize_cd_dynamax_diffusion(
    diffusion_coefficient,
    state_dim: int,
):
    """Return a diffusion coeff compatible with cd-dynamax's EnKF SDE solve.

    cd-dynamax's internal diffrax wrapper builds Brownian controls with shape
    equal to `y0.shape` (state_dim). For non-square diffusion coefficients
    (state_dim, bm_dim) with bm_dim != state_dim, pad/truncate columns so the
    returned matrix is always (state_dim, state_dim).
    """

    def _wrapped(x, u, t):
        L = diffusion_coefficient(x, u, t)
        if L.ndim == 1:
            L = jnp.diag(L)
        if L.ndim != 2:
            raise ValueError(
                "diffusion_coefficient must return a vector or matrix for cd-dynamax."
            )
        n_cols = L.shape[-1]
        if n_cols == state_dim:
            return L
        if n_cols < state_dim:
            return jnp.pad(L, ((0, 0), (0, state_dim - n_cols)))
        return L[:, :state_dim]

    return _wrapped


class _ConstantFunction(eqx.Module):
    value: Any

    def f(self, x=None, u=None, t=None):
        return self.value


class _CallableFunction(eqx.Module):
    fn: Callable = eqx.field(static=True)

    def f(self, x=None, u=None, t=None):
        return self.fn(x, u, t)


class _NumpyroDistributionAdapter:
    def __init__(self, base_distribution: dist.Distribution):
        self._base_distribution = base_distribution

    @property
    def distribution(self):
        # CD-Dynamax sometimes calls `obj.distribution.sample(seed=...)`.
        return self

    def log_prob(self, y):
        return self._base_distribution.log_prob(y)

    def sample(self, *args, seed=None, **kwargs):
        key = seed if seed is not None else kwargs.pop("key", None)
        sample_shape = kwargs.pop("sample_shape", ())
        if key is not None:
            return self._base_distribution.sample(key, sample_shape=sample_shape)
        return self._base_distribution.sample(
            *args, sample_shape=sample_shape, **kwargs
        )


class _ConditionalDistributionAdapter:
    def __init__(self, distribution_fn: Callable):
        self._distribution_fn = distribution_fn

    def log_prob(self, y, x=None, u=None, t=None):
        return self._distribution_fn(x, u, t).log_prob(y)

    def sample(self, x=None, u=None, t=None, *args, seed=None, **kwargs):
        key = seed if seed is not None else kwargs.pop("key", None)
        sample_shape = kwargs.pop("sample_shape", ())
        dist_xy = self._distribution_fn(x, u, t)
        if key is not None:
            return dist_xy.sample(key, sample_shape=sample_shape)
        return dist_xy.sample(*args, sample_shape=sample_shape, **kwargs)


def _as_emission_distribution(obs_model: Any) -> Any:
    if hasattr(obs_model, "log_prob") and hasattr(obs_model, "sample"):
        return obs_model
    if callable(obs_model):
        return _ConditionalDistributionAdapter(obs_model)
    return obs_model


def _fixed_param(value: Any) -> dict[str, Any]:
    return {"params": value, "props": ParameterProperties(trainable=False)}


def _as_learnable(value: Any) -> Any:
    if hasattr(value, "f"):
        return value
    if callable(value):
        return _CallableFunction(fn=value)
    return _ConstantFunction(value=jnp.asarray(value))


def _initialize_model_params(model: Any, **raw_kwargs: Any) -> Any:
    """Initialize cd_dynamax params, handling both raw and dict-wrapped APIs."""
    if isinstance(model, ContDiscreteLinearGaussianSSM):
        # HD-UQ/public expects `{params, props}` wrappers for CDLGSSM initialize,
        # but tensor-valued fields should remain raw arrays (not function wrappers).
        wrapped_linear_kwargs: dict[str, Any] = {}
        for key, value in raw_kwargs.items():
            wrapped_linear_kwargs[key] = (
                value if key == "dynamics_approx_order" else _fixed_param(value)
            )
        cdlg_params, _ = model.initialize(  # type: ignore[arg-type]
            **wrapped_linear_kwargs
        )
        return cdlg_params

    if isinstance(model, (ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM)):
        wrapped_nonlinear_kwargs: dict[str, Any] = {}
        learnable_fn_keys = {
            "initial_mean",
            "initial_cov",
            "dynamics_drift",
            "dynamics_diffusion_coefficient",
            "dynamics_diffusion_cov",
            "emission_function",
            "emission_cov",
        }
        for key, value in raw_kwargs.items():
            if key == "dynamics_approx_order":
                wrapped_nonlinear_kwargs[key] = value
            elif key in learnable_fn_keys:
                wrapped_nonlinear_kwargs[key] = _fixed_param(_as_learnable(value))
            else:
                wrapped_nonlinear_kwargs[key] = _fixed_param(value)
        cdnl_params, _ = model.initialize(  # type: ignore[arg-type]
            **wrapped_nonlinear_kwargs
        )
        return cdnl_params

    raise TypeError(
        "_initialize_model_params only supports "
        "ContDiscreteLinearGaussianSSM, "
        "ContDiscreteNonlinearGaussianSSM, and "
        "ContDiscreteNonlinearSSM; "
        f"got {type(model).__name__}."
    )


def dsx_to_cdlgssm_params(dsx_model: DynamicalModel) -> ParamsCDLGSSM:
    """Build ParamsCDLGSSM for CD-Dynamax's continuous-discrete KF.

    Requires:
    - drift is AffineDrift (A, B, b)
    - diffusion_coefficient is constant (callable returning same value for any x, u, t)
      returning same value for any x, u, t)
    - observation_model is LinearGaussianObservation
    - initial_condition is MultivariateNormal
    """
    state_evo = dsx_model.state_evolution
    if not isinstance(state_evo, ContinuousTimeStateEvolution):
        raise TypeError("dsx_to_cdlgssm_params requires ContinuousTimeStateEvolution.")
    drift = state_evo.drift
    if not isinstance(drift, AffineDrift):
        raise TypeError(
            f"dsx_to_cdlgssm_params requires AffineDrift, got {type(drift).__name__}."
        )
    if state_evo.diffusion_coefficient is None:
        raise ValueError("dsx_to_cdlgssm_params requires diffusion_coefficient.")

    # Extract constant L and use inferred Brownian dimension.
    x0 = jnp.zeros(dsx_model.state_dim)
    L = state_evo.diffusion_coefficient(x0, None, jnp.array(0.0))
    if state_evo.bm_dim is None:
        raise ValueError(
            "state_evolution.bm_dim is not set on ContinuousTimeStateEvolution."
        )
    Q = jnp.eye(state_evo.bm_dim)

    ic = dsx_model.initial_condition
    if not isinstance(ic, dist.MultivariateNormal):
        raise TypeError(
            "dsx_to_cdlgssm_params requires MultivariateNormal initial condition."
        )
    obs = dsx_model.observation_model
    if not isinstance(obs, LinearGaussianObservation):
        raise TypeError("dsx_to_cdlgssm_params requires LinearGaussianObservation.")

    B = (
        drift.B
        if drift.B is not None
        else jnp.zeros((dsx_model.state_dim, dsx_model.control_dim))
    )
    b = drift.b if drift.b is not None else jnp.zeros(dsx_model.state_dim)
    D = (
        obs.D
        if obs.D is not None
        else jnp.zeros((dsx_model.observation_dim, dsx_model.control_dim))
    )
    d = obs.bias if obs.bias is not None else jnp.zeros(dsx_model.observation_dim)

    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=dsx_model.state_dim,
        emission_dim=dsx_model.observation_dim,
        input_dim=dsx_model.control_dim,
    )
    return _initialize_model_params(
        cd_model,
        initial_mean=jnp.asarray(ic.loc),
        initial_cov=jnp.asarray(ic.covariance_matrix),
        dynamics_weights=drift.A,
        dynamics_input_weights=B,
        dynamics_bias=b,
        dynamics_diffusion_coefficient=L,
        dynamics_diffusion_cov=Q,
        emission_weights=obs.H,
        emission_input_weights=D,
        emission_bias=d,
        emission_cov=obs.R,
    )


def dsx_to_cd_dynamax(
    dsx_model: DynamicalModel, cd_model: SSMType | None = None
) -> tuple[Any, bool]:
    """
    Maps a dsx Dynamical Model to a CD-Dynamax-compatible model.
    """

    shared_params: dict[str, Any] = {}

    ## Map state evolution ##
    state_evo = dsx_model.state_evolution
    if isinstance(state_evo, ContinuousTimeStateEvolution):
        if state_evo.drift is not None or state_evo.potential is not None:
            shared_params.update(
                {
                    "dynamics_drift": state_evo.total_drift,
                }
            )
        else:
            raise ValueError("Both drift and potential are None; define at least one.")
        if state_evo.diffusion_coefficient is not None:
            if state_evo.bm_dim is None:
                raise ValueError(
                    "state_evolution.bm_dim is not set on ContinuousTimeStateEvolution."
                )
            diffusion_coeff = _normalize_cd_dynamax_diffusion(
                state_evo.diffusion_coefficient,
                dsx_model.state_dim,
            )
            shared_params.update(
                {
                    "dynamics_diffusion_coefficient": diffusion_coeff,
                    "dynamics_diffusion_cov": jnp.eye(dsx_model.state_dim),
                }
            )
    else:
        raise NotImplementedError(
            f"State evolution of type {type(state_evo)} is not supported yet."
        )

    uses_nonlinear_non_gaussian_api = isinstance(cd_model, ContDiscreteNonlinearSSM)

    ## Map initial condition ##
    ic = dsx_model.initial_condition
    if uses_nonlinear_non_gaussian_api:
        initial_distribution = (
            _NumpyroDistributionAdapter(ic) if isinstance(ic, dist.Distribution) else ic
        )
    else:
        if isinstance(ic, dist.MultivariateNormal):
            initial_mean = ic.loc  # type: ignore
            initial_cov = ic.covariance_matrix
        elif isinstance(ic, dist.Normal):
            initial_mean = ic.loc  # type: ignore
            initial_cov = jnp.square(ic.scale)
        else:
            raise NotImplementedError(
                f"Initial condition of type {type(ic)} is not supported yet."
            )

    ## Map observation model ##
    obs = dsx_model.observation_model
    non_gaussian_flag = False

    if isinstance(obs, LinearGaussianObservation):

        def emission_function(x, u, t):
            if x.ndim > 1:
                return x @ obs.H.T + (
                    obs.D @ u if obs.D is not None and u is not None else 0
                )
            else:
                return obs.H @ x + (
                    obs.D @ u if obs.D is not None and u is not None else 0
                )

        if uses_nonlinear_non_gaussian_api:
            emission_distribution = _ConditionalDistributionAdapter(
                lambda x=None, u=None, t=None: dist.MultivariateNormal(
                    loc=jnp.atleast_1d(jnp.asarray(emission_function(x, u, t))),
                    covariance_matrix=jnp.atleast_2d(jnp.asarray(obs.R)),
                )
            )

            model_params = {
                **shared_params,
                "initial_distribution": initial_distribution,
                "emission_distribution": emission_distribution,
            }
        else:
            model_params = {
                **shared_params,
                "initial_mean": initial_mean,
                "initial_cov": initial_cov,
                "emission_function": emission_function,
                "emission_cov": obs.R,  # type: ignore
            }
    else:
        # TODO: check for linear-gaussian observation models and extract H, R
        # TODO: check for Gaussian observation and use CDNLGSSM
        non_gaussian_flag = True
        model_params = {
            **shared_params,
            "initial_distribution": initial_distribution,
            "emission_distribution": _as_emission_distribution(
                dsx_model.observation_model
            ),
        }

    if cd_model is None:
        if non_gaussian_flag:
            model_to_use: SSMType = ContDiscreteNonlinearSSM(
                state_dim=dsx_model.state_dim,
                emission_dim=dsx_model.observation_dim,
                input_dim=dsx_model.control_dim,
            )
        else:
            model_to_use = ContDiscreteNonlinearGaussianSSM(
                state_dim=dsx_model.state_dim,
                emission_dim=dsx_model.observation_dim,
                input_dim=dsx_model.control_dim,
            )
    else:
        model_to_use = cd_model

    cd_dynamax_params = _initialize_model_params(model_to_use, **model_params)

    return cd_dynamax_params, non_gaussian_flag


def gaussian_to_nlgssm_params(dynamics: DynamicalModel) -> ParamsNLGSSM:
    """Build ParamsNLGSSM from a Gaussian discrete-time DynamicalModel.

    Supports linear or nonlinear dynamics:
    - `LinearGaussianStateEvolution` (LTI: f(x,u) = A@x + b + B@u with Gaussian noise),
    - `GaussianStateEvolution` (nonlinear: F(x, u, t_now, t_next) with Gaussian noise),
    with either `LinearGaussianObservation` (H x + d + D u) or `GaussianObservation`
    (arbitrary h(x,u,t) with Gaussian noise).

    Used by EKF/UKF in discrete filters. When control_dim is 0, u has shape (0,)
    and the B@u / D@u terms are omitted.
    """
    if not isinstance(
        dynamics.state_evolution, (LinearGaussianStateEvolution, GaussianStateEvolution)
    ) or not isinstance(
        dynamics.observation_model, (LinearGaussianObservation, GaussianObservation)
    ):
        raise TypeError(
            "gaussian_to_nlgssm_params expects a Gaussian DynamicalModel with "
            "state_evolution either LinearGaussianStateEvolution or "
            "GaussianStateEvolution, and observation_model either "
            "LinearGaussianObservation or GaussianObservation."
        )

    evo = dynamics.state_evolution
    obs = dynamics.observation_model
    ic = dynamics.initial_condition
    state_dim = dynamics.state_dim
    control_dim = dynamics.control_dim

    if isinstance(ic, dist.MultivariateNormal):
        initial_mean = jnp.asarray(ic.loc)
        initial_covariance = jnp.asarray(ic.covariance_matrix)
    elif isinstance(ic, dist.Normal):
        # dist.Normal: scalar Gaussian, treat as 1D state with variance scale^2.
        initial_mean = jnp.atleast_1d(jnp.asarray(ic.loc))
        initial_covariance = jnp.atleast_2d(jnp.square(jnp.asarray(ic.scale)))
    else:
        raise TypeError(
            "KF, EKF, and UKF require a Gaussian initial condition "
            "(MultivariateNormal or Normal) because they propagate mean and covariance. "
            "For non-Gaussian initial conditions, use filter_type='pf' (particle filter)."
        )

    # ----- Dynamics function -----
    if isinstance(evo, LinearGaussianStateEvolution):
        F = evo.A
        b = evo.bias if evo.bias is not None else jnp.zeros(state_dim)
        B = evo.B if evo.B is not None else jnp.zeros((state_dim, control_dim))

        def dynamics_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            out = F @ x + b
            if control_dim > 0 and u.size > 0:
                out = out + (B @ jnp.reshape(u, (-1, 1))).ravel()
            return out

    else:
        # GaussianStateEvolution: arbitrary nonlinear F(x, u, t_now, t_next) with Gaussian noise.
        def dynamics_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            # Discrete-time setting for EKF/UKF: ignore absolute time and pass dummy times.
            t_now = jnp.array(0.0, dtype=x.dtype)
            t_next = jnp.array(0.0, dtype=x.dtype)
            return evo.F(x, u, t_now, t_next)

    # ----- Emission function -----
    if isinstance(obs, LinearGaussianObservation):
        H = obs.H
        d = obs.bias if obs.bias is not None else jnp.zeros(dynamics.observation_dim)
        D = (
            obs.D
            if obs.D is not None
            else jnp.zeros((dynamics.observation_dim, control_dim))
        )

        def emission_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            out = H @ x + d
            if control_dim > 0 and u.size > 0:
                out = out + (D @ jnp.reshape(u, (-1, 1))).ravel()
            return out

    else:
        # GaussianObservation: y_t ~ N(h(x_t, u_t, t), R) with arbitrary h.
        def emission_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            _t = jnp.array(0.0, dtype=x.dtype)
            return obs.h(x, u, _t)  # warning: time is ignored

    return ParamsNLGSSM(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_function=dynamics_function,
        dynamics_covariance=evo.cov,
        emission_function=emission_function,
        emission_covariance=obs.R,
    )
