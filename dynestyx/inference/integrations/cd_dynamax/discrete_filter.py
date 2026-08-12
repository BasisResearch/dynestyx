"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF."""

from typing import Any, cast

import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    _condition_on as _kf_condition_on,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    _predict as _kf_predict,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    lgssm_filter,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    preprocess_params_and_inputs as _kf_preprocess_params,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    _condition_on as _ekf_condition_on,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    _predict as _ekf_predict,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    _compute_lambda,
    _compute_weights,
    unscented_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    _condition_on as _ukf_condition_on,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    _predict as _ukf_predict,
)
from jax import jacfwd
from jaxtyping import Array, Real

from dynestyx.inference.configs.filter import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    UKFConfig,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    _require_constant_linear_gaussian_fields,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.utils import squeeze_leading_singletons
from dynestyx.inference.utils.distribution_utils import _posterior_sequence_to_dists
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)


def _lti_to_lgssm_params(dynamics: DynamicalModel):
    """Build dynamax ParamsLGSSM from LinearGaussianSSM.initialize for an LTI model."""
    state_dim = dynamics.state_dim
    emission_dim = dynamics.observation_dim
    control_dim = dynamics.control_dim

    if (
        isinstance(dynamics.state_evolution, LinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, LinearGaussianObservation)
        and isinstance(dynamics.initial_condition, dist.MultivariateNormal)
    ):
        evo = dynamics.state_evolution
        obs = dynamics.observation_model
        ic = dynamics.initial_condition
        _require_constant_linear_gaussian_fields(
            evo,
            ("A", "B", "bias", "cov"),
            where="The cd_dynamax discrete Kalman filter",
        )
        _require_constant_linear_gaussian_fields(
            obs,
            ("H", "D", "bias", "R"),
            where="The cd_dynamax discrete Kalman filter",
        )
        model = LinearGaussianSSM(
            state_dim=state_dim,
            emission_dim=emission_dim,
            input_dim=control_dim,
            has_dynamics_bias=evo.bias is not None,
            has_emissions_bias=obs.bias is not None,
        )
        params, _ = model.initialize(
            initial_mean=squeeze_leading_singletons(ic.loc, 1),
            initial_covariance=squeeze_leading_singletons(ic.covariance_matrix, 2),
            dynamics_weights=evo.A,
            dynamics_bias=evo.bias,
            dynamics_input_weights=evo.B,
            dynamics_covariance=evo.cov,
            emission_weights=obs.H,
            emission_bias=obs.bias,
            emission_input_weights=obs.D,
            emission_covariance=obs.R,
        )
        return params
    raise TypeError(
        "filter_type='kf' expects a DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation and initial_condition as MultivariateNormal."
    )


def _prepare_inputs(
    dynamics: DynamicalModel,
    obs_values: Real[Array, "obs_time observation_dim"],
    obs_times: Real[Array, " obs_time"],
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None,
) -> tuple[
    Real[Array, "obs_time observation_dim"],
    Real[Array, "obs_time control_dim"],
]:
    """Prepare emissions and inputs arrays for cd-dynamax discrete filters."""
    emissions = obs_values
    t1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
        inputs = jnp.zeros((t1, control_dim))
    elif ctrl_values.shape[0] > t1:
        aligned_ctrl_times = cast(Real[Array, " ctrl_time"], ctrl_times)
        inds = jnp.searchsorted(aligned_ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        inputs = ctrl_values
    return emissions, inputs


def build_dynamax_filter(dynamics: DynamicalModel, filter_config: BaseFilterConfig):
    """Build a one-step (predict+update) function for `(dynamics, filter_config)`.

    Mirrors the per-step body that `lgssm_filter` / `extended_kalman_filter` /
    `unscented_kalman_filter` each build once (outside their own `lax.scan`)
    and then scan over -- e.g. `extended_kalman_filter` computes
    `F, H = jacfwd(f), jacfwd(h)` once before scanning
    (`inference_ekf.py:118-119`); this does the same setup, but returns the
    resulting step callable instead of scanning it. No stateful object is
    built: the return value is a plain `(function, tuple-of-arrays)` pair.

    Returns:
        `(step_fn, initial_state)`, where:
        - `step_fn(carry, u, y) -> (new_carry, (filtered_mean, filtered_cov))`,
          `carry = (pred_mean, pred_cov)` -- exactly what one `lax.scan`
          iteration of the underlying dynamax filter does: condition on `y`
          (producing the filtered belief for this step), then predict
          forward (producing the carry for the next step).
        - `initial_state = (pred_mean, pred_cov)` at t=0, taken directly from
          the model's prior -- the same value dynamax itself seeds its own
          scan carry with. Pass this as `prev_state` on the first call to
          `compute_cd_dynamax_discrete_filter_update`.
    """
    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        # lgssm_filter's own @preprocess_args wrapper zero-fills bias/input_weights
        # when they're None (e.g. no bias term, no controls); mirror that here since
        # we're calling _predict/_condition_on directly instead of through lgssm_filter.
        params, _ = _kf_preprocess_params(params, num_timesteps=1, inputs=None)
        F, B, b, Q = (
            params.dynamics.weights,
            params.dynamics.input_weights,
            params.dynamics.bias,
            params.dynamics.cov,
        )
        H, D, d, R = (
            params.emissions.weights,
            params.emissions.input_weights,
            params.emissions.bias,
            params.emissions.cov,
        )

        def step_fn(carry, u, y):
            pred_mean, pred_cov = carry
            filtered_mean, filtered_cov = _kf_condition_on(
                pred_mean, pred_cov, H, D, d, R, u, y
            )
            next_pred = _kf_predict(filtered_mean, filtered_cov, F, B, b, Q, u)
            return next_pred, (filtered_mean, filtered_cov)

        return step_fn, (params.initial.mean, params.initial.cov)

    # EKF and UKF share the same nonlinear params representation.
    params_nl = gaussian_to_nlgssm_params(dynamics)
    f, h = params_nl.dynamics_function, params_nl.emission_function
    Q, R = params_nl.dynamics_covariance, params_nl.emission_covariance

    if isinstance(filter_config, EKFConfig):
        # Same one-time Jacobian setup extended_kalman_filter itself does
        # before scanning (inference_ekf.py:119).
        F_jac, H_jac = jacfwd(f), jacfwd(h)

        def step_fn(carry, u, y):
            pred_mean, pred_cov = carry
            filtered_mean, filtered_cov = _ekf_condition_on(
                pred_mean, pred_cov, h, H_jac, R, u, y, 1
            )
            next_pred = _ekf_predict(filtered_mean, filtered_cov, f, F_jac, Q, u)
            return next_pred, (filtered_mean, filtered_cov)

        return step_fn, (params_nl.initial_mean, params_nl.initial_covariance)

    if isinstance(filter_config, UKFConfig):
        # Same one-time sigma-point weight setup unscented_kalman_filter
        # itself does before scanning (inference_ukf.py:169-172).
        state_dim = dynamics.state_dim
        lamb = _compute_lambda(filter_config.alpha, filter_config.kappa, state_dim)
        w_mean, w_cov = _compute_weights(
            state_dim, filter_config.alpha, filter_config.beta, lamb
        )

        def step_fn(carry, u, y):
            pred_mean, pred_cov = carry
            _, filtered_mean, filtered_cov = _ukf_condition_on(
                pred_mean, pred_cov, h, R, lamb, w_mean, w_cov, u, y
            )
            next_pred_mean, next_pred_cov, _ = _ukf_predict(
                filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, u
            )
            return (next_pred_mean, next_pred_cov), (filtered_mean, filtered_cov)

        return step_fn, (params_nl.initial_mean, params_nl.initial_covariance)

    raise ValueError(
        f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
        "Expected KFConfig, EKFConfig, or UKFConfig."
    )


def compute_cd_dynamax_discrete_filter_update(
    dynamics: DynamicalModel,
    filter_function,
    prev_state,
    *,
    y,
    u,
    t=None,
    t_prev=None,
):
    r"""One-step FilterUpdate: prev_state + u + y -> (new_state, filtered belief).

    `prev_state` is the `(pred_mean, pred_cov)` carry from the previous call,
    or `build_dynamax_filter`'s `initial_state` on the first call -- there is
    no `None`-sentinel bootstrap branch, because dynamax's own filters need
    none: the initial carry *is* the prior, and the first step just
    conditions it on `y_0` like any other step.

    `t`/`t_prev` are accepted for call-site parity with
    `compute_cuthbert_filter_update` but unused -- cd-dynamax's discrete
    dynamics/emission functions are already time-homogeneous per step (see
    `gaussian_to_nlgssm_params`'s warning about ignored absolute time).

    Returns:
        `(new_state, (filtered_mean, filtered_cov))` -- the same
        `(new_carry, output)` shape a `lax.scan` step returns. `new_state`
        feeds back in as `prev_state` on the next call; `(filtered_mean,
        filtered_cov)` is the filtered belief \(\hat x_{k|k}\) for external
        use (e.g. a control policy).
    """
    u_arr = jnp.zeros((dynamics.control_dim,)) if u is None else jnp.asarray(u)
    return filter_function(prev_state, u_arr, jnp.asarray(y))


def compute_cd_dynamax_discrete_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
) -> Any:
    """Pure-JAX cd-dynamax discrete filter computation (no numpyro side-effects)."""
    emissions, inputs = _prepare_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )

    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        return lgssm_filter(params, emissions, inputs=inputs)

    # EKF and UKF share the same nonlinear params representation.
    params_nl = gaussian_to_nlgssm_params(dynamics)

    if isinstance(filter_config, EKFConfig):
        return extended_kalman_filter(params_nl, emissions, inputs=inputs)
    if isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return unscented_kalman_filter(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )
    raise ValueError(
        f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
        "Expected KFConfig, EKFConfig, or UKFConfig."
    )


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[Real[Array, ""], object, list[dist.Distribution]]:
    """Run discrete-time filter via cd-dynamax (KF, EKF, UKF).

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - posterior: CD-Dynamax posterior object with filtered_means and
              filtered_covariances attributes.
            - filtered_dists: list of MultivariateNormal distributions p(x_t | y_{1:t})
              at each obs time, for posterior rollout.
    """
    posterior = compute_cd_dynamax_discrete_filter(
        dynamics,
        filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    filtered_dists = _posterior_sequence_to_dists(
        posterior,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=False,
        missing="empty",
    )
    return posterior.marginal_loglik, posterior, filtered_dists


__all__ = [
    "compute_cd_dynamax_discrete_filter",
    "compute_cd_dynamax_discrete_filter_update",
    "build_dynamax_filter",
    "run_discrete_filter",
    "_lti_to_lgssm_params",
    "_prepare_inputs",
]
