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
    r"""Build a one-step (predict+update) function for `(dynamics, filter_config)`.

    Uses dynamax's own `_predict`/`_condition_on` primitives (the same ones
    `lgssm_filter` / `extended_kalman_filter` / `unscented_kalman_filter`
    scan over internally), but in **predict-then-condition** order with the
    *filtered* belief as carry -- not dynamax's own native
    condition-then-predict order over the *predicted* belief. This
    reordering is required for online/closed-loop use: with dynamax's native
    order, the single `u` passed to a step is used both to condition on `y`
    (correct: `u_k`, the control that produced the state now observed) *and*
    to predict the state *after* that (which needs `u_{k+1}`, the control
    chosen from the belief this very call is producing -- not yet available
    in closed-loop control). Predict-then-condition avoids this: `u` plays
    exactly one role per call, matching `compute_cuthbert_filter_update`'s
    own documented semantics (`u` is the control that drove the transition
    *into* the state now being conditioned on `y`). No stateful object is
    built: the return value is a plain `(function, tuple-of-arrays)` pair.

    Returns:
        `(step_fn, initial_state)`, where:
        - `step_fn(carry, u, y) -> (filtered_mean, filtered_cov)`, `carry`
          is the previous filtered belief `(mean, cov)` (or `None` to
          bootstrap: see below). `step_fn`'s return value *is* the belief to
          pass back in as `carry` on the next call -- one object playing
          both roles, exactly like `compute_cuthbert_filter_update`'s single
          returned state.
        - `initial_state = (prior_mean, prior_cov)`, not yet conditioned on
          anything. When `carry is None`, `step_fn` conditions this prior
          directly on `y` with no predict step first (nothing has happened
          before $t_0$) -- matching cuthbert's own bootstrap, which skips the
          transition for the same reason (see
          `compute_cuthbert_filter_update`'s docstring).
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
        initial_state = (params.initial.mean, params.initial.cov)

        def step_fn(carry, u, y):
            if carry is None:
                pred_mean, pred_cov = initial_state
            else:
                pred_mean, pred_cov = _kf_predict(*carry, F, B, b, Q, u)
            return _kf_condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        return step_fn, initial_state

    # EKF and UKF share the same nonlinear params representation.
    params_nl = gaussian_to_nlgssm_params(dynamics)
    f, h = params_nl.dynamics_function, params_nl.emission_function
    Q, R = params_nl.dynamics_covariance, params_nl.emission_covariance
    initial_state = (params_nl.initial_mean, params_nl.initial_covariance)

    if isinstance(filter_config, EKFConfig):
        # Same one-time Jacobian setup extended_kalman_filter itself does
        # before scanning (inference_ekf.py:119).
        F_jac, H_jac = jacfwd(f), jacfwd(h)

        def step_fn(carry, u, y):
            if carry is None:
                pred_mean, pred_cov = initial_state
            else:
                pred_mean, pred_cov = _ekf_predict(*carry, f, F_jac, Q, u)
            return _ekf_condition_on(pred_mean, pred_cov, h, H_jac, R, u, y, 1)

        return step_fn, initial_state

    if isinstance(filter_config, UKFConfig):
        # Same one-time sigma-point weight setup unscented_kalman_filter
        # itself does before scanning (inference_ukf.py:169-172).
        state_dim = dynamics.state_dim
        lamb = _compute_lambda(filter_config.alpha, filter_config.kappa, state_dim)
        w_mean, w_cov = _compute_weights(
            state_dim, filter_config.alpha, filter_config.beta, lamb
        )

        def step_fn(carry, u, y):
            if carry is None:
                pred_mean, pred_cov = initial_state
            else:
                pred_mean, pred_cov, _ = _ukf_predict(
                    *carry, f, Q, lamb, w_mean, w_cov, u
                )
            _, filtered_mean, filtered_cov = _ukf_condition_on(
                pred_mean, pred_cov, h, R, lamb, w_mean, w_cov, u, y
            )
            return filtered_mean, filtered_cov

        return step_fn, initial_state

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
    r"""One-step FilterUpdate: prev_state + u + y -> new filtered belief.

    `u` is the control that drove the transition *into* the state now being
    filtered (`u_k`, producing `state_{k+1}` from `y_{k+1}`) -- the same
    convention `compute_cuthbert_filter_update` uses, not dynamax's own
    native same-index convention (see `build_dynamax_filter`'s docstring for
    why the reordering is necessary online).

    `prev_state` is the `(mean, cov)` belief returned by the previous call,
    or `None` to bootstrap: mirrors `compute_cuthbert_filter_update`'s own
    `prev_state=None` convention exactly. On bootstrap, the prior (from
    `build_dynamax_filter`'s `initial_state`, baked into `filter_function`)
    is conditioned on `y` directly, with no predict step -- nothing has
    happened before $t_0$.

    `t`/`t_prev` are accepted for call-site parity with
    `compute_cuthbert_filter_update` but unused -- cd-dynamax's discrete
    dynamics/emission functions are already time-homogeneous per step (see
    `gaussian_to_nlgssm_params`'s warning about ignored absolute time).

    Returns:
        `(filtered_mean, filtered_cov)` -- the new belief \(\hat x_{k+1|k+1}\).
        Feed this back in as `prev_state` on the next call, and/or use it
        directly for external consumption (e.g. a control policy).
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
