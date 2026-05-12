"""Minimal reproduction of GitHub issue #XXX.

Lotka-Volterra second-order model in a one-level hierarchy produces
"Warning: matrix is not positive definite" (via ``jax.debug.print``) on every
observation step when the drift is a batched equinox module inside
``dsx.plate``.  The flat (non-plate) version of the same model does not
trigger any warnings.

True parameters: α = 2/3, β = 4/3, γ = 1, δ = 1
Fixed point: (x*, y*) = (γ/δ, α/β) = (1, 1/2)
"""

import io
import sys

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import (
    DynamicalModel,
    ContinuousTimeStateEvolution,
    LinearGaussianObservation,
    Filter,
    SDESimulator,
)
from dynestyx.inference.filter_configs import ContinuousTimeEKFConfig, ContinuousTimeEnKFConfig

# ---------------------------------------------------------------------------
# Lotka-Volterra dynamics via quadratic feature expansion
# ---------------------------------------------------------------------------

# True LV weights (α = 2/3, β = 4/3, γ = 1, δ = 1), centred at fixed point
# (x*, y*) = (1, 0.5).  The linear part encodes α, -γ and the quadratic
# z₀·z₁ term captures the –β and +δ interactions.
_W_LINEAR = jnp.array([[2.0 / 3.0, 0.0], [0.0, -1.0]])
_W_QUAD = jnp.array([[0.0, -(4.0 / 3.0), 0.0], [1.0, 0.0, 0.0]])
_WEIGHTS_TRUE = jnp.concatenate([_W_LINEAR, _W_QUAD], axis=1)  # (2, 5)


class _LVDrift(eqx.Module):
    """Lotka-Volterra drift via second-order polynomial expansion around mu_i."""

    mu_i: jnp.ndarray  # centre point, shape (..., state_dim, 1) inside a plate
    weights: jnp.ndarray  # (state_dim, state_dim + n_quad)

    def __call__(self, x, u, t):
        mu_i = jnp.squeeze(self.mu_i)
        z = x - mu_i
        i_idx, j_idx = jnp.triu_indices(z.shape[0])
        features = jnp.concatenate([z, z[i_idx] * z[j_idx]])
        return self.weights @ features


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

_STATE_DIM = 2
_EMIT_SD = 0.5
_DIFF_COEF = 0.05


def _lv_flat_model(obs_times=None, obs_values=None, predict_times=None):
    """Single-trajectory LV model – *no* plate, used as reference baseline."""
    mu_i = numpyro.sample(
        "mu_i",
        dist.Normal(jnp.zeros((_STATE_DIM, 1)), jnp.ones((_STATE_DIM, 1))).to_event(2),
    )
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.zeros(_STATE_DIM), 0.5 * jnp.eye(_STATE_DIM)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=_LVDrift(mu_i=mu_i, weights=_WEIGHTS_TRUE),
            diffusion_coefficient=lambda x, u, t: _DIFF_COEF * jnp.eye(_STATE_DIM),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(_STATE_DIM), R=(_EMIT_SD**2) * jnp.eye(_STATE_DIM)
        ),
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _lv_hierarchical_model(
    obs_times=None, obs_values=None, predict_times=None, n_groups: int = 1
):
    """Hierarchical LV model with one ``dsx.plate`` level (N groups)."""
    with dsx.plate("groups", n_groups):
        mu_group = numpyro.sample(
            "mu_group",
            dist.Normal(
                jnp.zeros((_STATE_DIM, 1)), jnp.ones((_STATE_DIM, 1))
            )
            .expand([n_groups, _STATE_DIM, 1])
            .to_event(2),
        )
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.zeros(_STATE_DIM), 0.5 * jnp.eye(_STATE_DIM)
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=_LVDrift(mu_i=mu_group, weights=_WEIGHTS_TRUE),
                diffusion_coefficient=lambda x, u, t: _DIFF_COEF
                * jnp.eye(_STATE_DIM),
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.eye(_STATE_DIM), R=(_EMIT_SD**2) * jnp.eye(_STATE_DIM)
            ),
        )
        return dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PREDICT_TIMES = jnp.linspace(0.0, 5.0, 30)
_MU_TRUE = jnp.array([[[1.0], [0.5]]])  # (n_groups=1, state_dim=2, 1) – LV fixed point


@pytest.fixture(scope="module")
def lv_obs_values():
    """Synthetic LV observations generated at the true fixed-point centre."""
    with SDESimulator():
        synth = Predictive(
            _lv_hierarchical_model,
            params={"mu_group": _MU_TRUE},
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(10), n_groups=1, predict_times=_PREDICT_TIMES)
    # shape: (num_samples, n_groups, n_sim, T, obs_dim) → (n_groups, T, obs_dim)
    return synth["f_observations"][0, :, 0, :, :]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _count_pos_def_warnings(model_fn, filter_config, rng_key, params=None, **model_kwargs):
    """Run the model under *filter_config*, return (mll, n_pos_def_warnings).

    Parameters
    ----------
    model_fn:
        The NumpPyro model to run.
    filter_config:
        A filter configuration (e.g. ``ContinuousTimeEnKFConfig``).
    rng_key:
        JAX PRNG key.
    params:
        Optional dict of fixed parameter values forwarded to ``Predictive``.
    **model_kwargs:
        Keyword arguments forwarded to ``model_fn``.
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with Filter(filter_config=filter_config):
            out = Predictive(
                model_fn,
                params=params,
                num_samples=1,
                exclude_deterministic=False,
            )(rng_key, **model_kwargs)
    finally:
        captured = sys.stdout.getvalue()
        sys.stdout = old_stdout

    n_warnings = captured.count("not positive definite")
    mll = out["f_marginal_loglik"]
    return mll, n_warnings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Known bug: hierarchical LV model with dsx.plate triggers "
        "'Warning: matrix is not positive definite' on every observation step "
        "from the CD-dynamax backend.  Remove this xfail once fixed."
    ),
    strict=True,
)
@pytest.mark.parametrize(
    "filter_config",
    [
        ContinuousTimeEnKFConfig(warn=True),
        ContinuousTimeEKFConfig(warn=True),
    ],
    ids=["EnKF", "EKF"],
)
def test_lv_hierarchical_filter_no_pos_def_warnings(lv_obs_values, filter_config):
    """Hierarchical LV model must not produce 'matrix not positive definite' warnings.

    This is a regression test for an issue where the CD-dynamax backend emitted
    "Warning: matrix is not positive definite" (via ``jax.debug.print``) on
    *every* observation step when filtering a Lotka-Volterra SDE inside a
    ``dsx.plate`` context.
    """
    mll, n_warnings = _count_pos_def_warnings(
        _lv_hierarchical_model,
        filter_config,
        jr.PRNGKey(2),
        obs_times=_PREDICT_TIMES,
        obs_values=lv_obs_values,
        n_groups=1,
    )

    assert not jnp.isnan(mll).any(), "MLL must not be NaN"
    assert not jnp.isinf(mll).any(), "MLL must not be Inf"
    assert n_warnings == 0, (
        f"Expected 0 'matrix not positive definite' warnings from the filter backend, "
        f"but got {n_warnings}.  This typically means the filter covariance is "
        f"losing positive-definiteness when dynamics are batched inside dsx.plate."
    )


def test_lv_flat_and_hierarchical_mll_match(lv_obs_values):
    """MLL from the flat (no plate) and hierarchical models should agree.

    Both models share the same dynamics at the true parameters; the only
    difference is whether the trajectory is wrapped in a ``dsx.plate``.
    """
    # Flat model: observations are a single trajectory (T, obs_dim)
    obs_flat = lv_obs_values[0]  # (T, obs_dim)
    mu_i_true = _MU_TRUE[0]  # (state_dim, 1)

    mll_flat, _ = _count_pos_def_warnings(
        _lv_flat_model,
        ContinuousTimeEnKFConfig(warn=False),
        jr.PRNGKey(3),
        params={"mu_i": mu_i_true},
        obs_times=_PREDICT_TIMES,
        obs_values=obs_flat,
    )

    mll_hier, _ = _count_pos_def_warnings(
        _lv_hierarchical_model,
        ContinuousTimeEnKFConfig(warn=False),
        jr.PRNGKey(3),
        params={"mu_group": _MU_TRUE},
        obs_times=_PREDICT_TIMES,
        obs_values=lv_obs_values,
        n_groups=1,
    )

    assert not jnp.isnan(mll_flat).any()
    assert not jnp.isnan(mll_hier).any()

    flat_val = float(jnp.squeeze(mll_flat))
    hier_val = float(jnp.squeeze(mll_hier))
    assert abs(flat_val - hier_val) < 1.0, (
        f"Flat MLL ({flat_val:.3f}) and hierarchical MLL ({hier_val:.3f}) "
        f"should be close when both use the same dynamics at the true parameters."
    )
