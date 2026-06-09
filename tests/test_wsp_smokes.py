"""Smoke tests for the Weighted Sums Parameterization (WSP) box constraint.

WSP (``dynestyx.WSP`` + ``dynestyx.Box``) wraps a continuous-time SDE so its
solution stays inside an axis-aligned box. These tests cover the weight geometry,
that the wrapper refines correctly inside a ``DynamicalModel``, that the diffusion
vanishes at the boundary, that simulated trajectories are contained, and that the
wrapped model supports filtering and SVI for parameter inference.
"""

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from jax import Array
from numpyro.handlers import seed, trace
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

import dynestyx as dsx
from dynestyx import (
    WSP,
    Box,
    ContinuousTimeStateEvolution,
    DeterministicContinuousTimeStateEvolution,
    DynamicalModel,
    Filter,
    LinearGaussianObservation,
    ScalarDiffusion,
    SDESimulator,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.inference.filter_configs import ContinuousTimeDPFConfig
from dynestyx.models import WSPDrift


def _wsp_particle_filter(n_particles=32):
    """A small fixed-step particle filter, robust to WSP's boundary-degenerate diffusion.

    WSP's diffusion vanishes at the box faces, which makes covariance-propagating
    filters (KF/EKF/UKF/EnKF) singular there; a particle filter propagates through
    ``L dW`` directly and is unaffected. A fixed-step (non-adaptive) Euler solver
    keeps the run fast and differentiable.
    """
    return ContinuousTimeDPFConfig(
        n_particles=n_particles,
        diffeqsolve_dt0=0.05,
        diffeqsolve_max_steps=16,
        diffeqsolve_kwargs={
            "solver": diffrax.Euler(),
            "stepsize_controller": diffrax.ConstantStepSize(),
        },
    )


_UNIT_BOX = Box(jnp.array([0.0]), jnp.array([1.0]))


class _OUDrift(eqx.Module):
    """1-D Ornstein-Uhlenbeck drift ``theta * (mu - x)`` as a module (not a closure)."""

    theta: Array
    mu: Array

    def __call__(self, x, u, t):
        return self.theta * (self.mu - x)


def _ou_evolution(theta=1.5, mu=5.0, sigma=0.3):
    """A 1-D OU SDE that reverts toward ``mu`` (default outside [0, 1])."""
    return ContinuousTimeStateEvolution(
        drift=_OUDrift(jnp.asarray(theta), jnp.asarray(mu)),
        diffusion=ScalarDiffusion(sigma, bm_dim=1),
    )


def _wsp_model_factory(theta_prior=False):
    """Build a NumPyro model over a WSP-wrapped OU on the unit box."""

    def model(obs_times=None, obs_values=None, predict_times=None):
        theta = numpyro.sample("theta", dist.Uniform(0.2, 4.0)) if theta_prior else 1.5
        evolution = WSP(
            _ou_evolution(theta=theta, mu=0.5, sigma=0.35),
            _UNIT_BOX,
            alpha=8.0,
            beta=60.0,
            gamma=2.0,
            epsilon=0.05,
        )
        model_obj = DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.array([0.5]), jnp.array([[0.05**2]])
            ),
            state_evolution=evolution,
            observation_model=LinearGaussianObservation(
                H=jnp.eye(1), R=jnp.array([[0.04**2]])
            ),
        )
        return dsx.sample(
            "f",
            model_obj,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )

    return model


# --------------------------------------------------------------------------- #
# Geometry: weight and center pull
# --------------------------------------------------------------------------- #
def test_box_weight_in_unit_interval_and_vanishes_at_faces():
    box = _UNIT_BOX
    xs = jnp.linspace(0.0, 1.0, 101)[:, None]  # (101, 1)
    w = box.weight(xs, alpha=5.0, beta=1000.0)
    assert w.shape == xs.shape
    assert bool(jnp.all(w >= 0.0)) and bool(jnp.all(w <= 1.0))
    # Weight collapses to (near) zero at both faces.
    assert float(box.weight(box.lower, alpha=5.0, beta=1000.0)[0]) < 1e-3
    assert float(box.weight(box.upper, alpha=5.0, beta=1000.0)[0]) < 1e-3
    # ...and saturates to (near) one at the center for a stiff boundary.
    assert float(box.weight(box.center, alpha=5.0, beta=1000.0)[0]) > 0.9


def test_box_weight_preserves_shape_unbatched_and_batched():
    box = _UNIT_BOX
    assert box.weight(jnp.array([0.3]), 5.0, 50.0).shape == (1,)
    assert box.weight(jnp.zeros((7, 1)), 5.0, 50.0).shape == (7, 1)


def test_box_center_pull_points_inward():
    box = _UNIT_BOX
    # Near the lower face the pull is positive (toward the center at 0.5).
    pull_lo = box.center_pull(jnp.array([0.01]), gamma=2.0, epsilon=0.05)
    assert float(pull_lo[0]) > 0.0
    # Near the upper face the pull is negative.
    pull_hi = box.center_pull(jnp.array([0.99]), gamma=2.0, epsilon=0.05)
    assert float(pull_hi[0]) < 0.0


def test_box_rejects_mismatched_or_high_rank_bounds():
    with pytest.raises(ValueError):
        Box(jnp.array([0.0, 0.0]), jnp.array([1.0]))
    with pytest.raises(ValueError):
        Box(jnp.zeros((2, 2)), jnp.ones((2, 2)))


# --------------------------------------------------------------------------- #
# Construction / refinement
# --------------------------------------------------------------------------- #
def test_wsp_refines_to_stochastic_evolution():
    evolution = WSP(_ou_evolution(), _UNIT_BOX)
    model = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.array([0.5]), jnp.array([[0.05**2]])
        ),
        state_evolution=evolution,
        observation_model=LinearGaussianObservation(
            H=jnp.eye(1), R=jnp.array([[0.04**2]])
        ),
    )
    assert isinstance(model.state_evolution, StochasticContinuousTimeStateEvolution)
    assert model.state_dim == 1
    assert model.state_evolution.bm_dim == 1
    assert isinstance(model.state_evolution.drift, WSPDrift)


def test_wsp_ode_inner_refines_to_deterministic():
    ode_inner = ContinuousTimeStateEvolution(
        drift=_OUDrift(jnp.asarray(1.0), jnp.asarray(5.0))
    )
    evolution = WSP(ode_inner, _UNIT_BOX)
    model = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            jnp.array([0.5]), jnp.array([[0.05**2]])
        ),
        state_evolution=evolution,
        observation_model=LinearGaussianObservation(
            H=jnp.eye(1), R=jnp.array([[0.04**2]])
        ),
    )
    assert isinstance(model.state_evolution, DeterministicContinuousTimeStateEvolution)
    assert isinstance(model.state_evolution.drift, WSPDrift)


def test_wsp_rejects_non_continuous_state_evolution():
    with pytest.raises(TypeError):
        WSP(object(), _UNIT_BOX)  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------- #
# Diffusion vanishes at the boundary
# --------------------------------------------------------------------------- #
def test_wsp_diffusion_vanishes_at_boundary():
    sigma = 0.35
    evolution = WSP(_ou_evolution(sigma=sigma), _UNIT_BOX, alpha=8.0, beta=1000.0)
    diffusion = evolution.diffusion
    assert diffusion is not None

    def loading(x):
        return diffusion.as_matrix(x=x, u=None, t=0.0, state_dim=1)

    l_lo = loading(jnp.array([0.0]))
    l_hi = loading(jnp.array([1.0]))
    l_mid = loading(jnp.array([0.5]))
    assert l_mid.shape == (1, 1)
    # Near the faces the diffusion is (essentially) zero...
    assert float(jnp.abs(l_lo).max()) < 1e-3
    assert float(jnp.abs(l_hi).max()) < 1e-3
    # ...while at the center it recovers the inner diffusion scale.
    assert float(l_mid[0, 0]) == pytest.approx(sigma, abs=1e-2)


# --------------------------------------------------------------------------- #
# Simulation containment
# --------------------------------------------------------------------------- #
def test_wsp_simulation_stays_in_box_while_unconstrained_escapes():
    times = jnp.arange(0.0, 6.0, 0.005)
    tol = 5e-2

    # Unconstrained OU reverting to mu=5 escapes the unit box.
    def unconstrained(predict_times=None):
        model_obj = DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.array([0.6]), jnp.array([[0.02**2]])
            ),
            state_evolution=_ou_evolution(theta=1.5, mu=5.0, sigma=0.4),
            observation_model=LinearGaussianObservation(
                H=jnp.eye(1), R=jnp.array([[0.05**2]])
            ),
        )
        return dsx.sample("f", model_obj, predict_times=predict_times)

    # Same OU, wrapped in WSP, stays inside the unit box.
    def constrained(predict_times=None):
        model_obj = DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.array([0.6]), jnp.array([[0.02**2]])
            ),
            state_evolution=WSP(
                _ou_evolution(theta=1.5, mu=5.0, sigma=0.4),
                _UNIT_BOX,
                alpha=8.0,
                beta=50.0,
                gamma=2.0,
                epsilon=0.05,
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.eye(1), R=jnp.array([[0.05**2]])
            ),
        )
        return dsx.sample("f", model_obj, predict_times=predict_times)

    with SDESimulator(source="em_scan", n_simulations=8):
        unc = Predictive(unconstrained, num_samples=1, exclude_deterministic=False)(
            jr.PRNGKey(0), predict_times=times
        )
        con = Predictive(constrained, num_samples=1, exclude_deterministic=False)(
            jr.PRNGKey(0), predict_times=times
        )

    assert float(unc["f_states"].max()) > 1.0 + 0.5  # clearly leaves the box
    con_states = con["f_states"]
    assert float(con_states.min()) > -tol
    assert float(con_states.max()) < 1.0 + tol


# --------------------------------------------------------------------------- #
# Inference: filtering and SVI
# --------------------------------------------------------------------------- #
def _generate_observations(key, model, times, theta_true=2.0):
    gen = Predictive(
        model,
        params={"theta": jnp.array(theta_true)},
        num_samples=1,
        exclude_deterministic=False,
    )
    with SDESimulator(source="em_scan"):
        syn = gen(key, predict_times=times)
    return syn["f_observations"][0, 0]  # (T, 1)


def test_wsp_filter_marginal_loglik_is_finite():
    model = _wsp_model_factory(theta_prior=True)
    times = jnp.arange(0.0, 4.0, 0.1)
    obs_values = _generate_observations(jr.PRNGKey(1), model, times)

    def conditioned():
        with Filter(filter_config=_wsp_particle_filter()):
            return model(obs_times=times, obs_values=obs_values)

    tr = trace(seed(conditioned, jr.PRNGKey(2))).get_trace()
    mll = tr["f_marginal_loglik"]["value"]
    assert bool(jnp.isfinite(mll))


def test_wsp_filter_svi_runs():
    model = _wsp_model_factory(theta_prior=True)
    times = jnp.arange(0.0, 4.0, 0.1)
    obs_values = _generate_observations(jr.PRNGKey(1), model, times)

    def conditioned():
        with Filter(filter_config=_wsp_particle_filter()):
            return model(obs_times=times, obs_values=obs_values)

    guide = AutoNormal(conditioned)
    svi = SVI(conditioned, guide, optax.adam(5e-2), loss=Trace_ELBO())
    result = svi.run(jr.PRNGKey(3), 15, progress_bar=False)
    assert bool(jnp.all(jnp.isfinite(result.losses)))
