"""Plate-aware continuous model built from explicit drift and diffusion pieces."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Discretizer, Filter, SDESimulator
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeUKFConfig,
    EKFConfig,
    EnKFConfig,
    PFConfig,
)
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiagonalDiffusion,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianObservation,
    ScalarDiffusion,
)


def _make_diffusion(diffusion_form, sigma):
    state_dim = 2
    if diffusion_form == "scalar":
        coeff = sigma[..., None] if jnp.ndim(sigma) > 0 else sigma
        return ScalarDiffusion(coeff, bm_dim=state_dim)
    if diffusion_form == "diag":
        coeff = (sigma[..., None] if jnp.ndim(sigma) > 0 else sigma) * jnp.array(
            [1.0, 0.6]
        )
        return DiagonalDiffusion(coeff, bm_dim=state_dim)
    if diffusion_form == "full":
        base = jnp.array([[1.0, 0.0], [0.2, 0.7]])
        coeff = (sigma[..., None, None] if jnp.ndim(sigma) > 0 else sigma) * base
        return FullDiffusion(coeff)
    raise ValueError(f"Unknown diffusion form: {diffusion_form}")


def _manual_plate_vector_initial_mean_continuous_model(
    *,
    diffusion_form,
    alpha_mode,
    diffusion_mode,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=3,
):
    state_dim = 2
    # TODO: we should be able to handle this without broadcasting manually.
    initial_mean = jnp.broadcast_to(jnp.array([0.1, 0.05]), (M, state_dim))
    initial_cov = 0.15 * jnp.eye(state_dim)
    obs_cov = (0.08**2) * jnp.eye(state_dim)

    alpha_shared = None
    sigma_shared = None
    if alpha_mode == "shared":
        alpha_shared = numpyro.sample("alpha_shared", dist.Uniform(0.1, 0.8))
    if diffusion_mode == "shared":
        sigma_shared = numpyro.sample("sigma_shared", dist.Uniform(0.15, 0.25))

    with dsx.plate("trajectories", M):
        alpha = (
            alpha_shared
            if alpha_shared is not None
            else numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
        )
        sigma = (
            sigma_shared
            if sigma_shared is not None
            else numpyro.sample("sigma", dist.Uniform(0.15, 0.25))
        )

        def drift(x, u, t):
            x0 = x[..., 0]
            x1 = x[..., 1]
            lead_shape = jnp.broadcast_shapes(jnp.shape(alpha), jnp.shape(x0))
            first = -alpha * x0 + 0.1 * x1
            second = jnp.broadcast_to(-0.05 * x0 - 0.6 * x1, lead_shape)
            return jnp.stack([first, second], axis=-1)

        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=initial_mean,
                covariance_matrix=initial_cov,
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=drift,
                diffusion=_make_diffusion(diffusion_form, sigma),
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.eye(state_dim),
                R=obs_cov,
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _make_shared_shared_discretized_observations(diffusion_form):
    obs_times = jnp.arange(6.0)
    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(80)):
                _manual_plate_vector_initial_mean_continuous_model(
                    diffusion_form=diffusion_form,
                    alpha_mode="shared",
                    diffusion_mode="shared",
                    predict_times=obs_times,
                    M=3,
                )
    return obs_times, tr["f_observations"]["value"][:, 0]


def _make_shared_shared_continuous_observations(diffusion_form):
    obs_times = jnp.linspace(0.0, 0.5, 6)
    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(90)):
            _manual_plate_vector_initial_mean_continuous_model(
                diffusion_form=diffusion_form,
                alpha_mode="shared",
                diffusion_mode="shared",
                predict_times=obs_times,
                M=3,
            )
    return obs_times, tr["f_observations"]["value"][:, 0]


@pytest.mark.parametrize("diffusion_form", ["scalar", "diag", "full"])
@pytest.mark.parametrize(
    "alpha_mode",
    [
        pytest.param("shared", id="shared"),
        pytest.param(
            "plated",
            marks=pytest.mark.xfail(
                reason=(
                    "Explicit continuous-time drift callables still return "
                    "plate-batched values during single-member simulation when "
                    "the drift parameter alpha is plated."
                ),
                strict=True,
            ),
            id="plated",
        ),
    ],
)
@pytest.mark.parametrize(
    "diffusion_mode",
    [
        pytest.param("shared", id="shared"),
        pytest.param("plated", id="plated"),
    ],
)
def test_manual_continuous_model_discretizer_forward_shapes_and_sites(
    diffusion_form,
    alpha_mode,
    diffusion_mode,
):
    t = jnp.arange(6.0)
    M = 3

    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(60)):
                _manual_plate_vector_initial_mean_continuous_model(
                    diffusion_form=diffusion_form,
                    alpha_mode=alpha_mode,
                    diffusion_mode=diffusion_mode,
                    predict_times=t,
                    M=M,
                )

    expected_shape = (M, 1, len(t), 2)
    assert tr["f_states"]["value"].shape == expected_shape
    assert tr["f_observations"]["value"].shape == expected_shape

    alpha_site = "alpha_shared" if alpha_mode == "shared" else "alpha"
    sigma_site = "sigma_shared" if diffusion_mode == "shared" else "sigma"
    assert tr[alpha_site]["value"].shape == (() if alpha_mode == "shared" else (M,))
    assert tr[sigma_site]["value"].shape == (() if diffusion_mode == "shared" else (M,))


@pytest.mark.parametrize("diffusion_form", ["scalar", "diag", "full"])
def test_manual_continuous_model_shared_parameters_ct_enkf_shapes(diffusion_form):
    obs_times = jnp.linspace(0.0, 0.5, 6)
    M = 3

    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(70)):
            _manual_plate_vector_initial_mean_continuous_model(
                diffusion_form=diffusion_form,
                alpha_mode="shared",
                diffusion_mode="shared",
                predict_times=obs_times,
                M=M,
            )
    obs_values = tr["f_observations"]["value"][:, 0]

    with Filter(
        filter_config=ContinuousTimeEnKFConfig(
            n_particles=8,
            crn_seed=jr.PRNGKey(71),
        )
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(72)):
            _manual_plate_vector_initial_mean_continuous_model(
                diffusion_form=diffusion_form,
                alpha_mode="shared",
                diffusion_mode="shared",
                obs_times=obs_times,
                obs_values=obs_values,
                M=M,
            )

    assert tr["f_marginal_loglik"]["value"].shape == (M,)


@pytest.mark.parametrize("diffusion_form", ["scalar", "diag", "full"])
@pytest.mark.parametrize(
    "filter_config",
    [
        pytest.param(
            EKFConfig(filter_source="cuthbert"),
            marks=pytest.mark.xfail(
                reason=(
                    "The hand-written drift/diffusion model still hits the same "
                    "cuthbert EKF batched-initial-mean reshape bug after "
                    "discretization."
                ),
                strict=True,
            ),
            id="discretizer-ekf",
        ),
        pytest.param(
            EnKFConfig(
                filter_source="cuthbert",
                n_particles=8,
                crn_seed=jr.PRNGKey(81),
            ),
            marks=pytest.mark.xfail(
                reason=(
                    "The hand-written drift/diffusion model still hits the same "
                    "cuthbert EnKF plate-axis bug after discretization."
                ),
                strict=True,
            ),
            id="discretizer-enkf",
        ),
        pytest.param(
            PFConfig(
                filter_source="cuthbert",
                n_particles=16,
                crn_seed=jr.PRNGKey(81),
            ),
            marks=pytest.mark.xfail(
                reason=(
                    "The hand-written drift/diffusion model still hits the same "
                    "cuthbert PF batched state/control mismatch after "
                    "discretization."
                ),
                strict=True,
            ),
            id="discretizer-pf",
        ),
    ],
)
def test_manual_continuous_model_shared_parameters_discretizer_filters(
    diffusion_form,
    filter_config,
):
    obs_times, obs_values = _make_shared_shared_discretized_observations(diffusion_form)

    with Filter(filter_config=filter_config):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(82)):
                _manual_plate_vector_initial_mean_continuous_model(
                    diffusion_form=diffusion_form,
                    alpha_mode="shared",
                    diffusion_mode="shared",
                    obs_times=obs_times,
                    obs_values=obs_values,
                    M=3,
                )

    assert tr["f_marginal_loglik"]["value"].shape == (3,)


@pytest.mark.parametrize("diffusion_form", ["scalar", "diag", "full"])
@pytest.mark.parametrize(
    "filter_config",
    [
        pytest.param(
            ContinuousTimeEnKFConfig(
                n_particles=8,
                crn_seed=jr.PRNGKey(91),
            ),
            id="ct-enkf",
        ),
        pytest.param(ContinuousTimeEKFConfig(), id="ct-ekf"),
        pytest.param(
            ContinuousTimeDPFConfig(
                n_particles=16,
                crn_seed=jr.PRNGKey(91),
            ),
            marks=pytest.mark.xfail(
                reason=(
                    "The hand-written drift/diffusion model still triggers the "
                    "continuous-time PF batched solver-compatibility bug."
                ),
                strict=True,
            ),
            id="ct-pf",
        ),
        pytest.param(ContinuousTimeUKFConfig(), id="ct-ukf"),
    ],
)
def test_manual_continuous_model_shared_parameters_ct_filters(
    diffusion_form,
    filter_config,
):
    obs_times, obs_values = _make_shared_shared_continuous_observations(diffusion_form)

    with Filter(filter_config=filter_config):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(92)):
            _manual_plate_vector_initial_mean_continuous_model(
                diffusion_form=diffusion_form,
                alpha_mode="shared",
                diffusion_mode="shared",
                obs_times=obs_times,
                obs_values=obs_values,
                M=3,
            )

    assert tr["f_marginal_loglik"]["value"].shape == (3,)
