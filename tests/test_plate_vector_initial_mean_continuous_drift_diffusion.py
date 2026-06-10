"""Plate-aware continuous model built from explicit drift and diffusion pieces."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Discretizer, Filter, SDESimulator, Smoother
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeUKFConfig,
    EKFConfig,
    EnKFConfig,
    PFConfig,
)
from dynestyx.inference.smoother_configs import ContinuousTimeEKFSmootherConfig
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiagonalDiffusion,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianObservation,
    ScalarDiffusion,
)
from dynestyx.simulators import _slice_tree_for_plate_member
from dynestyx.utils import _has_any_batched_plate_source


class _AlphaDrift(eqx.Module):
    """Continuous-time drift whose (possibly plated) parameter is a module field.

    Storing ``alpha`` as an array field (instead of capturing it in a closure)
    makes it a sliceable pytree leaf, so plate machinery can give each trajectory
    its own ``alpha`` during both simulation and filtering. A closure capture would
    leak the plate axis into single-member computations. See the hierarchical
    tutorial's "sharp edges" section.
    """

    alpha: Array

    def __call__(self, x, u, t):
        x0 = x[..., 0]
        x1 = x[..., 1]
        first = -self.alpha * x0 + 0.1 * x1
        second = -0.05 * x0 - 0.6 * x1
        return jnp.stack([first, second], axis=-1)


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

        drift = _AlphaDrift(alpha=alpha)

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
            id="discretizer-ekf",
        ),
        pytest.param(
            EnKFConfig(
                filter_source="cuthbert",
                n_particles=8,
                crn_seed=jr.PRNGKey(81),
            ),
            id="discretizer-enkf",
        ),
        pytest.param(
            PFConfig(
                filter_source="cuthbert",
                n_particles=16,
                crn_seed=jr.PRNGKey(81),
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


# ---------------------------------------------------------------------------
# Per-member (genuinely batched) diffusion coefficients.
#
# These pin the behavior that the diffusion coefficient is a real, plate-batched
# pytree leaf (no longer a `static` field) and that the plate machinery slices /
# vmaps it per member by its intrinsic event rank. Before this fix, a per-member
# coefficient was hidden in the model's static aux-data, so per-member diffusion
# silently did not work (and a continuous-time filter over it crashed).
# ---------------------------------------------------------------------------


def _per_member_diffusion_model(
    *,
    sigma_vals: Array,
    diffusion_form: str = "full",
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    """Single-plate model whose diffusion scale varies deterministically per member."""
    state_dim = 2
    M = sigma_vals.shape[0]
    initial_mean = jnp.broadcast_to(jnp.array([0.1, 0.05]), (M, state_dim))
    initial_cov = 0.15 * jnp.eye(state_dim)
    obs_cov = (0.08**2) * jnp.eye(state_dim)

    with dsx.plate("trajectories", M):
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=initial_mean,
                covariance_matrix=initial_cov,
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=_AlphaDrift(alpha=jnp.full((M,), 0.4)),
                diffusion=_make_diffusion(diffusion_form, sigma_vals),
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


def _shared_obs_across_members(M, obs_times, diffusion_form):
    """Generate one observation trajectory and broadcast it identically to M members.

    Sharing identical data across members isolates the diffusion coefficient as the
    only thing that varies per member, so any spread in per-member marginal
    log-likelihood is attributable solely to per-member diffusion being used.
    """
    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(101)):
            _per_member_diffusion_model(
                sigma_vals=jnp.full((M,), 0.2),
                diffusion_form=diffusion_form,
                predict_times=obs_times,
            )
    single = tr["f_observations"]["value"][0, 0]  # (T, obs_dim) from member 0
    return jnp.broadcast_to(single, (M,) + single.shape)


@pytest.mark.parametrize("diffusion_form", ["scalar", "diag", "full"])
def test_per_member_diffusion_ct_enkf_uses_per_member_sigma(diffusion_form):
    obs_times = jnp.linspace(0.0, 0.6, 7)
    M = 3
    obs_values = _shared_obs_across_members(M, obs_times, diffusion_form)

    sigma_vals = jnp.array([0.05, 0.25, 0.8])
    with Filter(
        filter_config=ContinuousTimeEnKFConfig(
            n_particles=16,
            crn_seed=jr.PRNGKey(102),
        )
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(103)):
            _per_member_diffusion_model(
                sigma_vals=sigma_vals,
                diffusion_form=diffusion_form,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M,)
    assert jnp.all(jnp.isfinite(loglik))
    # Identical data + distinct per-member sigma => distinct marginal logliks.
    # Equal logliks would mean a single (shared) coefficient leaked to all members.
    assert float(jnp.ptp(loglik)) > 1e-2


def test_per_member_diffusion_discretizer_filter_uses_per_member_sigma():
    obs_times = jnp.arange(6.0)
    M = 3
    # Deterministic discretized observations, shared identically across members.
    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(104)):
                _per_member_diffusion_model(
                    sigma_vals=jnp.full((M,), 0.2),
                    diffusion_form="full",
                    predict_times=obs_times,
                )
    single = tr["f_observations"]["value"][0, 0]
    obs_values = jnp.broadcast_to(single, (M,) + single.shape)

    sigma_vals = jnp.array([0.05, 0.25, 0.8])
    with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(105)):
                _per_member_diffusion_model(
                    sigma_vals=sigma_vals,
                    diffusion_form="full",
                    obs_times=obs_times,
                    obs_values=obs_values,
                )

    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M,)
    assert jnp.all(jnp.isfinite(loglik))
    # The discretizer wraps the diffusion at `state_evolution.cte.diffusion`; the
    # per-member coefficient must still be vmapped per member through that wrapper.
    assert float(jnp.ptp(loglik)) > 1e-2


def _nested_per_member_diffusion_model(
    *,
    sigma_vals: Array,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    """Nested-plate (trajectories x groups) model with per-member diffusion."""
    state_dim = 2
    M, G = sigma_vals.shape
    initial_mean = jnp.broadcast_to(jnp.array([0.1, 0.05]), (M, G, state_dim))
    initial_cov = 0.15 * jnp.eye(state_dim)
    obs_cov = (0.08**2) * jnp.eye(state_dim)
    base = jnp.array([[1.0, 0.0], [0.2, 0.7]])

    with dsx.plate("groups", G):
        with dsx.plate("trajectories", M):
            coeff = sigma_vals[..., None, None] * base  # (M, G, state_dim, state_dim)
            dynamics = DynamicalModel(
                control_dim=0,
                initial_condition=dist.MultivariateNormal(
                    loc=initial_mean,
                    covariance_matrix=initial_cov,
                ),
                state_evolution=ContinuousTimeStateEvolution(
                    drift=_AlphaDrift(alpha=jnp.full((M, G), 0.4)),
                    diffusion=FullDiffusion(coeff),
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


def test_nested_per_member_diffusion_ct_enkf_uses_per_member_sigma():
    obs_times = jnp.linspace(0.0, 0.6, 7)
    # M, G distinct from state_dim/obs_dim (2) so the shared (2, 2) matrices
    # (H, base, covariances) do not collide with the plate sizes.
    M, G = 3, 2
    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(106)):
            _nested_per_member_diffusion_model(
                sigma_vals=jnp.full((M, G), 0.2),
                predict_times=obs_times,
            )
    single = tr["f_observations"]["value"][0, 0, 0]  # (T, obs_dim)
    obs_values = jnp.broadcast_to(single, (M, G) + single.shape)

    sigma_vals = jnp.array([[0.05, 0.1], [0.4, 0.8], [0.15, 0.6]])
    with Filter(
        filter_config=ContinuousTimeEnKFConfig(
            n_particles=16,
            crn_seed=jr.PRNGKey(107),
        )
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(108)):
            _nested_per_member_diffusion_model(
                sigma_vals=sigma_vals,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M, G)
    assert jnp.all(jnp.isfinite(loglik))
    # Nested vmap must strip both plate axes from the (M, G, 2, 2) coefficient.
    assert float(jnp.ptp(loglik)) > 1e-2


# Note: a separate, pre-existing ambiguity affects raw matrix-valued model fields
# (e.g. an observation matrix `H` of shape (state_dim, obs_dim)) whose shape
# coincides with the plate sizes; that is independent of the diffusion coefficient
# handled here, which is why this nested test uses non-colliding plate sizes.


def _sde_model_with_diffusion(diffusion):
    """Minimal single-trajectory SDE model wrapping a given diffusion."""
    state_dim = 2
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(state_dim),
            covariance_matrix=jnp.eye(state_dim),
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=_AlphaDrift(alpha=jnp.array(0.4)),
            diffusion=diffusion,
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(state_dim),
            R=0.1 * jnp.eye(state_dim),
        ),
    )


def test_slice_tree_selects_per_member_diffusion_coefficient():
    # Per-member, single plate (M=3): coefficient (3, 2, 2) -> member's (2, 2).
    coeff = jnp.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)
    se = _sde_model_with_diffusion(FullDiffusion(coeff)).state_evolution
    sliced = _slice_tree_for_plate_member(se, (3,), (1,))
    assert sliced.diffusion.coefficient.shape == (2, 2)
    assert jnp.array_equal(sliced.diffusion.coefficient, coeff[1])

    # Per-member, nested plate (M, G) = (2, 2): coefficient (2, 2, 2, 2).
    coeff_n = jnp.arange(2 * 2 * 2 * 2, dtype=float).reshape(2, 2, 2, 2)
    se_n = _sde_model_with_diffusion(FullDiffusion(coeff_n)).state_evolution
    sliced_n = _slice_tree_for_plate_member(se_n, (2, 2), (1, 0))
    assert sliced_n.diffusion.coefficient.shape == (2, 2)
    assert jnp.array_equal(sliced_n.diffusion.coefficient, coeff_n[1, 0])

    # Shared (2, 2) under a single plate (3,): NOT sliced.
    shared = 0.2 * jnp.eye(2)
    se_s = _sde_model_with_diffusion(FullDiffusion(shared)).state_evolution
    sliced_s = _slice_tree_for_plate_member(se_s, (3,), (1,))
    assert jnp.array_equal(sliced_s.diffusion.coefficient, shared)

    # Shape-collision regression: a shared (2, 2) coefficient under nested plates
    # (2, 2) must stay shared, not be sliced down to a scalar.
    se_c = _sde_model_with_diffusion(FullDiffusion(shared)).state_evolution
    sliced_c = _slice_tree_for_plate_member(se_c, (2, 2), (1, 1))
    assert sliced_c.diffusion.coefficient.shape == (2, 2)
    assert jnp.array_equal(sliced_c.diffusion.coefficient, shared)


def test_sde_model_build_emits_no_static_array_warning(recwarn):
    _ = _sde_model_with_diffusion(FullDiffusion(0.3 * jnp.eye(2)))
    static_warnings = [
        w for w in recwarn.list if "being set as static" in str(w.message)
    ]
    assert not static_warnings, [str(w.message) for w in static_warnings]


class _PerMemberScaledDiffusion(eqx.Module):
    """Callable (time-varying) diffusion coefficient with a per-member array field.

    The diffusion analogue of ``_AlphaDrift``: storing the per-member ``scale`` as
    an array field of an ``eqx.Module`` keeps it a sliceable pytree leaf, so the
    plate machinery recurses into the callable coefficient and gives each member
    its own scale (a plain closure would hide it).
    """

    scale: Array

    def __call__(self, x, u, t):
        base = jnp.array([[1.0, 0.0], [0.2, 0.7]])
        return self.scale[..., None, None] * jnp.exp(-0.1 * t) * base


def test_slice_tree_recurses_into_callable_module_diffusion():
    # A per-member callable coefficient (an eqx.Module with a (M,) field) is sliced
    # by recursing into the callable, not by treating the diffusion as opaque.
    scale = jnp.array([0.1, 0.2, 0.3])
    se = _sde_model_with_diffusion(
        FullDiffusion(_PerMemberScaledDiffusion(scale=scale), bm_dim=2)
    ).state_evolution
    sliced = _slice_tree_for_plate_member(se, (3,), (1,))
    assert sliced.diffusion.coefficient.scale.shape == ()
    assert float(sliced.diffusion.coefficient.scale) == float(scale[1])

    # A shared callable coefficient (plain lambda) carries no array leaves and is
    # left untouched (correctly treated as shared, never indexed).
    se_shared = _sde_model_with_diffusion(
        FullDiffusion(lambda x, u, t: 0.2 * jnp.eye(2), bm_dim=2)
    ).state_evolution
    sliced_shared = _slice_tree_for_plate_member(se_shared, (3,), (1,))
    assert callable(sliced_shared.diffusion.coefficient)


def _per_member_callable_diffusion_model(
    *,
    scale_vals: Array,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    """Single-plate model whose callable diffusion has a per-member scale field."""
    state_dim = 2
    M = scale_vals.shape[0]
    initial_mean = jnp.broadcast_to(jnp.array([0.1, 0.05]), (M, state_dim))
    obs_cov = (0.08**2) * jnp.eye(state_dim)

    with dsx.plate("trajectories", M):
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=initial_mean,
                covariance_matrix=0.15 * jnp.eye(state_dim),
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=_AlphaDrift(alpha=jnp.full((M,), 0.4)),
                diffusion=FullDiffusion(
                    _PerMemberScaledDiffusion(scale=scale_vals), bm_dim=state_dim
                ),
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


def test_per_member_callable_module_diffusion_ct_enkf_uses_per_member_scale():
    obs_times = jnp.linspace(0.0, 0.6, 7)
    M = 3
    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(140)):
            _per_member_callable_diffusion_model(
                scale_vals=jnp.full((M,), 0.2),
                predict_times=obs_times,
            )
    single = tr["f_observations"]["value"][0, 0]
    obs_values = jnp.broadcast_to(single, (M,) + single.shape)

    scale_vals = jnp.array([0.05, 0.25, 0.8])
    with Filter(
        filter_config=ContinuousTimeEnKFConfig(
            n_particles=16,
            crn_seed=jr.PRNGKey(141),
        )
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(142)):
            _per_member_callable_diffusion_model(
                scale_vals=scale_vals,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M,)
    assert jnp.all(jnp.isfinite(loglik))
    # The per-member scale lives inside a callable coefficient; identical data plus
    # distinct per-member scales must still yield distinct marginal logliks.
    assert float(jnp.ptp(loglik)) > 1e-2


def test_callable_diffusion_is_recognized_as_sole_plate_source():
    # Regression: the plate-source detector must agree with the slicer/vmap. A
    # callable eqx.Module diffusion (per-member ``scale``) as the ONLY plate-batched
    # source (shared initial condition, shared drift, non-colliding obs) must be
    # detected; otherwise the simulator/filter/smoother alignment guards spuriously
    # reject a model the slicers can handle.
    model = _sde_model_with_diffusion(
        FullDiffusion(_PerMemberScaledDiffusion(scale=jnp.array([0.05, 0.25, 0.8])))
    )
    assert _has_any_batched_plate_source(model, (3,)) is True
    # The slicer agrees: member 1's scale is selected.
    sliced = _slice_tree_for_plate_member(model.state_evolution, (3,), (1,))
    assert float(sliced.diffusion.coefficient.scale) == 0.25
    # A shared callable carries no per-member field, so it is not a plate source.
    shared = _sde_model_with_diffusion(FullDiffusion(lambda x, u, t: 0.2 * jnp.eye(2)))
    assert _has_any_batched_plate_source(shared, (3,)) is False


def _shared_ic_callable_diffusion_model(
    *,
    scale_vals: Array,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    """Plate model whose ONLY per-member source is a callable diffusion field.

    Initial condition, drift, and observation model are all shared, so the
    callable coefficient's per-member ``scale`` is the sole plate-batched source —
    the exact case that trips an alignment guard inconsistent with the slicers.
    """
    state_dim = 2
    M = scale_vals.shape[0]
    obs_cov = (0.08**2) * jnp.eye(state_dim)
    with dsx.plate("trajectories", M):
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.array([0.1, 0.05]),
                covariance_matrix=0.15 * jnp.eye(state_dim),
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=_AlphaDrift(alpha=jnp.asarray(0.4)),
                diffusion=FullDiffusion(
                    _PerMemberScaledDiffusion(scale=scale_vals), bm_dim=state_dim
                ),
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


def test_callable_diffusion_sole_plate_source_simulates_and_filters():
    M = 3
    times = jnp.linspace(0.0, 0.6, 6)
    # Forward simulation: the plated simulator gate must not reject the model.
    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(150)):
            _shared_ic_callable_diffusion_model(
                scale_vals=jnp.array([0.05, 0.25, 0.8]),
                predict_times=times,
            )
    states = tr["f_states"]["value"]
    assert states.shape[0] == M
    assert jnp.all(jnp.isfinite(states))

    # Filtering with SHARED observations: the only plate source is the callable
    # diffusion, so the alignment guard must accept it.
    obs_values = tr["f_observations"]["value"][0, 0]  # (T, obs_dim), shared
    with Filter(
        filter_config=ContinuousTimeEnKFConfig(n_particles=16, crn_seed=jr.PRNGKey(151))
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(152)):
            _shared_ic_callable_diffusion_model(
                scale_vals=jnp.array([0.05, 0.25, 0.8]),
                obs_times=times,
                obs_values=obs_values,
            )
    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M,)
    assert jnp.all(jnp.isfinite(loglik))
    assert float(jnp.ptp(loglik)) > 1e-2


def test_per_member_diffusion_ct_smoother_uses_per_member_sigma():
    # Smoothers share _make_plate_in_axes / the alignment guard with filters; pin
    # per-member diffusion through the continuous-time (CD-EKS) smoother vmap path.
    obs_times = jnp.linspace(0.0, 0.6, 7)
    M = 3
    obs_values = _shared_obs_across_members(M, obs_times, "full")

    sigma_vals = jnp.array([0.05, 0.25, 0.8])
    with Smoother(smoother_config=ContinuousTimeEKFSmootherConfig()):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(160)):
            _per_member_diffusion_model(
                sigma_vals=sigma_vals,
                diffusion_form="full",
                obs_times=obs_times,
                obs_values=obs_values,
            )

    loglik = tr["f_marginal_loglik"]["value"]
    assert loglik.shape == (M,)
    assert jnp.all(jnp.isfinite(loglik))
    assert float(jnp.ptp(loglik)) > 1e-2
