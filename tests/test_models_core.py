import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.handlers as nhandlers
import pytest

import dynestyx as dsx
from dynestyx.inference.integrations.cd_dynamax.utils import (
    _normalize_cd_dynamax_diffusion,
    dsx_to_cd_dynamax,
    gaussian_to_nlgssm_params,
)
from dynestyx.models.core import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.models.diffusions import (
    DiffusionType,
    diffusion_as_matrix,
    diffusion_covariance,
    evaluate_diffusion,
)
from dynestyx.simulators import DiscreteTimeSimulator


def _initial_condition_2d() -> dist.MultivariateNormal:
    return dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=jnp.eye(2),
    )


def _observation_model_2d(x, u, t):
    del u, t
    return dist.MultivariateNormal(
        loc=x,
        covariance_matrix=jnp.eye(2),
    )


def test_discrete_state_evolution_shape_validation() -> None:
    def bad_state_evolution(x, u, t_now, t_next):
        del x, u, t_now, t_next
        return dist.Normal(loc=0.0, scale=1.0)

    with pytest.raises(ValueError, match="State transition shape is inconsistent"):
        DynamicalModel(
            initial_condition=_initial_condition_2d(),
            state_evolution=bad_state_evolution,
            observation_model=_observation_model_2d,
            control_dim=0,
        )


def test_discrete_state_evolution_receives_probe_inputs() -> None:
    seen: dict[str, tuple[int, ...] | float] = {}

    def state_evolution(x, u, t_now, t_next):
        seen["x_shape"] = x.shape
        seen["u_shape"] = () if u is None else u.shape
        seen["t_now"] = float(t_now)
        seen["t_next"] = float(t_next)
        return dist.MultivariateNormal(
            loc=x + u,
            covariance_matrix=jnp.eye(2),
        )

    DynamicalModel(
        initial_condition=_initial_condition_2d(),
        state_evolution=state_evolution,
        observation_model=_observation_model_2d,
        control_dim=2,
    )

    assert seen["x_shape"] == (2,)
    assert seen["u_shape"] == (2,)
    assert seen["t_now"] == 0.0
    assert seen["t_next"] == 1.0


def test_gaussian_state_evolution_supports_callable_cov() -> None:
    def F(x, u, t_now, t_next):
        del u, t_now, t_next
        return x

    def cov_fn(x, u, t_now, t_next):
        del u, t_now, t_next
        return (1.0 + jnp.square(x[0])) * jnp.eye(2)

    evo = dsx.GaussianStateEvolution(F=F, cov=cov_fn)
    x = jnp.array([2.0, -1.0])
    d = evo(x=x, u=None, t_now=jnp.array(0.0), t_next=jnp.array(1.0))
    assert jnp.allclose(d.covariance_matrix, 5.0 * jnp.eye(2))


def test_gaussian_to_nlgssm_params_rejects_callable_cov() -> None:
    def F(x, u, t_now, t_next):
        del u, t_now, t_next
        return x

    def h(x, u, t):
        del u, t
        return x

    def cov_fn(x, u, t_now, t_next):
        del x, u, t_now, t_next
        return jnp.eye(2)

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=jnp.eye(2)
        ),
        state_evolution=dsx.GaussianStateEvolution(F=F, cov=cov_fn),
        observation_model=dsx.GaussianObservation(h=h, R=jnp.eye(2)),
    )

    with pytest.raises(TypeError, match="array-valued process covariance"):
        gaussian_to_nlgssm_params(dynamics)


def test_continuous_state_evolution_rejects_scalar_drift_for_1d_state() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: jnp.array(0.0),
    )

    def observation_model(x, u, t):
        del u, t
        return dist.Normal(loc=x[0], scale=1.0)

    with pytest.raises(ValueError, match="State drift shape is inconsistent"):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(1),
                covariance_matrix=jnp.eye(1),
            ),
            state_evolution=state_evolution,
            observation_model=observation_model,
            control_dim=0,
        )


def test_discrete_state_evolution_shape_validation_with_inferred_dims() -> None:
    def bad_state_evolution(x, u, t_now, t_next):
        del x, u, t_now, t_next
        return dist.Normal(loc=0.0, scale=1.0)

    with pytest.raises(ValueError, match="State transition shape is inconsistent"):
        DynamicalModel(
            initial_condition=_initial_condition_2d(),
            state_evolution=bad_state_evolution,
            observation_model=_observation_model_2d,
            control_dim=0,
        )


def test_continuous_state_evolution_rejects_scalar_drift_with_inferred_dims() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: jnp.array(0.0),
    )

    def observation_model(x, u, t):
        del u, t
        return dist.Normal(loc=x[0], scale=1.0)

    with pytest.raises(ValueError, match="State drift shape is inconsistent"):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(1),
                covariance_matrix=jnp.eye(1),
            ),
            state_evolution=state_evolution,
            observation_model=observation_model,
            control_dim=0,
        )


def test_discrete_categorical_state_space_infers_num_categories() -> None:
    seen_x_shape: tuple[int, ...] | None = None
    seen_x_dtype: jnp.dtype | None = None
    probs = jnp.array([0.2, 0.5, 0.3])
    transition_matrix = jnp.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])

    def state_evolution(x, u, t_now, t_next):
        nonlocal seen_x_shape, seen_x_dtype
        del u, t_now, t_next
        seen_x_shape = jnp.shape(x)
        seen_x_dtype = x.dtype
        # HMM-like indexing should work during constructor probing.
        return dist.Categorical(probs=transition_matrix[x])

    def observation_model(x, u, t):
        del u, t
        return dist.Normal(loc=jnp.asarray(x, dtype=jnp.float32), scale=1.0)

    model = DynamicalModel(
        initial_condition=dist.Categorical(probs=probs),
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_dim=0,
    )

    assert model.state_dim == 3
    assert model.categorical_state is True
    assert model.observation_dim == 1
    assert seen_x_shape == ()
    assert seen_x_dtype is not None
    assert jnp.issubdtype(seen_x_dtype, jnp.integer)


def test_categorical_state_override_compatible() -> None:
    model = DynamicalModel(
        initial_condition=dist.Categorical(probs=jnp.array([0.2, 0.8])),
        state_evolution=lambda x, u, t_now, t_next: dist.Categorical(
            probs=jnp.array([0.6, 0.4])
        ),
        observation_model=lambda x, u, t: dist.Normal(
            loc=jnp.asarray(x, dtype=jnp.float32), scale=1.0
        ),
        control_dim=0,
        categorical_state=True,
    )
    assert model.categorical_state is True


def test_categorical_state_override_incompatible_raises() -> None:
    with pytest.raises(
        ValueError, match="categorical_state does not match inferred initial_condition"
    ):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=jnp.eye(2),
            ),
            state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
                loc=x, covariance_matrix=jnp.eye(2)
            ),
            observation_model=lambda x, u, t: dist.MultivariateNormal(
                loc=x, covariance_matrix=jnp.eye(2)
            ),
            control_dim=0,
            categorical_state=True,
        )


def test_continuous_state_evolution_infers_bm_dim() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=lambda x, u, t: jnp.eye(2, 3),
    )

    def observation_model(x, u, t):
        del u, t
        return dist.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(2))

    _ = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=jnp.eye(2),
        ),
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_dim=0,
    )

    assert state_evolution.bm_dim == 3


def test_continuous_state_evolution_bm_dim_override_compatible() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=lambda x, u, t: jnp.eye(2, 3),
        bm_dim=3,
    )

    def observation_model(x, u, t):
        del u, t
        return dist.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(2))

    _ = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=jnp.eye(2),
        ),
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_dim=0,
    )

    assert state_evolution.bm_dim == 3


def test_continuous_state_evolution_bm_dim_override_mismatch_raises() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=lambda x, u, t: jnp.eye(2, 3),
        bm_dim=2,
    )

    def observation_model(x, u, t):
        del u, t
        return dist.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(2))

    with pytest.raises(
        ValueError, match="bm_dim does not match inferred diffusion_coefficient"
    ):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=jnp.eye(2),
            ),
            state_evolution=state_evolution,
            observation_model=observation_model,
            control_dim=0,
        )


def test_continuous_state_evolution_bm_dim_without_diffusion_raises() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        bm_dim=2,
    )

    def observation_model(x, u, t):
        del u, t
        return dist.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(2))

    with pytest.raises(
        ValueError, match="bm_dim cannot be set when diffusion_coefficient"
    ):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=jnp.eye(2),
            ),
            state_evolution=state_evolution,
            observation_model=observation_model,
            control_dim=0,
        )


def test_continuous_state_evolution_requires_explicit_bm_dim_for_diag_diffusion() -> (
    None
):
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=jnp.ones((2,)),
    )

    with pytest.raises(ValueError, match="diag diffusion requires explicit bm_dim"):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=jnp.eye(2),
            ),
            state_evolution=state_evolution,
            observation_model=_observation_model_2d,
            control_dim=0,
        )


def test_continuous_state_evolution_requires_explicit_bm_dim_for_scalar_diffusion() -> (
    None
):
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=jnp.array(0.3),
    )

    with pytest.raises(ValueError, match="scalar diffusion requires explicit bm_dim"):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=jnp.eye(2),
            ),
            state_evolution=state_evolution,
            observation_model=_observation_model_2d,
            control_dim=0,
        )


def test_continuous_state_evolution_rejects_invalid_shorthand_trailing_dim() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=jnp.ones((3,)),
    )

    with pytest.raises(ValueError, match="treated as scalar/diagonal shorthand"):
        DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=jnp.eye(2),
            ),
            state_evolution=state_evolution,
            observation_model=_observation_model_2d,
            control_dim=0,
        )


@pytest.mark.parametrize(
    (
        "diffusion_coefficient",
        "diffusion_type",
        "bm_dim",
        "expected_matrix",
        "expected_cov",
    ),
    [
        (
            jnp.array(0.5),
            "scalar",
            1,
            0.5 * jnp.ones((2, 1)),
            0.25 * jnp.ones((2, 2)),
        ),
        (
            jnp.array(0.5),
            "scalar",
            2,
            0.5 * jnp.eye(2),
            0.25 * jnp.eye(2),
        ),
        (
            jnp.array([0.2, 0.4]),
            "diag",
            1,
            jnp.array([[0.2], [0.4]]),
            jnp.array([[0.04, 0.08], [0.08, 0.16]]),
        ),
        (
            jnp.array([0.2, 0.4]),
            "diag",
            2,
            jnp.diag(jnp.array([0.2, 0.4])),
            jnp.diag(jnp.array([0.04, 0.16])),
        ),
    ],
)
def test_diffusion_helpers_preserve_shorthand_semantics(
    diffusion_coefficient,
    diffusion_type,
    bm_dim,
    expected_matrix,
    expected_cov,
) -> None:
    evaluated = evaluate_diffusion(
        diffusion_coefficient,
        diffusion_type=diffusion_type,
        bm_dim=bm_dim,
        x=jnp.zeros(2),
        u=None,
        t=jnp.array(0.0),
        state_dim=2,
    )

    assert jnp.allclose(
        diffusion_as_matrix(evaluated, state_dim=2),
        expected_matrix,
    )
    assert jnp.allclose(
        diffusion_covariance(evaluated, state_dim=2),
        expected_cov,
    )


@pytest.mark.parametrize("callable_form", ["scalar", "diag", "full"])
def test_diffusion_helpers_support_callable_shorthands(callable_form: str) -> None:
    diffusion_type: DiffusionType
    if callable_form == "scalar":
        diffusion_coefficient = lambda x, u, t: 0.5 + 0.0 * t
        diffusion_type = "scalar"
        bm_dim = 2
        expected = 0.5 * jnp.eye(2)
    elif callable_form == "diag":
        diffusion_coefficient = lambda x, u, t: jnp.array([0.2, 0.4]) + 0.0 * t
        diffusion_type = "diag"
        bm_dim = 2
        expected = jnp.diag(jnp.array([0.2, 0.4]))
    else:
        diffusion_coefficient = lambda x, u, t: jnp.eye(2) * (1.0 + 0.0 * t)
        diffusion_type = "full"
        bm_dim = None
        expected = jnp.eye(2)

    evaluated = evaluate_diffusion(
        diffusion_coefficient,
        diffusion_type=diffusion_type,
        bm_dim=bm_dim,
        x=jnp.zeros(2),
        u=None,
        t=jnp.array(1.0),
        state_dim=2,
    )

    assert jnp.allclose(diffusion_as_matrix(evaluated, state_dim=2), expected)


def test_cd_dynamax_rejects_diffusion_with_bm_dim_exceeds_state_dim() -> None:
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=jnp.ones((1, 2)),
        diffusion_type="full",
        bm_dim=2,
    )
    with pytest.raises(ValueError, match="bm_dim <= state_dim"):
        _normalize_cd_dynamax_diffusion(state_evolution, state_dim=1)(
            jnp.zeros(1),
            None,
            jnp.array(0.0),
        )


def test_continuous_cd_dynamax_rejects_diffusion_with_bm_dim_exceeds_state_dim_early() -> (
    None
):
    state_evolution = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: x,
        diffusion_coefficient=jnp.ones((1, 2)),
        diffusion_type="full",
        bm_dim=2,
    )
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1),
            covariance_matrix=jnp.eye(1),
        ),
        state_evolution=state_evolution,
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[1.0]]),
        ),
        control_dim=0,
    )

    with pytest.raises(
        ValueError, match="Continuous cd-dynamax filters require bm_dim <= state_dim"
    ):
        dsx_to_cd_dynamax(dynamics)


def test_discrete_state_evolution_bm_dim_override_raises() -> None:
    def state_evolution(x, u, t_now, t_next):
        del u, t_now, t_next
        return dist.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(2))

    state_evolution.bm_dim = 2  # type: ignore[attr-defined]

    with pytest.raises(
        ValueError, match="bm_dim can only be set for continuous-time models"
    ):
        DynamicalModel(
            initial_condition=_initial_condition_2d(),
            state_evolution=state_evolution,
            observation_model=_observation_model_2d,
            control_dim=0,
        )


def test_dirichlet_initial_condition_auto_infers_state_dim() -> None:
    k = 4

    def state_evolution(x, u, t_now, t_next):
        del u, t_now, t_next
        return dist.Dirichlet(concentration=x + 1.0)

    def observation_model(x, u, t):
        del u, t
        return dist.MultivariateNormal(
            loc=x,
            covariance_matrix=jnp.eye(k) * 1e-3,
        )

    model = DynamicalModel(
        initial_condition=dist.Dirichlet(concentration=jnp.ones(k)),
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_dim=0,
    )

    assert model.state_dim == k
    assert model.categorical_state is False


def test_dirichlet_initial_condition_state_dim_override_mismatch_raises() -> None:
    k = 3
    with pytest.raises(
        ValueError, match="state_dim does not match inferred initial_condition shape"
    ):
        DynamicalModel(
            initial_condition=dist.Dirichlet(concentration=jnp.ones(k)),
            state_evolution=lambda x, u, t_now, t_next: dist.Dirichlet(
                concentration=x + 1.0
            ),
            observation_model=lambda x, u, t: dist.MultivariateNormal(
                loc=x,
                covariance_matrix=jnp.eye(k) * 1e-3,
            ),
            control_dim=0,
            state_dim=k + 1,
        )


# ---------------------------------------------------------------------------
# t0 field and simulator validation tests
# ---------------------------------------------------------------------------


def _simple_discrete_model(t0=None):
    """Minimal 1-D discrete-time DynamicalModel for t0 tests."""
    return DynamicalModel(
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(x, 0.1),
        observation_model=lambda x, u, t: dist.Normal(x, 0.1),
        control_dim=0,
        t0=t0,
    )


def test_t0_defaults_to_none() -> None:
    model = _simple_discrete_model()
    assert model.t0 is None


def test_t0_stored_when_provided() -> None:
    model = _simple_discrete_model(t0=2.5)
    assert isinstance(model.t0, jax.Array)
    assert model.t0.shape == ()
    assert model.t0 == 2.5


def _run_model_with_simulator(
    dynamics,
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
):
    """Run a dynamics model inside DiscreteTimeSimulator and return the trace."""

    def model():
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
        )

    with DiscreteTimeSimulator():
        tr = nhandlers.trace(nhandlers.seed(model, rng_seed=0)).get_trace()
    return tr


def test_t0_no_error_when_matching_obs_times() -> None:
    predict_times = jnp.array([3.0, 4.0, 5.0])
    dynamics = _simple_discrete_model(t0=3.0)
    # Should not raise
    _run_model_with_simulator(dynamics, predict_times=predict_times)


def test_t0_mismatch_raises_informative_error() -> None:
    predict_times = jnp.array([3.0, 4.0, 5.0])
    dynamics = _simple_discrete_model(t0=0.0)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match=r"dynamics\.t0=0\.0 does not match .*\[0\]",
    ):
        _run_model_with_simulator(dynamics, predict_times=predict_times)


def test_obs_times_strictly_increasing_validation() -> None:
    predict_times = jnp.array([3.0, 2.0, 5.0])
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match=".*must be strictly increasing",
    ):
        _run_model_with_simulator(_simple_discrete_model(), predict_times=predict_times)


def test_ctrl_times_strictly_increasing_validation() -> None:
    ctrl_times = jnp.array([3.0, 2.0, 5.0])
    ctrl_values = jnp.zeros((3, 1))
    predict_times = jnp.array([3.0, 4.0, 5.0])
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="ctrl_times must be strictly increasing",
    ):
        _run_model_with_simulator(
            _simple_discrete_model(),
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
        )
