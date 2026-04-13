"""Smoke tests for hierarchical plate-aware simulator + discretizer behavior."""

import re

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx import (
    DiscreteTimeSimulator,
    Discretizer,
    Filter,
    ODESimulator,
    SDESimulator,
)
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEnKFConfig,
    EKFConfig,
    HMMConfig,
    KFConfig,
    PFConfig,
)
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianStateEvolution,
    LinearGaussianObservation,
)
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete
from dynestyx.models.state_evolution import AffineDrift


def _plate_discrete_lti_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    state_dim = 2
    Q = 0.1 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.25]])

    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
        A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _nested_plate_discrete_lti_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    G=2,
    M=2,
):
    state_dim = 2
    Q = 0.1 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.25]])

    with dsx.plate("groups", G):
        beta = numpyro.sample("beta", dist.Normal(0.0, 0.2))
        with dsx.plate("trajectories", M):
            alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
            A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
            A = jnp.broadcast_to(A_base, (M, G, 2, 2)).copy()
            A = A.at[:, :, 0, 0].set(alpha)
            A = A.at[:, :, 1, 1].set(0.8 + beta[None, :])
            dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )


def _plate_continuous_sde_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
        A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        L = 0.2 * jnp.array([[1.0], [0.5]])
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.25]])
        dynamics = LTI_continuous(A=A, L=L, H=H, R=R)
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _plate_continuous_ode_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
        A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        drift = AffineDrift(A=A)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2), covariance_matrix=jnp.eye(2)
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=drift, diffusion_coefficient=None
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.array([[1.0, 0.0]]), R=jnp.array([[0.25]])
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _plate_hmm_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    K = 2
    trans = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    means = jnp.array([-1.0, 1.0])

    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.Categorical(probs=jnp.ones(K) / K),
        state_evolution=lambda x, u, t_now, t_next: dist.Categorical(probs=trans[x]),
        observation_model=lambda x, u, t: dist.Normal(means[x], 0.3),
    )
    with dsx.plate("trajectories", M):
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


class _PlateNonlinearTransition(eqx.Module):
    beta: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        return 0.75 * x + self.beta * jnp.tanh(x)


def _plate_nonlinear_discrete_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    with dsx.plate("trajectories", M):
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 0.4))
        beta = 0.8 * (jnp.tanh(beta_raw) / 2.0)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(1), covariance_matrix=0.2 * jnp.eye(1)
            ),
            state_evolution=GaussianStateEvolution(
                F=_PlateNonlinearTransition(beta=beta),
                cov=0.05 * jnp.eye(1),
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.array([[1.0]]), R=jnp.array([[0.1]])
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _make_obs_values(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)


@pytest.mark.parametrize("source", ["diffrax", "em_scan"])
def test_plate_forward_discrete_ode_sde_shapes(source):
    t = jnp.arange(5.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            _plate_discrete_lti_model(predict_times=t, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (2, 1, len(t))
    assert tr["f_observations"]["value"].shape[:3] == (2, 1, len(t))

    with ODESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
            _plate_continuous_ode_model(predict_times=t, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (2, 1, len(t))
    assert tr["f_observations"]["value"].shape[:3] == (2, 1, len(t))

    with SDESimulator(source=source):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(2)):
            _plate_continuous_sde_model(predict_times=t, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (2, 1, len(t))
    assert tr["f_observations"]["value"].shape[:3] == (2, 1, len(t))


def test_plate_conditioning_discrete_single_and_nested():
    t = jnp.arange(5.0)
    obs_single = _make_obs_values((2, len(t), 1))
    obs_nested = _make_obs_values((2, 2, len(t), 1))

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(3)):
            _plate_discrete_lti_model(obs_times=t, obs_values=obs_single, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (2, 1, len(t))

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(4)):
            _nested_plate_discrete_lti_model(
                obs_times=t, obs_values=obs_nested, G=2, M=2
            )
    assert tr["f_times"]["value"].shape == (2, 2, 1, len(t))
    assert tr["f_states"]["value"].shape[:4] == (2, 2, 1, len(t))


def test_plate_conditioning_ode_single():
    t = jnp.linspace(0.0, 0.4, 5)
    obs = _make_obs_values((2, len(t), 1))
    with ODESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(5)):
            _plate_continuous_ode_model(obs_times=t, obs_values=obs, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (2, 1, len(t))


def test_plate_nonlinear_discrete_single_sample_under_plate():
    t = jnp.arange(6.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(16)):
            _plate_nonlinear_discrete_model(predict_times=t, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (2, 1, len(t))
    assert tr["f_observations"]["value"].shape[:3] == (2, 1, len(t))

    obs = tr["f_observations"]["value"][:, 0]
    with DiscreteTimeSimulator():
        with Filter(filter_config=EKFConfig(filter_source="cd_dynamax")):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(17)):
                _plate_nonlinear_discrete_model(
                    obs_times=t,
                    obs_values=obs,
                    predict_times=t,
                    M=2,
                )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(t))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(t))


def test_plate_sde_conditioning_policy_unchanged():
    t = jnp.linspace(0.0, 0.4, 5)
    obs = _make_obs_values((2, len(t), 1))
    with SDESimulator():
        with pytest.raises(ValueError, match="obs_times must not be provided"):
            with trace(), seed(rng_seed=jr.PRNGKey(6)):
                _plate_continuous_sde_model(obs_times=t, obs_values=obs, M=2)


def test_plate_rollout_discrete_gaussian_pf_hmm():
    obs_times = jnp.arange(4.0)
    predict_times = jnp.arange(6.0)
    obs_gaussian = _make_obs_values((2, len(obs_times), 1))
    obs_hmm = _make_obs_values((2, len(obs_times)))

    with DiscreteTimeSimulator():
        with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(7)):
                _plate_discrete_lti_model(
                    obs_times=obs_times,
                    obs_values=obs_gaussian,
                    predict_times=predict_times,
                    M=2,
                )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(predict_times))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(predict_times))

    with DiscreteTimeSimulator():
        with Filter(
            filter_config=PFConfig(
                n_particles=20,
                ess_threshold_ratio=0.5,
                crn_seed=jr.PRNGKey(0),
            )
        ):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(8)):
                _plate_discrete_lti_model(
                    obs_times=obs_times,
                    obs_values=obs_gaussian,
                    predict_times=predict_times,
                    M=2,
                )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(predict_times))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(predict_times))

    with DiscreteTimeSimulator():
        with Filter(filter_config=HMMConfig()):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(9)):
                _plate_hmm_model(
                    obs_times=obs_times,
                    obs_values=obs_hmm,
                    predict_times=predict_times,
                    M=2,
                )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(predict_times))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(predict_times))


def test_plate_rollout_continuous_gaussian_and_dpf():
    obs_times = jnp.linspace(0.0, 0.3, 4)
    predict_times = jnp.linspace(0.0, 0.5, 6)
    obs = _make_obs_values((2, len(obs_times), 1))

    with SDESimulator():
        with Filter(
            filter_config=ContinuousTimeEnKFConfig(
                n_particles=15,
                crn_seed=jr.PRNGKey(0),
            )
        ):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(10)):
                _plate_continuous_sde_model(
                    obs_times=obs_times,
                    obs_values=obs,
                    predict_times=predict_times,
                    M=2,
                )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(predict_times))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(predict_times))

    with SDESimulator():
        with Filter(
            filter_config=ContinuousTimeDPFConfig(
                n_particles=20,
                crn_seed=jr.PRNGKey(0),
            )
        ):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(11)):
                _plate_continuous_sde_model(
                    obs_times=obs_times,
                    obs_values=obs,
                    predict_times=predict_times,
                    M=2,
                )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(predict_times))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(predict_times))


def _plate_continuous_for_discretizer_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=2,
):
    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
        A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
        A = jnp.repeat(A_base[None], M, axis=0).at[:, 0, 0].set(alpha)
        L = 0.2 * jnp.array([[1.0], [0.5]])
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.25]])
        dynamics = LTI_continuous(A=A, L=L, H=H, R=R)
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def test_plate_discretizer_forward_and_rollout():
    obs_times = jnp.arange(4.0)
    predict_times = jnp.arange(6.0)

    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(12)):
                _plate_continuous_for_discretizer_model(predict_times=obs_times, M=2)
    assert tr["f_times"]["value"].shape == (2, 1, len(obs_times))
    obs = tr["f_observations"]["value"][:, 0]

    with DiscreteTimeSimulator():
        with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
            with Discretizer():
                with trace() as tr, seed(rng_seed=jr.PRNGKey(13)):
                    _plate_continuous_for_discretizer_model(
                        obs_times=obs_times,
                        obs_values=obs,
                        predict_times=predict_times,
                        M=2,
                    )
    assert tr["f_predicted_times"]["value"].shape == (2, 1, len(predict_times))
    assert tr["f_predicted_states"]["value"].shape[:3] == (2, 1, len(predict_times))


def _non_plate_discrete_model(predict_times=None):
    Q = 0.1 * jnp.eye(2)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.25]])
    A = jnp.array([[0.3, 0.1], [0.1, 0.8]])
    dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
    dsx.sample("f", dynamics, predict_times=predict_times)


def test_sample_site_parity_plate_and_non_plate():
    t = jnp.arange(4.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(14)):
            _non_plate_discrete_model(predict_times=t)
    assert "f_x_0" in tr
    assert "f_y_0" in tr

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(15)):
            _plate_discrete_lti_model(predict_times=t, M=2)
    member_x0_sites = [k for k in tr if re.fullmatch(r"f_p\d+_x_0", k)]
    member_y0_sites = [k for k in tr if re.fullmatch(r"f_p\d+_y_0", k)]
    assert len(member_x0_sites) == 2
    assert len(member_y0_sites) == 2
