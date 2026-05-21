"""Additional hierarchical inference coverage smokes for likely user patterns."""

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

import dynestyx as dsx
from dynestyx import (
    DiscreteTimeSimulator,
    Discretizer,
    Filter,
    Simulator,
    Smoother,
)
from dynestyx.inference.filter_configs import EnKFConfig, KFConfig
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import NUTSConfig
from dynestyx.inference.smoother_configs import KFSmootherConfig
from dynestyx.models import (
    GaussianStateEvolution,
    LinearGaussianObservation,
)
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete


def _squeeze_n_sim_axis(observations, plate_ndim):
    return jnp.squeeze(observations, axis=plate_ndim)


def _nested_plate_discrete_simulator_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    G=2,
    M=2,
):
    with dsx.plate("groups", G):
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 0.2))
        beta = 0.08 * jnp.tanh(beta_raw)
        with dsx.plate("trajectories", M):
            alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 0.2))
            alpha = 0.55 + 0.15 * jnn.sigmoid(alpha_raw)
            A = alpha[..., None, None]
            b = 0.05 * jnp.broadcast_to(beta[None, :, None], (M, G, 1))
            dynamics = LTI_discrete(
                A=A,
                Q=jnp.array([[0.04]]),
                H=jnp.array([[1.0]]),
                R=jnp.array([[0.04]]),
                b=b,
            )
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )


def _plate_linear_continuous_missing_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    with dsx.plate("trajectories", M):
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 0.2))
        alpha = 0.30 + 0.20 * jnn.sigmoid(alpha_raw)
        dynamics = LTI_continuous(
            A=-alpha[:, None, None],
            b=0.05 * alpha[:, None],
            L=0.07 * jnp.ones((1, 1)),
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.03]]),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _nested_plate_continuous_missing_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    G=2,
    M=2,
):
    with dsx.plate("groups", G):
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 0.2))
        beta = 0.05 + 0.10 * jnn.sigmoid(beta_raw)
        with dsx.plate("trajectories", M):
            alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 0.2))
            alpha = 0.25 + 0.15 * jnn.sigmoid(alpha_raw)
            a = -(alpha + beta[None, :])[..., None, None]
            b = 0.03 * jnp.tanh(alpha + beta[None, :])[..., None]
            dynamics = LTI_continuous(
                A=a,
                b=b,
                L=0.06 * jnp.ones((1, 1)),
                H=jnp.array([[1.0]]),
                R=jnp.array([[0.03]]),
            )
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )


def _plate_controls_continuous_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    with dsx.plate("trajectories", M):
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 0.2))
        alpha = 0.25 + 0.20 * jnn.sigmoid(alpha_raw)
        dynamics = LTI_continuous(
            A=-alpha[:, None, None],
            B=0.20 * jnp.ones((1, 1)),
            L=0.05 * jnp.ones((1, 1)),
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.03]]),
            D=jnp.array([[0.08]]),
            b=0.02 * alpha[:, None],
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
        )


class _PlateNonlinearTransition(eqx.Module):
    beta: jnp.ndarray

    def __call__(self, x, u, t_now, t_next):
        return 0.65 * x + 0.15 * jnp.tanh(x) + self.beta * jnp.tanh(x)


def _plate_nonlinear_gaussian_filter_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    with dsx.plate("trajectories", M):
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 0.3))
        beta = 0.15 * jnp.tanh(beta_raw)
        dynamics = dsx.DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(1), covariance_matrix=0.20 * jnp.eye(1)
            ),
            state_evolution=GaussianStateEvolution(
                F=_PlateNonlinearTransition(beta=beta),
                cov=0.04 * jnp.eye(1),
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.array([[1.0]]), R=jnp.array([[0.04]])
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _hierarchical_vector_local_noise_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    state_dim = 2
    A = jnp.array([[0.75, 0.05], [-0.04, 0.70]])
    H = jnp.eye(state_dim)
    R = 0.04 * jnp.eye(state_dim)

    with dsx.plate("trajectories", M):
        q_diag_raw = numpyro.sample(
            "q_diag_raw",
            dist.Normal(jnp.zeros(state_dim), 0.2 * jnp.ones(state_dim)).to_event(1),
        )
        q_diag = 0.02 + 0.03 * jnn.sigmoid(q_diag_raw)
        Q = jax.vmap(jnp.diag)(q_diag)
        dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R)
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _plate_batched_initial_condition_smoother_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    state_dim = 2
    A = jnp.array([[0.85, 0.05], [-0.02, 0.80]])

    with dsx.plate("trajectories", M):
        mu_i = numpyro.sample(
            "mu_i", dist.MultivariateNormal(jnp.zeros(state_dim), 0.15 * jnp.eye(2))
        )
        dynamics = LTI_discrete(
            A=A,
            Q=0.03 * jnp.eye(state_dim),
            H=jnp.eye(state_dim),
            R=0.04 * jnp.eye(state_dim),
            b=0.05 * mu_i,
            initial_mean=mu_i,
            initial_cov=0.20 * jnp.eye(state_dim),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _plate_vector_initial_mean_continuous_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    A = jnp.array([[-0.8]])
    L = 0.20 * jnp.eye(1)
    H = jnp.array([[1.0]])
    R = jnp.array([[0.08**2]])

    with dsx.plate("trajectories", M):
        mu_i = jnp.full((M, 1), 0.2)
        mu_0_i = jnp.full((M, 1), 0.1)
        b = -jnp.einsum("ij,...j->...i", A, mu_i)

        dynamics = LTI_continuous(
            A=A,
            L=L,
            H=H,
            R=R,
            b=b,
            initial_mean=mu_0_i,
            initial_cov=0.15 * jnp.eye(1),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _plate_linear_discrete_missing_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    with dsx.plate("trajectories", M):
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 0.2))
        alpha = 0.45 + 0.15 * jnn.sigmoid(alpha_raw)
        A = alpha[:, None, None]
        b = 0.03 * alpha[:, None]
        dynamics = LTI_discrete(
            A=A,
            Q=jnp.array([[0.03]]),
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.04]]),
            b=b,
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _plate_stable_continuous_hierarchical_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(0.2, 0.8))
        dynamics = LTI_continuous(
            A=-alpha[:, None, None],
            b=0.1 * alpha[:, None],
            L=0.1 * jnp.ones((1, 1)),
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.05]]),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _simulate_stable_plate_discretized_observations(obs_times, *, m=3):
    with DiscreteTimeSimulator():
        with Discretizer():
            samples = Predictive(
                _plate_stable_continuous_hierarchical_model,
                num_samples=1,
                exclude_deterministic=False,
            )(jr.PRNGKey(30), predict_times=obs_times, M=m)
    return samples["f_observations"][0, :, 0]


def test_hierarchical_nested_plate_simulator_mcmc_smoke():
    G = 2
    M = 2
    obs_times = jnp.arange(6.0)

    with Simulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(2)):
            _nested_plate_discrete_simulator_model(
                predict_times=obs_times,
                G=G,
                M=M,
            )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=2)

    with Simulator():
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
                init_strategy=init_to_value(
                    values={
                        "beta_raw": jnp.zeros(G),
                        "alpha_raw": jnp.zeros((M, G)),
                    }
                ),
            ),
            model=_nested_plate_discrete_simulator_model,
        )
        posterior = inference.run(
            jr.PRNGKey(3),
            obs_times,
            obs_values,
            G=G,
            M=M,
        )

    assert posterior["beta_raw"].shape == (1, G)
    assert posterior["alpha_raw"].shape == (1, M, G)
    assert not jnp.isnan(posterior["beta_raw"]).any()
    assert not jnp.isnan(posterior["alpha_raw"]).any()


def test_hierarchical_kf_smoother_missingness_smoke():
    M = 3
    obs_times = jnp.arange(6.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(4)):
            _plate_linear_discrete_missing_model(
                predict_times=obs_times,
                M=M,
            )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=1)
    obs_values = obs_values.at[0, 2:4, :].set(jnp.nan)
    obs_values = obs_values.at[1, 1, 0].set(jnp.nan)

    with Smoother(
        smoother_config=KFSmootherConfig(
            filter_source="cuthbert",
            record_smoothed_states_mean=True,
        )
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(5)):
            _plate_linear_discrete_missing_model(
                obs_times=obs_times,
                obs_values=obs_values,
                M=M,
            )

    assert tr["f_marginal_loglik"]["value"].shape == (M,)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"]).all()


def test_hierarchical_discretizer_nested_plate_missingness_filter_smoke():
    G = 2
    M = 2
    obs_times = jnp.arange(6.0)

    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(6)):
                _nested_plate_continuous_missing_model(
                    predict_times=obs_times,
                    G=G,
                    M=M,
                )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=2)
    obs_values = obs_values.at[0, 1, 1:3, :].set(jnp.nan)
    obs_values = obs_values.at[1, 0, 4, 0].set(jnp.nan)

    with Filter(
        filter_config=EnKFConfig(
            filter_source="cuthbert",
            n_particles=8,
            crn_seed=jr.PRNGKey(0),
        )
    ):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(7)):
                _nested_plate_continuous_missing_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    G=G,
                    M=M,
                )

    assert tr["f_marginal_loglik"]["value"].shape == (M, G)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"]).all()


def test_hierarchical_controls_filter_discretizer_smoke():
    M = 3
    obs_times = jnp.arange(6.0)
    ctrl_times = obs_times
    ctrl_values = jnp.stack(
        [jnp.sin(obs_times / 2.0 + 0.4 * i) for i in range(M)],
        axis=0,
    )[..., None]

    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(8)):
                _plate_controls_continuous_model(
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    predict_times=obs_times,
                    M=M,
                )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=1)

    with Filter(
        filter_config=EnKFConfig(
            filter_source="cuthbert",
            n_particles=8,
            crn_seed=jr.PRNGKey(1),
        )
    ):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(9)):
                _plate_controls_continuous_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    M=M,
                )

    assert tr["f_marginal_loglik"]["value"].shape == (M,)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"]).all()


def test_hierarchical_nonlinear_gaussian_enkf_smoke():
    M = 3
    obs_times = jnp.arange(8.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(10)):
            _plate_nonlinear_gaussian_filter_model(
                predict_times=obs_times,
                M=M,
            )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=1)

    def conditioned_model(
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        M=M,
    ):
        with Filter(
            filter_config=EnKFConfig(
                filter_source="cuthbert",
                n_particles=12,
                crn_seed=jr.PRNGKey(11),
            )
        ):
            _plate_nonlinear_gaussian_filter_model(
                obs_times=obs_times,
                obs_values=obs_values,
                M=M,
            )

    inference = MCMCInference(
        mcmc_config=NUTSConfig(
            num_samples=1,
            num_warmup=1,
            num_chains=1,
            mcmc_source="numpyro",
            init_strategy=init_to_value(values={"beta_raw": jnp.zeros(M)}),
        ),
        model=conditioned_model,
    )
    posterior = inference.run(jr.PRNGKey(12), obs_times, obs_values, M=M)

    assert posterior["beta_raw"].shape == (1, M)
    assert not jnp.isnan(posterior["beta_raw"]).any()


def test_hierarchical_vector_local_noise_parameters_smoke():
    M = 3
    obs_times = jnp.arange(8.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(13)):
            _hierarchical_vector_local_noise_model(
                predict_times=obs_times,
                M=M,
            )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=1)

    def conditioned_model(
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        M=M,
    ):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            _hierarchical_vector_local_noise_model(
                obs_times=obs_times,
                obs_values=obs_values,
                M=M,
            )

    inference = MCMCInference(
        mcmc_config=NUTSConfig(
            num_samples=1,
            num_warmup=1,
            num_chains=1,
            mcmc_source="numpyro",
            init_strategy=init_to_value(values={"q_diag_raw": jnp.zeros((M, 2))}),
        ),
        model=conditioned_model,
    )
    posterior = inference.run(jr.PRNGKey(14), obs_times, obs_values, M=M)

    assert posterior["q_diag_raw"].shape == (1, M, 2)
    assert not jnp.isnan(posterior["q_diag_raw"]).any()


def test_hierarchical_batched_initial_condition_smoother_smoke():
    M = 3
    obs_times = jnp.arange(7.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(15)):
            _plate_batched_initial_condition_smoother_model(
                predict_times=obs_times,
                M=M,
            )
    obs_values = _squeeze_n_sim_axis(tr["f_observations"]["value"], plate_ndim=1)

    with Smoother(
        smoother_config=KFSmootherConfig(
            filter_source="cuthbert",
            record_smoothed_states_mean=True,
        )
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(16)):
            _plate_batched_initial_condition_smoother_model(
                obs_times=obs_times,
                obs_values=obs_values,
                M=M,
            )

    assert tr["f_marginal_loglik"]["value"].shape == (M,)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"]).all()


def test_plate_discretizer_enkf_hierarchical_missingness_mcmc_smoke():
    m = 3
    obs_times = jnp.arange(6.0)
    obs_values = _simulate_stable_plate_discretized_observations(obs_times, m=m)
    obs_values = obs_values.at[0, 2:4, :].set(jnp.nan)
    obs_values = obs_values.at[1, 1, 0].set(jnp.nan)

    def conditioned_model(obs_times=None, obs_values=None, m=m):
        with Filter(
            filter_config=EnKFConfig(
                filter_source="cuthbert",
                n_particles=8,
                crn_seed=jr.PRNGKey(31),
            )
        ):
            with Discretizer():
                _plate_stable_continuous_hierarchical_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    M=m,
                )

    mcmc = MCMC(
        NUTS(conditioned_model),
        num_warmup=5,
        num_samples=5,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(32), obs_times=obs_times, obs_values=obs_values, m=m)

    posterior = mcmc.get_samples()
    assert posterior["alpha"].shape == (5, m)
    assert not jnp.isnan(posterior["alpha"]).any()


def test_plate_discretizer_enkf_batched_initial_mean_shared_cov_regression():
    """Minimal repro for the 08-hierarchical notebook EnKF+Discretizer failure."""
    obs_times = jnp.arange(6.0)

    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(40)):
                _plate_vector_initial_mean_continuous_model(
                    predict_times=obs_times,
                    M=3,
                )
    obs_values = tr["f_observations"]["value"][:, 0]

    with Filter(
        filter_config=EnKFConfig(
            filter_source="cuthbert",
            n_particles=8,
            crn_seed=jr.PRNGKey(41),
        )
    ):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(42)):
                _plate_vector_initial_mean_continuous_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    M=3,
                )

    assert tr["f_marginal_loglik"]["value"].shape == (3,)
