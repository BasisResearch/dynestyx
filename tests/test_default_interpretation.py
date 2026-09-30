"""Automatic interpretations agree with explicit effectful compositions."""

import warnings
from contextlib import ExitStack

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
from numpyro.optim import Adam

import dynestyx as dsx
from dynestyx._defaults import _default_handlers
from dynestyx.handlers import _condition_intp
from dynestyx.handlers import _DynestyxStackKind as Kind
from dynestyx.inference.configs.discretizer import DiffraxSampleConfig, ODEFlowConfig
from dynestyx.inference.configs.filter import EnKFConfig, KFConfig, PFConfig
from dynestyx.types import ConditionedResult, LatentStateResult


def _lti(continuous=False):
    if continuous:
        return dsx.LTI_continuous(
            A=-jnp.eye(1), L=jnp.eye(1) * 0.1, H=jnp.eye(1), R=jnp.eye(1) * 0.2
        )
    return dsx.LTI_discrete(
        A=jnp.eye(1) * 0.8, Q=jnp.eye(1) * 0.1, H=jnp.eye(1), R=jnp.eye(1) * 0.2
    )


def _nonlinear(*, continuous=False, ode=False, gaussian=True):
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=(
            dsx.ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -jnp.tanh(x),
                diffusion=None
                if ode
                else dsx.FullDiffusion(jnp.eye(1) * 0.1, bm_dim=1),
            )
            if continuous
            else lambda x, u, t_now, t_next: dist.MultivariateNormal(
                0.8 * x, jnp.eye(1)
            )
        ),
        observation_model=(
            dsx.GaussianObservation(lambda x, u, t: x, jnp.eye(1) * 0.2)
            if gaussian
            else lambda x, u, t: dist.StudentT(5, x, 0.3).to_event(1)
        ),
    )


@pytest.mark.parametrize(
    "size,dimension,expected",
    [
        (999, 1, dsx.LatentPathBuilder),
        (1000, 1, dsx.Filter),
        (499, 2, dsx.LatentPathBuilder),
        (500, 2, dsx.Filter),
    ],
)
def test_latent_path_cutoff(size, dimension, expected):
    dynamics = dsx.LTI_discrete(
        A=jnp.eye(dimension),
        Q=jnp.eye(dimension),
        H=jnp.eye(dimension),
        R=jnp.eye(dimension),
    )
    chosen = _default_handlers(
        dynamics,
        infer=True,
        predict=False,
        obs_times=jnp.arange(size),
        sample_mode=True,
        kinds=[],
    )
    assert isinstance(chosen[0], expected)


@pytest.mark.parametrize("kinds,sample_mode", [([], False), ([Kind.EVALUATION], True)])
def test_condition_and_evaluation_choose_filter(kinds, sample_mode):
    chosen = _default_handlers(
        _lti(),
        infer=True,
        predict=False,
        obs_times=jnp.arange(3),
        sample_mode=sample_mode,
        kinds=kinds,
    )
    assert isinstance(chosen[0], dsx.Filter)
    assert isinstance(chosen[0].filter_config, EnKFConfig if kinds else KFConfig)
    assert chosen[0].filter_config.filter_source == "cuthbert"


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("gaussian", [False, True])
def test_conservative_custom_transition_selection(continuous, gaussian):
    chosen = _default_handlers(
        _nonlinear(continuous=continuous, gaussian=gaussian),
        infer=True,
        predict=False,
        obs_times=jnp.arange(3),
        sample_mode=True,
        kinds=[],
    )
    assert isinstance(chosen[-1].filter_config, EnKFConfig if gaussian else PFConfig)
    if continuous:
        assert isinstance(chosen[0].discretizer_config, DiffraxSampleConfig)


def test_native_ode_and_sample_only_selection():
    ode = _nonlinear(continuous=True, ode=True)
    chosen = _default_handlers(
        ode,
        infer=True,
        predict=True,
        obs_times=jnp.arange(3),
        sample_mode=True,
        kinds=[],
    )
    assert [type(h) for h in chosen] == [dsx.LatentPathBuilder, dsx.Simulator]
    sde = dsx.discretize_dynamics(_lti(True), DiffraxSampleConfig())
    chosen = _default_handlers(
        sde,
        infer=True,
        predict=False,
        obs_times=jnp.arange(3),
        sample_mode=True,
        kinds=[],
    )
    assert isinstance(chosen[0].filter_config, EnKFConfig)
    chosen = _default_handlers(
        ode,
        infer=True,
        predict=False,
        obs_times=jnp.arange(3),
        sample_mode=False,
        kinds=[],
    )
    assert isinstance(chosen[0].discretizer_config, ODEFlowConfig)


@pytest.mark.parametrize("gaussian", [False, True])
def test_custom_transition_filter_matches_explicit(gaussian):
    dynamics = _nonlinear(gaussian=gaussian)
    config = EnKFConfig() if gaussian else PFConfig()
    with pytest.warns(UserWarning, match="Filter"):
        actual, _ = _run(dynamics, entry=dsx.condition, predictions=False)
    expected, _ = _run(
        dynamics, contexts=[dsx.Filter(config)], entry=dsx.condition, predictions=False
    )
    assert jnp.allclose(actual.marginal_loglik, expected.marginal_loglik)


def test_automatic_sde_ensemble_filter_matches_explicit():
    dynamics = _nonlinear(continuous=True)
    kwargs = dict(obs_times=jnp.arange(3.0) * 0.001, obs_values=jnp.zeros((3, 1)))
    with seed(rng_seed=0), pytest.warns(UserWarning, match="DiffraxSampleConfig"):
        actual = dsx.condition("f", dynamics, **kwargs)
    with (
        seed(rng_seed=0),
        dsx.Filter(EnKFConfig()),
        dsx.Discretizer(DiffraxSampleConfig()),
    ):
        expected = dsx.condition("f", dynamics, **kwargs)
    assert jnp.allclose(actual.marginal_loglik, expected.marginal_loglik)


def test_native_ode_latent_path_matches_explicit():
    dynamics = _nonlinear(continuous=True, ode=True)
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        actual, tr = _run(dynamics)
    expected, explicit_tr = _run(
        dynamics, contexts=[dsx.Simulator(), dsx.LatentPathBuilder()]
    )
    assert actual.state_path_params.shape == (1, 1)
    assert jnp.allclose(actual.joint_log_prob, expected.joint_log_prob)
    _assert_traces_equal(tr, explicit_tr)


def test_discretizer_only_conditions_observations():
    with pytest.warns(UserWarning, match="Filter"):
        actual, tr = _run(
            _lti(True),
            contexts=[dsx.Discretizer()],
            entry=dsx.condition,
            predictions=False,
        )
    expected, _ = _run(
        _lti(True),
        contexts=[dsx.Filter(KFConfig(filter_source="cuthbert")), dsx.Discretizer()],
        entry=dsx.condition,
        predictions=False,
    )
    assert jnp.isfinite(actual.marginal_loglik)
    assert jnp.allclose(actual.marginal_loglik, expected.marginal_loglik)
    assert tr == {}


def _run(
    dynamics,
    *,
    contexts=(),
    entry=dsx.sample,
    observations=True,
    predictions=True,
    plated=False,
):
    with ExitStack() as stack:
        stack.enter_context(seed(rng_seed=0))
        tr = stack.enter_context(trace())
        for context in contexts:
            stack.enter_context(context)
        if plated:
            stack.enter_context(dsx.plate("members", 2))
        y = jnp.zeros((2, 3, 1) if plated else (3, 1))
        out = entry(
            "f",
            dynamics,
            obs_times=jnp.arange(3.0) if observations else None,
            obs_values=y if observations else None,
            predict_times=jnp.array([2.0, 3.0]) if predictions else None,
        )
    return out, tr


def _assert_traces_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name]["type"] == expected[name]["type"]
        assert jnp.allclose(
            actual[name]["value"], expected[name]["value"], equal_nan=True
        ), name


@pytest.mark.parametrize("continuous", [False, True])
def test_prior_simulation_matches_explicit(continuous):
    dynamics = _lti(continuous)
    contexts = [dsx.Simulator(), *([dsx.Discretizer()] if continuous else [])]
    with pytest.warns(UserWarning, match="dynestyx selected.*Simulator"):
        _, actual = _run(dynamics, observations=False)
    _, expected = _run(dynamics, contexts=contexts, observations=False)
    _assert_traces_equal(actual, expected)


@pytest.mark.parametrize("plated", [False, True])
def test_latent_path_and_rollout_match_explicit(plated):
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        result, actual = _run(_lti(), plated=plated)
    explicit_result, expected = _run(
        _lti(), contexts=[dsx.Simulator(), dsx.LatentPathBuilder()], plated=plated
    )
    assert isinstance(result, LatentStateResult)
    assert jnp.allclose(result.joint_log_prob, explicit_result.joint_log_prob)
    _assert_traces_equal(actual, expected)


@pytest.mark.parametrize("continuous", [False, True])
def test_condition_matches_explicit_kf_and_has_no_sites(continuous):
    contexts = [
        dsx.Filter(KFConfig(filter_source="cuthbert")),
        *([dsx.Discretizer()] if continuous else []),
    ]
    with pytest.warns(UserWarning, match="Filter"):
        actual, tr = _run(_lti(continuous), entry=dsx.condition, predictions=False)
    expected, _ = _run(
        _lti(continuous), contexts=contexts, entry=dsx.condition, predictions=False
    )
    assert isinstance(actual, ConditionedResult)
    assert tr == {}
    assert jnp.allclose(actual.marginal_loglik, expected.marginal_loglik)


@pytest.mark.parametrize("stage", ["simulator", "filter", "evaluation"])
def test_partial_stacks_match_explicit(stage):
    factories = {
        "simulator": lambda: [dsx.Simulator()],
        "filter": lambda: [dsx.Filter(KFConfig(filter_source="cuthbert"))],
        "evaluation": lambda: [dsx.Evaluation(dsx.ObservationScoringConfig())],
    }
    explicit = (
        [dsx.Simulator(), dsx.LatentPathBuilder()]
        if stage == "simulator"
        else [dsx.Simulator(), dsx.Filter(KFConfig(filter_source="cuthbert"))]
    )
    if stage == "evaluation":
        explicit = [
            dsx.Evaluation(dsx.ObservationScoringConfig()),
            dsx.Simulator(),
            dsx.Filter(EnKFConfig()),
        ]
    with pytest.warns(UserWarning, match="dynestyx selected"):
        _, actual = _run(_lti(), contexts=factories[stage]())
    _, expected = _run(_lti(), contexts=explicit)
    _assert_traces_equal(actual, expected)


def test_ambient_effect_and_handler_execute_once():
    calls = []
    other = defop(lambda: "default")

    class Observer(ObjectInterpretation):
        @implements(_condition_intp)
        def observe(self, name, dynamics, **kwargs):
            calls.append(other())
            return fwd(name, dynamics, **kwargs)

    with handler({other: lambda: "preserved"}), handler(Observer()):
        with pytest.warns(UserWarning, match="dynestyx selected"):
            _run(_lti())
        assert other() == "preserved"
    assert calls == ["preserved"]


def test_explicit_handlers_are_quiet():
    with warnings.catch_warnings(record=True) as caught:
        _run(
            _lti(),
            contexts=[dsx.Simulator(), dsx.Filter(KFConfig(filter_source="cuthbert"))],
        )
    assert not any("dynestyx selected" in str(w.message) for w in caught)


@pytest.mark.parametrize(
    "simulator",
    [dsx.SDESimulator(), dsx.Simulator(simulator_config=dsx.SDESimulatorConfig())],
)
def test_defaults_do_not_override_explicit_native_simulator(simulator):
    with pytest.raises(ValueError, match="explicit simulator.*continuous-time"):
        _run(_lti(True), contexts=[simulator])


def test_explicit_native_simulator_prior_remains_native():
    simulator = dsx.Simulator(simulator_config=dsx.SDESimulatorConfig(dt0=0.1))
    with warnings.catch_warnings(record=True) as caught:
        _run(_lti(True), contexts=[simulator], observations=False)
    assert isinstance(simulator.simulator, dsx.SDESimulator)
    assert simulator.simulator.simulator_config.dt0 == 0.1
    assert not any("dynestyx selected" in str(w.message) for w in caught)


def test_default_evaluation_rejects_unsupported_observations():
    with pytest.raises(ValueError, match="Automatic Evaluation requires"):
        _run(
            _nonlinear(gaussian=False),
            contexts=[dsx.Evaluation(dsx.ObservationScoringConfig())],
        )


def test_observations_affect_density_without_a_handler():
    def model(y):
        return dsx.sample("f", _lti(), obs_times=jnp.arange(3.0), obs_values=y)

    params = {"f_state_path_params": jnp.zeros((3, 1))}
    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        plausible = log_density(model, (jnp.zeros((3, 1)),), {}, params)[0]
        absurd = log_density(model, (jnp.full((3, 1), 100.0),), {}, params)[0]
    assert plausible > absurd


def test_jit_condition_and_latent_missingness_metadata():
    m = _lti()
    t = jnp.arange(3.0)
    y = jnp.array([[0.0], [jnp.nan], [0.2]])
    with pytest.warns(UserWarning, match="Filter"):
        result = jax.jit(
            lambda y: dsx.condition("f", m, obs_times=t, obs_values=y).marginal_loglik
        )(y)
    assert jnp.isfinite(result)
    metadata = dsx.prepare_missing_observation_metadata(m, obs_times=t, obs_values=y)

    def model(y):
        with seed(rng_seed=0):
            return dsx.sample(
                "f", m, obs_times=t, obs_values=y, missing_obs_metadata=metadata
            ).joint_log_prob

    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        assert jnp.isfinite(jax.jit(model)(y))


def test_default_latent_path_mcmc_and_svi():
    m = _lti()
    t = jnp.arange(3.0)
    y = jnp.zeros((3, 1))

    def model():
        return dsx.sample("f", m, obs_times=t, obs_values=y)

    with pytest.warns(UserWarning, match="LatentPathBuilder"):
        mcmc = MCMC(NUTS(model), num_warmup=2, num_samples=2, progress_bar=False)
        mcmc.run(jr.PRNGKey(0))
        svi = SVI(model, AutoNormal(model), Adam(0.01), Trace_ELBO())
        state = svi.init(jr.PRNGKey(1))
        _, loss = svi.update(state)
    assert mcmc.get_samples()["f_state_path_params"].shape == (2, 3, 1)
    assert jnp.isfinite(loss)
