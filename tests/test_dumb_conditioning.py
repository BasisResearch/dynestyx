import pytest
import numpyro
import jax.numpy as jnp
from contextlib import nullcontext
from numpyro.handlers import trace
from effectful.ops.semantics import handler
from dsx.handlers import BaseCDDynamaxLogFactorAdder, Condition
from dsx.dynamical_models import DynamicalModel
from dsx.ops import Trajectory, sample_ds


class SimpleLogFactorAdder(BaseCDDynamaxLogFactorAdder):
    """A simple log factor adder that adds a log factor of 1."""

    def __init__(self, constant_log_prob: float = 1.) -> None:
        super().__init__()

        self.constant_log_prob = constant_log_prob
    
    def add_log_factors(self, name: str, dynamics: DynamicalModel, obs: Trajectory):
        numpyro.factor(f"{name}_simple_log_factor", self.constant_log_prob)


@pytest.fixture
def simple_log_factor_adder():
    """Pytest fixture that returns a SimpleLogFactorAdder instance."""
    return SimpleLogFactorAdder()


@pytest.fixture
def simple_model():
    """Pytest fixture that returns a model function with a single sample_ds call."""
    def model():
        # If sampling parameters, we would numpyro.sample here and pass them into DynamicalModel
        dynamics = DynamicalModel()
        return sample_ds("f", dynamics, None)
    
    return model


@pytest.mark.parametrize("constant_log_prob", [1.0, 2.0])
def test_condition_with_log_factor(simple_model, constant_log_prob):
    """Test that Condition handler with observation adds log factor correctly."""
    # Create dummy observation: Trajectory = (Times, States)
    times = jnp.array([0.0, 1.0, 2.0])
    states = {"x": jnp.array([0.0, 1.0, 2.0])}
    dummy_obs: Trajectory = (times, states)
    
    # Create composite handler using effectful coproduct
    condition_handler = handler(Condition({"f": dummy_obs}))
    log_factor_adder = handler(SimpleLogFactorAdder(constant_log_prob=constant_log_prob))
    
    # Run model with handler and trace
    # Note: trace() context manager returns the OrderedDict directly
    with trace() as trace_dict:
        with log_factor_adder:
            with condition_handler:
                simple_model()
    
    # Assert that the log factor is present
    assert "f_simple_log_factor" in trace_dict, "Log factor 'f_simple_log_factor' not found in trace"
    
    # numpyro.factor() creates a Unit distribution sample site
    # The factor value is stored in the Unit distribution's log_prob
    site = trace_dict["f_simple_log_factor"]
    assert isinstance(site["fn"], numpyro.distributions.distribution.Unit), \
        "Expected Unit distribution for factor"
    
    # Check that the log_prob equals the expected value (the factor value we added)
    # For Unit distribution, log_prob returns the log_factor value passed to numpyro.factor()
    log_prob = site["fn"].log_prob(site["value"])
    assert jnp.isclose(log_prob, constant_log_prob), \
        f"Expected log factor value {constant_log_prob}, got {log_prob}"


@pytest.mark.parametrize("condition_on_g", [True, False])
def test_log_factor_only_with_observation(simple_model, condition_on_g):
    """Test that log factor only appears when there's an actual observation."""
    # Create dummy observation: Trajectory = (Times, States)
    times = jnp.array([0.0, 1.0, 2.0])
    states = {"x": jnp.array([0.0, 1.0, 2.0])}
    dummy_obs: Trajectory = (times, states)
    
    # Create handlers
    # When condition_on_g is True, condition on "g" (which doesn't exist in the model)
    # When condition_on_g is False, don't use condition handler at all (use noop context manager)
    if condition_on_g:
        condition_data = {"g": dummy_obs}
        condition_handler = handler(Condition(condition_data))
    else:
        condition_handler = nullcontext()
    
    log_factor_adder = handler(SimpleLogFactorAdder(constant_log_prob=1.0))
    
    # Run model with handler and trace
    # Note: trace() context manager returns the OrderedDict directly
    with trace() as trace_dict:
        with log_factor_adder:
            with condition_handler:
                simple_model()
    
    # Assert that the log factor does NOT appear in the trace
    # Since the model samples "f" but we either condition on "g" (non-existent) 
    # or don't condition at all, there's no observation passed to add_log_factors
    assert "f_simple_log_factor" not in trace_dict, \
        "Log factor 'f_simple_log_factor' should NOT be present when there's no observation for 'f'"
