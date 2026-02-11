from contextlib import ExitStack

import jax.numpy as jnp
import jax.random as jr
from numpyro.handlers import seed
from numpyro.infer import Predictive

from dynestyx.discretizers import euler_maruyama
from dynestyx.dynamical_models import Context, Trajectory
from dynestyx.filters import (
    FilterBasedHMMMarginalLogLikelihood,
    FilterBasedMarginalLogLikelihood,
)
from dynestyx.handlers import Condition, Discretizer
from dynestyx.simulators import DiscreteTimeSimulator, ODESimulator, SDESimulator
from tests.combinatorial.model_factory import build_model, time_grid
from tests.combinatorial.specs import InferenceSpec, ModelSpec


def _enter_inference_contexts(stack: ExitStack, inf_spec: InferenceSpec):
    if inf_spec.runner == "discrete_sim":
        stack.enter_context(DiscreteTimeSimulator())
    elif inf_spec.runner == "ode_sim":
        stack.enter_context(ODESimulator())
    elif inf_spec.runner == "sde_sim":
        stack.enter_context(SDESimulator(key=jr.PRNGKey(1)))
    elif inf_spec.runner == "filter":
        filter_kwargs = {}
        ftype = inf_spec.filter_type or "default"
        if ftype.lower() == "pf":
            filter_kwargs["n_filter_particles"] = 32
        if ftype.lower() == "dpf":
            filter_kwargs["N_particles"] = 32
        if ftype.lower() in {"enkf", "default"}:
            filter_kwargs["enkf_N_particles"] = 16
        stack.enter_context(
            FilterBasedMarginalLogLikelihood(filter_type=ftype, **filter_kwargs)
        )
    elif inf_spec.runner == "filter_hmm":
        stack.enter_context(FilterBasedHMMMarginalLogLikelihood())

    if inf_spec.discretizer == "default":
        stack.enter_context(Discretizer())
    elif inf_spec.discretizer == "euler":
        stack.enter_context(Discretizer(discretize=euler_maruyama))


def run_forward_case(model_spec: ModelSpec, inf_spec: InferenceSpec, timesteps: int):
    model, dynamics = build_model(model_spec)
    times = time_grid(timesteps)
    obs_dim = dynamics.observation_dim
    obs_values = (
        jnp.linspace(0.0, 1.0, len(times))
        if obs_dim == 1
        else jnp.stack(
            [jnp.linspace(0.0, 1.0, len(times)), jnp.linspace(1.0, 2.0, len(times))],
            axis=-1,
        )
    )

    controls = Trajectory()
    if model_spec.uses_control:
        controls = Trajectory(times=times, values=jnp.ones((len(times), 2)))

    context = Context(
        observations=Trajectory(times=times, values=obs_values),
        controls=controls,
    )

    with seed(rng_seed=jr.PRNGKey(0)):
        with ExitStack() as stack:
            _enter_inference_contexts(stack, inf_spec)
            stack.enter_context(Condition(context))
            model()


def run_predictive_case(
    model_spec: ModelSpec,
    inf_spec: InferenceSpec,
    context_mode: str,
    timesteps: int,
):
    model, dynamics = build_model(model_spec)
    predictive = Predictive(model, num_samples=1, exclude_deterministic=False)
    times = time_grid(timesteps)

    obs_values = None
    if context_mode in {
        "obs_times_obs_values",
        "obs_times_obs_values_ctrl_times_ctrl_values",
    }:
        if dynamics.observation_dim == 1:
            obs_values = jnp.array([0.0, 1.0, 2.0])
        else:
            obs_values = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    controls = Trajectory()
    if context_mode == "obs_times_obs_values_ctrl_times_ctrl_values":
        controls = Trajectory(
            times=times,
            values=jnp.ones((len(times), dynamics.control_dim or 1)),
        )

    context = Context(
        observations=Trajectory(times=times, values=obs_values),
        controls=controls,
    )

    with seed(rng_seed=jr.PRNGKey(0)):
        with ExitStack() as stack:
            _enter_inference_contexts(stack, inf_spec)
            stack.enter_context(Condition(context))
            predictive(jr.PRNGKey(2))

