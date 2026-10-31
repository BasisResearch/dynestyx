"""Complete missing interpretation stages without replaying ambient handlers."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

from effectful.ops.semantics import coproduct
from effectful.ops.syntax import ObjectInterpretation
from effectful.ops.types import Term
from jaxtyping import Array, Real

from dynestyx.handlers import (
    _INFERENCE_KINDS,
    _condition_intp,
    _dynestyx_stack_kind,
    _DynestyxStackKind,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import Result

_LATENT_PATH_SIZE_CUTOFF = 1_000
_ConditionCallable = Callable[..., Result | Term[Result] | None]


def _default_handlers(
    dynamics: DynamicalModel,
    *,
    infer: bool,
    predict: bool,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None,
    sample_mode: bool,
    kinds: Sequence[_DynestyxStackKind],
) -> list[ObjectInterpretation]:
    """Choose handlers for missing inference and prediction stages.

    Categorical states use exact HMM filtering. Eligible small ``sample``
    sites use LatentPathBuilder; other inference requests use a structurally
    compatible filter. Prediction-only requests retain the model's native
    representation for Simulator's backend selection.

    Args:
        dynamics: Model at the current point in the handler stack.
        infer: Whether to supply an inference handler for observations.
        predict: Whether to supply a simulator for requested predictions.
        obs_times: Observation times, required when ``infer`` is true. The
            final axis supplies the per-trajectory size used by the LPB cutoff.
        sample_mode: Whether the caller is ``dsx.sample``. Only sample calls
            are eligible for automatic LatentPathBuilder selection.
        kinds: Active handler kinds, ordered from innermost to outermost.

    Returns:
        Handlers ordered from innermost to outermost: discretization when
        needed, inference, then simulation. The caller composes these handlers
        with the existing continuation.

    Raises:
        ValueError: If inference is unsupported for the model, Evaluation
            cannot obtain predictive outputs, or the PF fallback would use
            deterministic observations with continuous latent states.
    """
    # Imports stay local: these interpretations themselves import this module.
    import numpyro.distributions as dist

    from dynestyx.discretization.gaussian import _ConfiguredGaussianStateEvolution
    from dynestyx.discretization.ode_flow import _ODEFlowStateEvolution
    from dynestyx.discretizers import (
        Discretizer,
        _automatic_discretizer_config,
        discretize_dynamics,
    )
    from dynestyx.inference.checkers import _validate_inference_supported_model_classes
    from dynestyx.inference.configs.discretizer import DiffraxSampleConfig
    from dynestyx.inference.configs.filter import (
        EnKFConfig,
        HMMConfig,
        KFConfig,
        PFConfig,
    )
    from dynestyx.inference.filters import Filter
    from dynestyx.inference.latent.builder import LatentPathBuilder
    from dynestyx.models import (
        DeterministicContinuousTimeStateEvolution,
        DeterministicObservation,
        GaussianObservation,
        GaussianStateEvolution,
        LinearGaussianObservation,
        LinearGaussianStateEvolution,
        StochasticContinuousTimeStateEvolution,
    )
    from dynestyx.simulation import Simulator

    handlers = []
    if infer:
        # Observation inputs have already been validated by condition().
        obs_times = cast(Real[Array, "*obs_time_plate obs_time"], obs_times)
        # Reject unsupported model classes before choosing an inference algorithm.
        _validate_inference_supported_model_classes(dynamics)
        evolution = dynamics.state_evolution
        # Check for known structured transition classes that LPB can score.
        # Native ODEs are also eligible: LPB samples their initial state and
        # integrates the path instead of scoring stochastic transitions.
        density_known = isinstance(
            evolution,
            (
                LinearGaussianStateEvolution,
                GaussianStateEvolution,
                _ConfiguredGaussianStateEvolution,
                DeterministicContinuousTimeStateEvolution,
            ),
        ) or (
            # An already-discretized ODE flow has a transition density only
            # when its configured jitter adds nonzero process noise.
            isinstance(evolution, _ODEFlowStateEvolution)
            and evolution.config.jitter_scale > 0
        )
        # Choose LPB only for small sample sites with matching dimensions.
        # Count observations per trajectory, without multiplying by plate sizes.
        # condition() defaults and Evaluation use filtering instead.
        if dynamics.categorical_state:
            if _DynestyxStackKind.EVALUATION in kinds:
                raise ValueError(
                    "Automatic Evaluation does not support HMM predictive outputs."
                )
            handlers.append(Filter(HMMConfig()))
        elif (
            sample_mode
            and _DynestyxStackKind.EVALUATION not in kinds
            and dynamics.observation_dim == dynamics.state_dim
            and obs_times.shape[-1] * dynamics.observation_dim
            < _LATENT_PATH_SIZE_CUTOFF
            and density_known
        ):
            handlers.append(LatentPathBuilder())
        else:
            # Inspect the discrete representation to determine KF compatibility.
            # This probe does not replace the model; any required Discretizer
            # is added to the returned handler list below.
            discrete = (
                discretize_dynamics(dynamics) if dynamics.continuous_time else dynamics
            )
            # Canonical observation predictions currently support cuthbert EnKF,
            # but not cuthbert KF/PF, so Evaluation changes compatibility.
            if (
                _DynestyxStackKind.EVALUATION not in kinds
                and isinstance(discrete.state_evolution, LinearGaussianStateEvolution)
                and isinstance(discrete.observation_model, LinearGaussianObservation)
                and isinstance(discrete.initial_condition, dist.MultivariateNormal)
            ):
                # Exact Kalman filtering needs linear Gaussian transitions and
                # observations, with a multivariate Gaussian initial condition.
                config = KFConfig(filter_source="cuthbert")
            elif isinstance(
                dynamics.observation_model,
                (LinearGaussianObservation, GaussianObservation),
            ):
                # Gaussian observations permit EnKF when exact KF is unavailable.
                config = EnKFConfig()
            else:
                # General observations fall back to particle filtering, unless
                # Evaluation needs predictive diagnostics it cannot provide.
                if _DynestyxStackKind.EVALUATION in kinds:
                    raise ValueError(
                        "Automatic Evaluation requires a supported Gaussian "
                        "observation model and cuthbert EnKF predictive outputs. "
                        "Supply an explicit compatible Filter for this model."
                    )
                if isinstance(dynamics.observation_model, DeterministicObservation):
                    raise ValueError(
                        "Automatic particle filtering does not support deterministic "
                        "observations for continuous latent states. Add observation "
                        "noise explicitly with GaussianRelaxation(observation_model_cov=...), "
                        "or supply an explicit compatible inference handler."
                    )
                config = PFConfig()
            if dynamics.continuous_time:
                evolution = cast(
                    DeterministicContinuousTimeStateEvolution
                    | StochasticContinuousTimeStateEvolution,
                    evolution,
                )
                # Ensemble/particle SDE filters need sampled transitions. KF and
                # ODE models use the discretizer selected for their evolution.
                discretizer_config = (
                    DiffraxSampleConfig()
                    if isinstance(evolution, StochasticContinuousTimeStateEvolution)
                    and not isinstance(config, KFConfig)
                    else _automatic_discretizer_config(evolution)
                )
                handlers.append(Discretizer(discretizer_config))
            handlers.append(Filter(config))
    if predict:
        # Append generation after any inference so it can simulate posterior paths.
        handlers.append(Simulator())
    return handlers


def _apply_defaults(
    continuation: _ConditionCallable,
    before: _DynestyxStackKind | None,
    name: str,
    dynamics: DynamicalModel,
    *,
    _consumer: ObjectInterpretation | None = None,
    **kwargs: Any,
) -> Result | Term[Result] | None:
    """Supply missing stages and invoke the current handler continuation.

    Existing inference handlers, forwarded results, and completed predictions
    prevent duplicate work. New handlers are composed around the continuation
    without replaying the ambient stack. A warning names each automatic choice.

    Args:
        continuation: Bound handler method, or the terminal default operation.
        before: Consuming handler's kind, or ``None`` for the default operation.
        name: Dynamical-model sample-site name.
        dynamics: Model forwarded to this point in the stack.
        _consumer: Handler instance whose explicit simulator configuration
            must remain compatible with any automatic discretization.
        **kwargs: Operation inputs and forwarded results. Observation values,
            prediction times, and completion flags determine which stages
            need defaults; all arguments are passed to the continuation.

    Returns:
        The composed continuation's result, including symbolic or unhandled
        results when the surrounding stack does not consume the operation.

    Raises:
        ValueError: If default selection is unsupported or automatic
            discretization conflicts with the consumer's simulator config.
    """
    kinds = _dynestyx_stack_kind()
    conditioned = (
        kwargs.get("filtered_result") is not None
        or kwargs.get("smoothed_result") is not None
    )
    infer = (
        kwargs.get("obs_values") is not None
        and not conditioned
        and not _INFERENCE_KINDS.intersection(kinds)
    )
    predict = (
        before != _DynestyxStackKind.SIMULATOR
        and kwargs.get("predict_times") is not None
        and not kwargs.get("_dsx_prediction_done", False)
        and _DynestyxStackKind.SIMULATOR not in kinds
    )
    if not infer and not predict:
        return continuation(name, dynamics, **kwargs)

    handlers = _default_handlers(
        dynamics,
        infer=infer,
        predict=predict,
        obs_times=kwargs.get("obs_times"),
        sample_mode=kwargs.get("_dsx_sample_mode", False),
        kinds=kinds,
    )
    from dynestyx.discretizers import Discretizer

    if (
        before == _DynestyxStackKind.SIMULATOR
        and kwargs.get("predict_times") is not None
        and getattr(_consumer, "simulator_config", None) is not None
        and isinstance(handlers[0], Discretizer)
    ):
        raise ValueError(
            "Automatic inference discretizes this model, but the explicit "
            "simulator has a continuous-time configuration. Supply a compatible "
            "continuous-time Filter/Smoother, or use Simulator() without a "
            "solver configuration."
        )
    # A consumed native LPB path should also retain its continuous-time model.
    if conditioned and not infer:
        handlers = [h for h in handlers if not isinstance(h, Discretizer)]

    descriptions = []
    for interpretation in handlers:
        config = getattr(interpretation, "filter_config", None)
        if config is None:
            config = getattr(interpretation, "discretizer_config", None)
        config_name = "" if config is None else type(config).__name__ + "()"
        source = getattr(config, "filter_source", None)
        if source is not None:
            config_name = f"{type(config).__name__}(filter_source={source!r})"
        descriptions.append(f"{type(interpretation).__name__}({config_name})")
    warnings.warn(
        f"dynestyx selected {', '.join(descriptions)} for site {name!r}. "
        "Use explicit handler contexts to override these defaults.",
        UserWarning,
        stacklevel=3,
    )

    # The leftmost interpretation is the existing continuation. Calling the
    # composed method preserves its enclosing fwd prompt and unrelated effects.
    interpretation = {_condition_intp: continuation}
    for default in reversed(handlers):
        interpretation = coproduct(interpretation, default)
    return interpretation[_condition_intp](name, dynamics, **kwargs)


def _complete_defaults(
    before: _DynestyxStackKind,
) -> Callable[[_ConditionCallable], _ConditionCallable]:
    """Decorate a handler method to complete stages needed by its consumer.

    Apply this decorator beneath ``@implements(_condition_intp)`` on methods
    accepting ``self, name, dynamics, **kwargs``.

    Args:
        before: Consuming handler's kind. SIMULATOR needs inference completed
            before posterior simulation; EVALUATION needs filtered results
            and any requested simulation completed before scoring.

    Returns:
        A method decorator that binds the original handler to its instance
        and passes it to ``_apply_defaults``. Explicit handlers and completed
        stages are respected; the original method runs directly when no
        defaults are needed.
    """

    def decorate(implementation: _ConditionCallable) -> _ConditionCallable:
        """Wrap an unbound handler method while preserving its metadata."""

        @functools.wraps(implementation)
        def wrapped(
            self: ObjectInterpretation,
            name: str,
            dynamics: DynamicalModel,
            **kwargs: Any,
        ) -> Result | Term[Result] | None:
            """Complete missing stages before calling the bound handler."""
            return _apply_defaults(
                functools.partial(implementation, self),
                before,
                name,
                dynamics,
                _consumer=self,
                **kwargs,
            )

        return wrapped

    return decorate
