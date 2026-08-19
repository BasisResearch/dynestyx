"""Configuration-driven discretization of continuous-time models."""

from typing import Any

from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.discretization.diffrax_sample import _DiffraxSampleStateEvolution
from dynestyx.discretization.exact_affine import _ExactAffineStateEvolution
from dynestyx.discretization.gaussian import _ConfiguredGaussianStateEvolution
from dynestyx.discretization.ode_flow import _ODEFlowStateEvolution
from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.configs.discretizer import (
    BaseDiscretizerConfig,
    DiffraxSampleConfig,
    DiscretizerConfig,
    EulerMaruyamaConfig,
    ExactAffineConfig,
    LocalLinearizationConfig,
    MeanTrajectoryLinearizationConfig,
    ODEFlowConfig,
)
from dynestyx.models import (
    AffineDrift,
    DeterministicContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.core import StateEvolutionLike


def _automatic_discretizer_config(
    cte: DeterministicContinuousTimeStateEvolution
    | StochasticContinuousTimeStateEvolution,
) -> DiscretizerConfig:
    if isinstance(cte, DeterministicContinuousTimeStateEvolution):
        return ODEFlowConfig()
    if (
        isinstance(cte.drift, AffineDrift)
        and cte.potential is None
        and not callable(cte.diffusion.coefficient)
    ):
        return ExactAffineConfig()
    return EulerMaruyamaConfig()


def _discretize_state_evolution(
    cte: StateEvolutionLike,
    config: BaseDiscretizerConfig | None = None,
) -> DiscreteTimeStateEvolution:
    """Build the private discrete transition selected by a config."""
    if not isinstance(
        cte,
        (
            DeterministicContinuousTimeStateEvolution,
            StochasticContinuousTimeStateEvolution,
        ),
    ):
        raise TypeError(
            "Discretizer configs require a continuous-time state "
            f"evolution; got {type(cte).__name__}."
        )
    resolved = _automatic_discretizer_config(cte) if config is None else config
    if isinstance(cte, DeterministicContinuousTimeStateEvolution):
        if isinstance(resolved, ODEFlowConfig):
            return _ODEFlowStateEvolution(cte, resolved)
        raise TypeError(
            f"{type(resolved).__name__} requires a stochastic continuous-time "
            "state evolution; got DeterministicContinuousTimeStateEvolution."
        )
    if isinstance(resolved, ODEFlowConfig):
        raise TypeError(
            "ODEFlowConfig requires a deterministic continuous-time state "
            "evolution; got StochasticContinuousTimeStateEvolution."
        )
    if isinstance(resolved, ExactAffineConfig):
        return _ExactAffineStateEvolution(
            cte,
            covariance_jitter=resolved.covariance_jitter,
        )
    if isinstance(resolved, DiffraxSampleConfig):
        return _DiffraxSampleStateEvolution(cte, resolved)
    if isinstance(
        resolved,
        (
            EulerMaruyamaConfig,
            LocalLinearizationConfig,
            MeanTrajectoryLinearizationConfig,
        ),
    ):
        if isinstance(resolved, LocalLinearizationConfig) and callable(
            cte.diffusion.coefficient
        ):
            raise TypeError(
                "LocalLinearizationConfig requires structurally constant "
                "additive diffusion."
            )
        return _ConfiguredGaussianStateEvolution(cte, resolved)
    raise TypeError(
        "discretizer_config must be a concrete BaseDiscretizerConfig; "
        f"got {type(resolved).__name__}."
    )


class Discretizer(ObjectInterpretation, HandlesSelf):
    r"""Performs discretization of a continuous-time state evolution, converting it to a discrete-time state evolution.

    A `Discretizer` interpretation should be used inside an inference or simulation context. The outside inference/simulation
    context may then use the resulting `DiscreteTimeStateEvolution`:

    ```python
    from dynestyx.discretizers import (
        Discretizer,
        EnKFConfig,
        Filter,
        MeanTrajectoryLinearizationConfig,
    )
    with Filter(EnKFConfig()):
        with Discretizer(MeanTrajectoryLinearizationConfig()):
            model(...)
    ```

    When no config is provided, ODE models use their numerical flow, SDE models
    with an affine drift use an exact Gaussian discretization, and other SDE
    models use Euler--Maruyama.

    Attributes:
        discretizer_config: Explicit discretization config, or `None` for
            automatic routing.
    """

    def __init__(
        self,
        discretizer_config: BaseDiscretizerConfig | None = None,
    ):
        super().__init__()
        if discretizer_config is not None and not isinstance(
            discretizer_config, BaseDiscretizerConfig
        ):
            raise TypeError(
                "discretizer_config must be a BaseDiscretizerConfig or None, "
                f"got {type(discretizer_config).__name__}."
            )
        self.discretizer_config = discretizer_config

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> Any:
        if isinstance(
            dynamics.state_evolution,
            (
                DeterministicContinuousTimeStateEvolution,
                StochasticContinuousTimeStateEvolution,
            ),
        ):
            dynamics = DynamicalModel(
                initial_condition=dynamics.initial_condition,
                state_evolution=_discretize_state_evolution(
                    dynamics.state_evolution,
                    self.discretizer_config,
                ),
                observation_model=dynamics.observation_model,
                control_model=dynamics.control_model,
                control_dim=dynamics.control_dim,
                t0=dynamics.t0,
            )
        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )


__all__ = [
    "BaseDiscretizerConfig",
    "DiffraxSampleConfig",
    "Discretizer",
    "DiscretizerConfig",
    "EulerMaruyamaConfig",
    "ExactAffineConfig",
    "LocalLinearizationConfig",
    "MeanTrajectoryLinearizationConfig",
    "ODEFlowConfig",
]
