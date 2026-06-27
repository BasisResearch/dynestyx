"""Configuration objects for pure and handler-based simulation."""

from __future__ import annotations

import dataclasses
from typing import Literal

import diffrax as dfx
from jax import Array


@dataclasses.dataclass(frozen=True)
class SimulatorConfig:
    """Generic simulator options shared across all simulator backends."""

    pass


@dataclasses.dataclass(frozen=True)
class ODESimulatorConfig(SimulatorConfig):
    """Configuration for deterministic continuous-time simulation."""

    solver: dfx.AbstractSolver = dataclasses.field(default_factory=dfx.Tsit5)
    adjoint: dfx.AbstractAdjoint = dataclasses.field(
        default_factory=dfx.RecursiveCheckpointAdjoint
    )
    stepsize_controller: dfx.AbstractStepSizeController = dataclasses.field(
        default_factory=dfx.ConstantStepSize
    )
    dt0: float | int | Array = 1e-3
    max_steps: int = 100_000


@dataclasses.dataclass(frozen=True)
class SDESimulatorConfig(SimulatorConfig):
    """Configuration for stochastic continuous-time simulation."""

    solver: dfx.AbstractSolver = dataclasses.field(default_factory=dfx.Heun)
    stepsize_controller: dfx.AbstractStepSizeController = dataclasses.field(
        default_factory=dfx.ConstantStepSize
    )
    adjoint: dfx.AbstractAdjoint = dataclasses.field(
        default_factory=dfx.RecursiveCheckpointAdjoint
    )
    dt0: float | int | Array = 1e-4
    tol_vbt: float | int | Array | None = None
    max_steps: int | None = None
    source: Literal["diffrax", "em_scan"] = "em_scan"


__all__ = [
    "SimulatorConfig",
    "ODESimulatorConfig",
    "SDESimulatorConfig",
]
