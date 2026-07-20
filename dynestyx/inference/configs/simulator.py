"""Configuration objects for continuous-time simulators."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import diffrax as dfx
from jax import Array
from jaxtyping import Real

from dynestyx.types import as_scalar_time_array


@dataclasses.dataclass
class ODESimulatorConfig:
    """Structured ODE simulator settings.

    This config collects the solver settings used to reconstruct a deterministic
    state path

    ``x = (x(t_0), x(t_1), ..., x(t_T))``

    from path parameters such as an initial condition.
    """

    solver: dfx.AbstractSolver = dataclasses.field(default_factory=dfx.Tsit5)
    adjoint: dfx.AbstractAdjoint = dataclasses.field(
        default_factory=dfx.RecursiveCheckpointAdjoint
    )
    stepsize_controller: dfx.AbstractStepSizeController = dataclasses.field(
        default_factory=dfx.ConstantStepSize
    )
    dt0: float | int | Array = 1e-3
    max_steps: int = 100_000

    def diffeqsolve_settings(self) -> dict[str, Any]:
        """Return normalized Diffrax settings for ``diffeqsolve``."""
        return {
            "solver": self.solver,
            "stepsize_controller": self.stepsize_controller,
            "adjoint": self.adjoint,
            "dt0": as_scalar_time_array(self.dt0, name="dt0"),
            "max_steps": self.max_steps,
        }


@dataclasses.dataclass
class SDESimulatorConfig:
    """Structured SDE simulator settings.

    This config collects the backend and solver settings used to simulate an
    SDE path. It supports either a Diffrax-based solve or the faster
    Euler-Maruyama scan backend.
    """

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

    def __post_init__(self) -> None:
        if self.source not in {"diffrax", "em_scan"}:
            raise ValueError(
                "SDESimulatorConfig.source must be one of {'diffrax', 'em_scan'}, "
                f"got source={self.source!r}."
            )

    def diffeqsolve_settings(self) -> dict[str, Any]:
        """Return normalized Diffrax-style backend settings."""
        return {
            "solver": self.solver,
            "stepsize_controller": self.stepsize_controller,
            "adjoint": self.adjoint,
            "dt0": as_scalar_time_array(self.dt0, name="dt0"),
            "max_steps": self.max_steps,
        }

    def resolved_tol_vbt(self) -> Real[Array, ""] | None:
        """Return the resolved Brownian-tree tolerance for the active backend."""
        if self.source != "diffrax":
            return None

        dt0_arr = as_scalar_time_array(self.dt0, name="dt0")
        tol_vbt_arr = (
            dt0_arr / 2.0
            if self.tol_vbt is None
            else as_scalar_time_array(self.tol_vbt, name="tol_vbt")
        )
        if bool(tol_vbt_arr >= dt0_arr):
            raise ValueError(
                "tol_vbt must be smaller than dt0 for statistically correct simulation."
            )
        return tol_vbt_arr


type SimulatorConfig = ODESimulatorConfig | SDESimulatorConfig
