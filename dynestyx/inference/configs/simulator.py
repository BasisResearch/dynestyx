"""Configuration objects for continuous-time simulators."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import diffrax as dfx
from jaxtyping import Array, Real

from dynestyx.types import as_scalar_time_array
from dynestyx.utils import _raise_now_or_error_if


@dataclasses.dataclass
class ODESimulatorConfig:
    """Configuration object for ODE simulators.

    Attributes:
        solver (diffrax.AbstractSolver): Diffrax solver used for integration.
            Defaults to `diffrax.Tsit5()`. See Diffrax's
            [solver guide](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/)
            when selecting an alternative.
        adjoint (diffrax.AbstractAdjoint): Strategy used to differentiate
            through the solve. Defaults to
            `diffrax.RecursiveCheckpointAdjoint()`.
        stepsize_controller (diffrax.AbstractStepSizeController): Step-size
            policy used by Diffrax. Defaults to
            `diffrax.ConstantStepSize()`; supply an adaptive controller when
            error-controlled stepping is required.
        dt0 (float | int | jax.Array): Initial step size passed to
            `diffrax.diffeqsolve`. With the default constant-step controller,
            this is the fixed integration step. Defaults to `1e-3`.
        max_steps (int): Maximum number of integration steps permitted by
            `diffrax.diffeqsolve`. Defaults to `100_000`.

    Properties:
        diffeqsolve_settings (dict[str, Any]): Normalized keyword arguments passed
            to `diffrax.diffeqsolve`; scalar time values are converted to JAX
            arrays.
    """

    solver: dfx.AbstractSolver = dataclasses.field(default_factory=dfx.Tsit5)
    adjoint: dfx.AbstractAdjoint = dataclasses.field(
        default_factory=dfx.RecursiveCheckpointAdjoint
    )
    stepsize_controller: dfx.AbstractStepSizeController = dataclasses.field(
        default_factory=dfx.ConstantStepSize
    )
    dt0: float | int | Real[Array, ""] = 1e-3
    max_steps: int = 100_000

    @property
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
    """SDE Solver Settings for SDE Simulation. Supports diffrax-based solvers
    or a faster, hand-rolled Euler-Maruyama scan backend.

    !! Note: The choice of solver can imply convergence to different paths
       for the same model. For example, the default `diffrax.Heun()` converges
       to the Stratonovich SDE, while `diffrax.Euler()` converges to the Itô SDE.
       This likely doesn't matter for most models, but can cause issues with state-dependent diffusions.

    Attributes:
        solver (diffrax.AbstractSolver): Diffrax SDE solver. Defaults to
            `diffrax.Heun()`. This setting is used only when `source="diffrax"`.
        stepsize_controller (diffrax.AbstractStepSizeController): Diffrax
            step-size policy. Defaults to `diffrax.ConstantStepSize()` and is
            used only when `source="diffrax"`.
        adjoint (diffrax.AbstractAdjoint): Strategy used to differentiate
            through the Diffrax solve. Defaults to
            `diffrax.RecursiveCheckpointAdjoint()` and is used only when
            `source="diffrax"`.
        dt0 (float | int | jax.Array): Integration step size. It is passed to
            Diffrax as the initial step size and used as the fixed
            Euler-Maruyama step by the `"em_scan"` backend. Defaults to `1e-4`.
        tol_vbt (float | int | jax.Array | None): Tolerance for Diffrax's
            `VirtualBrownianTree`. When `source="diffrax"`, `None` resolves to
            `dt0 / 2`; an explicit value must be smaller than `dt0` for
            statistically correct simulation. Ignored by `"em_scan"`.
        max_steps (int | None): Maximum number of Diffrax integration steps.
            `None` leaves the Diffrax default in effect. The `"em_scan"`
            backend does not use this setting.
        source (Literal["diffrax", "em_scan"]): Simulation backend.
            `"diffrax"` uses the configured Diffrax solver and a virtual
            Brownian tree. `"em_scan"` uses a fixed-step Euler-Maruyama
            `jax.lax.scan` and is the default for speed.

    Properties:
        diffeqsolve_settings (dict[str, Any]): Normalized Diffrax keyword
            arguments derived from the config; scalar time values are
            converted to JAX arrays.
        resolved_tol_vbt (jax.Array | None): Effective virtual-Brownian-tree
            tolerance for the selected backend. Returns `None` for `"em_scan"`
            and validates the tolerance for `"diffrax"`.
    """

    solver: dfx.AbstractSolver = dataclasses.field(default_factory=dfx.Heun)
    stepsize_controller: dfx.AbstractStepSizeController = dataclasses.field(
        default_factory=dfx.ConstantStepSize
    )
    adjoint: dfx.AbstractAdjoint = dataclasses.field(
        default_factory=dfx.RecursiveCheckpointAdjoint
    )
    dt0: float | int | Real[Array, ""] = 1e-4
    tol_vbt: float | int | Real[Array, ""] | None = None
    max_steps: int | None = None
    source: Literal["diffrax", "em_scan"] = "em_scan"

    def __post_init__(self) -> None:
        if self.source not in {"diffrax", "em_scan"}:
            raise ValueError(
                "SDESimulatorConfig.source must be one of {'diffrax', 'em_scan'}, "
                f"got source={self.source!r}."
            )

    @property
    def diffeqsolve_settings(self) -> dict[str, Any]:
        """Return normalized Diffrax-style backend settings."""
        return {
            "solver": self.solver,
            "stepsize_controller": self.stepsize_controller,
            "adjoint": self.adjoint,
            "dt0": as_scalar_time_array(self.dt0, name="dt0"),
            "max_steps": self.max_steps,
        }

    @property
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
        tol_vbt_arr = _raise_now_or_error_if(
            tol_vbt_arr,
            tol_vbt_arr >= dt0_arr,
            "tol_vbt must be smaller than dt0 for statistically correct simulation.",
        )
        return tol_vbt_arr


type SimulatorConfig = ODESimulatorConfig | SDESimulatorConfig
