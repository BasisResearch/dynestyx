import math

from cd_dynamax import ContDiscreteNonlinearGaussianSSM as CDNLGSSM
from cd_dynamax import ContDiscreteNonlinearSSM as CDNLSSM
from jax import Array

from dynestyx.dynamical_models import DynamicalModel

type SSMType = CDNLGSSM | CDNLSSM


def _should_record_field(
    record_val: bool | None, shape: tuple[int, ...], max_elems: int
) -> bool:
    """
    Decide whether to record a field based on user preference and size.

    - If record_val is True: always record (obey user).
    - If record_val is False: never record (obey user).
    - If record_val is None (unspecified): record only if math.prod(shape) <= max_elems.
    """
    if record_val is True:
        return True
    if record_val is False:
        return False
    return math.prod(shape) <= max_elems


def _validate_control_dim(dynamics: DynamicalModel, ctrl_values: Array | None) -> None:
    """
    Validate that control_dim is set in DynamicalModel when controls are present.

    Args:
        dynamics: DynamicalModel instance
        ctrl_values: Control values array or None

    Raises:
        ValueError: If controls are provided but control_dim is not set or is 0
    """
    if ctrl_values is not None:
        if dynamics.control_dim is None or dynamics.control_dim == 0:
            # Try to infer from shape
            if ctrl_values.ndim >= 2:
                inferred_dim = ctrl_values.shape[1]
                raise ValueError(
                    f"Controls are provided (shape: {ctrl_values.shape}), but "
                    f"dynamics.control_dim is {dynamics.control_dim}. "
                    f"Please set control_dim={inferred_dim} when creating the DynamicalModel."
                )
            else:
                raise ValueError(
                    f"Controls are provided, but dynamics.control_dim is {dynamics.control_dim}. "
                    "Please set control_dim when creating the DynamicalModel."
                )


def _validate_controls(
    obs_times: Array,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
) -> None:
    """
    Validate that ctrl_times/ctrl_values align with obs_times if provided.

    Raises:
        ValueError: If control times length doesn't match observation times length.
    """
    if ctrl_times is None:
        if ctrl_values is not None:
            raise ValueError(
                "ctrl_values is not None, but ctrl_times is None. "
                "Provide both ctrl_times and ctrl_values together."
            )
        return
    if ctrl_values is None:
        raise ValueError(
            "ctrl_times is not None, but ctrl_values is None. "
            "Provide both ctrl_times and ctrl_values together."
        )
    if len(ctrl_times) != len(obs_times):
        raise ValueError(
            f"Control times length ({len(ctrl_times)}) must match "
            f"observation times length ({len(obs_times)})"
        )


def _get_val_or_None(values: Array | None, t_idx: int) -> Array | None:
    """
    Safely get value at index t_idx, returning None if values is None.

    Args:
        values: Values array or None
        t_idx: Time index to access

    Returns:
        Value at index t_idx, or None if values is None
    """
    return values[t_idx] if values is not None else None
