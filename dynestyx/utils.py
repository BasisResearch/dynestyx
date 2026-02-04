from cd_dynamax import ContDiscreteNonlinearGaussianSSM as CDNLGSSM
from cd_dynamax import ContDiscreteNonlinearSSM as CDNLSSM
from jax import Array

from dynestyx.dynamical_models import DynamicalModel
from dynestyx.ops import Context

type SSMType = CDNLGSSM | CDNLSSM


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


def _get_controls(
    context: Context, obs_times: Array
) -> tuple[Array | None, Array | None]:
    """
    Extract and validate controls from context.

    Args:
        context: Context containing controls trajectory
        obs_times: Observation times array for validation

    Returns:
        Tuple of (ctrl_times, ctrl_values). Both are None if no controls are provided.
        If controls are provided, ctrl_times and ctrl_values are extracted and validated.

    Raises:
        ValueError: If control times length doesn't match observation times length,
                    or if ctrl_values is a dict.
    """
    # Pull control trajectory from context
    # Only validate controls if they actually have times
    # If controls is a Trajectory with times=None, treat it as no controls
    ctrl_traj = context.controls
    ctrl_times = ctrl_traj.times if ctrl_traj is not None else None

    if ctrl_times is None:
        if ctrl_traj.values is not None:
            raise ValueError(
                "ctrl_traj.values is not None, but ctrl_times is None. This is likely a bug in the context creation."
            )
        # No controls provided
        return None, None
    elif ctrl_traj.values is None:
        raise ValueError(
            "ctrl_traj.values is None, but ctrl_times is not None. This is likely a bug in the context creation."
        )

    # Check lengths match (concrete check, safe in traced context)
    if len(ctrl_times) != len(obs_times):
        raise ValueError(
            f"Control times length ({len(ctrl_times)}) must match "
            f"observation times length ({len(obs_times)})"
        )
    # Note: Full equality check would require jnp.array_equal which creates
    # traced booleans. We trust that if lengths match, times match (validated
    # at fixture/context creation time).

    # Controls are provided (have times), extract and validate
    ctrl_values = ctrl_traj.values

    # Validate ctrl_values is not a dict
    if isinstance(ctrl_values, dict):
        raise ValueError("ctrl_values must be an Array or None, not a dict")

    return ctrl_times, ctrl_values


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
