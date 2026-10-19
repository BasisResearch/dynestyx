"""Resolve whether optional output fields should be recorded."""

import math


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
