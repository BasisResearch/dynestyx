"""Utility functions for tests."""

from datetime import datetime
from pathlib import Path


def get_output_dir(test_name: str) -> Path:
    """Create and return a datetime-based output directory for a test.

    Args:
        test_name: Name of the test (e.g., "test_discreteTime_generic")

    Returns:
        Path to the output directory (e.g., ".output/test_discreteTime_generic/20240101_120000")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(".output") / test_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
