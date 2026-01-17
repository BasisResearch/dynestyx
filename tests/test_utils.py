"""Utility functions for tests."""

import os
from datetime import datetime
from pathlib import Path


def get_output_dir(test_name: str) -> Path:
    """Create and return an output directory for a test within the master output directory.

    Args:
        test_name: Name of the test (e.g., "test_discreteTime_generic")

    Returns:
        Path to the output directory (e.g., ".output/20240101_120000_abc123/test_discreteTime_generic")
    """
    # Get master output directory from environment variable, or use default
    master_dir_str = os.environ.get("TEST_OUTPUT_MASTER_DIR", None)
    if master_dir_str is None:
        # Fallback: create timestamp-based directory if not set by test script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_dir = Path(".output") / timestamp
        master_dir.mkdir(parents=True, exist_ok=True)
    else:
        master_dir = Path(master_dir_str)

    # Create test-specific subdirectory within master directory
    output_dir = master_dir / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
