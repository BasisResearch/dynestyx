#!/bin/bash
set -euxo pipefail

# Create master output directory with datetime and random suffix
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RANDOM_SUFFIX=$(openssl rand -hex 4)
MASTER_OUTPUT_DIR=".output/${TIMESTAMP}_${RANDOM_SUFFIX}"
mkdir -p "$MASTER_OUTPUT_DIR"

# Export for use by get_output_dir
export TEST_OUTPUT_MASTER_DIR="$MASTER_OUTPUT_DIR"

echo "Master output directory: $MASTER_OUTPUT_DIR"

DYNESTYX_SMOKE_TEST=1 DYNESTYX_PF_PARTICLES_SCALE=0.01 pytest tests/ -n auto --ignore=tests/test_science
