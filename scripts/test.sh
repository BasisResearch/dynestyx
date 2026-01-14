#!/bin/bash
set -euxo pipefail

pytest tests/ -n auto --ignore=tests/test_science
