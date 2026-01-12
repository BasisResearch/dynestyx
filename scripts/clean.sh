#!/bin/bash
set -euxo pipefail

SRC="dsx tests"
ruff check --fix $SRC
ruff format $SRC
