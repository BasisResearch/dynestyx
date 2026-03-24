#!/bin/bash
set -euxo pipefail

SRC="tests/ dynestyx/"

ty check $SRC
ruff check $SRC
ruff format --diff $SRC
