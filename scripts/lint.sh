#!/bin/bash
set -euxo pipefail

SRC="tests/ dynestyx/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
