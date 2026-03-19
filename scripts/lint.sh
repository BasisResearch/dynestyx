#!/bin/bash
set -euxo pipefail

SRC="tests/ dynestyx/ docs/ scripts/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
