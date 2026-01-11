#!/bin/bash
set -euxo pipefail

SRC="tests/ dsx/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
