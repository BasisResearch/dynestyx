#!/bin/bash
set -euxo pipefail

SRC="dynestyx/ tests/"
ruff check --fix $SRC
ruff format $SRC
