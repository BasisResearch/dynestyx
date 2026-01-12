#!/bin/bash
set -euxo pipefail

pytest tests/ -n auto
