#!/usr/bin/env bash
# The test invocation, in one place, so ci.yml and pr-pipeline.yml cannot drift.
# Extra arguments are appended.
set -euo pipefail
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
exec "$PYTHON" -m pytest -ra --strict-markers --strict-config "$@"
