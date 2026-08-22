#!/usr/bin/env bash
# The offline test invocation, in one place.
#
# ci.yml and pr-pipeline.yml both call this so they cannot drift apart. Only the OFFLINE
# tier runs here: the GPU tier needs real devices that a GitHub runner does not have, and
# it is evidenced in the PR body instead (see .claude/WORKFLOW.md).
#
# Extra arguments are appended, which is how the coverage job adds its flags.
set -euo pipefail

# `python` on a CI runner (setup-python provides it), `python3` on a distro that does not
# ship the unversioned name, and `$PYTHON` when a caller wants a specific interpreter.
# Guessing wrong here fails with "python: not found", which reads like a broken test suite.
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

# `-m "not gpu"` is already the default in pyproject, but state it explicitly: a future edit
# to addopts must not silently start requiring a GPU in CI.
exec "$PYTHON" -m pytest -ra --strict-markers --strict-config \
  -m "not gpu and not multigpu" "$@"
