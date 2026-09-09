#!/usr/bin/env bash
# The offline test invocation, in one place.
#
# ci.yml and pr-pipeline.yml both call this so they cannot drift apart. Only the OFFLINE
# tier runs here: the GPU tier needs real devices that a GitHub runner does not have, and
# it is evidenced in the PR body instead (see .claude/WORKFLOW.md).
#
# Extra arguments are appended, which is how the coverage job adds its flags.
set -euo pipefail

# The repository's own virtualenv first, because that is the one with the dependencies in it.
# Falling through to whatever `python` is on PATH found the system interpreter and failed with
# "No module named pytest" — which reads like a broken test suite rather than a wrong
# interpreter, and it only surfaced when the repository moved and PATH no longer happened to
# carry the venv. `$PYTHON` still wins, for a caller that means a specific one; then a CI
# runner's `python` (setup-python provides it), then `python3` on a distro without the
# unversioned name.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python || command -v python3)"
fi

# Hide the GPUs, even on a box that has eight of them.
#
# The offline tier is *defined* as the part that runs with no accelerator, and the only
# honest way to check that is to run it with no accelerator. Deselecting the `gpu` marker
# is not the same thing: an unmarked test can still take a CUDA path without meaning to,
# pass on a dev box and fail on the runner. That is not hypothetical — it is how
# `torch.empty(pin_memory=True)` reached CI, where it raises rather than falling back.
#
# A developer who wants the GPU tier asks for it explicitly with `pytest -m gpu`, which
# does not go through this script.
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# `-m "not gpu"` is already the default in pyproject, but state it explicitly: a future edit
# to addopts must not silently start requiring a GPU in CI.
exec "$PYTHON" -m pytest -ra --strict-markers --strict-config \
  -m "not gpu and not multigpu" "$@"
