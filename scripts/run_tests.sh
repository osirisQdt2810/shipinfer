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
