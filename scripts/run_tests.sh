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
SEARCHED=0
if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
  SEARCHED=1
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  # An ACTIVATED venv outranks the search below: a caller who activated one has said which
  # interpreter they mean nearly as plainly as `PYTHON=`, and without this a stale `.venv` in
  # the primary checkout silently wins over it from a linked worktree.
  PYTHON="$VIRTUAL_ENV/bin/python"
  SEARCHED=1
else
  SEARCHED=1
  # A GIT WORKTREE HAS NO `.venv` OF ITS OWN, and that is the failure above recurring: run
  # from one, this fell through to the system interpreter and said "No module named pytest".
  # The main worktree's venv is the one with the dependencies in it, and `--git-common-dir`
  # is how you find it from any linked worktree (it points at the primary `.git`).
  MAIN_ROOT="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  MAIN_ROOT="${MAIN_ROOT%/.git}"
  if [ -n "$MAIN_ROOT" ] && [ -x "$MAIN_ROOT/.venv/bin/python" ]; then
    PYTHON="$MAIN_ROOT/.venv/bin/python"
  else
    # `|| true`, because `set -e` on a failed substitution aborts with ZERO output -- and
    # silence is the whole thing this block exists to remove. The check below speaks instead.
    PYTHON="$(command -v python || command -v python3 || true)"
  fi
fi

# AND SAY WHICH PROBLEM IT IS. CI has no `.venv` and relies on `setup-python`'s interpreter,
# so falling through is legitimate there -- what is not legitimate is the message it used to
# fail with, because "No module named pytest" reads like a broken suite rather than a wrong
# interpreter. This is the whole reason the search above exists, so it is asserted.
if [ -z "${PYTHON:-}" ]; then
  echo "run_tests.sh: found no python interpreter at all." >&2
  echo "  Looked for a venv at $REPO_ROOT/.venv, an activated VIRTUAL_ENV, the main" >&2
  echo "  worktree's venv, and \`python\`/\`python3\` on PATH." >&2
  exit 1
fi
if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
  echo "run_tests.sh: $PYTHON has no pytest." >&2
  if [ "$SEARCHED" = "1" ]; then
    # Only when a search actually happened. Printing where it looked after the caller named
    # the interpreter with `PYTHON=` describes work nobody did.
    echo "  Looked for a venv at $REPO_ROOT/.venv, an activated VIRTUAL_ENV, and, from a" >&2
    echo "  worktree, the main checkout's." >&2
  fi
  echo "  Install the dev extra (\`pip install -e '.[dev,cli]'\`) or name an" >&2
  echo "  interpreter: PYTHON=/path/to/python bash scripts/run_tests.sh" >&2
  exit 1
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
