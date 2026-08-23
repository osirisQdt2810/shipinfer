#!/usr/bin/env bash
# Container entrypoint.
#
# Thin on purpose: it validates the one thing that is worth failing fast on, then execs.
# `exec` matters — without it the shell stays PID 1's child and SIGTERM never reaches the
# server, so an orchestrator's graceful stop becomes a 30-second wait and a SIGKILL in the
# middle of a batch.
set -euo pipefail

REPO="${SHIPINFER_MODEL_REPOSITORY:-/workspace/model_repository}"

if [ ! -d "$REPO" ]; then
  echo "error: model repository not found at $REPO" >&2
  echo "       mount one with -v /path/to/model_repository:$REPO" >&2
  exit 78   # EX_CONFIG: a configuration problem, not a crash
fi

# One line at start-up that answers most of the questions a support ticket would ask.
shipinfer doctor || true

case "${1:-serve}" in
  serve|bench|doctor|repo|backends|policies|queues)
    exec shipinfer "$@"
    ;;
  test)
    shift
    exec bash scripts/run_tests.sh "$@"
    ;;
  shell|bash)
    exec /bin/bash
    ;;
  *)
    # Anything else is run verbatim, so `docker run ... python -c ...` still works.
    exec "$@"
    ;;
esac
