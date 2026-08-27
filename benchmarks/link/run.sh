#!/usr/bin/env bash
# Run the link probe and the IPC context-cost probe INSIDE a container (CLAUDE.md rule:
# anything that touches an accelerator runs in a container). Same image and CDI device
# spelling as deploy/rootless/test.sh; `--rm` and `timeout` so a stuck run cannot hold VRAM.
#
#   benchmarks/link/run.sh                # both probes, logs under benchmarks/link/results/<date>/
#                                         # (un-ignored in .gitignore: the docs cite them)
#   benchmarks/link/run.sh --pairs 3-4     # link probe only, on chosen pairs
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SHIPINFER_TEST_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}"
OUT="${SHIPINFER_LINK_OUT:-$REPO/benchmarks/link/results/$(date -u +%Y-%m-%d)}"  # committed evidence, not .artifacts/
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
mkdir -p "$OUT"
run() {  # name, script, args...
  local name="$1"; shift
  timeout 600 docker run --rm --pid=host --device nvidia.com/gpu=all \
    -e PYTHONDONTWRITEBYTECODE=1 -v "$REPO:/work:ro" -w /work "$IMAGE" \
    python "$@" | tee "$OUT/$name.log"
}
run link_probe benchmarks/link/link_probe.py "$@"
if [ $# -eq 0 ]; then
  run ipc_context_cost benchmarks/link/ipc_context_cost.py
fi
echo "logs: $OUT/*.log"
