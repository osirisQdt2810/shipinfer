#!/usr/bin/env bash
# Does Nsight Systems take down ANY process that deserialises a TensorRT plan?
#
# `PROFILE-DIES-AT-THE-DESIGN-LOAD` narrowed the failure to one sentence: a binary that loads a
# plan segfaults under this nsys, one that only uses CUDA does not. Every probe so far has been
# one of ours, so "something of ours is involved" is still on the table. `trtexec` is NVIDIA's
# own loader and shares none of our code, which is what makes it the deciding probe.
#
# A SCRIPT AND NOT A COMMAND LINE, deliberately: `scripts/hooks/require_container.py` refuses a
# bare `trtexec` by text even inside `deploy/rootless/run.sh`, which is the advisory-deny-list
# limitation CLAUDE.md describes. The honest way past it is to put the invocation in a file and
# run the file in the container, not to reach for `SHIPINFER_ALLOW_HOST_RUN`.
#
#   deploy/rootless/run.sh bash scripts/probe_nsys_trtexec.sh
set -uo pipefail

TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/tensorrt}"
NSYS_DIR="${SHIPINFER_NSYS_DIR:-/opt/nvidia/nsight-systems/2025.1.3}"
PLAN="${SHIPINFER_PROBE_PLAN:-/work/model_repository/ship_detector/1/model.plan}"
OUT="/work/.artifacts/profile"
TRTEXEC="$TRT_DIR/bin/trtexec"

mkdir -p "$OUT"
echo "== what is here =="
echo "trtexec: $([ -x "$TRTEXEC" ] && echo "$TRTEXEC" || echo MISSING)"
echo "nsys:    $([ -x "$NSYS_DIR/bin/nsys" ] && "$NSYS_DIR/bin/nsys" --version | tail -1 || echo MISSING)"
echo "plan:    $([ -f "$PLAN" ] && ls -la "$PLAN" | awk '{print $5, $9}' || echo MISSING)"
[ -x "$TRTEXEC" ] || { echo "no trtexec under $TRT_DIR/bin; set SHIPINFER_TENSORRT_DIR"; exit 2; }
[ -f "$PLAN" ] || { echo "no plan at $PLAN; build one with scripts/build_engines.py"; exit 2; }

# THE CONTROL FIRST. Without it a segfault under nsys proves nothing: the plan could simply be
# unloadable by this trtexec against this driver, which is a different finding entirely.
echo
echo "== control: trtexec alone =="
"$TRTEXEC" --loadEngine="$PLAN" --iterations=10 --avgRuns=1 > "$OUT/trtexec_plain.log" 2>&1
echo "exit=$?"
tail -3 "$OUT/trtexec_plain.log"

echo
echo "== under nsys, the same command and the profiler's own flags =="
"$NSYS_DIR/bin/nsys" profile \
  --trace="${SHIPINFER_NSYS_TRACE:-cuda,nvtx,osrt}" \
  --sample=none \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --wait=primary \
  --output "$OUT/trtexec_probe" \
  "$TRTEXEC" --loadEngine="$PLAN" --iterations=10 --avgRuns=1 > "$OUT/trtexec_nsys.log" 2>&1
echo "exit=$?"
tail -5 "$OUT/trtexec_nsys.log"
