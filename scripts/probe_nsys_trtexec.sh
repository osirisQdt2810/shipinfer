#!/usr/bin/env bash
# Does Nsight Systems take down ANY process that deserialises a TensorRT plan?
#
# `PROFILE-DIES-AT-THE-DESIGN-LOAD` narrowed the failure to one sentence: a binary that loads a
# plan segfaults under this nsys, one that only uses CUDA does not. Every probe so far has been
# one of ours, so "something of ours is involved" is still on the table. `trtexec` is NVIDIA's
# own loader and shares none of our code, which is what makes it the deciding probe.
#
# A SCRIPT because the probe is three commands with a control, not because the hook demanded
# one. The ledger said the hook "refuses a bare `trtexec` even inside `deploy/rootless/run.sh`";
# it does not -- `_is_containerised` matches the segment's own executable and exempts the whole
# segment, so `run.sh trtexec …` was always allowed. That claim is corrected there too.
#
# WHAT THIS FILE DOES NEED is its own gate, and it is below rather than in the hook alone: a
# shell wrapper hides `trtexec` from a deny-list over command text, and `trtexec` is NVIDIA's
# binary with no `runtime/containment.h` of its own. The hook now knows this filename as well,
# but the check that cannot be spelled around is the one in the process that would do the work.
#
#   deploy/rootless/run.sh bash scripts/probe_nsys_trtexec.sh
set -uo pipefail

# THE PROJECT'S OWN GATE, called rather than reimplemented. `containment.require_container`
# is two-of-three -- marker file, pid 1's cgroup, overlay root -- because `/.dockerenv` alone
# is a file anyone can touch; a shell copy of that rule would be a second definition to drift.
# Refusing here matters because a wrapper hides `trtexec` from a deny-list over command text.
python -c 'from shipinfer.runtime import containment
containment.require_container("the nsys x TensorRT probe")' || exit 2

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
