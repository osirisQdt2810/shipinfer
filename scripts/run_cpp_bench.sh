#!/usr/bin/env bash
# The standard C++ data-plane measurement, so the flags are in one place rather than retyped.
#
#   scripts/run_cpp_bench.sh <label> [extra flags...]
#
# 50 cameras x 20 fps = the design load, 70 s with the analysis's 10 s warmup. Instances match
# the four `model_repository/*/config.yaml` files unless overridden on the command line.
set -euo pipefail

LABEL="${1:?usage: run_cpp_bench.sh <label> [extra flags...]}"
shift || true
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/.artifacts/cpp"

timeout "${SHIPINFER_BENCH_TIMEOUT:-900}" "$REPO/deploy/rootless/cpp.sh" \
  --person-frames /work/benchmarks/baseline/data/person_2K \
  --ship-frames   /work/benchmarks/baseline/data/ship_2K \
  --det-engine    /work/models/yolo26n_fp32.engine \
  --seg-engine    /work/models/yolo26n-seg_fp32.engine \
  --emb-engine    /work/models/reid_r50_fp32.engine \
  --ship-emb-engine /work/models/reid_r50_fp32.engine \
  --gpu-ids 2,3,4,5 \
  --cameras "${SHIPINFER_BENCH_CAMERAS:-50}" \
  --fps "${SHIPINFER_BENCH_FPS:-20}" \
  --seconds "${SHIPINFER_BENCH_SECONDS:-70}" \
  --workers "${SHIPINFER_BENCH_WORKERS:-48}" \
  --seg-instances "${SHIPINFER_SEG_INSTANCES:-3}" \
  --emb-instances "${SHIPINFER_EMB_INSTANCES:-3}" \
  --ship-emb-instances "${SHIPINFER_SHIP_EMB_INSTANCES:-3}" \
  --log-jsonl "/work/.artifacts/cpp/${LABEL}.jsonl" \
  "$@" > "$REPO/.artifacts/cpp/${LABEL}.log" 2>&1

echo "exit=$?"
grep -E '^(startup_s|frames_read|frames_dropped|frames_accepted|frames_failed|events_emitted|events_complete|events_incomplete|queue_rejected|collector_)' \
  "$REPO/.artifacts/cpp/${LABEL}.log" || true
echo "--- final occupancy ---"
tail -1 "$REPO/.artifacts/cpp/${LABEL}.jsonl"
