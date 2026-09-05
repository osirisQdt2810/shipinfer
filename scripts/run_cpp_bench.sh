#!/usr/bin/env bash
# The standard C++ data-plane measurement, so the flags are in one place rather than retyped.
#
#   scripts/run_cpp_bench.sh <label> [extra flags...]
#
# 50 cameras x 20 fps = the design load, 70 s with the analysis's 10 s warmup.
#
# The instance counts, the batch windows and the engine paths are NOT here any more: they are
# `model_repository/*/config.yaml`'s and they arrive in the resolved plan (P5-B). They used to
# be restated on this command line and the numbers disagreed -- 3/3/3 against the repository's
# 2/2/1, and one global 2000 us against its 5000/8000/8000/3000 -- so the two planes were
# measured at configurations neither file described.
#
# The WORKER COUNT went the same way (P5-C), with the queue capacities and the reassembly
# window beside it: they are `core/settings/`'s, `shipinfer plan` reads them below, and the
# binary refuses a plan that states none. `SHIPINFER_BENCH_WORKERS` therefore sets the
# SETTING -- the plan is written in this shell, so the env var reaches it the ordinary way --
# rather than a flag the binary would have had to reconcile with the file.
set -euo pipefail

LABEL="${1:?usage: run_cpp_bench.sh <label> [extra flags...]}"
shift || true
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/.artifacts/cpp"

# The plan, written by the control plane from the chain file and the repository. Rewritten on
# every run rather than committed: a stale plan is a measurement of a configuration the
# repository no longer describes, which is the defect this replaced.
CHAIN="${SHIPINFER_BENCH_CHAIN:-$REPO/topology/ship_person_cpu.yaml}"
PLAN="$REPO/.artifacts/cpp/${LABEL}.plan"
SHIPINFER_PIPELINE__WORKERS="${SHIPINFER_BENCH_WORKERS:-48}" \
  python -m shipinfer plan -t "$CHAIN" -r "$REPO/model_repository" -o "$PLAN"

# `set -e` would abort on a non-zero exit before the status is printed, so the run that most
# needs reading — a timeout (124), a crash — would leave no line and no summary. Capture it.
status=0
timeout "${SHIPINFER_BENCH_TIMEOUT:-900}" "$REPO/deploy/rootless/cpp.sh" \
  --person-frames /work/benchmarks/baseline/data/person_2K \
  --ship-frames   /work/benchmarks/baseline/data/ship_2K \
  --plan          "/work/.artifacts/cpp/${LABEL}.plan" \
  --repository    /work/model_repository \
  --gpu-ids "${SHIPINFER_BENCH_GPUS:-2,3,4,5}" \
  --cameras "${SHIPINFER_BENCH_CAMERAS:-50}" \
  --fps "${SHIPINFER_BENCH_FPS:-20}" \
  --seconds "${SHIPINFER_BENCH_SECONDS:-70}" \
  --log-jsonl "/work/.artifacts/cpp/${LABEL}.jsonl" \
  "$@" > "$REPO/.artifacts/cpp/${LABEL}.log" 2>&1 || status=$?
echo "exit=$status"
grep -E '^(startup_s|frames_read|frames_dropped|frames_accepted|frames_failed|events_emitted|events_complete|events_incomplete|queue_rejected|collector_)' \
  "$REPO/.artifacts/cpp/${LABEL}.log" || true
echo "--- final occupancy ---"
tail -1 "$REPO/.artifacts/cpp/${LABEL}.jsonl"
