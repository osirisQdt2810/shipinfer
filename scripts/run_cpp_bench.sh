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

GPU_IDS="${SHIPINFER_BENCH_GPUS:-2,3,4,5}"
NGPUS="$(printf '%s' "$GPU_IDS" | tr ',' '\n' | grep -c .)"

# PER GPU, because the workers feed per-GPU model instances and this file's GPU set is a
# variable. 23 x 7 = 161 is the measurement below; a fixed 160 would have been right only for
# the seven-device set the sweep used and wrong at this file's own default of four.
#
# MEASURED at 50 x 20 fps on GPUs 0-6 (7 is `GPU7-DEGRADED`), 70 s each, one variable. At the
# old 48 the run retired 597 img/s and rejected 40% at the ingest queue while every model queue
# sat near empty -- the workers were the binding constraint, not the GPUs:
#
#   total  48 (~7/gpu) -> 597 img/s   (27 974 rejected, 40%)
#   total 112 (16/gpu) -> 858 img/s   ( 9 722 rejected, 14%)
#   total 160 (23/gpu) -> 937 img/s   ( 4 015 rejected, 5.7%)   <- the plateau starts here
#   total 192 (27/gpu) -> 940 img/s   ( 3 761 rejected, 5.4%)
#   total 224 (32/gpu) -> 690 img/s   (21 118 rejected, 30%)    <- the segmenter backs up
#
# Past ~27/gpu it inverts: `ship_segmenter_buffer_size` goes 30 -> 123 while the others stay
# near ten, so the contention has moved into one model's queue. 23 is the start of the plateau
# rather than its top -- 27 buys 0.3% for 20% more threads on a box that is shared.
#
# THE PER-GPU FORM IS AN ASSUMPTION, not a measurement: the sweep ran on one device count, and
# what generalises is that the workers feed per-GPU instances. Re-sweep before trusting it on a
# very different device count.
WORKERS_PER_GPU="${SHIPINFER_BENCH_WORKERS_PER_GPU:-23}"
SHIPINFER_PIPELINE__WORKERS="${SHIPINFER_BENCH_WORKERS:-$((WORKERS_PER_GPU * NGPUS))}" \
  python -m shipinfer plan -t "$CHAIN" -r "$REPO/model_repository" -o "$PLAN"

# `set -e` would abort on a non-zero exit before the status is printed, so the run that most
# needs reading — a timeout (124), a crash — would leave no line and no summary. Capture it.
status=0
timeout "${SHIPINFER_BENCH_TIMEOUT:-900}" "$REPO/deploy/rootless/cpp.sh" \
  --person-frames /work/benchmarks/baseline/data/person_2K \
  --ship-frames   /work/benchmarks/baseline/data/ship_2K \
  --plan          "/work/.artifacts/cpp/${LABEL}.plan" \
  --repository    /work/model_repository \
  --gpu-ids "$GPU_IDS" \
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
