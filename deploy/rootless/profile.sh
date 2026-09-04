#!/usr/bin/env bash
# Profile one run under Nsight Systems, in the container, on real GPUs.
#
# WHY THIS EXISTS
# ---------------
# Every optimisation in this repository so far was chosen by reasoning from a symptom. The
# buffer-growth log says *which queue grows*; it does not say *where the time goes*, and the
# operator was right that the answer should be a profile. `benchmarks/stages.py` gives the
# per-stage host timings; this gives the device timeline underneath them — kernel durations,
# memcpy directions, stream overlap, and the gaps where nothing is running.
#
# WHAT IT CAN AND CANNOT TRACE HERE
# ---------------------------------
# `/proc/sys/kernel/perf_event_paranoid` is 4 on this box, so CPU **sampling** needs
# CAP_PERFMON and is switched off (`--sample=none`). CUDA and NVTX tracing do not use perf
# events and work as they are. That is the useful half: it answers "is the GPU idle, and if so
# between what", which is the question standing behind the 390 img/s ceiling.
#
# Nsight is mounted from the host rather than baked into the image, exactly as TensorRT is:
# the image is not rebuilt in this environment (see `deploy/rootless/test.sh`).
#
#   deploy/rootless/profile.sh --systems shipinfer --cameras 50 --fps 20 --seconds 30
#   deploy/rootless/profile.sh --cpp -- --cameras 50 --fps 20 --seconds 30   # the C++ plane
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Which GPUs this container may see, and why one degraded card is not a dead tier.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gpus.sh"
IMAGE="${SHIPINFER_BENCH_IMAGE:-shipinfer-gst:jammy}"
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"
LIBS="$REPO/benchmarks/build/baseline-libs"
NSYS_DIR="${SHIPINFER_NSYS_DIR:-$(ls -d /opt/nvidia/nsight-systems/* 2>/dev/null | sort -V | tail -1)}"
OUT="${SHIPINFER_PROFILE_OUT:-$REPO/.artifacts/profile}"

TARGET="python"
if [ "${1:-}" = "--cpp" ]; then
  TARGET="cpp"
  shift
  [ "${1:-}" = "--" ] && shift
fi

if [ -z "$NSYS_DIR" ] || [ ! -x "$NSYS_DIR/bin/nsys" ]; then
  echo "no Nsight Systems found. Set SHIPINFER_NSYS_DIR to an install root containing bin/nsys." >&2
  exit 1
fi
if [ ! -d "$TRT_DIR/lib" ]; then
  echo "TensorRT not found at $TRT_DIR — set SHIPINFER_TENSORRT_DIR." >&2
  exit 1
fi
mkdir -p "$OUT"

# `--sample=none` for the reason in the header; `--cuda-memory-usage` because a stall that is
# actually an allocator serialising is invisible without it, and this project has already been
# bitten by a per-frame `cudaMalloc` on the dispatch path.
NSYS_ARGS=(
  "$NSYS_DIR/bin/nsys" profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --cuda-memory-usage=true
  --force-overwrite=true
  --output "/work/.artifacts/profile/run"
)

if [ "$TARGET" = "cpp" ]; then
  BINARY="$REPO/csrc/build/shipinfer_pipeline"
  if [ ! -x "$BINARY" ]; then
    echo "no binary at $BINARY — run: python scripts/build_csrc.py" >&2
    exit 1
  fi
  COMMAND=(/work/csrc/build/shipinfer_pipeline "$@")
else
  COMMAND=(python /work/benchmarks/run_bench.py "$@")
fi

mount_libs=()
path_libs=""
if [ -d "$LIBS" ]; then
  mount_libs=(-v "$LIBS:/baseline-libs:ro")
  path_libs=":/baseline-libs"
fi

exec docker run --rm --pid=host "${GPU_DEVICES[@]}" \
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/tensorrt/lib:/usr/local/cuda-12.6/lib64${path_libs}" \
  -e PYTHONPATH=/work/src:/work \
  -e SHIPINFER_IN_CONTAINER=1 \
  -e SHIPINFER_CUDA_GRAPHS="${SHIPINFER_CUDA_GRAPHS:-off}" \
  -v "$REPO:/work" \
  -v "$TRT_DIR:/tensorrt:ro" \
  -v "$WHEELS:/wheels:ro" \
  -v "$NSYS_DIR:$NSYS_DIR:ro" \
  "${mount_libs[@]}" \
  -w /work "$IMAGE" \
  bash -c '
    python -c "import tensorrt" 2>/dev/null || \
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels tensorrt >/dev/null 2>&1 || true
    python -c "import pydantic" 2>/dev/null || \
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels \
        pydantic pydantic-settings typer pyyaml >/dev/null 2>&1 || true
    exec "$@"
  ' bash "${NSYS_ARGS[@]}" "${COMMAND[@]}"
