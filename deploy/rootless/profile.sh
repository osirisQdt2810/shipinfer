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
#   SHIPINFER_CPP_COMMAND='bash /work/scripts/cpp_bench_over_rtsp.sh' \
#     deploy/rootless/profile.sh --cpp -- --plan /work/.artifacts/cpp/run.plan …  # over RTSP
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Which GPUs this container may see, and why one degraded card is not a dead tier.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gpus.sh"
IMAGE="${SHIPINFER_BENCH_IMAGE:-shipinfer-gst:jammy}"
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"
LIBS="$REPO/benchmarks/build/baseline-libs"
# NOT SIMPLY THE NEWEST, which is how this picked the broken one for a week. MEASURED
# 12 Sep with `scripts/probe_nsys_trtexec.sh`: NVIDIA's OWN `trtexec --loadEngine` segfaults
# (exit 139) under nsys 2025.1.3 and exits 0 under 2024.5.1 and 2024.6.2 -- same container,
# same plan, same flags -- so nothing of ours is involved and the profiler is the variable.
# `PROFILE-DIES-AT-THE-DESIGN-LOAD` has the table.
#
# A LIST OF ONE, and it is a recorded measurement rather than a guess: re-run that probe when
# a new Nsight lands, and delete the entry when the version it names is gone from the box.
NSYS_BROKEN_VERSIONS="${SHIPINFER_NSYS_BROKEN:-2025.1.3}"
pick_nsys() {
  local newest="" candidate
  for candidate in $(ls -d /opt/nvidia/nsight-systems/* 2>/dev/null | sort -V); do
    [ -x "$candidate/bin/nsys" ] || continue
    case " $NSYS_BROKEN_VERSIONS " in
      *" $(basename "$candidate") "*)
        echo "nsys $(basename "$candidate") skipped: it segfaults on TensorRT engine load" \
             "(scripts/probe_nsys_trtexec.sh). Set SHIPINFER_NSYS_DIR to use it anyway." >&2
        continue
        ;;
    esac
    newest="$candidate"
  done
  printf '%s' "$newest"
}
NSYS_DIR="${SHIPINFER_NSYS_DIR:-$(pick_nsys)}"
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
  # NARROWABLE, because the tracer is a suspect: the C++ bench segfaults under this profiler
  # (exit 139) at twelve cameras and at fifty while the same binary runs clean without it, and
  # finding which tracer takes it down means running them one at a time
  # (`PROFILE-DIES-AT-THE-DESIGN-LOAD`).
  "--trace=${SHIPINFER_NSYS_TRACE:-cuda,nvtx,osrt}"
  --sample=none
  --cuda-memory-usage=true
  --force-overwrite=true
  # PRIMARY ONLY. The RTSP route's wrapper `exec`s the bench and leaves its two servers
  # running as siblings (`cpp_bench_over_rtsp.sh`: the container's exit reaps them), so the
  # default `--wait=all` sits waiting for processes that outlive the measurement and no report
  # is written at all -- measured 11 Sep: "One or more process it created re-parented".
  --wait=primary
  --output "/work/.artifacts/profile/run"
)

if [ "$TARGET" = "cpp" ]; then
  # `cpp.sh`'s contract, spelled the same way on purpose: V168 makes optimisation a loop --
  # benchmark, then profile -- and a profiler that cannot launch what the benchmark launches
  # breaks the loop at the join. It named `shipinfer_pipeline`, which `scripts/build_csrc.py`
  # has never produced (the binaries are `csrc/shipinfer/cli/*.cpp`), so `--cpp` could not run
  # at all; and the mandated route needs `SHIPINFER_CPP_COMMAND`, because the RTSP servers
  # start in the SAME container as the bench (see `cpp_bench_over_rtsp.sh`).
  CPP_BINARY="${SHIPINFER_CPP_BINARY:-bench}"
  BINARY="$REPO/csrc/build/$CPP_BINARY"
  if [ ! -x "$BINARY" ]; then
    echo "no binary at $BINARY — run: python scripts/build_csrc.py" >&2
    exit 1
  fi
  if [ -n "${SHIPINFER_CPP_COMMAND:-}" ]; then
    # Deliberately word-split: the value is a command line, not a path.
    # shellcheck disable=SC2206
    COMMAND=(${SHIPINFER_CPP_COMMAND} "$@")
  else
    COMMAND=("/work/csrc/build/$CPP_BINARY" "$@")
  fi
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
  -e SHIPINFER_CPP_BINARY -e SHIPINFER_CUDA_BLOCKING_SYNC -e SHIPINFER_DEVICE_FOLD \
  -e SHIPINFER_RTSP_PERSON_DATA -e SHIPINFER_RTSP_SHIP_DATA -e SHIPINFER_RTSP_PORT \
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
