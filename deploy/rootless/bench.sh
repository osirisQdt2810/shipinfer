#!/usr/bin/env bash
# Run the head-to-head benchmark INSIDE a container, on real GPUs.
#
#   deploy/rootless/bench.sh                          # both systems, 50x20, 70 s
#   deploy/rootless/bench.sh --systems baseline       # one at a time, uncontended
#   deploy/rootless/bench.sh --systems shipinfer --seconds 40
#
# WHY THIS IS A SEPARATE SCRIPT FROM test.sh
#
# `test.sh` mounts the repository read-only, which is right for tests and wrong here: a
# benchmark's whole output is files. This mounts the repository read-write and writes only
# under `.artifacts/bench/`. It also needs three things the test image does not carry:
#
#   the baseline binary      built on the host by `benchmarks/harness/baseline.py`, because
#                            this box cannot build container images and the containers have
#                            no network, so `apt install g++ libopencv-dev` is impossible.
#                            A compiler is not a measurement; the measurement still runs here.
#   its shared-library set   staged into `benchmarks/build/baseline-libs`, minus libc,
#                            libstdc++ and the driver — shadowing the image's own would break
#                            torch in the same container.
#   TensorRT                 both systems load real plans, and no image on this box ships
#                            TensorRT. The host has 10.14.1 under /usr/local/TensorRT, and
#                            the image is jammy like the host, so its libs are mounted and
#                            the matching cp311 wheel (already staged in the wheel dir) is
#                            installed. Same trick as the baseline libraries, same reason.
#
# CUDA GRAPHS default to off here. The TensorRT execute path calls `stream.synchronize()`
# and can allocate a pinned staging buffer inside `fetch_output`, and CUDA forbids both
# inside a capture region — so capture cannot succeed for this path as written. Override
# with SHIPINFER_CUDA_GRAPHS=on to measure the difference once that is restructured.
#
# GPU HYGIENE: `--rm` so the container cannot outlive the run, and the driver reclaims every
# context when it exits. This box is shared and its VRAM is recorded continuously.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SHIPINFER_BENCH_IMAGE:-shipinfer-gst:jammy}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"
LIBS="$REPO/benchmarks/build/baseline-libs"
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon unreachable at $DOCKER_HOST — run deploy/rootless/setup.sh first" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image '$IMAGE' is absent. Set SHIPINFER_BENCH_IMAGE to one carrying TensorRT." >&2
  exit 1
fi

# The baseline binary is a host build. Say so plainly if it is missing rather than letting
# the run fail deep inside the harness.
if [ ! -x "$REPO/benchmarks/build/sim_pipeline_v2" ]; then
  echo "baseline binary missing. Build it on the host first:" >&2
  echo "  python -c 'from benchmarks.harness import baseline; baseline.build_binary()'" >&2
  exit 1
fi

if [ ! -d "$TRT_DIR/lib" ]; then
  echo "TensorRT not found at $TRT_DIR — set SHIPINFER_TENSORRT_DIR." >&2
  exit 1
fi

mkdir -p "$REPO/.artifacts/bench"

# `LD_LIBRARY_PATH` puts the staged closure *after* the image's own directories, so the
# image's libc and libstdc++ still win and only the baseline's extra dependencies come from
# the host.
exec docker run --rm --pid=host --device nvidia.com/gpu=all \
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/tensorrt/lib:/baseline-libs" \
  -e PYTHONPATH=/work/src:/work \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e SHIPINFER_IN_CONTAINER=1 \
  -e SHIPINFER_CUDA_GRAPHS="${SHIPINFER_CUDA_GRAPHS:-off}" \
  -v "$REPO:/work" \
  -v "$LIBS:/baseline-libs:ro" \
  -v "$TRT_DIR:/tensorrt:ro" \
  -v "$WHEELS:/wheels:ro" \
  -w /work "$IMAGE" \
  bash -c '
    set -e
    python -c "import tensorrt" 2>/dev/null || \
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels tensorrt \
        >/dev/null 2>&1 || true
    python -c "import pydantic" 2>/dev/null || \
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels \
        pydantic pydantic-settings typer pyyaml >/dev/null 2>&1 || true
    exec python benchmarks/run_bench.py "$@"
  ' bash "$@"
