#!/usr/bin/env bash
# Run the C++ data plane INSIDE a container, on real GPUs.
#
# Same arrangement as `bench.sh`: the binary is built on the host (see `scripts/build_csrc.py`
# for why), its library closure is already staged under `benchmarks/build/baseline-libs` by the
# baseline harness, and the container mounts that plus the host's TensorRT.
#
#   deploy/rootless/cpp.sh --cameras 12 --fps 10 --seconds 40 --log-jsonl /work/out.jsonl
#   SHIPINFER_CPP_BINARY=test_dataplane deploy/rootless/cpp.sh    # the test binary
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SHIPINFER_BENCH_IMAGE:-shipinfer-gst:jammy}"
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
LIBS="$REPO/benchmarks/build/baseline-libs"
BINARY="$REPO/csrc/build/${SHIPINFER_CPP_BINARY:-bench}"

if [ ! -x "$BINARY" ]; then
  echo "no binary at $BINARY — run: python scripts/build_csrc.py" >&2
  exit 1
fi
if [ ! -d "$TRT_DIR/lib" ]; then
  echo "TensorRT not found at $TRT_DIR — set SHIPINFER_TENSORRT_DIR." >&2
  exit 1
fi

mount_libs=()
path_libs=""
if [ -d "$LIBS" ]; then
  mount_libs=(-v "$LIBS:/baseline-libs:ro")
  path_libs=":/baseline-libs"
fi

exec docker run --rm --pid=host --device nvidia.com/gpu=all \
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/tensorrt/lib:/usr/local/cuda-12.6/lib64${path_libs}" \
  -e SHIPINFER_IN_CONTAINER=1 \
  -v "$REPO:/work" \
  -v "$TRT_DIR:/tensorrt:ro" \
  "${mount_libs[@]}" \
  -w /work "$IMAGE" \
  "/work/csrc/build/${SHIPINFER_CPP_BINARY:-bench}" "$@"
