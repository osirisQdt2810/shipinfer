# The container every runner here starts, defined once.
#
# `_inside.sh` was factored out with the argument that two copies of a wheel list is how a
# container ends up without grpcio while the run reads as green. That argument applies one
# level up, and it was proved the hard way: `run.sh`'s first version copied this block from
# `test.sh` and had already drifted at review time -- TensorRT mounted at `/opt/tensorrt`
# where all four siblings use `/tensorrt`, so a `trtexec` line copied out of `cpp.sh` failed
# in the one door built to run it.
#
# Leaves `DOCKER_ARGV` holding everything before the image, and `IMAGE` set. A caller adds its
# own `-e`/`-v` and its `exec`:
#
#   source .../_container.sh
#   exec docker run --rm "${DOCKER_ARGV[@]}" -e MINE=1 -w /work "$IMAGE" bash -c '...'
#
# Inputs, all optional: `SHIPINFER_TEST_IMAGE`, `SHIPINFER_WHEELS`, `SHIPINFER_TENSORRT_DIR`,
# `SHIPINFER_GPUS` (via `_gpus.sh`), and `CONTAINER_MOUNT` (`ro` default) set by the caller
# BEFORE sourcing.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$_here/../.." && pwd)"

# Which GPUs this container may see, and why one degraded card is not a dead tier.
source "$_here/_gpus.sh"

IMAGE="${SHIPINFER_TEST_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"

# The ROOTLESS socket, exported here and never left to a shell profile. `setup.sh` tells the
# operator to put it in one, and a profile is read by an interactive login shell and by
# NOTHING else -- so a runner that omits this works when a human types it and dies in every
# script, cron entry, CI job and agent, with docker's generic message about
# `/var/run/docker.sock` (the rootful daemon this account cannot reach).
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon unreachable at $DOCKER_HOST — run deploy/rootless/setup.sh first" >&2
  exit 1
fi
if [ ! -d "$WHEELS" ] || [ -z "$(ls -A "$WHEELS" 2>/dev/null)" ]; then
  echo "no wheels at $WHEELS — run deploy/rootless/wheels.sh first" >&2
  exit 1
fi

# The host's TensorRT, mounted rather than installed: the image has torch and CUDA but no
# TensorRT, and the accelerator seam this tier exists to cover IS that path. `/tensorrt` is the
# path every runner here uses, so a `trtexec` or `-DTENSORRT_ROOT=` line copied from one script
# works in another.
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
_trt_mount=()
TRT_LD_PATH=""
if [ -d "$TRT_DIR/lib" ]; then
  _trt_mount=(-v "$TRT_DIR:/tensorrt:ro")
  TRT_LD_PATH=":/tensorrt/lib"
fi

# The submodule on PYTHONPATH when it is checked out, and an empty string when it is not:
# ADR-001's promise is that a plain `pytest` needs no GPU, not that it needs no submodule.
SHIPVISION_PATH=""
if [ -f "$REPO/3rdparty/shipvision/pyproject.toml" ]; then
  SHIPVISION_PATH=":/work/3rdparty/shipvision"
fi

# READ-ONLY unless the caller says otherwise: a test that writes into the source tree pollutes
# the next run, and `tmp_path` exists so it need not. `run.sh` sets `rw` because the jobs it
# exists for produce artefacts -- `build_engines.py` writes `model_repository/<m>/1/model.plan`.
CONTAINER_MOUNT="${CONTAINER_MOUNT:-ro}"
case "$CONTAINER_MOUNT" in
  rw|ro) ;;
  *) echo "CONTAINER_MOUNT is 'rw' or 'ro'; got '$CONTAINER_MOUNT'" >&2; exit 2 ;;
esac

DOCKER_ARGV=(
  --pid=host
  "${GPU_DEVICES[@]}"
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${TRT_LD_PATH}"
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH="/work/src${SHIPVISION_PATH}"
  -e SHIPINFER_TEST_GPUS
  -v "$REPO:/work:$CONTAINER_MOUNT"
  -v "$WHEELS:/wheels:ro"
  "${_trt_mount[@]}"
)
