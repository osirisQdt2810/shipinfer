#!/usr/bin/env bash
# One command, inside the tier's container. The door that did not exist.
#
#   deploy/rootless/run.sh python scripts/build_engines.py --check
#   deploy/rootless/run.sh                                   # an interactive shell
#   SHIPINFER_GPUS=0,1,2,3 deploy/rootless/run.sh nvidia-smi -L
#
# WHY THIS EXISTS
#
# CLAUDE.md and `container.md` both tell a reader `make shell`, and there is no Makefile
# anywhere in this repository -- the compose setup those lines describe was replaced by the
# scripts here, because this kernel refuses `docker build` from an unprivileged user namespace
# (see `setup.sh`, KERNEL LIMIT). So the container rule had no door for anything that is not
# `pytest` or a benchmark: an engine build, `shipinfer repo ls`, one `python -c`, a shell to
# look around in. The documented command failed and the honest ones did not exist, which is
# how somebody ends up reaching for `SHIPINFER_ALLOW_HOST_RUN=1`.
#
# The container is `test.sh`'s, exactly: same image, same read-only repo, same wheels, same
# TensorRT mount, same `_inside.sh` preamble. The two differ in one line -- what they `exec`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Which GPUs this container may see, and why one degraded card is not a dead tier.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gpus.sh"

IMAGE="${SHIPINFER_TEST_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"

# The ROOTLESS socket, exported here and not left to a shell profile. `setup.sh` tells the
# operator to put this in their profile, and a profile is read by an interactive login shell
# and by NOTHING else -- so without this line the door works when a human types it and dies in
# every script, cron entry, CI job and agent with docker's generic "Cannot connect to the
# Docker daemon at unix:///var/run/docker.sock" (the rootful daemon, which this account cannot
# reach; `setup.sh` exists because of that). Every sibling script here exports it; this one
# forgot, and it is the exact failure the `-it` comment below argues about.
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon unreachable at $DOCKER_HOST — run deploy/rootless/setup.sh first" >&2
  exit 1
fi
if [ ! -d "$WHEELS" ] || [ -z "$(ls -A "$WHEELS" 2>/dev/null)" ]; then
  echo "no wheels at $WHEELS — run deploy/rootless/wheels.sh first" >&2
  exit 1
fi

# WRITABLE, and that is the one deliberate difference from `test.sh`. A test that writes into
# the source tree pollutes the next run and `tmp_path` exists so it need not -- but the whole
# point of this door is the jobs that DO produce artefacts: `build_engines.py` writes
# `model_repository/<model>/1/model.plan`, and a read-only mount would make the documented
# command fail in a way that reads as a bug in the script.
#
# NOTE FOR A WORKTREE. `models/*` and `model_repository/*/*/*.plan` are both gitignored, so a
# `git worktree` has neither the ONNX to build from nor anywhere the build has already been
# done -- `build_engines.py --check` there reports every ONNX MISSING and it is telling the
# truth about that tree. Engine work runs in the primary checkout.
MOUNT="${SHIPINFER_RUN_MOUNT:-rw}"
case "$MOUNT" in
  rw|ro) ;;
  *) echo "SHIPINFER_RUN_MOUNT is 'rw' (default) or 'ro'; got '$MOUNT'" >&2; exit 2 ;;
esac

# The host's TensorRT, mounted rather than installed, exactly as `test.sh` does it: the image
# has torch and CUDA but no TensorRT, and the accelerator seam this tier covers IS that path.
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
trt_mount=()
trt_path=""
if [ -d "$TRT_DIR/lib" ]; then
  trt_mount=(-v "$TRT_DIR:/opt/tensorrt:ro")
  trt_path=":/opt/tensorrt/lib"
fi

shipvision_path=""
if [ -f "$REPO/3rdparty/shipvision/pyproject.toml" ]; then
  shipvision_path=":/work/3rdparty/shipvision"
fi

# `bash -l` and not `bash`, so an interactive shell gets a prompt and history rather than a
# bare `sh`-like one that looks broken.
if [ "$#" -eq 0 ]; then
  set -- bash -l
fi

# `-it` only when there IS a terminal. Unconditionally, docker refuses with "cannot attach
# stdin to a TTY-enabled container because stdin is not a terminal" -- so the door would work
# by hand and fail in every script, CI job and agent that used it, which is the half that
# matters most for a command whose whole purpose is to be reachable.
tty=()
if [ -t 0 ] && [ -t 1 ]; then
  tty=(-it)
fi

exec docker run --rm "${tty[@]}" --pid=host "${GPU_DEVICES[@]}" \
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${trt_path}" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="/work/src${shipvision_path}" \
  -e SHIPINFER_TEST_GPUS \
  -v "$REPO:/work:$MOUNT" \
  -v "$WHEELS:/wheels:ro" \
  "${trt_mount[@]}" \
  -w /work "$IMAGE" \
  bash -c '
    set -e
    . /work/deploy/rootless/_inside.sh
    exec "$@"
  ' bash "$@"
