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
# The container is `test.sh`'s, through the same `_container.sh`: same image, same wheels, the
# same `/tensorrt` mount every runner here uses, the same `_inside.sh` preamble. Two
# deliberate differences, both below: the repo is mounted **rw** (`SHIPINFER_RUN_MOUNT=ro` to
# opt out), because the jobs this exists for produce artefacts; and `-it` only when there is a
# terminal.
set -euo pipefail

# WRITABLE, and that is the one deliberate difference from `test.sh`. The jobs this door
# exists for produce artefacts: `build_engines.py` writes `model_repository/<m>/1/model.plan`,
# and a read-only mount would make the documented command fail in a way that reads as a bug in
# the script. `SHIPINFER_RUN_MOUNT=ro` for a job that should not write.
#
# NOTE FOR A WORKTREE. `models/*` and `model_repository/*/*/*.plan` are both gitignored, so a
# `git worktree` has neither the ONNX to build from nor a plan already built --
# `build_engines.py --check` there reports every ONNX MISSING and is telling the truth about
# that tree. Engine work runs in the primary checkout.
CONTAINER_MOUNT="${SHIPINFER_RUN_MOUNT:-rw}"

# The container every runner here starts, defined once -- image, wheels, socket, GPU knob,
# TensorRT at `/tensorrt`, PYTHONPATH.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_container.sh"

# Forwarded so `run.sh python -m pytest -m gpu tests/system` reaches the same footage
# `test.sh` would mount, instead of skipping the tier for a reason the caller cannot see.
video_mount=()
if [ -n "${SHIPINFER_SYSTEM_VIDEO:-}" ]; then
  if [ ! -e "$SHIPINFER_SYSTEM_VIDEO" ]; then
    echo "SHIPINFER_SYSTEM_VIDEO=$SHIPINFER_SYSTEM_VIDEO does not exist" >&2
    exit 1
  fi
  video_mount=(-v "$(cd "$(dirname "$SHIPINFER_SYSTEM_VIDEO")" && pwd)/$(basename "$SHIPINFER_SYSTEM_VIDEO"):/footage:ro")
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

exec docker run --rm "${tty[@]}" "${DOCKER_ARGV[@]}" \
  -e SHIPINFER_SYSTEM_VIDEO="${SHIPINFER_SYSTEM_VIDEO:+/footage}" \
  "${video_mount[@]}" \
  -w /work "$IMAGE" \
  bash -c '
    set -e
    . /work/deploy/rootless/_inside.sh
    exec "$@"
  ' bash "$@"
