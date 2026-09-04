#!/usr/bin/env bash
# Run the test suite INSIDE a container, on real GPUs. The only sanctioned way to run it.
#
#   deploy/rootless/test.sh                    # the offline tier
#   deploy/rootless/test.sh -m gpu             # the GPU tier
#   deploy/rootless/test.sh -m multigpu
#   SHIPINFER_TEST_GPUS=0,1 deploy/rootless/test.sh -m multigpu   # which of the CONTAINER's
#                                              # devices a test may take (default 0,1). Not
#                                              # physical ordinals: SHIPINFER_GPUS decides
#                                              # which cards are here, and they are renumbered
#                                              # from 0 inside.
#   deploy/rootless/test.sh tests/ingest -q    # any pytest arguments
#   SHIPINFER_SYSTEM_VIDEO=references/.../clip.mp4 deploy/rootless/test.sh -m gpu tests/system
#                                              # the real chain, decode to output, on real
#                                              # footage; without it that tier skips itself
#
# WHY A CONTAINER IS NOT OPTIONAL HERE
#
# A host virtualenv accumulates whatever happens to be installed, and a test then passes for
# a reason nobody chose. That is not hypothetical: `tests/ingest/test_sources_replay.py` had
# two tests asserting the error message for a missing path and an empty directory, and both
# passed on the host only because OpenCV happened to be importable. In a clean container they
# failed, revealing that the source imported OpenCV *before* validating the path — so a
# typo'd URI reported "install OpenCV", which is advice that does not fix the problem. The
# container found that; the host had hidden it for as long as the code existed.
#
# The host also cannot run the production path at all: `nvcc` here is 11.5 against a 12.6
# driver, so `sm_89` is unbuildable, and TensorRT is not installed. See container.md.
#
# ROOTLESS, AND WHY THE FLAGS LOOK ODD
#
# This account is not in the `docker` group and has no passwordless sudo, so the daemon is
# rootless — see deploy/rootless/setup.sh, which must be run once. Two consequences:
#
#   --pid=host                    this kernel refuses to mount /proc from an unprivileged
#                                 user namespace, so the container cannot have its own PID
#                                 namespace. Reproduce with:
#                                   unshare --user --map-root-user --mount --pid --fork \
#                                       sh -c 'mount -t proc proc /proc'
#   "${GPU_DEVICES[@]}"          CDI instead of `--gpus all`, because the legacy hook
#                                 (`SHIPINFER_GPUS`, `_gpus.sh` -- one sick card must not
#                                 chroots in to refresh the loader cache and hits the same
#                                 restriction.
#
# Dependencies come from a mounted wheel directory rather than PyPI: `--skip-iptables` (the
# `iptables` binary is absent and installing it needs root) leaves containers with no
# outbound network. `deploy/rootless/wheels.sh` populates it.
set -euo pipefail

# The container every runner here starts, defined once -- image, wheels, socket, GPU knob,
# TensorRT mount, PYTHONPATH, and a READ-ONLY repo. A test that writes into the source tree
# pollutes the next run, and `tmp_path` exists precisely so it does not have to.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_container.sh"

# The one mount only this runner needs. The system tier (`-m gpu tests/system`) runs the real
# chain on real footage, and footage is not in the repository: `references/` is gitignored, and
# committing frames of real people to make a test runnable is the wrong trade. So the operator
# points at a video or a frame directory and it is mounted read-only; unset, the tier skips
# itself and says what is missing.
video_mount=()
if [ -n "${SHIPINFER_SYSTEM_VIDEO:-}" ]; then
  if [ ! -e "$SHIPINFER_SYSTEM_VIDEO" ]; then
    echo "SHIPINFER_SYSTEM_VIDEO=$SHIPINFER_SYSTEM_VIDEO does not exist" >&2
    exit 1
  fi
  video_mount=(-v "$(cd "$(dirname "$SHIPINFER_SYSTEM_VIDEO")" && pwd)/$(basename "$SHIPINFER_SYSTEM_VIDEO"):/footage:ro")
fi

args=("$@")

exec docker run --rm "${DOCKER_ARGV[@]}" \
  -e SHIPINFER_SYSTEM_VIDEO="${SHIPINFER_SYSTEM_VIDEO:+/footage}" \
  "${video_mount[@]}" \
  -w /work "$IMAGE" \
  bash -c '
    set -e
    . /work/deploy/rootless/_inside.sh
    exec python -m pytest -ra --strict-markers -p no:cacheprovider "$@"
  ' bash "${args[@]}"
