#!/usr/bin/env bash
# Run the test suite INSIDE a container, on real GPUs. The only sanctioned way to run it.
#
#   deploy/rootless/test.sh                    # the offline tier
#   deploy/rootless/test.sh -m gpu             # the GPU tier
#   deploy/rootless/test.sh -m multigpu
#   deploy/rootless/test.sh tests/ingest -q    # any pytest arguments
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
#   --device nvidia.com/gpu=all   CDI instead of `--gpus all`, because the legacy hook
#                                 chroots in to refresh the loader cache and hits the same
#                                 restriction.
#
# Dependencies come from a mounted wheel directory rather than PyPI: `--skip-iptables` (the
# `iptables` binary is absent and installing it needs root) leaves containers with no
# outbound network. `deploy/rootless/wheels.sh` populates it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SHIPINFER_TEST_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon unreachable at $DOCKER_HOST — run deploy/rootless/setup.sh first" >&2
  exit 1
fi
if [ ! -d "$WHEELS" ] || [ -z "$(ls -A "$WHEELS" 2>/dev/null)" ]; then
  echo "no wheels at $WHEELS — run deploy/rootless/wheels.sh first" >&2
  exit 1
fi

# Default to the offline tier, matching scripts/run_tests.sh, unless the caller passes -m.
args=("$@")
if ! printf '%s\n' "${args[@]}" | grep -q '^-m$'; then
  args=(-m "not gpu and not multigpu" "${args[@]}")
fi

# The repository is mounted READ-ONLY on purpose: a test that writes into the source tree is
# a test that pollutes the next run, and `tmp_path` exists precisely so it does not have to.
exec docker run --rm --pid=host --device nvidia.com/gpu=all \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  -e PYTHONPATH=/work/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/work:ro" \
  -v "$WHEELS:/wheels:ro" \
  -w /work "$IMAGE" \
  bash -c '
    set -e
    pip install -q --root-user-action=ignore --no-index --find-links=/wheels \
      pydantic pydantic-settings typer pyyaml pytest pytest-timeout pytest-asyncio \
      fastapi httpx starlette uvicorn anyio opencv-python-headless scipy >/dev/null 2>&1 || \
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels \
        pydantic pydantic-settings typer pyyaml pytest pytest-timeout pytest-asyncio
    exec python -m pytest -ra --strict-markers -p no:cacheprovider "$@"
  ' bash "${args[@]}"
