#!/usr/bin/env bash
# Run the test suite INSIDE a container, on real GPUs. The only sanctioned way to run it.
#
#   deploy/rootless/test.sh                    # the offline tier
#   deploy/rootless/test.sh -m gpu             # the GPU tier
#   deploy/rootless/test.sh -m multigpu
#   SHIPINFER_TEST_GPUS=3,4 deploy/rootless/test.sh -m multigpu   # which physical GPUs a
#                                              # multi-process test may take (default 0,1)
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
# TensorRT is mounted from the host. No image here carries it, and without it the GPU tier
# silently reduces to "the tests that do not need an engine" — which is most of the value
# missing, since the accelerator seam this tier exists to cover is the TensorRT path. The
# host is jammy like the image, so its libraries load; same arrangement as bench.sh.
TRT_DIR="${SHIPINFER_TENSORRT_DIR:-/usr/local/TensorRT}"
# The system tier (`-m gpu tests/system`) runs the real chain on real footage, and footage is
# not in the repository: `references/` is gitignored, and committing frames of real people to
# make a test runnable is the wrong trade. So the operator points at a video or a frame
# directory and it is mounted read-only; unset, the tier skips itself and says what is missing.
video_mount=()
if [ -n "${SHIPINFER_SYSTEM_VIDEO:-}" ]; then
  if [ ! -e "$SHIPINFER_SYSTEM_VIDEO" ]; then
    echo "SHIPINFER_SYSTEM_VIDEO=$SHIPINFER_SYSTEM_VIDEO does not exist" >&2
    exit 1
  fi
  video_mount=(-v "$(cd "$(dirname "$SHIPINFER_SYSTEM_VIDEO")" && pwd)/$(basename "$SHIPINFER_SYSTEM_VIDEO"):/footage:ro")
fi

trt_mount=()
trt_path=""
if [ -d "$TRT_DIR/lib" ]; then
  trt_mount=(-v "$TRT_DIR:/tensorrt:ro")
  trt_path=":/tensorrt/lib"
fi

# The tracking plane is the one part of the system whose correctness argument is about
# *threads*, and its tests were skipping in every tier that runs: the image does not install
# the submodule and CI deliberately does not check it out (ADR-001). Putting it on PYTHONPATH
# — pure Python, no build, no install — is enough for `shipvision.tracking`, so those tests
# run here even though the compiled `_C` is absent.
#
# Read-only and additive on purpose: the submodule not being checked out is still a supported
# state, and then this is an empty string and nothing changes. ADR-001's promise is that a
# plain `pytest` needs no GPU, not that it needs no submodule.
shipvision_path=""
if [ -f "$REPO/3rdparty/shipvision/pyproject.toml" ]; then
  shipvision_path=":/work/3rdparty/shipvision"
fi

exec docker run --rm --pid=host --device nvidia.com/gpu=all \
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${trt_path}" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="/work/src${shipvision_path}" \
  -e SHIPINFER_TEST_GPUS \
  -e SHIPINFER_SYSTEM_VIDEO="${SHIPINFER_SYSTEM_VIDEO:+/footage}" \
  -v "$REPO:/work:ro" \
  -v "$WHEELS:/wheels:ro" \
  "${trt_mount[@]}" "${video_mount[@]}" \
  -w /work "$IMAGE" \
  bash -c '
    set -e
    # Two groups, and the split matters. The single list this replaced fell back to a
    # minimal set when ANY package in it was unavailable, so one missing wheel silently
    # dropped fastapi, opencv and scipy as well -- and the tests needing them skipped, which
    # looks identical to passing. Required fails loudly; optional is installed one at a time
    # so a gap costs exactly that package and SAYS SO.
    pip install -q --root-user-action=ignore --no-index --find-links=/wheels \
      pydantic pydantic-settings typer pyyaml pytest pytest-timeout pytest-asyncio
    # grpcio/protobuf are here because without them 150 tests (tests/launch, the shard
    # service, test_priority) never COLLECT in the container, so the container tier was not a
    # superset of the host tier and nobody could see the difference.
    # grpcio-tools is listed even though no wheel is staged today: tests/launch/
    # test_proto_is_current.py importorskips `grpc_tools`, so its 7 tests are the last of the
    # host tier the container cannot run, and listing it means they start running the moment
    # the wheel is staged rather than waiting for someone to notice the gap again.
    for package in fastapi httpx starlette uvicorn anyio opencv-python-headless scipy \
                   grpcio protobuf grpcio-tools; do
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels "$package" \
        >/dev/null 2>&1 || echo "NOTE: $package is not in /wheels; tests needing it will skip" >&2
    done
    python -c "import tensorrt" 2>/dev/null || \
      pip install -q --root-user-action=ignore --no-index --find-links=/wheels tensorrt \
        >/dev/null 2>&1 || true
    exec python -m pytest -ra --strict-markers -p no:cacheprovider "$@"
  ' bash "${args[@]}"
