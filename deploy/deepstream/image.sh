#!/usr/bin/env bash
# Bake the DeepStream image this project's `deepstream` topology runs in, WITHOUT `docker build`.
#
# WHY THIS EXISTS
#
# `src/shipinfer/pipeline/deepstream/` generates nvinfer configs, walks NvDsBatchMeta and
# publishes PerceptionEvents — all of it verified offline, none of it ever run, because there
# is no DeepStream image on this box and `pyds` is not on PyPI. This is the recipe that closes
# that gap, in the same shape as `deploy/rootless/gst-image.sh` and for the same reasons.
#
# TWO HOST QUIRKS THIS WORKS AROUND (identical to gst-image.sh — read its header for detail)
#
#   docker build       impossible here: buildkit gives each RUN its own PID namespace and this
#                      kernel refuses to mount /proc from an unprivileged userns. `docker run`
#                      + `docker commit` is the same thing without a namespace per step.
#   --network=host     containers on the default bridge have no outbound network, because the
#                      rootless daemon was installed --skip-iptables and therefore has no NAT.
#
# WHAT LANDS IN THE IMAGE
#
#   nvcr.io/nvidia/deepstream (base)   nvurisrcbin / nvstreammux / nvinfer / nvtracker and
#                      libnvds_nvmultiobjecttracker.so. The `-triton-multiarch` flavour is the
#                      large one and is what `nvinferserver` would need; `-samples` is enough
#                      for `nvinfer`, which is what this topology uses (see
#                      docs/design/topology-deepstream.md §1).
#   pyds               the Python bindings, from the SDK's own bindings wheel. NOT on PyPI:
#                      /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps holds
#                      the prebuilt wheel on recent images, and building it from source needs
#                      the SDK headers, which only exist inside this container.
#   shipinfer          installed from the repository, so the child a `Fleet` spawns
#                      (`python -m shipinfer deepstream`) exists inside the image.
#
# WHAT IS STILL MISSING AFTERWARDS, AND CANNOT BE FIXED HERE
#
#   the bbox parser    the shipped detector is end-to-end (yolo26: one decoded 300x6 tensor),
#                      and nvinfer's built-in parsers cannot read it. A ~60-line C++ `.so`
#                      against nvdsinfer_custom_impl.h is required, compiled INSIDE this image
#                      because that is where the SDK headers are. Config generation refuses
#                      until `topology.deepstream.bbox_parser` names it, so this cannot be
#                      forgotten into a run that silently reports zero detections.
#   the engines        `model.plan` is valid only for the TensorRT version that built it, and
#                      this image's is unlikely to be the host's. Declare `onnx_file` in the
#                      model's `parameters` and nvinfer rebuilds on first start.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${SHIPINFER_DS_BASE_IMAGE:-nvcr.io/nvidia/deepstream:7.1-samples-multiarch}"
IMAGE="${SHIPINFER_DS_IMAGE:-shipinfer-deepstream:7.1}"
CONTAINER="shipinfer-ds-bake-$$"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

if docker image inspect "$IMAGE" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
  echo "$IMAGE already exists; FORCE=1 to rebake" >&2
  exit 0
fi

trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

# One line each: interpolated into a `bash -c` string, where a newline inside the variable
# would end the command rather than separate two arguments.
APT_PACKAGES="python3-pip python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 \
gir1.2-gst-plugins-base-1.0 libcairo2-dev libgirepository1.0-dev"

docker run --name "$CONTAINER" --pid=host --network=host \
  -v "$REPO:/workspace:ro" \
  "$BASE" bash -euxc "
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends $APT_PACKAGES
    rm -rf /var/lib/apt/lists/*

    # The bindings ship as a wheel inside the SDK on 6.3+. Older images carry the source and
    # need cmake + the SDK headers; if the wheel is absent this fails loudly rather than
    # committing an image whose shard cannot start.
    wheel=\$(find /opt/nvidia/deepstream -name 'pyds-*.whl' | head -1)
    test -n \"\$wheel\" || { echo 'no pyds wheel in this image; build it from '
                            'sources/deepstream_python_apps/bindings' >&2; exit 1; }
    pip install --root-user-action=ignore \"\$wheel\"

    pip install --root-user-action=ignore /workspace
    # A registry built here would be a LIE: nvinfer's plugin_init probes CUDA, and this bake
    # container has no GPU, so the plugin registers zero features and GStreamer caches that —
    # after which 'nvinfer is not installed' in every container from this image. Same trap as
    # gst-image.sh; same fix.
    rm -rf /root/.cache/gstreamer-1.0
    python3 -c \"
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
import pyds
print('DeepStream bindings', pyds.__file__)
for element in ('nvurisrcbin', 'nvstreammux', 'nvinfer', 'nvtracker', 'fakesink'):
    print(f'  {element:16s} {Gst.ElementFactory.find(element) is not None}')
\"
    rm -rf /root/.cache/gstreamer-1.0
  "

docker commit \
  --change 'ENV GST_DEBUG=1' \
  --change 'ENV PYTHONDONTWRITEBYTECODE=1' \
  --change 'ENV USE_NEW_NVSTREAMMUX=no' \
  "$CONTAINER" "$IMAGE"
echo "==> committed $IMAGE"
echo
echo "Next, inside it (one shard per GPU; --dry-run first needs no device):"
echo "  docker run --rm --network=host --gpus all -v $REPO:/workspace $IMAGE \\"
echo "    python3 -m shipinfer deepstream -r /workspace/model_repository --dry-run"
