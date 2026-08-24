#!/usr/bin/env bash
# Bake the GStreamer/PyGObject test image on top of the pytorch base, WITHOUT `docker build`.
#
# WHY THIS EXISTS
#
# `src/shipinfer/ingest/sources/gstreamer.py` had never executed a single pipeline, because
# `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` carries no GStreamer and no PyGObject —
# and neither does the host. So the RTSP path was strings in a docstring.
#
# The base image is Ubuntu 22.04 with conda Python 3.11 and, usefully, **no** libglib /
# libgobject of its own (only libffi), so apt's GLib is the only one in the process and
# PyGObject built against it does not fight a conda copy. That is why this layers onto the
# existing base instead of asking for a new one.
#
# TWO HOST QUIRKS THIS WORKS AROUND
#
#   docker build       impossible here: buildkit gives each RUN its own PID namespace and
#                      this kernel refuses to mount /proc from an unprivileged userns (see
#                      setup.sh, KERNEL LIMIT). `docker run` + `docker commit` is the same
#                      thing without a namespace per step.
#   --network=host     containers on the default bridge have no outbound network, because
#                      the rootless daemon was installed --skip-iptables and therefore has
#                      no NAT. The host network namespace does have connectivity (through
#                      rootlesskit's slirp4netns), so apt and pip work there and only there.
#                      Verified: bridge -> TCP fails, host -> TCP to 1.1.1.1:443 succeeds.
#
# WHAT LANDS IN THE IMAGE
#
#   gstreamer1.0-plugins-{base,good,bad,ugly} + libav   decode, incl. the `nvcodec` plugin
#                      (nvh264dec/nvh265dec/cudaconvert/cudadownload) which dlopens the
#                      driver's libnvcuvid.so.1 — bind-mounted by the CDI device, so NVDEC
#                      is real inside the container.
#   PyGObject 3.50     pinned: 3.52 dropped girepository-1.0 for girepository-2.0, which
#                      needs GLib 2.80. Jammy ships 2.72, so 3.52 does not build here.
#   gst-rtsp-server    the RTSP server `scripts/rtsp_serve.py` stands up, so the ingest
#                      tests talk to a real DESCRIBE/SETUP/PLAY session over a real socket
#                      instead of to a scripted double. ffmpeg's own `-f rtsp` muxer cannot
#                      do this: in 4.4 `rtsp_flags listen` is a DEMUXER option, so ffmpeg can
#                      only ANNOUNCE to a server, never serve a PLAY.
#   ffmpeg             encodes the real 1920x1080 JPEGs into the H.264 fixture the server
#                      loops, so the bitstream under test is x264's and not GStreamer's.
#
# `nvvideoconvert` and `memory:NVMM` are DeepStream elements and are NOT here; upstream
# nvcodec's device memory is `memory:CUDAMemory`. `ingest/sources/gstreamer.py` probes for
# both, which is why it runs on either install.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${SHIPINFER_BASE_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}"
IMAGE="${SHIPINFER_GST_IMAGE:-shipinfer-gst:jammy}"
WHEELS="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"
CONTAINER="shipinfer-gst-bake-$$"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

if docker image inspect "$IMAGE" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
  echo "$IMAGE already exists; FORCE=1 to rebake" >&2
  exit 0
fi

trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

# One line each: these are interpolated into a `bash -c` string, where a newline inside the
# variable would end the command rather than separate two arguments.
APT_PACKAGES="build-essential pkg-config libgirepository1.0-dev libcairo2-dev gir1.2-glib-2.0 \
gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-0 gstreamer1.0-rtsp \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev ffmpeg"

PIP_PACKAGES="pydantic pydantic-settings typer pyyaml pytest pytest-timeout pytest-asyncio \
fastapi httpx starlette uvicorn anyio opencv-python-headless scipy"

docker run --name "$CONTAINER" --pid=host --network=host \
  -v "$WHEELS:/wheels:ro" \
  "$BASE" bash -euxc "
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends $APT_PACKAGES
    rm -rf /var/lib/apt/lists/*
    pip install --root-user-action=ignore 'pycairo' 'PyGObject==3.50.0'
    pip install --root-user-action=ignore --no-index --find-links=/wheels $PIP_PACKAGES
    # A registry built here would be a LIE. `plugin_init` for nvcodec probes CUDA, and this
    # bake container has no GPU, so the plugin registers zero features and GStreamer caches
    # that — after which `nvh264dec` is \"not installed\" in every container from this image,
    # even with eight A5000s attached. Cached feature lists are invalidated by the plugin
    # file's mtime and size, never by the machine changing. So the cache is removed and each
    # container rebuilds it (~11 s, once) against the devices it actually has.
    rm -rf /root/.cache/gstreamer-1.0
    python -c \"
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
print('GStreamer', '.'.join(map(str, Gst.version()[:3])))
for element in ('rtspsrc', 'rtph264depay', 'h264parse', 'avdec_h264',
                'nvh264dec', 'nvv4l2decoder', 'cudaconvert', 'cudadownload',
                'nvvideoconvert', 'videoconvert', 'appsink'):
    print(f'  {element:16s} {Gst.ElementFactory.find(element) is not None}')
\"
    rm -rf /root/.cache/gstreamer-1.0
  "

docker commit \
  --change 'ENV GST_DEBUG=1' \
  --change 'ENV PYTHONDONTWRITEBYTECODE=1' \
  "$CONTAINER" "$IMAGE"
echo "==> committed $IMAGE"
