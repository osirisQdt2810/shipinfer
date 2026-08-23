#!/usr/bin/env bash
# Populate the wheel directory the containerised test runner installs from.
#
# Containers here have no outbound network — rootless Docker was installed with
# --skip-iptables because the `iptables` binary is absent and installing it needs root, so
# slirp4netns cannot NAT. Dependencies therefore arrive as a mounted directory of wheels.
#
# Downloaded for the CONTAINER's interpreter, not this host's: the image ships Python 3.11
# and the host venv is 3.10, so a wheel built for the host would be silently wrong for any
# package with a compiled component — pydantic-core and numpy among them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${SHIPINFER_WHEELS:-/tmp/wheels-py311}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
ABI="${SHIPINFER_CONTAINER_ABI:-cp311}"
VERSION="${SHIPINFER_CONTAINER_PY:-3.11}"

mkdir -p "$DEST"
echo "==> downloading ${ABI} wheels into $DEST"

# torch is deliberately absent: the image already carries a CUDA build of it, and a second
# copy would be several gigabytes and might not match the image's CUDA.
"$PY" -m pip download --dest "$DEST" --only-binary=:all: \
  --python-version "$VERSION" --implementation cp --abi "$ABI" \
  --platform manylinux2014_x86_64 \
  pydantic pydantic-settings typer pyyaml numpy scipy \
  pytest pytest-timeout pytest-asyncio \
  fastapi httpx starlette uvicorn anyio \
  opencv-python-headless

printf '==> %s wheels\n' "$(ls -1 "$DEST"/*.whl | wc -l)"
