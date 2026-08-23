#!/usr/bin/env bash
# Bring up rootless Docker with working GPUs, on a host where we have no root.
#
# The project requires everything to run in a container, and the account here is not in the
# `docker` group and has no passwordless sudo. This script is the path that needs neither.
# Every step below was arrived at by hitting the failure it works around, and each one says
# which failure — so if a future host behaves differently, the reason a step exists is
# recoverable rather than folklore.
#
# WHAT THIS BUYS AND WHAT IT DOES NOT
#
#   works : running containers, with all GPUs visible, bind-mounting the repository
#   fails : `docker build` — see the KERNEL LIMIT note at the bottom
#
# Idempotent. Run it again after a reboot (or enable linger, see below).
set -euo pipefail

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------------------
log "Prerequisites"
# ---------------------------------------------------------------------------------------
missing=0
for tool in dockerd-rootless-setuptool.sh newuidmap nvidia-ctk nvidia-container-cli; do
  if command -v "$tool" >/dev/null; then printf '  ok   %s\n' "$tool"
  else printf '  MISSING %s\n' "$tool"; missing=1; fi
done
for f in /etc/subuid /etc/subgid; do
  if grep -q "^${USER}:" "$f"; then printf '  ok   %s has an entry for %s\n' "$f" "$USER"
  else printf '  MISSING an entry for %s in %s (needs root)\n' "$USER" "$f"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo; echo "Cannot proceed without the above."; exit 1; }

# ---------------------------------------------------------------------------------------
log "nvidia-container-cli, configured for rootless"
# ---------------------------------------------------------------------------------------
# Read in preference to /etc/nvidia-container-runtime/config.toml, so the system file is
# untouched. `no-cgroups` is required: rootless cannot place devices into a cgroup, and
# without it the hook fails with "cgroup subsystem devices not found".
mkdir -p "$HOME/.config/nvidia-container-runtime"
cat > "$HOME/.config/nvidia-container-runtime/config.toml" <<'TOML'
[nvidia-container-cli]
no-cgroups = true
ldconfig = "/sbin/ldconfig.real"
TOML
printf '  wrote %s\n' "$HOME/.config/nvidia-container-runtime/config.toml"

# ---------------------------------------------------------------------------------------
log "The rootless daemon"
# ---------------------------------------------------------------------------------------
# --skip-iptables because the `iptables` binary is absent and installing it needs root. The
# cost is no NAT port publishing; bind-mounts and --network=host-style access still work, and
# nothing here needs a published port.
if [ ! -S "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock" ]; then
  dockerd-rootless-setuptool.sh install --force --skip-iptables
else
  printf '  already installed\n'
fi
systemctl --user enable --now docker.service >/dev/null 2>&1 || true
export DOCKER_HOST="unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock"
printf '  DOCKER_HOST=%s\n' "$DOCKER_HOST"

# ---------------------------------------------------------------------------------------
log "GPUs, via CDI rather than the legacy hook"
# ---------------------------------------------------------------------------------------
# The legacy `--gpus all` path runs nvidia-container-cli, which chroots into the container to
# refresh the loader cache and therefore mounts /proc — refused here (see KERNEL LIMIT). CDI
# describes the devices declaratively instead, and Docker applies it without that hook.
mkdir -p "$HOME/.cdi" "$HOME/.config/docker"
nvidia-ctk cdi generate --output="$HOME/.cdi/nvidia.yaml" >/dev/null 2>&1
printf '  generated %s\n' "$HOME/.cdi/nvidia.yaml"

# nvidia-ctk still emits one createContainer hook, `update-ldcache`, which mounts /proc for
# the same reason. Strip it; the driver libraries are bind-mounted at their host paths, so
# the loader finds them from LD_LIBRARY_PATH without a cache refresh.
python3 - "$HOME/.cdi/nvidia.yaml" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
lines = path.read_text().splitlines()
out, i, removed = [], 0, 0
while i < len(lines):
    if lines[i].strip() == "- hookName: createContainer" and any(
        "update-ldcache" in lines[j] for j in range(i, min(i + 8, len(lines)))
    ):
        indent = len(lines[i]) - len(lines[i].lstrip())
        i += 1
        while i < len(lines):
            stripped, cur = lines[i].strip(), len(lines[i]) - len(lines[i].lstrip())
            if stripped.startswith("- ") and cur <= indent:
                break
            if stripped and cur <= indent - 2:
                break
            i += 1
        removed += 1
        continue
    out.append(lines[i]); i += 1
path.write_text("\n".join(out) + "\n")
print(f"  stripped {removed} update-ldcache hook(s)")
PY

cat > "$HOME/.config/docker/daemon.json" <<JSON
{
  "features": { "cdi": true },
  "cdi-spec-dirs": ["$HOME/.cdi"]
}
JSON
printf '  wrote %s\n' "$HOME/.config/docker/daemon.json"
systemctl --user restart docker.service
sleep 5

# ---------------------------------------------------------------------------------------
log "Verify"
# ---------------------------------------------------------------------------------------
docker version --format '  daemon {{.Server.Version}} (rootless)'
docker run --rm --pid=host --device nvidia.com/gpu=all \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  nvidia/cuda:12.6.3-base-ubuntu22.04 \
  nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  gpu /'

cat <<'NOTE'

==> Done. Put these in your shell profile:

    export DOCKER_HOST=unix://${XDG_RUNTIME_DIR}/docker.sock

    Every `docker run` needs two flags on this host:
        --pid=host                       (see KERNEL LIMIT)
        --device nvidia.com/gpu=all      (instead of --gpus all)

==> KERNEL LIMIT — why --pid=host, and why `docker build` does not work

This kernel refuses to mount /proc from an unprivileged user namespace. Reproduce it with
no containers involved at all:

    unshare --user --map-root-user --mount --pid --fork \
        sh -c 'mount -t proc proc /proc'
    mount: /proc: permission denied.

A container needs a fresh /proc for its own PID namespace, so `--pid=host` sidesteps it by
not creating one. That weakens isolation — the container sees host processes — which is
acceptable for a build/test container on a trusted machine and is NOT acceptable in
production.

`docker build` has no equivalent flag: buildkit runs each RUN step in its own namespace and
cannot be told to share the host's. So images cannot be built here; they must be pulled, or
built on a host where this restriction does not apply.

==> The clean fix, one command from an administrator

    sudo usermod -aG docker $USER && newgrp docker

Rootful Docker creates namespaces as root, so the /proc restriction does not apply: builds
work, `--gpus all` works, and `--pid=host` is no longer needed. That is the configuration
`deploy/docker/Dockerfile` was written for, and until it is available that Dockerfile stays
unexercised.

==> To undo everything this script did

    systemctl --user disable --now docker.service
    dockerd-rootless-setuptool.sh uninstall
    rm -rf ~/.cdi ~/.config/docker/daemon.json ~/.config/nvidia-container-runtime
NOTE
