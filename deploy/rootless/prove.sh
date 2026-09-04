#!/usr/bin/env bash
# Prove, to someone who does not trust the claim, that work really ran in a container on GPUs.
#
# The problem with "it ran in a container on the GPU" is that it is unfalsifiable from the
# outside. This script makes it falsifiable two ways at once:
#
#   1. **From inside**, it records what only a process inside the container can see —
#      /.dockerenv, its cgroup, its hostname, its PID namespace, the devices CUDA reports —
#      into a file on a bind mount, so the evidence lands outside where it can be read.
#
#   2. **From outside**, it draws a VRAM signature that an INDEPENDENT recorder catches. The
#      operator's ~/workspaces/tools/vram_log.sh samples `nvidia-smi memory.used` every 0.5 s
#      into a CSV this script never touches. So the script allocates a deliberately odd,
#      announced-in-advance staircase across the four target GPUs and holds it. If the CSV
#      shows that staircase at the stated wall-clock time, the allocation happened; if it does
#      not, the claim is false. The witness is the operator's tool, not mine.
#
# The staircase is uneven on purpose. A flat allocation could be anything; 1.5/3/4.5/6 GiB
# across gpu2..gpu5 in that order is a shape nothing else on a shared box produces by accident.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Which GPUs this container may see, and why one degraded card is not a dead tier.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gpus.sh"
OUT="${1:-$REPO/.artifacts/attestation}"
GPUS="${SHIPINFER_BENCH_GPUS:-2,3,4,5}"
HOLD_S="${HOLD_S:-12}"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"

mkdir -p "$OUT"
STAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT="$OUT/attestation_$STAMP.txt"

echo "==> Attestation will be written to $REPORT"
echo "==> Cross-check it against your own VRAM log; the signature is announced below."
printf '\n'

# Announced BEFORE the run, so it cannot be fitted to the result afterwards.
cat <<ANNOUNCE | tee "$REPORT"
ShipInfer container/GPU attestation
===================================
announced at (host clock) : $(date '+%Y-%m-%d %H:%M:%S %Z')
target GPUs               : $GPUS
expected VRAM signature   : an uneven staircase, held for ${HOLD_S}s
                            gpu$(echo "$GPUS" | cut -d, -f1) ~1536 MiB
                            gpu$(echo "$GPUS" | cut -d, -f2) ~3072 MiB
                            gpu$(echo "$GPUS" | cut -d, -f3) ~4608 MiB
                            gpu$(echo "$GPUS" | cut -d, -f4) ~6144 MiB
how to verify             : grep the window below out of your vram CSV; the four columns
                            must rise in that order and return to idle afterwards.
ANNOUNCE

printf '\n==> running inside the container now\n\n'

docker run --rm --pid=host "${GPU_DEVICES[@]}" \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e HOLD_S="$HOLD_S" \
  -v "$REPO:/work:ro" \
  -v "$OUT:/out" \
  -w /work pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime \
  python /work/deploy/rootless/_attest.py "/out/attestation_$STAMP.txt" \
  | tee -a "$REPORT"

printf '\n==> done. Now compare with your recorder:\n'
printf '    the window is in %s\n' "$REPORT"
