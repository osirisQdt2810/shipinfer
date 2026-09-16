# Which physical GPUs a container may see. Sourced by every script here that runs docker.
#
# WHY THIS EXISTS, and it is not a convenience. Every script used to hard-code
# `--device nvidia.com/gpu=all`, and on 1 Sep a single degraded card took the whole GPU tier
# down for three days:
#
#   RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
#   at ATen/cuda/CUDAContext.cpp:52 ... device=7, num_gpus=7
#
# The driver enumerates 8 devices, so `torch.cuda.device_count()` says 8, while CUDA can
# only open 7 — and `torch.cuda.__init__` queues `_check_capability`, which walks EVERY
# visible device. So a test that wanted GPUs 0-3 failed at CUDA init because of a card it
# never asked for. `CUDA_VISIBLE_DEVICES` alone does not help when the device node is
# mounted: the fix is to not hand the container the card at all.
#
#   SHIPINFER_GPUS=0,1,2,3 deploy/rootless/test.sh -m gpu   # route around a sick card
#   SHIPINFER_GPUS=all     (the default)
#
# Leaves `GPU_DEVICES` set to the `docker run` arguments, and `CUDA_VISIBLE_DEVICES` unset
# deliberately: the container then numbers what it was given from 0, so a test asking for
# `cuda:0` gets the first HEALTHY card rather than a hole.

# AND THE DEFAULT ROUTES AROUND IT NOW, because documenting the incantation was not enough:
# the header above has described this exact assert, from this exact card, since 1 Sep, and the
# default stayed `all` -- so it took the tier down again on 16 Sep for a session that did not
# know to set the variable. `scripts/usable_gpus.py` asks the CUDA runtime which cards it can
# actually open and matches them to the driver's indices by PCI bus id; it prints nothing and
# exits non-zero when the two agree, when either enumeration is unavailable, or when the answer
# would be empty. So this only ever fires on a box that is already broken, and an explicit
# `SHIPINFER_GPUS` -- including `SHIPINFER_GPUS=all` -- is never second-guessed. EMPTY counts as
# "not chosen" rather than as a choice, which is the one behaviour that changed for a value
# somebody might have set on purpose: `SHIPINFER_GPUS= ` now consults the probe, and
# `SHIPINFER_GPUS=all` is the way to say "every device, whatever CUDA thinks".
if [ -z "${SHIPINFER_GPUS:-}" ]; then
  # `python3 <path>`, not the shebang: this must not depend on an exec bit surviving a
  # checkout, and a missing interpreter has to degrade to the old default rather than fail.
  #
  # `SHIPINFER_GPU_PROBE` is the escape and the seam: `0` or empty turns the detection off and
  # restores the unconditional `all`, and any other value is the probe to run -- which is how
  # `tests/test_deploy_gpu_selection.py` asserts both branches without a sick card to hand.
  _probe="${SHIPINFER_GPU_PROBE-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/usable_gpus.py}"
  if [ "$_probe" = "0" ] || [ -z "$_probe" ]; then
    _usable=""
  else
    # ONCE, with the streams split into a file. Running it twice -- stdout, then stderr --
    # doubled a fork and a dlopen per container launch, could name a different card from the
    # one dropped if the two disagreed, and the second assignment had no `|| ...`: under the
    # `set -e` every caller here uses, a probe that failed on its second call aborted the
    # script with nothing on either stream, which is worse than the assert it replaces.
    _err="$(mktemp)"
    _usable="$(python3 "$_probe" 2>"$_err")" || _usable=""
    _missing="$(cat "$_err" 2>/dev/null)" || _missing=""
    rm -f "$_err"
  fi
  if [ -n "$_usable" ]; then
    echo "SHIPINFER_GPUS: CUDA cannot open GPU(s) $_missing -- and a card torch enumerates but" \
      "cannot open takes the whole tier down (see _gpus.sh). Using $_usable; set" \
      "SHIPINFER_GPUS explicitly to override." >&2
    SHIPINFER_GPUS="$_usable"
  fi
  unset _usable _missing _probe _err
fi

GPU_DEVICES=()
if [ "${SHIPINFER_GPUS:-all}" = "all" ]; then
  GPU_DEVICES=(--device nvidia.com/gpu=all)
else
  # One `--device` per index rather than a comma list: CDI accepts both, but a typo in a
  # comma list is a device named `0,1` that resolves to nothing and silently gives the
  # container no GPU at all — which reads exactly like a machine with no driver.
  _saved_ifs="$IFS"
  IFS=','
  for _index in ${SHIPINFER_GPUS}; do
    case "$_index" in
      ''|*[!0-9]*)
        echo "SHIPINFER_GPUS='$SHIPINFER_GPUS' is a comma-separated list of device indices" \
          "(or 'all'); '$_index' is not an index" >&2
        exit 2
        ;;
    esac
    GPU_DEVICES+=(--device "nvidia.com/gpu=$_index")
  done
  IFS="$_saved_ifs"
  unset _saved_ifs _index
fi
