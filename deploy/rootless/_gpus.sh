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
