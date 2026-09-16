#!/usr/bin/env bash
# Narrow the container's visible device set to the cards a run actually uses, and work out what
# to CALL each of them in the output. Sourced by `scripts/run_cpp_bench.sh`; a separate file so
# it can be sourced by a test too, the way `_gpus.sh` is (`tests/test_deploy_gpu_selection.py`).
#
# WHY IT EXISTS: `cpp.sh` passes `--device nvidia.com/gpu=all` by default, and CUDA's per-process
# init enumerates every visible device -- measured 16 Sep, first context 9.96/10.07 s with all
# eight cards against 0.658/0.665 s with three, on a box where four are held by other tenants.
# At fifty cameras that blows the camera-start budget and the run reads ZERO frames.
#
# SORTED, and that is not tidiness. The container does NOT order the visible set by the order the
# `--device` flags were passed: measured by UUID, `SHIPINFER_GPUS=6,3,2` gives container ordinal
# 0 = host 2, 1 = host 3, 2 = host 6 -- ASCENDING BY HOST INDEX. A positional mapping printed
# host 2's counters under `6:`, inverted, which is the one output a reader cannot tell from a
# correct one (#294 round 1). So narrowing NORMALISES: `6,3,2` and `2,3,6` are the same run.
#
# Reads NARROW and GPU_IDS (host ids). Sets GPU_IDS (what the binary is told), GPU_LABELS (what
# it prints), and exports SHIPINFER_GPUS for `_gpus.sh`.
narrow_devices() {
  if [ "${NARROW:-1}" != "1" ] || [ -n "${SHIPINFER_GPUS:-}" ]; then
    # Either opted out, or the operator picked the visible set by hand -- in which case
    # `GPU_IDS` is already whatever they mean the binary to address.
    GPU_LABELS="$GPU_IDS"
    return 0
  fi
  GPU_LABELS="$(printf '%s' "$GPU_IDS" | tr ',' '\n' | sort -n | paste -sd,)"
  export SHIPINFER_GPUS="$GPU_LABELS"
  # CUDA's default is FASTEST_FIRST, which only ties by bus id because every card here is the
  # same model. Pinned so the ascending mapping above is a fact rather than a coincidence.
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  # FIELDS, not lines: `printf '%s'` emits no trailing newline, so `wc -l` undercounts by one
  # and a three-card run was told `--gpu-ids 0,1`. The test for this is what caught it.
  GPU_IDS="$(seq -s, 0 "$(( $(printf '%s' "$GPU_LABELS" | awk -F, '{print NF}') - 1 ))")"
}
