"""Which GPU a device-tier test may take.

Asked of torch, not written down. Two tests carried ``DEVICE = 5`` -- "the one the operator
keeps free" -- which broke the day the container stopped being handed every card:
``SHIPINFER_GPUS=0,1,2,3`` (#128) leaves torch reporting ``[0, 1, 2, 3]``, and the settings
correctly refuse ``visible_gpus: [5]``.

``SHIPINFER_TEST_GPUS`` is the knob that already means "which devices a test may take"
(``deploy/rootless/_container.sh`` passes it through), so this is the ONE place that reads
it -- ``test_service_multigpu.py`` held the second copy, defaulting to ``0,1`` and so asking
a one-card container for ``cuda:1``.

The name avoids a ``test_`` prefix on purpose: pytest collects such an imported name as a
test in the importing module.
"""

from __future__ import annotations

import os

import pytest

__all__ = ["a_test_device", "visible_devices"]


def visible_devices() -> list[int]:
    """Every device this container has, narrowed by ``SHIPINFER_TEST_GPUS`` when it is set.

    Container ordinals, not physical ones: `--device nvidia.com/gpu=2` is `cuda:0` inside, so
    a physical index is not a thing a test can ask for and should not pretend to.

    An explicit request that cannot be honoured **skips**, naming both sides -- it does not
    fall back. `test_service_multigpu.py` takes the knob at face value and skips, and two
    readers of one knob with opposite semantics is how `DEVICE = 5` happened: a silent
    fallback would put a test on `cuda:0` after the operator's own command excluded it.
    """
    import torch

    available = list(range(torch.cuda.device_count()))
    wanted = [
        int(piece)
        for piece in os.environ.get("SHIPINFER_TEST_GPUS", "").split(",")
        if piece.strip()
    ]
    if not wanted:
        return available
    honoured = [index for index in wanted if index in available]
    if not honoured:
        pytest.skip(
            f"SHIPINFER_TEST_GPUS asks for {wanted} and this container has {available}; "
            f"these are CONTAINER ordinals, so with SHIPINFER_GPUS=4,5 the devices are 0,1"
        )
    return honoured


def a_test_device() -> int:
    """One device to run on, or a named skip when the container was given none."""
    devices = visible_devices()
    if not devices:
        pytest.skip(
            "no CUDA device is visible in this container; run through "
            "deploy/rootless/test.sh, and see SHIPINFER_GPUS if one is faulted"
        )
    return devices[0]
