"""`_narrow.sh`: what the container sees, and what the tables call it.

The mapping this asserts is the correctness core of the narrowing change, and round 1 of #294
got it WRONG in a way nothing caught: labels were applied positionally, so
`SHIPINFER_BENCH_GPUS=6,3,2` printed host 2's counters under `6:` -- an inverted `per_device`
table in the one file whose value is cross-run comparison. Sourced and asserted here, the way
`test_deploy_gpu_selection.py` does for `_gpus.sh`, because a container run cannot be a gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "deploy" / "rootless" / "_narrow.sh"


def _narrow(ids: str, *, narrow: str = "1", visible: str | None = None) -> dict[str, str]:
    """Source the helper and print the three values a caller depends on."""
    env = {"PATH": "/usr/bin:/bin", "NARROW": narrow, "GPU_IDS": ids}
    if visible is not None:
        env["SHIPINFER_GPUS"] = visible
    done = subprocess.run(
        [
            "bash",
            "-c",
            f'. "{HELPER}" && narrow_devices && '
            'echo "ids=$GPU_IDS" && echo "labels=$GPU_LABELS" && '
            'echo "visible=${SHIPINFER_GPUS:-}" && echo "order=${CUDA_DEVICE_ORDER:-}"',
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return dict(line.split("=", 1) for line in done.stdout.strip().splitlines())


class TestNarrowingSortsBecauseTheContainerDoes:
    def test_out_of_order_host_ids_are_sorted_and_labelled_in_that_order(self) -> None:
        """THE round-1 regression. `6,3,2` must not label ordinal 0 as host 6."""
        out = _narrow("6,3,2")

        assert out["visible"] == "2,3,6", "the container is given the cards ascending"
        assert out["ids"] == "0,1,2", "and addresses them as the ordinals it will see"
        assert (
            out["labels"] == "2,3,6"
        ), "so label i is the i-th SMALLEST host id, not the i-th typed"

    def test_already_ascending_is_unchanged_by_the_sort(self) -> None:
        out = _narrow("1,3,4,6")

        assert out["visible"] == "1,3,4,6"
        assert out["ids"] == "0,1,2,3"
        assert out["labels"] == "1,3,4,6"

    def test_one_device_still_gets_a_label(self) -> None:
        out = _narrow("5")

        assert (out["ids"], out["labels"], out["visible"]) == ("0", "5", "5")

    def test_the_device_order_is_pinned_so_the_mapping_is_not_a_coincidence(self) -> None:
        """CUDA's default FASTEST_FIRST only ties by bus id because the cards match here."""
        assert _narrow("2,3,6")["order"] == "PCI_BUS_ID"


class TestTheEscapeHatchesLeaveItAlone:
    def test_narrow_zero_passes_the_host_ids_straight_through(self) -> None:
        out = _narrow("6,3,2", narrow="0")

        assert out["ids"] == "6,3,2", "the binary is told exactly what the caller asked for"
        assert out["labels"] == "6,3,2", "and labels equal ids, so `label_of` is the identity"
        assert out["visible"] == "", "nothing narrowed, so `_gpus.sh` still sees `all`"

    def test_an_explicit_visible_set_wins(self) -> None:
        """The operator picked the visible set, so `GPU_IDS` already means container ordinals."""
        out = _narrow("0,1,2", visible="1,2,6")

        assert out["ids"] == "0,1,2"
        assert out["labels"] == "0,1,2"
        assert out["visible"] == "1,2,6", "left exactly as the operator set it"
