"""Which GPU a device-tier test may take — the four outcomes, all of them offline.

`tests/support/devices.py` decides that for every device-tier test, and a GPU run exercises
exactly one of its branches: the one where the knob is unset. The other three are what the
file exists for, and they are reachable with no driver at all by monkeypatching
`torch.cuda.device_count` — torch is in the test image, only the cards are absent.

NOT marked `gpu`, deliberately. A test of the code that picks a device must not need one.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tests.support.devices import a_test_device, visible_devices


@pytest.fixture()
def cards(monkeypatch: pytest.MonkeyPatch):
    """Pretend this container was given ``count`` devices, whatever the host really has."""

    def given(count: int) -> None:
        cuda = SimpleNamespace(device_count=lambda: count)
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    return given


class TestTheKnobNarrowsWhatTorchReports:
    def test_unset_means_every_device_this_container_has(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(4)
        monkeypatch.delenv("SHIPINFER_TEST_GPUS", raising=False)

        assert visible_devices() == [0, 1, 2, 3]

    def test_a_subset_is_honoured_in_the_order_asked_for(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(4)
        monkeypatch.setenv("SHIPINFER_TEST_GPUS", "2,1")

        assert visible_devices() == [2, 1], "not sorted; the operator's order is the answer"

    def test_a_partial_overlap_keeps_what_can_be_honoured(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(2)
        monkeypatch.setenv("SHIPINFER_TEST_GPUS", "1,7")

        assert visible_devices() == [1]

    def test_whitespace_and_empty_entries_are_ignored(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(2)
        monkeypatch.setenv("SHIPINFER_TEST_GPUS", " 1 , ")

        assert visible_devices() == [1]


class TestAnUnhonourableRequestSkips:
    """It does not fall back, and that is the whole point of the file.

    Two readers of one knob with opposite semantics is how `DEVICE = 5` happened: a silent
    fallback would put a test on `cuda:0` after the operator's own command excluded it.
    """

    def test_nothing_honourable_skips_and_names_both_sides(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(2)
        monkeypatch.setenv("SHIPINFER_TEST_GPUS", "4,5")

        with pytest.raises(pytest.skip.Exception, match=r"asks for \[4, 5\].*has \[0, 1\]"):
            visible_devices()

    def test_the_skip_explains_the_ordinals_are_the_containers(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The genuinely surprising part, so the message says it rather than the docstring."""
        cards(2)
        monkeypatch.setenv("SHIPINFER_TEST_GPUS", "4,5")

        with pytest.raises(pytest.skip.Exception, match=r"SHIPINFER_GPUS=4,5"):
            visible_devices()


class TestOneDevice:
    def test_the_first_honoured_device_is_the_one(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(4)
        monkeypatch.setenv("SHIPINFER_TEST_GPUS", "3,2")

        assert a_test_device() == 3

    def test_a_container_with_no_card_skips_and_names_the_runner(
        self, cards, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cards(0)
        monkeypatch.delenv("SHIPINFER_TEST_GPUS", raising=False)

        with pytest.raises(pytest.skip.Exception, match=r"deploy/rootless/test\.sh"):
            a_test_device()
