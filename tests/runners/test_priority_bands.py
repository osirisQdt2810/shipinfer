"""``PriorityBands``: which lane a camera is admitted into, and who said so.

Its own file because the unit is its own object now. Every case here was previously
reachable only by driving a whole runner, which is why the two-table rule below -- the one
that keeps a DELETE-then-POST from inheriting a dead placement's ``tracking_critical`` --
had no direct test at all.
"""

from __future__ import annotations

import logging
import threading

import pytest

from shipinfer.core.request import Priority
from shipinfer.runners.bands import PriorityBands


@pytest.fixture()
def bands() -> PriorityBands:
    return PriorityBands()


class TestThePrecedence:
    """Placement, then this process's config, then the learned default."""

    def test_a_placement_outranks_the_config(self, bands: PriorityBands) -> None:
        bands.configure({"cam0": Priority.BACKGROUND})
        bands.record_placement("cam0", Priority.HIGH)

        assert bands.for_camera("cam0") is Priority.HIGH

    def test_the_config_answers_when_no_placement_has(self, bands: PriorityBands) -> None:
        bands.configure({"cam0": Priority.BACKGROUND})

        assert bands.for_camera("cam0") is Priority.BACKGROUND

    def test_an_unknown_camera_gets_the_fleet_default(self, bands: PriorityBands) -> None:
        assert bands.for_camera("cam0") is Priority.NORMAL


class TestTrackingCriticalIsZeroAndThereforeFalsy:
    """ADR-005's trap: the shorter spelling demotes the one camera priorities exist for.

    The *placed* case is the one that discriminates -- rewrite `for_camera` with ``or`` and
    it reddens. The configured one survives that mutation, because the fall-through reaches
    the memo, which reads the same table with its own ``is not None`` and answers correctly
    by a different route. It is kept because it states the contract, not because it catches
    that particular slip.
    """

    def test_a_placed_tracking_critical_is_not_read_as_absent(
        self, bands: PriorityBands
    ) -> None:
        bands.configure({"cam0": Priority.BACKGROUND})
        bands.record_placement("cam0", Priority.TRACKING_CRITICAL)

        assert bands.for_camera("cam0") is Priority.TRACKING_CRITICAL

    def test_a_configured_tracking_critical_is_not_read_as_absent(
        self, bands: PriorityBands
    ) -> None:
        bands.configure({"cam0": Priority.TRACKING_CRITICAL})

        assert bands.for_camera("cam0") is Priority.TRACKING_CRITICAL

    def test_restoring_a_tracking_critical_writes_it_back(self, bands: PriorityBands) -> None:
        bands.record_placement("cam0", Priority.TRACKING_CRITICAL)
        previous = bands.placed("cam0")
        bands.record_placement("cam0", Priority.BACKGROUND)
        bands.restore("cam0", previous)

        assert bands.placed("cam0") is Priority.TRACKING_CRITICAL


class TestAPlacementCanBeUnsaid:
    """The reason there are two tables and not one dict with a write order."""

    def test_a_spec_with_no_band_erases_the_previous_placement(
        self, bands: PriorityBands
    ) -> None:
        bands.configure({"cam0": Priority.NORMAL})
        bands.record_placement("cam0", Priority.TRACKING_CRITICAL)
        bands.record_placement("cam0", None)

        assert bands.placed("cam0") is None
        assert bands.for_camera("cam0") is Priority.NORMAL

    def test_forgetting_a_placement_leaves_the_config_standing(
        self, bands: PriorityBands
    ) -> None:
        bands.configure({"cam0": Priority.BACKGROUND})
        bands.record_placement("cam0", Priority.HIGH)
        bands.forget_placement("cam0")

        assert bands.for_camera("cam0") is Priority.BACKGROUND

    def test_clearing_every_placement_leaves_every_configured_band(
        self, bands: PriorityBands
    ) -> None:
        bands.configure({"cam0": Priority.BACKGROUND, "cam1": Priority.HIGH})
        bands.record_placement("cam0", Priority.TRACKING_CRITICAL)
        bands.record_placement("cam1", Priority.TRACKING_CRITICAL)
        bands.clear_placements()

        assert (bands.for_camera("cam0"), bands.for_camera("cam1")) == (
            Priority.BACKGROUND,
            Priority.HIGH,
        )

    def test_restoring_nothing_pops_rather_than_writing(self, bands: PriorityBands) -> None:
        """``previous is None`` means there was no placement, not ``NORMAL``."""
        previous = bands.placed("cam0")
        bands.record_placement("cam0", Priority.HIGH)
        bands.restore("cam0", previous)

        assert bands.placed("cam0") is None


class TestTheLearnedDefault:
    def test_it_is_memoised_and_logged_once(
        self, bands: PriorityBands, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Paying for the discovery per frame would make a config gap a performance bug."""
        with caplog.at_level(logging.INFO, logger="shipinfer.runners.bands"):
            for _ in range(5):
                assert bands.for_camera("cam0") is Priority.NORMAL

        said = [r for r in caplog.records if "not in the ingest config" in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in said]

    def test_a_later_placement_still_outranks_it(self, bands: PriorityBands) -> None:
        """It records that the *config* is silent, which a placement does not change."""
        assert bands.for_camera("cam0") is Priority.NORMAL
        bands.record_placement("cam0", Priority.TRACKING_CRITICAL)

        assert bands.for_camera("cam0") is Priority.TRACKING_CRITICAL

    def test_removing_the_camera_does_not_unlearn_it(
        self, bands: PriorityBands, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="shipinfer.runners.bands"):
            bands.for_camera("cam0")
            bands.forget_placement("cam0")
            bands.for_camera("cam0")

        said = [r for r in caplog.records if "not in the ingest config" in r.getMessage()]
        assert len(said) == 1, "the config is still silent about it; that is not news twice"


class TestItIsSafeFromAnyThread:
    def test_concurrent_first_sightings_agree_and_announce_once(
        self, bands: PriorityBands, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Twenty threads meeting the same unknown camera at once (the submit path)."""
        seen: list[Priority] = []
        start = threading.Event()

        def ask() -> None:
            start.wait(5.0)
            seen.append(bands.for_camera("cam0"))

        with caplog.at_level(logging.INFO, logger="shipinfer.runners.bands"):
            workers = [threading.Thread(target=ask) for _ in range(20)]
            for worker in workers:
                worker.start()
            start.set()
            for worker in workers:
                worker.join(5.0)

        assert seen == [Priority.NORMAL] * 20
        said = [r for r in caplog.records if "not in the ingest config" in r.getMessage()]
        assert len(said) == 1, "the memo is under the lock, so exactly one thread announces"
