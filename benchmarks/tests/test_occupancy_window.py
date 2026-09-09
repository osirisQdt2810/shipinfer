"""Occupancy is a percentage of the window it was measured over, and here is that arithmetic.

The counter behind it is cumulative microseconds, and both readers used to divide it by
`--seconds` — which contains `warmup_s`, while every other counter the analysis rates is
differenced against an at-warmup snapshot. So the printed percentage charged the warm-up's
idle time to the busy window and *understated* the steady one: the one direction in this
reading that is not conservative, and the reading that redirected the next optimisation away
from the GPU (`NOT-GPU-BOUND-AT-FIVE-GPUS`).

The lesson from the system tier applies: **the arithmetic is where a benchmark lies.** So the
division lives in one pure function and is pinned here, with no GPU and no engines.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness.shipinfer import _summed_by_device, busy_pct

HARNESS = Path(__file__).resolve().parents[1] / "harness" / "shipinfer.py"


class _Instance:
    def __init__(self, device: str, compute_us: float, rows: int) -> None:
        self._stats = {"device": device, "compute_us": compute_us, "rows": rows}

    def stats(self) -> dict:
        return self._stats


class _Handle:
    def __init__(self, *instances: _Instance) -> None:
        self.instances = instances


class TestTheSnapshotSumsTheRightField:
    """`_summed_by_device` is what makes the boundary snapshot possible at all."""

    def test_instances_on_one_device_add_and_devices_stay_apart(self) -> None:
        handles = {
            "det": _Handle(_Instance("cuda:0", 10.0, 1), _Instance("cuda:0", 5.0, 2)),
            "emb": _Handle(_Instance("cuda:1", 7.5, 3)),
        }

        assert _summed_by_device(handles, "compute_us") == {
            "det": {"cuda:0": 15.0},
            "emb": {"cuda:1": 7.5},
        }

    def test_it_reads_the_field_it_is_asked_for(self) -> None:
        """Named rather than hard-coded, so a caller cannot get occupancy when it asked for
        rows -- which is the mistake that would make the subtraction silently meaningless."""
        handles = {"det": _Handle(_Instance("cuda:0", 10.0, 4))}

        assert _summed_by_device(handles, "rows") == {"det": {"cuda:0": 4.0}}


class TestTheBoundarySnapshotCarriesOccupancy:
    """The one link the pure tests above cannot reach, and it is the link that broke twice.

    `counters()` is a closure over a live server, so this reads the source instead. It is a
    weaker test and says so -- what it catches is the specific omission #167 and #170 both
    made: a counter computed and then not carried, which leaves the subtraction a no-op and
    the percentage silently back to the whole window.
    """

    def test_the_snapshot_and_its_fallback_both_name_the_counter(self) -> None:
        # Whitespace stripped, because `black` wraps a long call and a guard the project's own
        # formatter breaks is a guard that gets deleted rather than fixed. It did break the
        # first draft of this test.
        dense = "".join(HARNESS.read_text("utf-8").split())

        assert '"compute_us":_summed_by_device(handles,"compute_us")' in dense, (
            "the at-warmup snapshot no longer collects occupancy, so `busy_pct` subtracts "
            "nothing and the percentage is over the whole window again"
        )
        assert dense.count('"compute_us":{}') == 1, (
            "the shorter-than-warmup fallback must still supply the key, or the run that has "
            "no steady window raises instead of reporting the whole one"
        )
        assert (
            'busy_pct(at_end["compute_us"],at_warmup["compute_us"],steady_s)' in dense
        ), "the result is no longer built from the two snapshots"


class TestOccupancyIsRatedOverTheSteadyWindow:
    def test_the_warmup_is_subtracted_from_both_halves(self) -> None:
        """A 70 s run with a 10 s warm-up in which the device did 1 s of work, and 36 s of
        work by the end. The steady answer is 35 s over 60 s; dividing the end value by the
        whole window gives 51.4%, which is the understatement this exists to remove."""
        pct = busy_pct(
            {"m": {"cuda:0": 36_000_000.0}}, {"m": {"cuda:0": 1_000_000.0}}, steady_s=60.0
        )

        assert pct["m"]["cuda:0"] == 35_000_000.0 / 60_000_000.0 * 100.0
        assert round(pct["m"]["cuda:0"], 1) == 58.3
        assert round(36_000_000.0 / 70_000_000.0 * 100.0, 1) == 51.4, (
            "the old arithmetic, kept here as the contrast: without it this test would pass "
            "on numbers where the two agree"
        )

    def test_no_window_is_no_number_rather_than_a_zero(self) -> None:
        """A run shorter than its own warm-up has no steady seconds. `0.0%` would print as
        "idle", which is a claim; silence is the truth."""
        assert busy_pct({"m": {"cuda:0": 5.0}}, {}, steady_s=0.0) == {}
        assert busy_pct({"m": {"cuda:0": 5.0}}, {}, steady_s=-1.0) == {}

    def test_a_model_absent_at_the_boundary_is_charged_in_full(self) -> None:
        """A model whose first instance answered after the boundary has no snapshot row. Its
        whole total is steady work, and a missing row must not be a `KeyError`."""
        pct = busy_pct({"late": {"cuda:1": 30_000_000.0}}, {"other": {}}, steady_s=60.0)

        assert pct == {"late": {"cuda:1": 50.0}}

    def test_a_counter_that_appears_to_go_backwards_reads_zero(self) -> None:
        """Not expected — `compute_us` only grows — but a negative percentage would send a
        reader looking for a GPU problem instead of at the snapshot that produced it."""
        pct = busy_pct({"m": {"cuda:0": 1.0}}, {"m": {"cuda:0": 9.0}}, steady_s=60.0)

        assert pct == {"m": {"cuda:0": 0.0}}

    def test_every_device_of_every_model_is_carried_through(self) -> None:
        """The shape is the same table the printer and a shard's `summary.json` carry, so a
        model or a device dropped here is a row missing from the balancing evidence."""
        pct = busy_pct(
            {"det": {"cuda:0": 30_000_000.0, "cuda:1": 6_000_000.0}, "emb": {"cuda:1": 0.0}},
            {},
            steady_s=60.0,
        )

        assert pct == {"det": {"cuda:0": 50.0, "cuda:1": 10.0}, "emb": {"cuda:1": 0.0}}
