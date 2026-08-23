"""The one number the whole benchmark exists to produce, and how it can be wrong.

These are pure-logic tests over synthetic occupancy logs, so they belong in the offline
tier: no GPU, no engines, no baseline binary. That is deliberate. The measurement is the
part most worth testing and the part least in need of hardware — if the arithmetic that
turns a buffer log into "5x" is wrong, running it on real GPUs only makes the wrong number
more convincing.

The failure this file is mostly about is **double counting**. ShipInfer's occupancy log
carries one row per module, and two of those modules see the same image: every camera frame
enters the pipeline queue and then becomes a detector request. Summing modules the way the
baseline's own report legitimately does would report roughly twice ShipInfer's real
throughput — a 2x speed-up conjured out of a naming decision, in the direction that flatters
us. That is why the sum is not a shared helper: it is a per-system decision, stated once in
``run_bench.system_throughput``, and asserted here in both directions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import run_bench
from benchmarks.harness import analysis
from benchmarks.harness import shipinfer as shipinfer_harness
from benchmarks.harness.config import BenchConfig


def write_log(path: Path, rows: list[dict[str, float]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def steady_log(path: Path, modules: dict[str, float], *, seconds: int = 40) -> Path:
    """A log whose every buffer holds a constant depth — nothing is falling behind."""
    return write_log(
        path,
        [
            {"t": float(t), **{f"{m}_buffer_size": d for m, d in modules.items()}}
            for t in range(seconds)
        ],
    )


def growing_log(path: Path, modules: dict[str, float], *, seconds: int = 40) -> Path:
    """A log whose buffers grow linearly at the given rate — the saturation signature."""
    return write_log(
        path,
        [
            {"t": float(t), **{f"{m}_buffer_size": rate * t for m, rate in modules.items()}}
            for t in range(seconds)
        ],
    )


#: Which module an image enters at, per system. The same declaration `run_bench` makes.
ENTRIES = {"baseline": ("det", "seg"), "shipinfer": ("pipeline",)}


def analyse(path: Path, system: str, offered: dict[str, float | None]):
    return analysis.analyse(
        analysis.read_log(path, sample_interval_s=1.0),
        system=system,
        warmup_s=5.0,
        offered=offered,
        capacity=65536,
        entry_modules=ENTRIES[system],
    )


class TestTheBaselineSumsDisjointStreams:
    """Half the baseline's sources feed `det`, half feed `seg`. No image is in both."""

    def test_a_keeping_up_baseline_sustains_the_whole_offered_rate(self, tmp_path: Path):
        log = steady_log(tmp_path / "b.jsonl", {"det": 12, "seg": 9})
        run = analyse(log, "baseline", {"det": 500.0, "seg": 500.0})

        throughput = run_bench.system_throughput(run)

        assert throughput.is_rate
        assert throughput.images_per_s == pytest.approx(1000.0, abs=1.0)

    def test_growth_is_subtracted_from_the_offered_rate(self, tmp_path: Path):
        """`det` falls behind by 100 img/s; that is exactly what it fails to sustain."""
        log = growing_log(tmp_path / "b.jsonl", {"det": 100.0, "seg": 0.0})
        run = analyse(log, "baseline", {"det": 500.0, "seg": 500.0})

        throughput = run_bench.system_throughput(run)

        assert throughput.images_per_s == pytest.approx(900.0, abs=5.0)
        assert throughput.saturated, "a growing buffer is the definition of saturated"
        assert not throughput.is_rate, "a saturated number is a bound, not a rate"


class TestShipInferIsNotDoubleCounted:
    """The failure that would flatter us, asserted directly."""

    def test_the_detector_is_not_added_to_the_pipeline_queue(self, tmp_path: Path):
        """Both modules see every frame. Counting both would report twice the truth.

        `total_sustained` used to sum every module with a known offered rate — here
        1000 + 1000 — and `render()` printed it as a TOTAL row directly above the
        comparison block. Summing only the declared entry modules closes that: the detector
        sees the frame the pipeline queue already counted.
        """
        log = steady_log(tmp_path / "s.jsonl", {"pipeline": 30, "ship_detector": 11})
        run = analyse(log, "shipinfer", {"pipeline": 1000.0, "ship_detector": 1000.0})

        throughput = run_bench.system_throughput(run)

        assert throughput.images_per_s == pytest.approx(1000.0, abs=1.0)
        assert run.total_sustained == pytest.approx(
            1000.0, abs=1.0
        ), "the total counts the pipeline queue once, not the queue plus the detector"

    def test_the_rendered_table_never_shows_a_double_counted_total(self, tmp_path: Path):
        """The production path, not just the property: the printed table is what a reader
        divides by eye, so it is what has to be right."""
        log = steady_log(
            tmp_path / "s.jsonl", {"pipeline": 30, "ship_detector": 11, "ship_embedder": 40}
        )
        run = analyse(
            log,
            "shipinfer",
            {"pipeline": 1000.0, "ship_detector": 1000.0, "ship_embedder": 3000.0},
        )

        table = analysis.render([run])

        totals = [line for line in table.splitlines() if "TOTAL" in line]
        assert len(totals) == 1, table
        assert "5000" not in table and "2000" not in table, (
            "a five-figure sum of frames and crops has no meaning and invites the division "
            "`compare()` refuses:\n" + table
        )

    def test_crop_fed_models_never_enter_the_image_count(self, tmp_path: Path):
        """Segmenter and embedder rates are crops per second, not images per second."""
        log = steady_log(
            tmp_path / "s.jsonl",
            {"pipeline": 30, "ship_detector": 11, "ship_embedder": 40, "person_embedder": 90},
        )
        run = analyse(
            log,
            "shipinfer",
            {
                "pipeline": 1000.0,
                "ship_detector": 1000.0,
                "ship_embedder": 3000.0,
                "person_embedder": 12000.0,
            },
        )

        throughput = run_bench.system_throughput(run)

        assert throughput.images_per_s == pytest.approx(1000.0, abs=1.0), (
            "15 000 crops/s is the sizing target, not a throughput — folding it into an "
            "image count would report a 16x speed-up from the fan-out alone"
        )

    def test_a_missing_pipeline_module_is_unmeasured_not_zero(self, tmp_path: Path):
        log = steady_log(tmp_path / "s.jsonl", {"ship_detector": 11})
        run = analyse(log, "shipinfer", {"ship_detector": 1000.0})

        throughput = run_bench.system_throughput(run)

        assert throughput.images_per_s is None
        assert not throughput.is_rate


class TestTheVerdictRefusesToInvent:
    """A ratio of two bounds is not a speed-up."""

    def _throughput(self, system: str, value: float | None, *, saturated: bool):
        return run_bench.SystemThroughput(system, value, saturated, None, "test")

    def test_a_real_speedup_is_reported_against_the_target(self):
        text = run_bench.compare(
            self._throughput("baseline", 200.0, saturated=False),
            self._throughput("shipinfer", 1000.0, saturated=False),
            target=5.0,
        )
        assert "5.00x" in text
        assert "MET" in text and "NOT MET" not in text

    def test_falling_short_of_the_target_says_so(self):
        text = run_bench.compare(
            self._throughput("baseline", 400.0, saturated=False),
            self._throughput("shipinfer", 1000.0, saturated=False),
            target=5.0,
        )
        assert "2.50x" in text
        assert "NOT MET" in text

    def test_a_saturated_side_yields_no_ratio(self):
        """The number exists but is an upper bound, so dividing it would be a fiction."""
        text = run_bench.compare(
            self._throughput("baseline", 775.5, saturated=True),
            self._throughput("shipinfer", 1000.0, saturated=False),
            target=5.0,
        )
        assert "NOT AVAILABLE" in text
        assert "x" not in text.split("Speed-up:")[1].split("\n")[0].replace("baseline", "")

    def test_an_unmeasured_side_yields_no_ratio(self):
        text = run_bench.compare(
            self._throughput("baseline", None, saturated=False),
            self._throughput("shipinfer", 1000.0, saturated=False),
            target=5.0,
        )
        assert "NOT AVAILABLE" in text


class TestTheFitDistinguishesTheThreeCases:
    """Sanity on the measurement underneath, so a metric test cannot pass on a broken fit."""

    def test_a_flat_buffer_is_sustained(self, tmp_path: Path):
        run = analyse(steady_log(tmp_path / "l.jsonl", {"det": 10}), "baseline", {"det": 500.0})
        assert run.verdict == analysis.SUSTAINED

    def test_a_growing_buffer_is_saturated(self, tmp_path: Path):
        run = analyse(
            growing_log(tmp_path / "l.jsonl", {"det": 50.0}), "baseline", {"det": 500.0}
        )
        assert run.verdict == analysis.SATURATED

    def test_one_saturated_module_saturates_the_run(self, tmp_path: Path):
        """A pipeline is bounded by its slowest stage, so any growth is the verdict."""
        rows = [
            {"t": float(t), "det_buffer_size": 10.0, "seg_buffer_size": 40.0 * t}
            for t in range(40)
        ]
        run = analyse(
            write_log(tmp_path / "l.jsonl", rows), "baseline", {"det": 500.0, "seg": 500.0}
        )
        assert run.verdict == analysis.SATURATED
        assert run.binding_module is not None
        assert run.binding_module.module == "seg"


class TestTheAnalysisRefusesWhatItCannotMeasure:
    """Three ways a number could be published that the data does not support."""

    def test_a_window_too_short_to_bound_a_slope_is_refused(self, tmp_path: Path):
        """Two samples give one difference and no interval, which reads as flat whatever
        the slope. Raising beats reporting."""
        rows = [{"t": float(t), "det_buffer_size": 900.0 * t} for t in range(12)]
        with pytest.raises(ValueError, match="fewer than"):
            analysis.analyse(
                analysis.read_log(write_log(tmp_path / "l.jsonl", rows), sample_interval_s=1.0),
                system="baseline",
                warmup_s=10.0,
                offered={"det": 1000.0},
                entry_modules=("det",),
            )

    def test_an_unbounded_estimate_is_unmeasured_not_sustained(self):
        """A single first difference has an infinite half-width, which compares as
        not-significantly-positive — so a queue growing at 890/s used to come back
        SUSTAINED and have its number published as a rate."""
        fit = analysis.fit_growth([0.0, 1.0], [0.0, 890.0])

        assert not math.isfinite(fit.half_width)
        assert fit.verdict == analysis.UNMEASURED

    def test_a_draining_queue_cannot_report_more_than_was_offered(self, tmp_path: Path):
        """A queue that empties at 150/s against a 1000 img/s offer used to print 1150 —
        and a 1.35x speed-up out of nothing but drainage."""
        rows = [
            {"t": float(t), "det_buffer_size": max(0.0, 5000.0 - 150.0 * t)} for t in range(30)
        ]
        run = analysis.analyse(
            analysis.read_log(write_log(tmp_path / "l.jsonl", rows), sample_interval_s=1.0),
            system="baseline",
            warmup_s=5.0,
            offered={"det": 1000.0},
            entry_modules=("det",),
        )

        module = run.modules[0]
        assert module.fit.verdict == analysis.DRAINING
        assert module.sustained == pytest.approx(1000.0), "clamped to what was offered"
        assert module.headroom == pytest.approx(150.0, abs=5.0), "the spare shows as headroom"


class TestAStarvedGeneratorIsRefused:
    """The load has to be delivered before a throughput may be derived from it."""

    def _result(self, read: int, elapsed: float):
        return shipinfer_harness.ShipInferResult(
            log=Path("/dev/null"),
            startup_s=1.0,
            elapsed_s=elapsed,
            frames_accepted=read,
            events_emitted=read,
            frames_read=read,
        )

    def test_a_generator_that_delivered_the_load_passes(self):
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        shipinfer_harness.check_offer(config, self._result(read=30000, elapsed=30.0))

    def test_a_starved_generator_raises_rather_than_reporting(self):
        """40% of the target delivered, and the analysis would otherwise divide by 1000."""
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        with pytest.raises(RuntimeError, match="never offered"):
            shipinfer_harness.check_offer(config, self._result(read=12000, elapsed=30.0))

    def test_the_offered_rate_used_is_the_measured_one(self):
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        result = self._result(read=24000, elapsed=30.0)

        assert shipinfer_harness.achieved_offer(config, result) == pytest.approx(800.0)
