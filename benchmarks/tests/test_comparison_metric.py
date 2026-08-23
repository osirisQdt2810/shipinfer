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
from pathlib import Path

import pytest

from benchmarks import run_bench
from benchmarks.harness import analysis


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


def analyse(path: Path, system: str, offered: dict[str, float | None]):
    return analysis.analyse(
        analysis.read_log(path, sample_interval_s=1.0),
        system=system,
        warmup_s=5.0,
        offered=offered,
        capacity=65536,
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
        """Both modules see every frame. Counting both reports twice the truth.

        This is the whole reason ``RunAnalysis.total_sustained`` is not used for ShipInfer:
        it sums every module with a known offered rate, which here is 1000 + 1000.
        """
        log = steady_log(tmp_path / "s.jsonl", {"pipeline": 30, "ship_detector": 11})
        run = analyse(log, "shipinfer", {"pipeline": 1000.0, "ship_detector": 1000.0})

        throughput = run_bench.system_throughput(run)

        assert throughput.images_per_s == pytest.approx(1000.0, abs=1.0)
        assert run.total_sustained == pytest.approx(2000.0, abs=2.0), (
            "guard on the trap itself: total_sustained really does sum to 2000 here, so a "
            "future refactor that reaches for it instead would double the reported speed-up"
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
