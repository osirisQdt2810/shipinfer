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
        assert throughput.kind == run_bench.CAPACITY, (
            "a saturated run is the *only* regime in which `offered - growth` is exact; "
            "refusing it as 'a bound' is what made the speed-up unreachable"
        )


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


class TestTheRatioKnowsWhichWayItIsWrong:
    """Four ways two numbers combine, and only one of them is a plain division.

    The previous version of this class refused every saturated run as "a bound", which had
    the methodology backwards and made the PR's headline deliverable unreachable: both
    systems are offered the same 1000 img/s by construction, so either neither saturated
    (each reported its offered rate back, ratio 1.00x) or one did (comparison refused). Its
    green `5.00x` came from a hand-built baseline that was simultaneously at 200 img/s and
    not saturated, which a 1000 img/s run cannot be.
    """

    def _throughput(self, system: str, value: float | None, verdict: str):
        return run_bench.SystemThroughput(
            system, value, verdict == analysis.SATURATED, None, "test", verdict
        )

    def _capacity(self, system: str, value: float):
        return self._throughput(system, value, analysis.SATURATED)

    def _floor(self, system: str, value: float):
        return self._throughput(system, value, analysis.SUSTAINED)

    def test_two_capacities_divide_exactly(self):
        text = run_bench.compare(
            self._capacity("baseline", 200.0), self._capacity("shipinfer", 1000.0), target=5.0
        )
        assert "Speed-up: 5.00x" in text
        assert "MET" in text and "NOT MET" not in text
        headline = text.split("Speed-up:")[1].split("\n")[0]
        assert (
            ">= 5.00x" not in headline and "<= 5.00x" not in headline
        ), "an exact ratio must not be dressed as a bound"

    def test_two_capacities_can_miss_the_target(self):
        text = run_bench.compare(
            self._capacity("baseline", 400.0), self._capacity("shipinfer", 1000.0), target=5.0
        )
        assert "Speed-up: 2.50x" in text
        assert "NOT MET" in text

    def test_a_floor_over_a_capacity_is_a_floor_and_can_meet_the_target(self):
        """The regime a real run lands in: the baseline hits its wall at 200 while we are
        still keeping up with everything offered. The true ratio is 5x *or better*, and a
        floor above the target is enough to claim it."""
        text = run_bench.compare(
            self._capacity("baseline", 200.0), self._floor("shipinfer", 1000.0), target=5.0
        )
        headline = text.split("Speed-up:")[1].split("\n")[0]
        assert ">= 5.00x" in headline
        assert "MET" in headline and "NOT MET" not in headline

    def test_a_floor_below_the_target_is_inconclusive_not_a_failure(self):
        """A floor can meet a target but cannot miss one — the answer is more load."""
        text = run_bench.compare(
            self._capacity("baseline", 500.0), self._floor("shipinfer", 1000.0), target=5.0
        )
        headline = text.split("Speed-up:")[1].split("\n")[0]
        assert ">= 2.00x" in headline
        assert "INCONCLUSIVE" in headline
        assert "NOT MET" not in headline

    def test_a_capacity_over_a_floor_is_a_ceiling_and_can_only_miss(self):
        """We saturated and they did not: their number is a floor, so the ratio is an
        over-estimate. It is enough to fail a target and never enough to pass one."""
        text = run_bench.compare(
            self._floor("baseline", 500.0), self._capacity("shipinfer", 900.0), target=5.0
        )
        headline = text.split("Speed-up:")[1].split("\n")[0]
        assert "<= 1.80x" in headline
        assert "NOT MET" in headline

    def test_a_ceiling_above_the_target_still_claims_nothing(self):
        text = run_bench.compare(
            self._floor("baseline", 100.0), self._capacity("shipinfer", 900.0), target=5.0
        )
        headline = text.split("Speed-up:")[1].split("\n")[0]
        assert "<= 9.00x" in headline
        assert "INCONCLUSIVE" in headline

    def test_two_floors_are_an_artefact_of_the_offered_rate(self):
        """The case the harness actually produced every time, printed as 1.00x NOT MET.

        Both sides are handed the same load by construction, so when neither saturates the
        ratio is 1.00 no matter how much headroom either has. That is a fact about the
        offer, not about either system, and the report has to say so and point at --sweep.
        """
        text = run_bench.compare(
            self._floor("baseline", 1000.0), self._floor("shipinfer", 1000.0), target=5.0
        )
        assert "NOT AVAILABLE" in text
        assert "--sweep" in text
        assert "1.00x" not in text

    def test_an_unmeasured_side_yields_no_ratio(self):
        text = run_bench.compare(
            self._throughput("baseline", None, analysis.UNMEASURED),
            self._floor("shipinfer", 1000.0),
            target=5.0,
        )
        assert "NOT AVAILABLE" in text

    def test_a_capped_run_is_unmeasured_whatever_its_number(self):
        """A capped buffer sheds instead of growing, so its slope is not a rate — and the
        number it carries is the queue's capacity, not the system's."""
        capped = self._throughput("shipinfer", 999.0, analysis.UNMEASURED)

        assert capped.kind == run_bench.NOTHING
        assert not capped.is_rate
        assert "NOT AVAILABLE" in run_bench.compare(
            self._capacity("baseline", 100.0), capped, target=5.0
        )


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


class TestDroppedWorkIsNotCountedAsThroughput:
    """A frame backpressure refused never entered the queue, so it cannot be retired."""

    def _result(self, *, read: int, dropped: int, elapsed: float = 30.0):
        return shipinfer_harness.ShipInferResult(
            log=Path("/dev/null"),
            startup_s=1.0,
            elapsed_s=elapsed,
            frames_accepted=read - dropped,
            events_emitted=read - dropped,
            frames_read=read,
            frames_dropped=dropped,
        )

    def test_the_offered_rate_excludes_what_was_refused(self):
        """Read 1000/s, refuse 700/s: the load the pipeline saw is 300/s, and a flat buffer
        used to make that report as 1000 img/s SUSTAINED — a 3.3x overstatement, one-sided
        because the baseline is a separate binary measured a different way."""
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        result = self._result(read=30000, dropped=21000)

        assert shipinfer_harness.achieved_offer(config, result) == pytest.approx(300.0)

    def test_heavy_dropping_refuses_the_run_outright(self):
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)

        with pytest.raises(RuntimeError, match="shedding system"):
            shipinfer_harness.check_offer(config, self._result(read=30000, dropped=21000))

    def test_a_trickle_of_drops_is_still_a_measurement(self):
        """A little shedding is the system working as designed."""
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        shipinfer_harness.check_offer(config, self._result(read=30000, dropped=150))


class TestACappedBufferIsNotAMeasurement:
    """Once occupancy sits at the bound the queue sheds instead of growing."""

    def test_a_buffer_at_its_capacity_forces_unmeasured(self, tmp_path: Path):
        """A genuinely saturated run used to report SUSTAINED once the cap was reached,
        turning the queue's capacity into a throughput result."""
        rows = [{"t": float(t), "det_buffer_size": 1000.0} for t in range(30)]
        run = analysis.analyse(
            analysis.read_log(write_log(tmp_path / "l.jsonl", rows), sample_interval_s=1.0),
            system="baseline",
            warmup_s=5.0,
            offered={"det": 500.0},
            capacity=1000,
            entry_modules=("det",),
        )

        assert run.modules[0].capped
        assert run.verdict == analysis.UNMEASURED

    def test_a_buffer_well_below_capacity_is_measured_normally(self, tmp_path: Path):
        rows = [{"t": float(t), "det_buffer_size": 40.0} for t in range(30)]
        run = analysis.analyse(
            analysis.read_log(write_log(tmp_path / "l.jsonl", rows), sample_interval_s=1.0),
            system="baseline",
            warmup_s=5.0,
            offered={"det": 500.0},
            capacity=65536,
            entry_modules=("det",),
        )

        assert not run.modules[0].capped
        assert run.verdict == analysis.SUSTAINED


class TestRefusedWorkCannotBecomeThroughput:
    """The failure a reviewer reproduced end to end: a fabricated 5x out of refusals.

    When the scheduler rejects work rather than queueing it, nothing accumulates. The
    pipeline queue stays flat because frames are being refused; a model's queue plateaus at
    its own bound rather than growing. Both slopes read zero, so `offered - growth` publishes
    the entire offered rate as sustained throughput — and the bias runs toward us, because
    the baseline is a separate binary that is not measured this way.

    Three independent guards, because each catches a different shape of it.
    """

    def _result(self, **overrides):
        base = {
            "log": Path("/dev/null"),
            "startup_s": 1.0,
            "elapsed_s": 30.0,
            "frames_accepted": 30000,
            "events_emitted": 30000,
            "frames_read": 30000,
            "frames_dropped": 0,
            "requests_total": {},
            "requests_rejected": {},
        }
        base.update(overrides)
        return shipinfer_harness.ShipInferResult(**base)

    def test_heavy_scheduler_rejection_refuses_the_run(self):
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        result = self._result(requests_rejected={"ship_detector": 20000.0})

        with pytest.raises(RuntimeError, match="refused"):
            shipinfer_harness.check_offer(config, result)

    def test_a_trickle_of_rejection_is_still_a_measurement(self):
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)
        shipinfer_harness.check_offer(
            config, self._result(requests_rejected={"ship_detector": 100.0})
        )

    def test_a_claim_far_above_the_emitted_rate_is_refused(self):
        """The decisive check: an event either came out of the pipeline or it did not."""
        result = self._result(events_emitted=7500)  # 250/s out, 1000/s claimed

        with pytest.raises(RuntimeError, match="came out of the pipeline"):
            shipinfer_harness.reconcile(result, claimed=1000.0)

    def test_a_claim_matching_the_emitted_rate_passes(self):
        shipinfer_harness.reconcile(self._result(), claimed=1000.0)

    def test_a_small_gap_is_tolerated(self):
        """Reassembly holds a frame briefly, so the two counts never match exactly."""
        shipinfer_harness.reconcile(self._result(events_emitted=28000), claimed=1000.0)

    def test_the_claim_is_not_silently_clamped(self):
        """Substituting the emitted rate would hide that the two estimates disagreed."""
        with pytest.raises(RuntimeError) as caught:
            shipinfer_harness.reconcile(self._result(events_emitted=1500), claimed=1000.0)

        assert "1000.0" in str(caught.value) and "50.0" in str(caught.value)


class TestTheCapacityGuardUsesTheRightBound:
    """One capacity for every module could never trip for the queues that saturate first."""

    def test_a_model_queue_bound_is_its_instances_times_the_per_instance_size(self):
        """Keyed by the module names the ShipInfer log actually carries. Keying off
        `instances_per_gpu` used the *baseline's* vocabulary — `det`, `seg` — so the mapping
        matched nothing our side samples and the guard was dead for every queue it was
        added for. This test pinned the wrong key and stayed green through that."""
        config = BenchConfig(gpus=(2, 3, 4, 5))

        capacities = shipinfer_harness.per_module_capacity(config)

        assert capacities["ship_detector"] == 64 * 2 * 4, "8 instances x the 64-deep default"
        assert capacities["ship_segmenter"] == 64 * 1 * 4

    def test_the_pipeline_queue_keeps_its_own_much_larger_bound(self):
        config = BenchConfig(buffer_capacity=65536)

        capacities = shipinfer_harness.per_module_capacity(config)

        assert capacities[shipinfer_harness.PIPELINE_MODULE] == 65536

    def test_the_two_bounds_differ_by_orders_of_magnitude(self):
        """Which is why using one for both made the plateau guard unreachable."""
        capacities = shipinfer_harness.per_module_capacity(BenchConfig(gpus=(2, 3, 4, 5)))

        assert capacities["ship_detector"] * 100 < capacities[shipinfer_harness.PIPELINE_MODULE]

    def test_an_unnamed_module_gets_no_guard_rather_than_the_wrong_one(self, tmp_path: Path):
        run = analysis.analyse(
            analysis.read_log(
                write_log(
                    tmp_path / "l.jsonl",
                    [{"t": float(t), "mystery_buffer_size": 10.0} for t in range(30)],
                ),
                sample_interval_s=1.0,
            ),
            system="shipinfer",
            warmup_s=5.0,
            offered={"mystery": 100.0},
            capacity={"pipeline": 65536},
            entry_modules=("mystery",),
        )

        assert run.modules[0].capacity is None
        assert not run.modules[0].capped


class TestAnUnmeasurableRunIsNotARate:
    """UNMEASURED is the verdict the analysis raises to say "this cannot support a number".

    `is_rate` was `not saturated`, so UNMEASURED — the verdict for a buffer pegged at its
    bound, or a fit that could not be bounded — read as a rate. A pipeline queue at
    65000/65536 is shedding and certainly saturated, and it printed
    `Speed-up: 5.26x (MET)` with no bound label.
    """

    def _throughput(self, verdict: str):
        return run_bench.SystemThroughput("shipinfer", 1000.0, False, None, "test", verdict)

    def test_unmeasured_is_not_a_rate(self):
        assert not self._throughput(analysis.UNMEASURED).is_rate

    def test_sustained_and_draining_are(self):
        assert self._throughput(analysis.SUSTAINED).is_rate
        assert self._throughput(analysis.DRAINING).is_rate

    def test_an_unknown_future_verdict_is_not(self):
        """An allow-list, so a verdict added later does not silently become a rate."""
        assert not self._throughput("SOMETHING_NEW").is_rate

    def test_compare_refuses_to_divide_an_unmeasured_run(self):
        text = run_bench.compare(
            run_bench.SystemThroughput("baseline", 200.0, False, None, "t", analysis.SUSTAINED),
            self._throughput(analysis.UNMEASURED),
            target=5.0,
        )

        assert "NOT AVAILABLE" in text
        assert "5.00x" not in text
        assert "UNMEASURED" in text, "and it names why"


class TestEverySampledModuleHasACapacity:
    """The plateau guard was keyed on the baseline's vocabulary and matched nothing.

    `instances_per_gpu` names `det` and `seg`; the ShipInfer log carries `pipeline` plus
    `GRAPH_MODELS`. So `_capacity_for` returned None for all four model queues, `capped` was
    permanently False, and a detector queue at its 512-deep bound read SUSTAINED — the exact
    failure the function's docstring says it exists to prevent, landed on the wrong keys.
    """

    def test_no_sampled_module_is_left_without_a_bound(self):
        config = BenchConfig(gpus=(2, 3, 4, 5))

        capacities = shipinfer_harness.per_module_capacity(config)

        sampled = (shipinfer_harness.PIPELINE_MODULE, *shipinfer_harness.GRAPH_MODELS)
        missing = [name for name in sampled if name not in capacities]
        assert not missing, f"these queues have no plateau guard: {missing}"

    def test_a_model_queue_at_its_bound_is_now_caught(self, tmp_path: Path):
        capacity = shipinfer_harness.per_module_capacity(BenchConfig(gpus=(2, 3, 4, 5)))
        rows = [
            {"t": float(t), "pipeline_buffer_size": 30.0, "ship_detector_buffer_size": 512.0}
            for t in range(30)
        ]
        run = analysis.analyse(
            analysis.read_log(write_log(tmp_path / "l.jsonl", rows), sample_interval_s=1.0),
            system="shipinfer",
            warmup_s=5.0,
            offered={"pipeline": 1000.0, "ship_detector": 1000.0},
            capacity=capacity,
            entry_modules=("pipeline",),
        )

        assert run.verdict == analysis.UNMEASURED, "a pegged detector queue is not sustained"
        assert not run_bench.system_throughput(run).is_rate


class TestADeliveryOfNothingIsRefused:
    def _empty(self):
        return shipinfer_harness.ShipInferResult(
            log=Path("/dev/null"),
            startup_s=1.0,
            elapsed_s=30.0,
            frames_accepted=0,
            events_emitted=0,
            frames_read=0,
            frames_dropped=0,
        )

    def test_the_achieved_offer_is_zero_not_the_target(self):
        """Returning the target made the measurement equal what it was compared against."""
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)

        assert shipinfer_harness.achieved_offer(config, self._empty()) == 0.0

    def test_check_offer_refuses_it(self):
        """The 50-camera run where every per-device counter stayed at zero used to pass."""
        config = BenchConfig(cameras=50, fps=20.0, seconds=30.0)

        with pytest.raises(RuntimeError, match="no delivered frames"):
            shipinfer_harness.check_offer(config, self._empty())


class TestTheSweepClimbsUntilSomethingSaturates:
    """One offered rate cannot settle the comparison, so `--sweep` climbs.

    Both systems are handed the same load by construction. If neither saturates, both report
    that load back and the ratio is 1.00 however much headroom either has — a fact about the
    offer, not about either system. The ladder's job is to reach the rung where somebody's
    buffer grows, because that rung is the only place a capacity is measurable.
    """

    def _throughput(self, kind: str, value: float):
        verdict = analysis.SATURATED if kind == run_bench.CAPACITY else analysis.SUSTAINED
        return run_bench.SystemThroughput(
            "shipinfer", value, kind == run_bench.CAPACITY, None, "test", verdict
        )

    def _ladder(self, monkeypatch, outcomes):
        """Drive `sweep_system` over a scripted sequence, recording the rungs it ran."""
        seen: list[float] = []

        def measure(cfg, _out_dir):
            seen.append(cfg.fps)
            kind, value = outcomes[len(seen) - 1]
            return object(), self._throughput(kind, value)

        monkeypatch.setitem(run_bench.MEASURE, "shipinfer", measure)
        return seen

    def test_it_stops_at_the_first_saturated_rung(self, monkeypatch, tmp_path: Path):
        """The rung that saturated *is* the capacity. Climbing past it only saturates
        harder, and on a shared box every extra rung is minutes of four GPUs spent to
        learn nothing."""
        seen = self._ladder(
            monkeypatch,
            [
                (run_bench.FLOOR, 250.0),
                (run_bench.FLOOR, 500.0),
                (run_bench.CAPACITY, 780.0),
                (run_bench.CAPACITY, 790.0),
            ],
        )
        cfg = BenchConfig(cameras=50, fps=20.0, out_dir=tmp_path)

        runs, throughput = run_bench.sweep_system(
            "shipinfer", cfg, tmp_path, [0.25, 0.5, 1.0, 2.0]
        )

        assert seen == [5.0, 10.0, 20.0], "it ran a rung past the one that saturated"
        assert len(runs) == 3
        assert throughput.kind == run_bench.CAPACITY
        assert throughput.images_per_s == 780.0

    def test_rungs_are_climbed_low_to_high_whatever_order_they_arrive_in(
        self, monkeypatch, tmp_path: Path
    ):
        """Descending would saturate on the first rung and report the top of the ladder as
        the capacity, which is the ladder's own number rather than the system's."""
        seen = self._ladder(
            monkeypatch, [(run_bench.FLOOR, 250.0), (run_bench.CAPACITY, 600.0)]
        )
        cfg = BenchConfig(cameras=50, fps=20.0, out_dir=tmp_path)

        run_bench.sweep_system("shipinfer", cfg, tmp_path, [2.0, 0.5, 0.25])

        assert seen == [5.0, 10.0]

    def test_a_ladder_nobody_saturates_reports_the_highest_floor(
        self, monkeypatch, tmp_path: Path
    ):
        """Not promoted to a capacity. The system kept up with everything it was offered,
        which says its capacity is above the top rung and nothing more."""
        self._ladder(
            monkeypatch,
            [(run_bench.FLOOR, 250.0), (run_bench.FLOOR, 500.0), (run_bench.FLOOR, 1000.0)],
        )
        cfg = BenchConfig(cameras=50, fps=20.0, out_dir=tmp_path)

        _, throughput = run_bench.sweep_system("shipinfer", cfg, tmp_path, [0.25, 0.5, 1.0])

        assert throughput.kind == run_bench.FLOOR
        assert throughput.images_per_s == 1000.0

    def test_each_rung_gets_its_own_directory(self, monkeypatch, tmp_path: Path):
        """Two rungs writing one occupancy log would fit a line through both of them."""
        dirs: list[Path] = []

        def measure(_cfg, out_dir):
            dirs.append(out_dir)
            return object(), self._throughput(run_bench.FLOOR, 100.0)

        monkeypatch.setitem(run_bench.MEASURE, "shipinfer", measure)
        cfg = BenchConfig(cameras=50, fps=20.0, out_dir=tmp_path)

        run_bench.sweep_system("shipinfer", cfg, tmp_path, [0.5, 1.0])

        assert len(set(dirs)) == 2


class TestARungIsTheSameExperimentAtADifferentLoad:
    def test_the_rate_scales_with_the_multiplier(self):
        cfg = BenchConfig(cameras=50, fps=20.0)

        assert cfg.at_offer(0.5).offered_total == 500.0
        assert cfg.at_offer(2.0).offered_total == 2000.0

    def test_the_camera_count_does_not_move(self):
        """Cameras are the topology — source workers, queue lanes, how the baseline's two
        queues are fed. Scaling them changes the experiment rather than the load."""
        cfg = BenchConfig(cameras=50, fps=20.0)

        rung = cfg.at_offer(0.25)

        assert rung.cameras == 50
        assert rung.fps == 5.0

    def test_a_rung_offering_nothing_is_refused(self):
        with pytest.raises(ValueError, match="offer something"):
            BenchConfig(cameras=50, fps=20.0).at_offer(0)


class TestEveryRateUsesTheWindowTheFitUses:
    """Offered rate and growth rate have to describe the same seconds.

    The growth fit skips `t < warmup_s`; the counters were cumulative from process start and
    were divided by the whole run. With 50 cameras staggering their first decode and four
    GPUs' worth of instances deserialising engines through the warmup, a run that offered a
    clean 1000 img/s in steady state measured 60000/70 = 857 — which either raised
    `check_offer`'s "never offered the load" and destroyed a good run, or understated
    capacity by 143 img/s inside `sustained = offered - growth`.
    """

    def _result(self, **overrides):
        base = {
            "log": Path("unused.jsonl"),
            "startup_s": 1.0,
            "elapsed_s": 70.0,
            "frames_accepted": 60_000,
            "events_emitted": 60_000,
            "frames_read": 60_000,
            "frames_dropped": 0,
            "steady_s": 60.0,
            "steady_frames_read": 60_000,
            "steady_frames_dropped": 0,
            "steady_events_emitted": 60_000,
        }
        base.update(overrides)
        return shipinfer_harness.ShipInferResult(**base)

    def test_the_warmup_shortfall_does_not_depress_the_offered_rate(self):
        """Nothing read in the first 10 s, a clean 1000 img/s for the next 60."""
        cfg = BenchConfig(cameras=50, fps=20.0, seconds=70.0, warmup_s=10.0)

        rate = shipinfer_harness.achieved_offer(cfg, self._result())

        assert rate == pytest.approx(1000.0), "the warmup was charged against steady state"

    def test_rating_the_whole_run_is_what_produced_857(self):
        """The arithmetic that made this worth fixing, pinned so the regression is legible.

        60 000 frames is a clean 1000 img/s over the 60 s steady window and 857 img/s over
        the 70 s run — a 14% understatement, silently, on every measurement the harness
        makes.
        """
        whole_run = 60_000 / 70.0
        assert pytest.approx(857.1, abs=0.1) == whole_run

    def test_a_run_that_never_reached_its_warmup_boundary_rates_the_whole_window(self):
        """`BenchConfig` refuses a warmup longer than the run, so this can only come from a
        run that ended early. Rate what there is rather than divide by zero — and note it
        still cannot invent a number: `steady_frames_read` is 0, so the fallback reads the
        cumulative counters, which is the honest thing to do with the only data there is."""
        cfg = BenchConfig(cameras=50, fps=20.0, seconds=70.0, warmup_s=10.0)

        rate = shipinfer_harness.achieved_offer(
            cfg,
            self._result(elapsed_s=5.0, steady_s=0.0, frames_read=5_000, steady_frames_read=0),
        )

        assert rate == pytest.approx(1000.0)

    def test_a_run_with_no_window_at_all_reports_nothing(self):
        cfg = BenchConfig(cameras=50, fps=20.0, seconds=70.0, warmup_s=10.0)

        rate = shipinfer_harness.achieved_offer(cfg, self._result(elapsed_s=0.0, steady_s=0.0))
        assert rate == 0.0

    def test_downstream_models_are_rated_over_the_steady_window_too(self):
        """`requests_total` is cumulative from process start, so dividing by the whole run
        charged every model for the warmup it spent waiting on engines to deserialise."""
        cfg = BenchConfig(cameras=50, fps=20.0, seconds=70.0, warmup_s=10.0)
        result = self._result(
            requests_total={"person_embedder": 600_000.0},
            steady_requests_total={"person_embedder": 600_000.0},
        )

        rates = shipinfer_harness.offered_rates(cfg, result)

        assert rates["person_embedder"] == pytest.approx(10_000.0)

    def test_reconcile_compares_two_rates_from_the_same_window(self):
        """Averaging the emitted rate over a run that includes start-up would forgive a
        disagreement that is real — which is the one thing this check exists to catch."""
        # 60 000 events over the 60 s steady window is 1000/s: a 1000 img/s claim agrees.
        shipinfer_harness.reconcile(self._result(), 1000.0)

        # The same events over the whole 70 s would read 857/s and refuse it.
        with pytest.raises(RuntimeError, match="does not support a throughput claim"):
            shipinfer_harness.reconcile(self._result(steady_events_emitted=30_000), 1000.0)


class TestThePlateauGuardUsesTheRunsOwnInstanceCount:
    """A bound transcribed from a config file is a copy that will go stale.

    The guard exists to notice a queue sitting *at* its bound — such a queue stops growing,
    so its slope stops meaning anything and its offered rate would publish as throughput.
    Comparing against a hardcoded 512 when the real bound is 256 defeats it exactly.
    """

    def test_the_bound_follows_the_instances_the_run_started(self):
        cfg = BenchConfig(cameras=50, fps=20.0, gpus=(2, 3, 4, 5))

        capacities = shipinfer_harness.per_module_capacity(
            cfg, instances={"ship_detector": 4, "person_embedder": 16}
        )

        assert capacities["ship_detector"] == 4 * 64, "4 instances x max_queue_size"
        assert capacities["person_embedder"] == 16 * 64

    def test_halving_the_instances_halves_the_bound(self):
        """The reviewer's case: `count: 1` on the detector makes the real bound 256 while a
        transcribed table still says 512."""
        cfg = BenchConfig(cameras=50, fps=20.0, gpus=(2, 3, 4, 5))

        eight = shipinfer_harness.per_module_capacity(cfg, instances={"ship_detector": 8})
        four = shipinfer_harness.per_module_capacity(cfg, instances={"ship_detector": 4})

        assert eight["ship_detector"] == 2 * four["ship_detector"]

    def test_without_a_count_it_falls_back_to_the_config_defaults(self):
        """The fallback still has to be right, because a caller that forgets is worse off
        with `None` than with a slightly stale number."""
        cfg = BenchConfig(cameras=50, fps=20.0, gpus=(2, 3, 4, 5))

        capacities = shipinfer_harness.per_module_capacity(cfg)

        assert capacities["ship_detector"] == 2 * 4 * 64
        assert capacities["ship_segmenter"] == 1 * 4 * 64

    def test_every_sampled_module_gets_a_bound(self):
        """A model added to the graph and left out of the mapping loses its guard silently."""
        cfg = BenchConfig(cameras=50, fps=20.0, gpus=(2, 3))

        capacities = shipinfer_harness.per_module_capacity(cfg, instances={})

        for module in (shipinfer_harness.PIPELINE_MODULE, *shipinfer_harness.GRAPH_MODELS):
            assert capacities.get(module), f"{module} has no bound"


class TestARungFailureCostsOneRung:
    """A refusal on a high rung must not discard the rungs already measured.

    With the load generator's own ceiling this is the reachable case, not a hypothetical:
    the top rung raises, and the lower rungs — the point of the ladder — used to go with it,
    along with `summary.json`.
    """

    def _measure(self, outcomes):
        """Scripted per-rung outcomes: a float sustains, an exception is raised."""
        seen = []

        def measure(cfg, _out_dir):
            outcome = outcomes[len(seen)]
            seen.append(cfg.fps)
            if isinstance(outcome, Exception):
                raise outcome
            return object(), run_bench.SystemThroughput(
                "shipinfer", outcome, False, None, "test", analysis.SUSTAINED
            )

        return measure, seen

    def test_the_measured_rungs_survive_a_later_refusal(self, monkeypatch, tmp_path: Path):
        measure, seen = self._measure(
            [60.0, 120.0, RuntimeError("the load generator delivered 87 img/s of 240")]
        )
        monkeypatch.setitem(run_bench.MEASURE, "shipinfer", measure)
        cfg = BenchConfig(cameras=12, fps=10.0, out_dir=tmp_path)

        runs, throughput = run_bench.sweep_system("shipinfer", cfg, tmp_path, [0.5, 1.0, 2.0])

        assert len(seen) == 3, "it should have attempted the failing rung"
        assert len(runs) == 2, "the two good rungs were discarded"
        assert throughput.images_per_s == 120.0
        assert throughput.kind == run_bench.FLOOR

    def test_a_short_window_is_caught_too(self, monkeypatch, tmp_path: Path):
        """`analyse` raises `ValueError`, not `RuntimeError`, when the steady window holds
        fewer samples than a fit needs — a different exception for the same situation."""
        measure, _ = self._measure([60.0, ValueError("leaves 0 of 1 sample(s)")])
        monkeypatch.setitem(run_bench.MEASURE, "shipinfer", measure)
        cfg = BenchConfig(cameras=12, fps=10.0, out_dir=tmp_path)

        runs, throughput = run_bench.sweep_system("shipinfer", cfg, tmp_path, [1.0, 2.0])

        assert len(runs) == 1
        assert throughput.images_per_s == 60.0

    def test_a_first_rung_failure_yields_no_measurement_rather_than_a_crash(
        self, monkeypatch, tmp_path: Path
    ):
        measure, _ = self._measure([RuntimeError("nothing delivered")])
        monkeypatch.setitem(run_bench.MEASURE, "shipinfer", measure)
        cfg = BenchConfig(cameras=12, fps=10.0, out_dir=tmp_path)

        runs, throughput = run_bench.sweep_system("shipinfer", cfg, tmp_path, [1.0])

        assert runs == []
        assert throughput.kind == run_bench.NOTHING


class TestTheRunRecordsTheBoxItRanOn:
    """A number taken on a contended box is not comparable with one taken on an idle one,
    and the two systems are not equally affected: ours is CPU-bound Python and the baseline
    is a GPU-bound C++ binary, so a noisy neighbour depresses ours much more."""

    def test_the_load_average_is_in_the_metadata(self):
        meta = BenchConfig(cameras=50, fps=20.0).as_dict()

        assert len(meta["load_average"]) == 3
        assert all(isinstance(v, float) for v in meta["load_average"])
        assert meta["cpu_count"] and meta["cpu_count"] > 0

    def test_a_quiet_box_gets_a_plain_line(self, monkeypatch):
        monkeypatch.setattr(run_bench.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
        monkeypatch.setattr(run_bench.os, "cpu_count", lambda: 48)

        note = run_bench.load_note(BenchConfig(cameras=50, fps=20.0))

        assert "48 cpus" in note
        assert "BUSY" not in note

    def test_a_busy_box_says_so_in_the_run_output(self, monkeypatch):
        """Silently recording it in JSON is not enough — the person reading the console is
        the one about to quote the ratio."""
        monkeypatch.setattr(run_bench.os, "getloadavg", lambda: (35.0, 34.0, 33.0))
        monkeypatch.setattr(run_bench.os, "cpu_count", lambda: 48)

        note = run_bench.load_note(BenchConfig(cameras=50, fps=20.0))

        assert "BUSY" in note
        assert "indicative only" in note


class TestTheBaselinesOfferIsLabelledAsAsserted:
    """The one asymmetry the harness cannot close from outside, stated on every run.

    Our offered rate is measured from ingest counters and gated at 98%; the baseline's is
    read off its own configuration, because its log carries buffer depths and no arrival
    counter and it is run unchanged on purpose.
    """

    def _capacity(self, system: str, value: float):
        return run_bench.SystemThroughput(system, value, True, None, "test", analysis.SATURATED)

    def test_the_comparison_says_the_baseline_offer_was_not_measured(self):
        text = run_bench.compare(
            self._capacity("baseline", 868.0), self._capacity("shipinfer", 81.0), target=5.0
        )

        assert "asserted from its configuration, not" in text
        assert "errs in the baseline's favour" in text

    def test_it_is_omitted_when_there_is_no_baseline_number_to_qualify(self):
        unmeasured = run_bench.SystemThroughput(
            "baseline", None, False, None, "test", analysis.UNMEASURED
        )

        text = run_bench.compare(unmeasured, self._capacity("shipinfer", 81.0), target=5.0)

        assert "errs in the baseline's favour" not in text
