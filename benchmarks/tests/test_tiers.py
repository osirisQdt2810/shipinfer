"""The algo and kernel tiers, tested where they can be: offline, on their arithmetic.

R44 asks for three benchmark tiers and only the system one existed. These two are new, and
the lesson from the system tier is that **the arithmetic is where a benchmark lies** — every
defect review found in `run_bench.py` was a formula that produced a plausible number from a
run that did not support it, not a broken measurement loop. So the formulas are pinned here,
with no GPU and no engines, exactly as `test_comparison_metric.py` pins the system tier's.
"""

from __future__ import annotations

import pytest

from benchmarks import kernels, stages
from benchmarks.harness.config import BenchConfig


class TestTheKernelTierReportsWhatItCouldNotMeasure:
    """A shorter table with no explanation is how "we never measured it" becomes "it is not
    faster". Every implementation that did not run has to appear with a reason."""

    def test_an_implementation_that_cannot_be_constructed_is_reported(self) -> None:
        result = kernels.OpResult(op="letterbox")
        result.skipped["native"] = "the fused kernels are unavailable"

        text = kernels.render([result])

        assert "native" in text
        assert "skipped" in text
        assert "the fused kernels are unavailable" in text

    def test_a_missing_baseline_yields_no_ratio_rather_than_a_wrong_one(self) -> None:
        """numpy is the readable implementation a fused one has to beat. Without it there is
        nothing to be faster *than*, and inventing a denominator is the failure mode."""
        result = kernels.OpResult(op="letterbox")
        only = kernels.Measurement("letterbox", "torch", 0.001, 0.01, 10)
        result.measurements.append(only)

        assert result.baseline is None
        assert result.speedup(only) is None

    def test_the_ratio_is_against_numpy(self) -> None:
        result = kernels.OpResult(op="letterbox")
        base = kernels.Measurement("letterbox", "numpy", 0.004, 0.01, 10)
        fast = kernels.Measurement("letterbox", "torch", 0.001, 0.01, 10)
        result.measurements += [base, fast]

        assert result.speedup(fast) == pytest.approx(4.0)
        assert result.speedup(base) == pytest.approx(1.0)


class TestTheKernelTierFlagsNumbersItDoesNotTrust:
    """This box is shared and a microbenchmark is the first thing to stop being reproducible.
    The first run of the kernel tier was taken at load 41 of 48 with spreads reaching 76%."""

    def test_a_wide_spread_is_marked_noisy(self) -> None:
        result = kernels.OpResult(op="nms")
        result.measurements.append(kernels.Measurement("nms", "numpy", 0.001, 0.65, 10))

        assert "(noisy)" in kernels.render([result])

    def test_a_tight_spread_is_not(self) -> None:
        result = kernels.OpResult(op="nms")
        result.measurements.append(kernels.Measurement("nms", "numpy", 0.001, 0.02, 10))

        assert "(noisy)" not in kernels.render([result])

    def test_a_busy_host_says_so(self, monkeypatch) -> None:
        monkeypatch.setattr(kernels.os, "getloadavg", lambda: (40.0, 40.0, 40.0))
        monkeypatch.setattr(kernels.os, "cpu_count", lambda: 48)

        assert "BUSY" in kernels.load_note()

    def test_a_quiet_host_does_not(self, monkeypatch) -> None:
        monkeypatch.setattr(kernels.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
        monkeypatch.setattr(kernels.os, "cpu_count", lambda: 48)

        assert "BUSY" not in kernels.load_note()

    def test_the_report_says_a_kernel_ratio_is_not_a_system_ratio(self) -> None:
        """The one sentence that keeps this tier from being quoted as a speed-up: an op that
        is 2% of the frame budget caps out at 2% however fast it gets."""
        text = kernels.render([kernels.OpResult(op="nms")])

        assert "not a system speed-up" in text


class _FakeHistogram:
    """The two methods the profile reads, and nothing else."""

    def __init__(self, per_stage: dict[str, tuple[int, float, float]]) -> None:
        self._per_stage = per_stage

    def snapshot(self, **labels: str) -> tuple[int, float]:
        calls, p50, _ = self._per_stage.get(labels["stage"], (0, 0.0, 0.0))
        return calls, calls * p50

    def quantile(self, q: float, **labels: str) -> float:
        _calls, p50, p95 = self._per_stage.get(labels["stage"], (0, 0.0, 0.0))
        return p95 if q >= 0.9 else p50


class _FakeMetrics:
    def __init__(self, per_stage: dict[str, tuple[int, float, float]]) -> None:
        self.stage_latency_us = _FakeHistogram(per_stage)


class _FakeResult:
    def __init__(self, frames: int, read: int, steady_s: float) -> None:
        self.frames_accepted = frames
        self.steady_frames_read = read
        self.steady_s = steady_s
        self.elapsed_s = steady_s
        self.frames_read = read


class TestTheAlgoTierChargesEachStageWhatItActuallyCosts:
    """`calls_per_frame` is the whole point. A stage costing 8 ms per call that runs on one
    frame in three costs 2.7 ms per frame, and assuming one call per frame would overstate
    the cheap stages and understate the expensive ones by the same factor."""

    def _profile(self, per_stage, *, frames=100, read=100, steady=1.0, stages_run=None):
        config = BenchConfig(cameras=10, fps=10.0)
        return stages.profile_from(
            _FakeResult(frames, read, steady),
            _FakeMetrics(per_stage),
            config,
            stages_run or tuple(per_stage),
        )

    def test_a_conditional_branch_is_charged_pro_rata(self) -> None:
        # `ship_segmenter` ran on a third of the frames: 8 ms a call is 2.67 ms a frame.
        profile = self._profile({"ship_segmenter": (33, 8000.0, 9000.0)}, frames=100)

        (cost,) = profile.stages
        assert cost.calls_per_frame == pytest.approx(0.33)
        assert cost.per_frame_us == pytest.approx(2640.0)

    def test_a_stage_that_runs_more_than_once_a_frame_is_charged_more(self) -> None:
        """The embedders run once per object batch, not once per frame."""
        profile = self._profile({"person_embedder": (300, 1000.0, 1200.0)}, frames=100)

        (cost,) = profile.stages
        assert cost.calls_per_frame == pytest.approx(3.0)
        assert cost.per_frame_us == pytest.approx(3000.0)

    def test_a_stage_that_never_ran_is_omitted_not_zeroed(self) -> None:
        """A zero row reads as "free", which is a different claim from "did not run"."""
        profile = self._profile({"detect": (100, 500.0, 700.0), "absent": (0, 0.0, 0.0)})

        assert [c.stage for c in profile.stages] == ["detect"]

    def test_the_table_is_ordered_by_per_frame_cost(self) -> None:
        """The reader wants the expensive stage first; per *call* would put a rare, slow
        stage above a cheap one that runs fifteen times."""
        profile = self._profile(
            {
                "rare_but_slow": (10, 9000.0, 9000.0),  # 0.1/frame -> 900us
                "cheap_but_often": (1500, 200.0, 300.0),  # 15/frame -> 3000us
            },
            frames=100,
        )

        assert [c.stage for c in profile.stages] == ["cheap_but_often", "rare_but_slow"]

    def test_the_shares_sum_to_one(self) -> None:
        profile = self._profile(
            {"a": (100, 1000.0, 1000.0), "b": (100, 3000.0, 3000.0)}, frames=100
        )

        assert sum(profile.share(c) for c in profile.stages) == pytest.approx(1.0)


class TestTheAlgoTierRefusesToProfileASaturatedRun:
    """Under saturation a stage's latency includes the time it waited behind other frames, so
    a queueing artefact reads as an expensive stage. A profile wants service time."""

    def test_a_run_that_kept_up_is_not_warned_about(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)  # 100 img/s offered
        profile = stages.profile_from(
            _FakeResult(1000, 1000, 10.0),
            _FakeMetrics({"detect": (1000, 500.0, 600.0)}),
            config,
            ("detect",),
        )

        assert profile.achieved == pytest.approx(100.0)
        assert profile.kept_up
        assert "WARNING" not in stages.render(profile)

    def test_a_run_that_fell_behind_is_warned_about_loudly(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)  # 100 offered, 60 delivered
        profile = stages.profile_from(
            _FakeResult(600, 600, 10.0),
            _FakeMetrics({"detect": (600, 500.0, 600.0)}),
            config,
            ("detect",),
        )

        assert not profile.kept_up
        text = stages.render(profile)
        assert "WARNING" in text
        assert "queueing" in text

    def test_the_bar_is_the_same_98_percent_the_system_tier_uses(self) -> None:
        """One threshold, so a run the system tier accepts is one this tier profiles."""
        config = BenchConfig(cameras=10, fps=10.0)
        just_under = stages.profile_from(
            _FakeResult(979, 979, 10.0), _FakeMetrics({}), config, ()
        )
        just_over = stages.profile_from(
            _FakeResult(981, 981, 10.0), _FakeMetrics({}), config, ()
        )

        assert not just_under.kept_up
        assert just_over.kept_up


class TestSerialAgainstWallIsWhatConcurrencyBought:
    def test_the_serial_total_is_the_sum_of_the_per_frame_costs(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)
        profile = stages.profile_from(
            _FakeResult(100, 100, 1.0),
            _FakeMetrics({"a": (100, 1000.0, 0.0), "b": (100, 2000.0, 0.0)}),
            config,
            ("a", "b"),
        )

        assert profile.serial_per_frame_us == pytest.approx(3000.0)

    def test_the_wall_per_frame_is_the_run_divided_by_the_frames(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)
        profile = stages.profile_from(_FakeResult(100, 100, 1.0), _FakeMetrics({}), config, ())

        assert profile.wall_per_frame_us == pytest.approx(10_000.0)

    def test_a_run_with_no_frames_divides_by_nothing(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)
        profile = stages.profile_from(_FakeResult(0, 0, 1.0), _FakeMetrics({}), config, ())

        assert profile.wall_per_frame_us == 0.0
        assert profile.serial_per_frame_us == 0.0

    def test_the_report_explains_what_the_gap_means(self) -> None:
        """If serial and wall are close, adding workers will not help — that is the sentence
        this tier exists to let someone say."""
        config = BenchConfig(cameras=10, fps=10.0)
        text = stages.render(
            stages.profile_from(_FakeResult(100, 100, 1.0), _FakeMetrics({}), config, ())
        )

        assert "no concurrency at all" in text
