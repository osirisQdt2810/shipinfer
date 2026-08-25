"""The algo and kernel tiers, tested where they can be: offline, on their arithmetic.

R44 asks for three benchmark tiers and only the system one existed. These two are new, and
the lesson from the system tier is that **the arithmetic is where a benchmark lies** — every
defect review found in `run_bench.py` was a formula that produced a plausible number from a
run that did not support it, not a broken measurement loop. So the formulas are pinned here,
with no GPU and no engines, exactly as `test_comparison_metric.py` pins the system tier's.
"""

from __future__ import annotations

import numpy as np
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


def _cells(per_stage: dict[str, list[float]]) -> dict:
    """Real histograms, observed and read back the way the harness reads them.

    The first version of these tests used a fake whose `quantile` returned the exact value it
    was handed — which removed the bucket quantisation that *was* the defect (CONVENTIONS 2.9:
    a fake must not be wider than the contract). A real `Histogram` keeps the quantisation.
    """
    from benchmarks.harness.histograms import read_cell
    from shipinfer.core.metrics.histogram import Histogram

    histogram = Histogram("shipinfer_pipeline_stage_latency_us", "test")
    for stage, values in per_stage.items():
        for value in values:
            histogram.observe(value, stage=stage)
    return {stage: read_cell(histogram, stage=stage) for stage in per_stage}


class _Result:
    """The fields `profile_from` reads.

    The whole-run counters are deliberately *not* the steady ones — twice the frames, three
    times the duration — so a profile that reads the wrong window produces a wrong number
    rather than the right one by coincidence.
    """

    def __init__(
        self,
        *,
        frames: int,
        read: int,
        steady_s: float,
        cells: dict | None = None,
        whole_run: bool = False,
    ) -> None:
        self.steady_frames_accepted = frames
        self.steady_frames_read = read
        self.steady_s = steady_s
        self.steady_stage_latency = dict(cells or {})
        self.steady_is_whole_run = whole_run
        self.frames_accepted = frames * 2
        self.frames_read = read * 2
        self.elapsed_s = steady_s * 3


def _profile(per_stage, *, frames=100, read=100, steady=1.0, stages_run=None, whole_run=False):
    config = BenchConfig(cameras=10, fps=10.0)
    result = _Result(
        frames=frames, read=read, steady_s=steady, cells=_cells(per_stage), whole_run=whole_run
    )
    return stages.profile_from(result, None, config, stages_run or tuple(per_stage))


class TestTheAlgoTierChargesEachStageWhatItActuallyCosts:
    """`calls_per_frame` is the whole point. A stage costing 8 ms per call that runs on one
    frame in three costs 2.7 ms per frame, and assuming one call per frame would overstate
    the cheap stages and understate the expensive ones by the same factor."""

    def test_a_conditional_branch_is_charged_pro_rata(self) -> None:
        profile = _profile(
            {"detect": [1000.0] * 100, "ship_segmenter": [2000.0] * 50}, frames=100
        )
        by_name = {c.stage: c for c in profile.stages}

        assert by_name["ship_segmenter"].calls_per_frame == pytest.approx(0.5)
        assert by_name["ship_segmenter"].per_frame_us == pytest.approx(1000.0)

    def test_a_stage_that_runs_more_than_once_a_frame_is_charged_more(self) -> None:
        profile = _profile({"person_embedder": [1000.0] * 300}, frames=100)

        (cost,) = profile.stages
        assert cost.calls_per_frame == pytest.approx(3.0)
        assert cost.per_frame_us == pytest.approx(3000.0)

    def test_a_stage_that_never_ran_is_omitted_not_zeroed(self) -> None:
        profile = _profile({"detect": [1000.0] * 100}, stages_run=("detect", "crop"))

        assert [c.stage for c in profile.stages] == ["detect"]

    def test_the_table_is_ordered_by_per_frame_cost(self) -> None:
        profile = _profile(
            {"rare_but_slow": [5000.0] * 10, "cheap_but_often": [100.0] * 1000}, frames=100
        )

        assert [c.stage for c in profile.stages] == ["cheap_but_often", "rare_but_slow"]

    def test_the_cost_is_the_exact_mean_not_the_bucket_edge(self) -> None:
        """Two stages in one histogram bucket share a p50 and differ by 2.3x in cost. The
        first version charged both the bucket's upper edge and rendered the difference as a
        tie; the cost is `sum / count`, which the histogram carries exactly."""
        profile = _profile({"a": [1050.0] * 100, "b": [2400.0] * 100}, frames=100)
        by_name = {c.stage: c for c in profile.stages}

        assert by_name["a"].p50_us == by_name["b"].p50_us  # the bucket's resolution
        assert by_name["a"].per_frame_us == pytest.approx(1050.0)
        assert by_name["b"].per_frame_us == pytest.approx(2400.0)
        assert by_name["a"].mean_us == pytest.approx(1050.0)

    def test_shares_follow_the_exact_totals(self) -> None:
        """1000 us and 3000 us per frame are a quarter and three quarters. Charged by bucket
        edge they would be 1000 and 5000 — a sixth and five sixths — so this fails on p50."""
        profile = _profile({"a": [1000.0] * 100, "b": [3000.0] * 100}, frames=100)
        by_name = {c.stage: c for c in profile.stages}

        assert profile.share(by_name["a"]) == pytest.approx(0.25)
        assert profile.share(by_name["b"]) == pytest.approx(0.75)


class TestTheProfileReadsOneWindow:
    """A steady-window duration over a whole-run frame count is two windows in one number —
    the mistake `ShipInferResult.steady_*` exists to prevent, reintroduced by the first version
    of this tier. Every number here comes from the steady window, and the report says so."""

    def test_per_frame_cost_uses_the_steady_frame_count(self) -> None:
        profile = _profile({"detect": [1000.0] * 100}, frames=100)

        (cost,) = profile.stages
        assert profile.frames == 100  # not the whole-run 200
        assert cost.per_frame_us == pytest.approx(1000.0)  # not 500

    def test_wall_per_frame_uses_the_steady_window(self) -> None:
        profile = _profile({}, frames=100, read=100, steady=1.0)

        assert profile.wall_per_frame_us == pytest.approx(
            10_000.0
        )  # 1.0 s / 100, not 3.0 s / 200

    def test_the_window_is_named_in_the_report(self) -> None:
        profile = _profile({"detect": [1000.0] * 100})

        assert profile.window == stages.STEADY_WINDOW
        assert stages.STEADY_WINDOW in stages.render(profile)

    def test_a_run_shorter_than_its_warmup_says_so(self) -> None:
        profile = _profile({"detect": [1000.0] * 100}, whole_run=True)

        assert "whole run" in profile.window
        assert "whole run" in stages.render(profile)

    def test_the_steady_cells_are_the_difference_of_two_snapshots(self) -> None:
        """What the harness does at the warm-up boundary and the end, in miniature."""
        from benchmarks.harness.histograms import read_cell
        from shipinfer.core.metrics.histogram import Histogram

        histogram = Histogram("h", "test")
        for value in (100.0, 100.0, 100.0):  # warm-up: cheap
            histogram.observe(value, stage="detect")
        at_warmup = read_cell(histogram, stage="detect")
        for value in (2000.0, 2000.0):  # steady: what the profile must see
            histogram.observe(value, stage="detect")
        steady = read_cell(histogram, stage="detect").minus(at_warmup)

        assert steady.count == 2
        assert steady.mean == pytest.approx(2000.0)
        assert steady.quantile(0.5) >= 2000.0  # the bucket that holds 2000, not the 100s
        assert read_cell(histogram, stage="detect").mean == pytest.approx(860.0)  # whole run

    def test_a_snapshot_cannot_be_subtracted_from_an_earlier_one(self) -> None:
        from benchmarks.harness.histograms import read_cell
        from shipinfer.core.metrics.histogram import Histogram

        histogram = Histogram("h", "test")
        histogram.observe(100.0, stage="detect")
        earlier = read_cell(histogram, stage="detect")
        histogram.observe(100.0, stage="detect")
        later = read_cell(histogram, stage="detect")

        with pytest.raises(ValueError, match="only grows"):
            earlier.minus(later)


class TestTheAlgoTierRefusesToProfileASaturatedRun:
    """Under saturation a stage's latency includes the time it waited behind other frames, so
    a queueing artefact reads as an expensive stage. A profile wants service time."""

    def test_a_run_that_kept_up_is_not_warned_about(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)
        profile = stages.profile_from(
            _Result(frames=1000, read=1000, steady_s=10.0), None, config, ()
        )

        assert profile.kept_up
        assert "WARNING" not in stages.render(profile)

    def test_a_run_that_fell_behind_is_warned_about_loudly(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)
        profile = stages.profile_from(
            _Result(frames=600, read=600, steady_s=10.0), None, config, ()
        )

        assert not profile.kept_up
        assert "WARNING" in stages.render(profile)
        assert "queueing" in stages.render(profile)

    def test_the_bar_is_the_same_98_percent_the_system_tier_uses(self) -> None:
        config = BenchConfig(cameras=10, fps=10.0)  # offers 100 img/s
        below = stages.profile_from(
            _Result(frames=979, read=979, steady_s=10.0), None, config, ()
        )
        above = stages.profile_from(
            _Result(frames=981, read=981, steady_s=10.0), None, config, ()
        )

        assert not below.kept_up
        assert above.kept_up


class TestSerialAgainstWallIsWhatConcurrencyBought:
    def test_the_serial_total_is_the_sum_of_the_per_frame_costs(self) -> None:
        profile = _profile({"a": [1000.0] * 100, "b": [3000.0] * 100}, frames=100)

        assert profile.serial_per_frame_us == pytest.approx(4000.0)

    def test_the_wall_per_frame_is_the_run_divided_by_the_frames(self) -> None:
        profile = _profile({}, frames=100, read=100, steady=1.0)

        assert profile.wall_per_frame_us == pytest.approx(10_000.0)

    def test_a_run_with_no_frames_divides_by_nothing(self) -> None:
        profile = _profile({}, frames=0, read=0, steady=1.0)

        assert profile.wall_per_frame_us == 0.0
        assert profile.serial_per_frame_us == 0.0

    def test_the_report_explains_what_the_gap_means(self) -> None:
        text = stages.render(_profile({}, frames=100, read=100, steady=1.0))

        assert "serial per frame" in text
        assert "wall per frame" in text
        assert "cheaper stage" in text


class TestTheRunRecordsWhichSourceItMeasured:
    """R55 makes RTSP mandatory for the benchmark, not only for the tests — and the reason is
    that a replay run measures the inference plane with the **decode path removed**.

    A deployment reads fifty RTSP cameras, so NVDEC, the jitter buffer, reconnects and the
    NV12 conversion are part of the real cost and none of them appear in a replay run. A
    replay number is an upper bound on the RTSP one. The two must never be compared as though
    they measured the same thing, which is why the source is recorded rather than implied.
    """

    def test_replay_is_the_default_and_is_recorded(self) -> None:
        assert BenchConfig(cameras=4, fps=5.0).as_dict()["source"] == "replay"

    def test_an_rtsp_run_says_so_in_its_metadata(self) -> None:
        assert BenchConfig(cameras=4, fps=5.0, source="rtsp").as_dict()["source"] == "rtsp"

    def test_a_replay_camera_points_at_a_folder(self) -> None:
        from benchmarks.harness import shipinfer as harness

        cameras = harness._cameras(BenchConfig(cameras=4, fps=5.0))

        assert len(cameras) == 4
        assert all(c["source"] == "replay" for c in cameras)
        assert all(not c["uri"].startswith("rtsp://") for c in cameras)

    def test_an_rtsp_camera_points_at_the_local_server(self) -> None:
        from benchmarks.harness import shipinfer as harness

        cameras = harness._cameras(
            BenchConfig(cameras=4, fps=5.0, source="rtsp", rtsp_port=9001)
        )

        assert len(cameras) == 4
        # Person content on the configured port, ship content on the next one — the same
        # halves replay reads off disk (see `test_the_rtsp_split_matches_the_replay_split`).
        assert [c["uri"].startswith("rtsp://127.0.0.1:9001/") for c in cameras] == [
            True,
            True,
            False,
            False,
        ]
        assert all(c["uri"].startswith("rtsp://127.0.0.1:9002/") for c in cameras[2:])
        # Left unset on purpose: the point of an RTSP run is to exercise the registry's own
        # decoder selection, the way production does, rather than to pin one.
        assert all("source" not in c for c in cameras)

    def test_every_camera_gets_its_own_stream(self) -> None:
        """Fifty cameras sharing one URI would measure one decode fifty times."""
        from benchmarks.harness import shipinfer as harness

        cameras = harness._cameras(BenchConfig(cameras=8, fps=5.0, source="rtsp"))

        assert len({c["uri"] for c in cameras}) == 8

    def test_the_camera_ids_match_between_the_two_sources(self) -> None:
        """So a replay run and an RTSP run are comparable per camera, even though their
        totals are not comparable to each other."""
        from benchmarks.harness import shipinfer as harness

        replay = harness._cameras(BenchConfig(cameras=6, fps=5.0))
        rtsp = harness._cameras(BenchConfig(cameras=6, fps=5.0, source="rtsp"))

        assert [c["camera_id"] for c in replay] == [c["camera_id"] for c in rtsp]

    def test_the_rtsp_split_matches_the_replay_split(self) -> None:
        """The content split decides the crop fan-out, and the crop fan-out is the downstream
        load. One server fed with person frames starved the ship branch and the analysis
        blamed the detector — so the ship half must reach the ship server, camera for camera."""
        from benchmarks.harness import rtsp
        from benchmarks.harness import shipinfer as harness

        config = BenchConfig(cameras=6, fps=5.0, source="rtsp", rtsp_port=9001)
        resolved = config.resolved()
        replay = harness._cameras(BenchConfig(cameras=6, fps=5.0))
        streamed = harness._cameras(config)

        for index in range(6):
            is_person = replay[index]["uri"] == str(resolved.person_frames)
            expected_port = config.rtsp_port if is_person else rtsp.ship_port(config)
            assert streamed[index]["uri"].startswith(f"rtsp://127.0.0.1:{expected_port}/"), (
                index,
                streamed[index]["uri"],
            )
        assert sum(r["uri"] == str(resolved.person_frames) for r in replay) == 3
        assert sum(r["uri"] == str(resolved.ship_frames) for r in replay) == 3


class TestTheRtspServerIsRefusedRatherThanToleratedWhenItFails:
    """A run whose cameras cannot connect produces a clean-looking zero, and this project has
    already published one of those. So both failure modes raise at start-up rather than
    surfacing as a mysteriously empty measurement forty seconds later."""

    def test_a_replay_run_starts_nothing(self, monkeypatch) -> None:
        from benchmarks.harness import rtsp

        started: list[object] = []
        monkeypatch.setattr(rtsp.subprocess, "Popen", lambda *a, **_k: started.append(a))

        with rtsp.serving(BenchConfig(cameras=4, fps=5.0)):
            pass

        assert started == [], "a replay run must not start an RTSP server"

    def test_a_server_that_never_accepts_is_refused(self, monkeypatch, tmp_path) -> None:
        from benchmarks.harness import rtsp

        class _Alive:
            returncode = None
            stdout = None

            def poll(self) -> None:
                return None

            def terminate(self) -> None: ...

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None: ...

        monkeypatch.setattr(rtsp.subprocess, "Popen", lambda *_a, **_k: _Alive())
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: False)

        with (
            pytest.raises(RuntimeError, match="did not accept a connection"),
            rtsp.serving(
                BenchConfig(cameras=4, fps=5.0, source="rtsp", out_dir=tmp_path), timeout_s=0.5
            ),
        ):
            pass

    def test_a_server_that_exits_early_reports_its_output(self, monkeypatch, tmp_path) -> None:
        """The reason it died is the whole diagnosis — a missing GStreamer plugin, a port in
        use — and swallowing it costs an afternoon."""
        import io

        from benchmarks.harness import rtsp

        class _Dead:
            returncode = 1
            stdout = io.StringIO("gst_parse_launch: no element rtph264pay")

            def poll(self) -> int:
                return 1

            def terminate(self) -> None: ...

            def wait(self, timeout: float | None = None) -> int:
                return 1

            def kill(self) -> None: ...

        def popen(_argv, **kwargs):
            # The server writes to the file handle it was given, the way the real one does.
            kwargs["stdout"].write("gst_parse_launch: no element rtph264pay")
            return _Dead()

        monkeypatch.setattr(rtsp.subprocess, "Popen", popen)
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: False)

        with (
            pytest.raises(RuntimeError, match="no element rtph264pay"),
            rtsp.serving(
                BenchConfig(cameras=4, fps=5.0, source="rtsp", out_dir=tmp_path), timeout_s=5.0
            ),
        ):
            pass

    def test_the_server_is_stopped_even_when_the_run_raises(
        self, monkeypatch, tmp_path
    ) -> None:
        """A GLib loop left holding the port makes the *next* run fail with an address
        already in use, minutes later and nowhere near the cause."""
        from benchmarks.harness import rtsp

        stopped: list[str] = []

        class _Server:
            returncode = None
            stdout = None

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                stopped.append("terminate")

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                stopped.append("kill")

        monkeypatch.setattr(rtsp.subprocess, "Popen", lambda *_a, **_k: _Server())
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: True)

        with (
            pytest.raises(ValueError, match="the run failed"),
            rtsp.serving(BenchConfig(cameras=4, fps=5.0, source="rtsp", out_dir=tmp_path)),
        ):
            raise ValueError("the run failed")

        assert "terminate" in stopped

    def test_a_server_that_ignores_terminate_is_killed(self, monkeypatch, tmp_path) -> None:
        from benchmarks.harness import rtsp

        stopped: list[str] = []

        class _Stubborn:
            returncode = None
            stdout = None
            _terminated = False

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                stopped.append("terminate")
                self._terminated = True

            _killed = False

            def wait(self, timeout: float | None = None) -> int:
                if self._terminated and not self._killed:
                    raise rtsp.subprocess.TimeoutExpired("rtsp", timeout or 0)
                return 0

            def kill(self) -> None:
                stopped.append("kill")
                self._killed = True

        monkeypatch.setattr(rtsp.subprocess, "Popen", lambda *_a, **_k: _Stubborn())
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: True)

        with rtsp.serving(BenchConfig(cameras=4, fps=5.0, source="rtsp", out_dir=tmp_path)):
            pass

        # Two servers — person and ship content — each terminated, then killed.
        assert stopped == ["terminate", "kill", "terminate", "kill"]

    def test_each_content_half_is_served_from_its_own_directory(
        self, monkeypatch, tmp_path
    ) -> None:
        from benchmarks.harness import rtsp

        started: list[list[str]] = []

        class _Server:
            returncode = None
            stdout = None

            def poll(self) -> None:
                return None

            def terminate(self) -> None: ...

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None: ...

        def popen(argv, **_kwargs):
            started.append([str(a) for a in argv])
            return _Server()

        monkeypatch.setattr(rtsp.subprocess, "Popen", popen)
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: True)
        config = BenchConfig(
            cameras=6, fps=5.0, source="rtsp", rtsp_port=9001, out_dir=tmp_path
        )

        with rtsp.serving(config):
            pass

        resolved = config.resolved()
        by_port = {argv[argv.index("--port") + 1]: argv for argv in started}
        assert set(by_port) == {"9001", "9002"}
        person, ship = by_port["9001"], by_port["9002"]
        assert person[person.index("--data") + 1] == str(resolved.person_frames)
        assert ship[ship.index("--data") + 1] == str(resolved.ship_frames)
        assert person[person.index("--streams") + 1] == "3"
        assert ship[ship.index("--streams") + 1] == "3"

    def test_a_failing_server_is_named(self, monkeypatch, tmp_path) -> None:
        """Two servers means the message has to say which one died."""
        import io

        from benchmarks.harness import rtsp

        class _Dead:
            returncode = 1
            stdout = io.StringIO("Address already in use")

            def poll(self) -> int:
                return 1

            def terminate(self) -> None: ...

            def wait(self, timeout: float | None = None) -> int:
                return 1

            def kill(self) -> None: ...

        monkeypatch.setattr(rtsp.subprocess, "Popen", lambda *_a, **_k: _Dead())
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: False)

        with (
            pytest.raises(RuntimeError, match=r"person RTSP server \(port 9001\)"),
            rtsp.serving(
                BenchConfig(cameras=4, fps=5.0, source="rtsp", rtsp_port=9001, out_dir=tmp_path)
            ),
        ):
            pass


class TestTheKernelTierMeasuresWhatProductionRuns:
    def test_synchronisation_follows_on_device_not_the_description(self) -> None:
        """`runtime/ops/base.py`: callers branch on `on_device`. Sniffing `describe()` for
        "cuda" or "torch" worked for the two implementations that mention them and would have
        left any other device implementation timing its launches."""

        class _HostOpsThatTalksAboutCuda:
            on_device = False

            def describe(self) -> str:
                return "torch kernels on cuda:0"  # prose, not a device

        def call() -> int:
            return 1

        assert kernels._synchronised(_HostOpsThatTalksAboutCuda(), call) is call

    def test_inputs_are_seeded_and_shared_across_implementations(self) -> None:
        """NMS cost depends on the overlap structure of the boxes, so numpy and native timed on
        different random box sets carry a data difference in the `vs numpy` ratio."""
        first, second = kernels._inputs(7), kernels._inputs(7)

        np.testing.assert_array_equal(first.nms_boxes, second.nms_boxes)
        np.testing.assert_array_equal(first.nms_scores, second.nms_scores)
        np.testing.assert_array_equal(first.frame, second.frame)
        np.testing.assert_array_equal(first.boxes, second.boxes)
        assert not np.array_equal(kernels._inputs(7).frame, kernels._inputs(8).frame)

    def test_the_device_fair_column_can_be_selected_on_its_own(self) -> None:
        assert kernels.parse_args(["--op", "letterbox_to_device"]).op == "letterbox_to_device"
        assert kernels.parse_args([]).seed == 0


class TestRtspAppliesToShipInferOnly:
    def test_rtsp_with_the_baseline_is_refused(self, capsys) -> None:
        """The baseline reads JPEGs off disk; a head-to-head under rtsp charges one system for
        NVDEC, the jitter buffer and NV12 conversion and not the other, and the table would
        still render it as a comparison."""
        from benchmarks import run_bench

        assert run_bench.main(["--source", "rtsp", "--systems", "baseline,shipinfer"]) == 2
        assert "applies to shipinfer only" in capsys.readouterr().err


class TestTheKernelTierMeasuresTheDeviceItWasAskedFor:
    """Round 2 of the review: `--device 2` measured cuda:0. No `ImageOps` exposes a public
    `device`, so the destination fell through to the current device and the synchronise waited
    on cuda:0 — the native path skipped (a cross-device write, refused) and the torch path ran
    on the contended GPU while the table named the requested one."""

    def test_the_destination_is_the_requested_device(self) -> None:
        torch = pytest.importorskip("torch")

        assert kernels._destination(2) == torch.device("cuda", 2)
        assert kernels._destination(2).index == 2
        assert kernels._destination(None) == torch.device("cuda")

    def test_the_synchronise_waits_on_the_requested_device(self, monkeypatch) -> None:
        torch = pytest.importorskip("torch")
        waited: list[object] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda device=None: waited.append(device)
        )

        class _DeviceOps:
            on_device = True

        result = kernels._synchronised(_DeviceOps(), lambda: "done", 3)()

        assert result == "done"
        assert waited == [3]

    def test_the_index_reaches_every_case_that_needs_it(self) -> None:
        """Structural, because allocating on cuda:2 needs cuda:2: the two call sites thread
        the index rather than guess it."""
        import inspect

        source = inspect.getsource(kernels.measure)
        assert "_cases(ops, inputs, device)" in source
        assert "_synchronised(ops, cases[op], device)" in source
        assert "_to_device_case(ops, frame, params, device)" in inspect.getsource(
            kernels._cases
        )


class TestTheServerLogIsAFileNotAPipe:
    def test_the_servers_write_to_files_under_the_run_directory(
        self, monkeypatch, tmp_path
    ) -> None:
        """A pipe nobody drains fills at 64 KiB — one GST_DEBUG away — and then the server
        blocks on write and every camera stalls, which the table reports as a shortfall."""
        from benchmarks.harness import rtsp

        handed: list[object] = []

        class _Server:
            returncode = None
            stdout = None

            def poll(self) -> None:
                return None

            def terminate(self) -> None: ...

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None: ...

        def popen(_argv, **kwargs):
            handed.append(kwargs["stdout"])
            kwargs["stdout"].write("serving\n")
            return _Server()

        monkeypatch.setattr(rtsp.subprocess, "Popen", popen)
        monkeypatch.setattr(rtsp, "_accepting", lambda *_a, **_k: True)

        with rtsp.serving(BenchConfig(cameras=4, fps=5.0, source="rtsp", out_dir=tmp_path)):
            pass

        assert len(handed) == 2  # one handle per server, and each was a real file, not a pipe
        assert sorted(p.name for p in tmp_path.glob("rtsp-*.log")) == [
            "rtsp-person.log",
            "rtsp-ship.log",
        ]
        assert (tmp_path / "rtsp-person.log").read_text() == "serving\n"
