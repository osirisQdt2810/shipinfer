"""A C++ run whose shutdown abandons a camera still reports what it measured.

`cli/bench.cpp` cannot unwind when a camera is abandoned -- a detached thread still holds
references into the frame -- so it `_Exit`s, and it used to do that before printing anything.
Measured 9 Sep at 50 cameras x 20 fps x 70 s: `gstreamer` abandons and reports NO counters,
while `nvdec` -- V156's route, per `sources/nvdec.h`'s first line -- reports in full at the
same load. So what was lost is the host-decode fallback's reading, but the loss belongs to
neither source: any camera that hangs past the deadline discards the whole run, on any arm.

Two halves: the report is hoisted so both exits print the ingest and queue counters, and the
fleet's stop deadline scales with the fleet rather than sharing a fixed 5 s between fifty
GStreamer pipelines. Source-read rather than run -- reaching the path needs a camera that
hangs -- so weaker than a run, and what it catches is the report going back to one exit or
being copied to two so they drift.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp"
RUNNER = ROOT / "scripts" / "run_cpp_bench.sh"


class TestOneReporterServesBothExits:
    def test_the_ingest_counters_are_printed_from_one_place(self) -> None:
        """A second copy for the abandoned path is the report that drifts: this file's own
        subject is two exits that must say the same thing about the same run."""
        text = BENCH.read_text(encoding="utf-8")

        assert text.count('"frames_read "') == 1, (
            "`frames_read` is printed from more than one place, so the two exits can come to "
            "disagree about the same run"
        )
        assert text.count("report_ingest_and_queue();") == 2, (
            "the hoisted reporter is called from a number of places other than two, so one "
            "of the exits prints nothing or something else"
        )

    def test_the_abandoned_exit_reports_before_it_leaves(self) -> None:
        """The whole point. `_Exit` runs no destructors and returns to nobody, so anything
        after it is unreachable -- the print has to precede it on that path."""
        text = BENCH.read_text(encoding="utf-8")
        before_exit = text[: text.index("std::_Exit(1)")]
        after_abandonment = before_exit.rsplit("abandoned past the stop deadline", 1)[-1]

        assert "report_ingest_and_queue();" in after_abandonment, (
            "the abandoned exit still leaves without printing, so a run that loses one camera "
            "of fifty discards the whole measurement"
        )

    def test_the_reassembly_half_is_absent_rather_than_wrong_there(self) -> None:
        """`collector_*` and `events_complete` need `collector.drain()`, and draining would
        block on the very threads that did not stop. Absent is honest; zero is a claim."""
        text = BENCH.read_text(encoding="utf-8")

        assert text.index("std::_Exit(1)") < text.index('"collector_reported "'), (
            "the abandoned path now prints reassembly counters, which it cannot have: the "
            "drain that seals them has not run"
        )

    def test_the_report_is_flushed_because_that_exit_does_not(self) -> None:
        """`_Exit` skips atexit and every stdio buffer, and a run's stdout is redirected to a
        log -- so fully buffered. The first version printed the whole report into a buffer
        that was then discarded; only the unbuffered `std::cerr` line came through, so the
        run still said nothing. Forced with `--stop-deadline-ms 1` to find it."""
        text = BENCH.read_text(encoding="utf-8")
        between = text[text.index("report_ingest_and_queue();") : text.index("std::_Exit(1)")]

        assert "std::flush" in between, (
            "nothing flushes stdout before `_Exit`, so the partial report is written into a "
            "buffer that `_Exit` discards and the run reports nothing after all"
        )

    def test_the_refused_map_is_read_under_its_mutex(self) -> None:
        """The old read needed no lock BECAUSE OF WHERE IT STOOD -- after `stopping.store` and
        the worker joins, so no writer existed. Hoisting it moved it before both, with every
        worker still spinning and a detached actor still pushing frames, so a refused
        `collector.open` can insert a new key mid-iteration. Positional safety does not
        survive a move, which is why the invariant is pinned rather than the position.
        """
        lines = BENCH.read_text(encoding="utf-8").splitlines()
        uses = [
            n
            for n, row in enumerate(lines)
            # The declaration is the one mention that needs no lock, and it is the line that
            # names the type -- so it is recognised by that rather than by its line number.
            if "open_refused_by_camera" in row and "std::map<" not in row
        ]

        # A lower bound, not an exact count: the printed line names the counter in a string
        # literal too, and pinning the number would break on a reworded message.
        assert len(uses) >= 2, f"expected at least a write and a read, found {len(uses)}"
        for n in uses:
            near = "\n".join(lines[max(0, n - 4) : n])
            assert "lock(refused_mutex)" in near, (
                f"bench.cpp:{n + 1} touches `open_refused_by_camera` with no `refused_mutex` "
                "in the four lines above it; the workers write it under that lock while the "
                "abandoned path now reads it with them still running"
            )

    def test_the_missing_stage_tally_is_written_and_read_under_its_mutex(self) -> None:
        """`events_incomplete` says how many frames lost a stage and never which one -- which
        reads as a timeout and is not one: measured 11 Sep, 1 483 of 9 520 with
        `collector_timeouts 0`. The tally is written from the collector's emit callback, which
        runs on whichever worker sealed the frame, and read at the end; both sides take
        `missing_lock`, and the reason the lock is pinned rather than its position is the one
        `test_the_refused_map_is_read_under_its_mutex` gives.
        """
        lines = BENCH.read_text(encoding="utf-8").splitlines()
        uses = [
            n
            for n, row in enumerate(lines)
            if "missing_by_stage" in row
            and "std::map<" not in row
            and "&missing_lock" not in row
        ]

        assert len(uses) >= 2, f"expected at least a write and a read, found {len(uses)}"
        for n in uses:
            near = "\n".join(lines[max(0, n - 4) : n])
            assert "lock(missing_lock)" in near, (
                f"bench.cpp:{n + 1} touches `missing_by_stage` with no `missing_lock` in the "
                "four lines above it; every worker writes it as it seals a frame"
            )

    def test_the_report_names_the_stage_rather_than_only_counting(self) -> None:
        """A number nobody can act on is not a diagnostic: the line has to carry the stage's
        own name, which is what `FrameResult::missing` holds."""
        text = BENCH.read_text(encoding="utf-8")

        assert 'events_missing_stage " << stage' in text
        assert "for (const std::string& stage : result.missing)" in text, (
            "the tally has to come from the collector's own missing list rather than be "
            "inferred from the reason"
        )

    def test_the_message_says_which_report_this_is(self) -> None:
        """Otherwise a reader takes a partial record for a full one -- and the counters that
        are missing are the ones a throughput claim is made from."""
        text = BENCH.read_text(encoding="utf-8")

        assert "reporting the ingest " in text and "reassembly half" in text


class TestTheFleetsStopDeadlineScalesWithTheFleet:
    """`IngestManager::stop` spends ONE deadline on every camera, on purpose: per-actor would
    turn one stuck decoder into fifty consecutive waits. The consequence is that the library's
    5 s default is shared, and fifty GStreamer pipelines do not fit in it.

    Measured at 50 x 20 x 70 s: 31 of 50 abandoned at 5 s, 1 of 50 at 30 s.
    """

    def _stop_ms(self, cameras: int) -> int:
        """Evaluates the runner's own line, so this tests the arithmetic and not a copy."""
        line = next(
            row
            for row in RUNNER.read_text(encoding="utf-8").splitlines()
            if row.startswith("STOP_MS=")
        )
        done = subprocess.run(
            # The override is UNSET first: the shipped line honours
            # `SHIPINFER_BENCH_STOP_DEADLINE_MS`, so an operator shell that exports it would
            # make the floor test fail and let the design-load test pass for the wrong reason.
            [
                "bash",
                "-c",
                f"unset SHIPINFER_BENCH_STOP_DEADLINE_MS; CAMERAS={cameras}; {line}; "
                'echo "$STOP_MS"',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(done.stdout.strip())

    def test_the_design_load_gets_far_more_than_the_library_default(self) -> None:
        assert (
            self._stop_ms(50) >= 15000
        ), "fifty cameras share this deadline, and 5 s of it abandoned 31 of them"

    def test_a_small_fleet_is_not_made_slower_to_shut_down(self) -> None:
        """A floor, not a formula everywhere: eight cameras stop inside a second, and a
        deadline is a budget for the ones that hang rather than a wait every run pays."""
        assert self._stop_ms(8) == 5000

    def test_it_is_passed_to_the_binary_and_a_caller_can_still_override(self) -> None:
        """The binary takes the last spelling of a flag and `"$@"` is appended after these,
        so a run that wants its own deadline keeps it."""
        text = RUNNER.read_text(encoding="utf-8")
        args = text.split("deploy/rootless/cpp.sh")[1]

        assert '--stop-deadline-ms "$STOP_MS"' in args
        assert args.index('--stop-deadline-ms "$STOP_MS"') < args.index('"$@"')
