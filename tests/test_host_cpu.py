"""``scripts/host_cpu.py`` separates a run's own CPU from its load generators'.

Why this file exists: `NOT-GPU-BOUND-AT-FIVE-GPUS` ruled the GPUs out as the limit and left
the host cost recorded as "unknown, split between the RTSP servers (ours to discount) and our
own decode threads (ours to optimise)". The split decides how much of the RTSP arm's ~17%
penalty is a benchmark artefact, so the arithmetic that produces it is the thing worth
pinning -- the same argument `benchmarks/tests/test_occupancy_window.py` makes about the
occupancy percentage.

Offline by design (ADR-001): `/proc` and `wait4` are the kernel's, not a driver's. The
subprocesses here spin on the CPU for fractions of a second and touch no device.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from tests.support.subprocess_env import checkout_env

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "host_cpu.py"

#: Long enough that a tick-resolution counter (usually 100 Hz) reads several ticks, short
#: enough that the whole file stays under a second of CPU.
SPIN_S = 0.35

_SPIN = (
    "import time\n"
    "end = time.monotonic() + {seconds}\n"
    "while time.monotonic() < end:\n"
    "    pass\n"
)


@pytest.fixture(scope="module")
def host_cpu() -> ModuleType:
    """Import by path, and registered first -- `scripts/` is not a package on `sys.path`."""
    spec = importlib.util.spec_from_file_location("scripts.host_cpu", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.host_cpu"] = module
    spec.loader.exec_module(module)
    return module


def _spinner(seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _SPIN.format(seconds=seconds)], env=checkout_env()
    )


class TestReadingOneProcessesCpu:
    def test_a_comm_with_spaces_and_parentheses_does_not_shift_the_fields(
        self, host_cpu: ModuleType
    ) -> None:
        """`comm` is whatever the process wrote to `/proc/self/comm`, and a `split()` on the
        whole line reports another field's value as utime. 13 and 7 are fields 14 and 15."""
        fields = [str(n) for n in range(3, 53)]
        fields[14 - 3] = "13"
        fields[15 - 3] = "7"
        line = "4242 (py (weird) name) S " + " ".join(fields[1:])

        assert host_cpu.ticks_from_stat(line) == 20.0

    def test_this_process_has_used_some_cpu(self, host_cpu: ModuleType) -> None:
        assert (host_cpu.cpu_seconds(os.getpid()) or 0.0) > 0.0

    def test_a_process_that_is_gone_reads_none_rather_than_zero(
        self, host_cpu: ModuleType
    ) -> None:
        """Zero would say "this generator was free", which is the opposite of the truth."""
        done = _spinner(0.0)
        done.wait()

        assert host_cpu.cpu_seconds(done.pid) is None


class TestTheGeneratorsCpuIsAWindowAndNotALifetime:
    """The property the whole measurement rests on.

    A server is started before the run -- it builds its fixture cache and serves the first
    connects -- so charging its lifetime total to the run overstates the penalty this exists
    to quantify, in the direction that flatters us.
    """

    def test_only_the_cpu_inside_the_window_is_charged(self, host_cpu: ModuleType) -> None:
        spinner = _spinner(SPIN_S * 3)
        try:
            time.sleep(SPIN_S)
            before = {spinner.pid: host_cpu.cpu_seconds(spinner.pid)}
            time.sleep(SPIN_S)
            after = {spinner.pid: host_cpu.cpu_seconds(spinner.pid)}
        finally:
            spinner.kill()
            spinner.wait()

        charged = host_cpu.window(before, after)[str(spinner.pid)]
        lifetime = after[spinner.pid] or 0.0

        assert 0.0 < charged < lifetime, (
            f"charged {charged:.3f}s of a {lifetime:.3f}s lifetime; a delta must be strictly "
            "less than the total when the process was already burning CPU beforehand"
        )
        assert charged == pytest.approx(SPIN_S, abs=SPIN_S)

    def test_a_generator_that_died_mid_run_is_not_counted_as_free(
        self, host_cpu: ModuleType
    ) -> None:
        """`nan` and not `0.0`, so `generator_cpu_s` under-reports visibly rather than
        silently: a server killed by the OOM killer took its CPU with it."""
        charged = host_cpu.window({7: 1.0}, {7: None})

        assert math.isnan(charged["7"])

    def test_a_counter_that_appears_to_go_backwards_reads_zero(
        self, host_cpu: ModuleType
    ) -> None:
        """Not expected -- these only grow -- but a negative would send a reader looking for
        a kernel problem instead of at the pid that was reused."""
        assert host_cpu.window({7: 9.0}, {7: 1.0}) == {"7": 0.0}


class TestTheAccountingIsDivisible:
    def test_cores_busy_is_cpu_seconds_over_the_wall_window(self, host_cpu: ModuleType) -> None:
        """CPU-seconds alone cannot say whether a host was saturated: 280 over 70 s is four
        busy cores here and impossible on a two-core box, which is why `cores` is reported."""
        out = host_cpu.report(280.0, {"1": 70.0, "2": 70.0}, 70.0)

        assert out["command_cores_busy"] == 4.0
        assert out["generator_cpu_s"] == 140.0
        assert out["generator_cores_busy"] == 2.0
        assert out["cores"] == os.cpu_count()

    def test_a_dead_generator_does_not_poison_the_sum(self, host_cpu: ModuleType) -> None:
        """`nan + 1.0` is `nan`, and one dead server would erase the whole accounting."""
        out = host_cpu.report(1.0, {"1": 2.0, "2": float("nan")}, 1.0)

        assert out["generator_cpu_s"] == 2.0
        assert math.isnan(out["generators"]["2"])

    def test_a_zero_window_reports_no_rate_rather_than_dividing(
        self, host_cpu: ModuleType
    ) -> None:
        out = host_cpu.report(1.0, {}, 0.0)

        assert out["command_cores_busy"] == 0.0
        assert out["generator_cores_busy"] == 0.0


class TestTheWrapperIsTransparentToTheCommand:
    """It sits between the harness and the bench, so its own behaviour must be invisible."""

    def test_it_measures_the_command_and_passes_its_exit_code_through(
        self, host_cpu: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        out = tmp_path / "nested" / "cpu.json"

        code = host_cpu.main(
            ["--out", str(out), "--", sys.executable, "-c", _SPIN.format(seconds=SPIN_S)]
        )

        assert code == 0
        accounting = json.loads(out.read_text(encoding="utf-8"))
        assert accounting["command_cpu_s"] == pytest.approx(SPIN_S, abs=SPIN_S)
        assert accounting["command_cores_busy"] > 0.5, accounting
        assert "host cpu:" in capsys.readouterr().err

    def test_a_failing_command_fails_the_wrapper(self, host_cpu: ModuleType) -> None:
        assert host_cpu.main(["--", sys.executable, "-c", "raise SystemExit(3)"]) == 3

    def test_a_signalled_command_is_not_reported_as_a_clean_exit(
        self, host_cpu: ModuleType
    ) -> None:
        """`docker stop` on a bench must not leave a run's status saying it finished."""
        code = host_cpu.main(
            [
                "--",
                sys.executable,
                "-c",
                "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
            ]
        )

        assert code == 128 + 15

    def test_no_command_is_a_usage_error(self, host_cpu: ModuleType) -> None:
        """The exit code the harness's own tests use for a malformed command line."""
        assert host_cpu.main([]) == 2
        assert host_cpu.main(["--pid", "1"]) == 2


class TestBothArmsAreActuallyWiredToIt:
    """The one link the tests above cannot reach, and it is the link that goes missing.

    A measurement whose wiring is deleted leaves every unit test green and the number gone --
    `benchmarks/tests/test_occupancy_window.py` exists because that happened twice to the
    occupancy counter. Weaker than the tests above and says so: what it catches is a wrapper
    dropped from one arm, or a line printed into a log and into no summary.
    """

    RTSP = ROOT / "scripts" / "cpp_bench_over_rtsp.sh"
    RUNNER = ROOT / "scripts" / "run_cpp_bench.sh"

    def test_the_rtsp_arm_accounts_for_both_of_its_servers(self) -> None:
        text = self.RTSP.read_text(encoding="utf-8")

        assert "scripts/host_cpu.py" in text
        assert '--pid "$person_pid" --pid "$ship_pid"' in text, (
            "the RTSP arm must charge BOTH servers; one of them is half the load and half "
            "the CPU this exists to discount"
        )

    def test_the_replay_arm_is_wrapped_too(self) -> None:
        """Without this the two arms are not comparable, and their difference -- our own
        decode cost -- is the whole reason the accounting was added."""
        text = self.RUNNER.read_text(encoding="utf-8")

        assert "host_cpu.py" in text.split('if [ "$SOURCE" = "replay" ]')[1].split("else")[0]

    def test_the_number_reaches_the_summary_and_not_only_the_log(self) -> None:
        """`run_cpp_bench.sh` says this in its own words about the `chain` line: it was in
        every log and in none of these summaries, so a reader of the documented output could
        quote a figure without ever seeing the caveat. Same alternation, same reason."""
        assert "host cpu:" in self.RUNNER.read_text(encoding="utf-8").split("grep -E")[1]
