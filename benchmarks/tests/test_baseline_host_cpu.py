"""The baseline reports the CPU it used, so a ratio has a like-for-like denominator.

`C1-WHAT-IS-THE-5x-AGAINST?` recorded that none existed: "GPU-seconds is the honest measure,
`InstanceStats::ewma_latency_us` holds ours and the bench does not print it -- but
`sim_pipeline_v2` reports no counterpart, so there is nothing to divide by". The binary reports
none, and it is run unchanged on purpose (ADR: the baseline is someone else's tree). The kernel
keeps a counterpart for any process, which is what this uses.

Offline by design: `RUSAGE_CHILDREN` is the kernel's, the subprocess here spins on the CPU for
a fraction of a second, and nothing touches a device or the baseline binary.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time

import pytest

from benchmarks.harness.baseline import children_cpu_since
from benchmarks.run_bench import host_cpu_line
from tests.support.subprocess_env import checkout_env

SPIN_S = 0.35

_SPIN = (
    "import time\nend = time.monotonic() + {seconds}\nwhile time.monotonic() < end:\n    pass\n"
)


class TestTheChildsCpuIsMeasuredExactly:
    """`RUSAGE_CHILDREN` accumulates a child's total when it is reaped, so a delta around a
    window that spawns and reaps one child is that child's CPU -- shutdown included.

    `os.wait4` would give the same number and cannot be used here: `baseline._terminate`
    reaps the process itself on the ordinary path, because the binary has no `--seconds` and
    is always stopped by signal.
    """

    def test_a_spinning_child_is_charged_what_it_burned(self) -> None:
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        child = subprocess.Popen(
            [sys.executable, "-c", _SPIN.format(seconds=SPIN_S)], env=checkout_env()
        )
        child.wait()

        assert children_cpu_since(before) == pytest.approx(SPIN_S, abs=SPIN_S)

    def test_a_child_reaped_before_the_window_is_not_charged_to_it(self) -> None:
        """The reason the reading is taken after `command_line` and not at the top of
        `run_baseline`: `build_binary`, `stage_runtime_libs` and the `pkg-config` probes all
        fork, and their CPU is the harness's rather than the measurement's."""
        earlier = subprocess.Popen(
            [sys.executable, "-c", _SPIN.format(seconds=SPIN_S)], env=checkout_env()
        )
        earlier.wait()

        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        time.sleep(0.05)

        assert children_cpu_since(before) == pytest.approx(0.0, abs=0.02)


class TestTheLineSaysWhatTheNumbersSpan:
    def test_cores_and_per_image_are_both_reported(self) -> None:
        line = host_cpu_line("baseline", cpu_s=280.0, elapsed_s=70.0, images_per_s=1000.0)

        assert "280.0 CPU-s over 70.0 s = 4.00 cores" in line
        assert "4.00 ms CPU/image" in line

    def test_the_caveat_travels_with_the_number(self) -> None:
        """A per-image figure whose numerator paid for start-up and whose denominator did not
        is a comparison between systems, not an absolute -- so the qualifier is printed rather
        than filed in a docstring nobody reads at the terminal."""
        line = host_cpu_line("baseline", cpu_s=280.0, elapsed_s=70.0, images_per_s=1000.0)

        assert "(both spanning start-up)" in line

    def test_an_unmeasured_rate_reports_cores_and_no_ratio(self) -> None:
        """`images_per_s` is `None` for an UNMEASURED run, and inventing a per-image cost
        there would be the "empty result means failure" shape ADR-005 refuses."""
        line = host_cpu_line("shipinfer", cpu_s=100.0, elapsed_s=50.0, images_per_s=None)

        assert "= 2.00 cores" in line
        assert "CPU/image" not in line

    def test_a_zero_window_divides_by_nothing(self) -> None:
        line = host_cpu_line("baseline", cpu_s=1.0, elapsed_s=0.0, images_per_s=10.0)

        assert "0.00 cores" in line
        assert "CPU/image" not in line
