"""The arithmetic on top of Triton's three phase timings, and the gate that guards it.

`PhaseTimings` is pure Python, so all of this is offline — which is the point: the module's
own docstring calls `is_measured` "load-bearing, not a convenience", and 284 lines carrying
that claim shipped with no test at all.

The failure it guards against is specific. With phase timing off, `device_busy_us` is 0, so
a naive `1 - busy/wall` reports **100% idle** — a confident falsehood about the one quantity
this module exists to establish, and one that would land in a metrics histogram and be
averaged into every aggregate over it. Removing the gate would leave the offline suite green
and the dashboard reading "the device is never busy".
"""

from __future__ import annotations

import pytest

from shipinfer.runtime.profiling import PHASES, PhaseTimings


class TestIsMeasured:
    def test_nothing_timed_is_not_a_measurement(self) -> None:
        assert not PhaseTimings(wall_us=1000.0).is_measured

    def test_one_timed_phase_is(self) -> None:
        assert PhaseTimings(per_phase={"compute_infer": 5.0}, wall_us=10.0).is_measured

    def test_a_zero_length_phase_still_counts_as_measured(self) -> None:
        """A phase that ran and took no measurable time is data; an absent one is not. The
        distinction is the whole reason this is `bool(per_phase)` and not `device_busy_us`."""
        assert PhaseTimings(per_phase={"compute_input": 0.0}, wall_us=10.0).is_measured


class TestIdleFraction:
    def test_an_unmeasured_batch_reports_zero_not_one(self) -> None:
        """Zero rather than one, because an unmeasured span must not read as a problem —
        and zero rather than NaN, because one NaN poisons every aggregate over the
        histogram this reaches."""
        assert PhaseTimings(wall_us=1000.0).idle_fraction == 0.0

    def test_a_fully_busy_device_is_zero_idle(self) -> None:
        timings = PhaseTimings(
            per_phase={"compute_input": 100.0, "compute_infer": 800.0, "compute_output": 100.0},
            wall_us=1000.0,
        )
        assert timings.idle_fraction == pytest.approx(0.0)

    def test_half_the_wall_clock_on_device_is_half_idle(self) -> None:
        timings = PhaseTimings(per_phase={"compute_infer": 500.0}, wall_us=1000.0)
        assert timings.idle_fraction == pytest.approx(0.5)

    def test_a_zero_wall_clock_cannot_be_divided_by(self) -> None:
        assert PhaseTimings(per_phase={"compute_infer": 5.0}, wall_us=0.0).idle_fraction == 0.0

    def test_busy_exceeding_wall_clock_clamps_at_zero(self) -> None:
        """Overlapping streams, or a stopwatch that disagrees with itself. A negative idle
        fraction in a report is a bug that looks like a result."""
        timings = PhaseTimings(per_phase={"compute_infer": 2000.0}, wall_us=1000.0)
        assert timings.idle_fraction == 0.0


class TestDeviceBusy:
    def test_it_sums_only_the_three_triton_phases(self) -> None:
        """A stray key must not inflate the device's share: the numbers exist to line up
        with `nv_inference_compute_*_duration_us`, which names exactly three spans."""
        timings = PhaseTimings(
            per_phase={"compute_infer": 100.0, "something_else": 900.0}, wall_us=1000.0
        )
        assert timings.device_busy_us == pytest.approx(100.0)

    def test_a_missing_phase_contributes_nothing(self) -> None:
        timings = PhaseTimings(per_phase={"compute_infer": 10.0}, wall_us=20.0)
        assert timings.device_busy_us == pytest.approx(10.0)

    def test_the_phase_names_are_tritons(self) -> None:
        """Renaming one silently drops it out of `device_busy_us` and out of the metric
        label that a Triton deployment's dashboards are keyed on."""
        assert PHASES == ("compute_input", "compute_infer", "compute_output")
