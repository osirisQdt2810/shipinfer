"""The offline tier must be the same run on a GPU host as on a runner with no driver."""

from __future__ import annotations

import os

import pytest


class TestTheOfflineTierHidesTheAccelerators:
    """``pytest`` with no device marker selected must not open a CUDA context (ADR-001)."""

    @pytest.mark.parametrize(
        ("expression", "requested"),
        [
            ("not gpu and not multigpu", False),  # the default in pyproject.toml
            ("scheduling and not gpu and not multigpu", False),
            ("", True),  # no expression selects everything, device tests included
            ("gpu", True),
            ("multigpu", True),
            ("gpu and slow", True),
            ("not gpu", True),  # still selects a multigpu-only test
            ("not multigpu", True),  # still selects a gpu-only test
            ("gpu or multigpu", True),
            ("not slow", True),  # selects every fast device test
            ("slow and not gpu and not multigpu", False),
            # a multigpu test that also carries `scheduling` is selected; identifiers other
            # than the device markers are free, and the predicate must say what *could* run
            ("(gpu or scheduling) and not gpu", True),
        ],
    )
    def test_the_expression_decides_whether_a_device_tier_is_wanted(
        self, tier_predicate, expression: str, requested: bool
    ) -> None:
        assert tier_predicate(expression) is requested

    def test_an_unparseable_expression_does_not_hide_anything(self, tier_predicate) -> None:
        # pytest rejects the expression itself; the predicate must not pre-empt that by
        # hiding a device from a run that may have meant to use one.
        assert tier_predicate("gpu and (") is True

    def test_this_run_sees_no_accelerator(self, request, tier_predicate) -> None:
        """The live assertion: the process this test runs in has no device.

        This test carries no device marker, so it is part of the offline tier; if that tier
        selected a device test as well (``-m "not gpu"`` on a multi-GPU box) the hook leaves
        the devices alone by design, and this assertion would be about a different run.
        """
        if tier_predicate(request.config.getoption("markexpr") or ""):
            pytest.skip("this run selected a device tier; the devices are meant to be visible")
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert os.environ.get("HIP_VISIBLE_DEVICES") == ""
        torch = pytest.importorskip("torch")
        assert torch.cuda.is_available() is False, "the offline tier opened a CUDA device"


class TestAskingTheDriverCannotAbortCollection:
    """`_device_count_or_zero` takes its probe as an argument for exactly this test."""

    def test_a_driver_that_answers_is_believed(self, device_count_or_zero) -> None:
        assert device_count_or_zero(lambda: 3) == 3

    def test_a_driver_that_raises_is_a_machine_with_none(self, device_count_or_zero) -> None:
        def broken():
            raise RuntimeError("CUDA driver initialization failed")

        with pytest.warns(UserWarning, match="could not ask the driver"):
            assert device_count_or_zero(broken) == 0

    def test_the_failure_is_kept_for_the_skip_reason(self, probe_device_count) -> None:
        def broken():
            raise RuntimeError("CUDA driver initialization failed")

        with pytest.warns(UserWarning):
            count, failure = probe_device_count(broken)
        assert count == 0
        assert failure is not None and "CUDA driver initialization failed" in failure
        assert probe_device_count(lambda: 4) == (4, None)
