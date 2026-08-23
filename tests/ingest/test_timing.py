"""The two clock policies, asserted as sequences rather than as "it waited".

These are the tests that make the reconnect behaviour trustworthy. "The camera retried" is
easy to observe and says nothing; "it retried at 0.5, 1, 2, 4, 5, 5 seconds" is the policy.
"""

from __future__ import annotations

import random

import pytest

from shipinfer.ingest.timing import DeadlinePacer, ExponentialBackoff

from .conftest import FakeClock


class TestReconnectBackoff:
    """Retry delays grow, are jittered, and stop at a cap."""

    def test_delays_grow_geometrically_and_stop_at_the_cap(self):
        backoff = ExponentialBackoff(0.5, 5.0, factor=2.0, jitter=0.0)
        delays = [backoff.next_delay() for _ in range(7)]
        assert delays == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0]

    def test_a_success_returns_to_the_first_delay(self):
        backoff = ExponentialBackoff(0.1, 10.0, factor=3.0, jitter=0.0)
        assert [backoff.next_delay() for _ in range(3)] == [0.1, 0.30000000000000004, 0.9]
        backoff.reset()
        assert backoff.attempts == 0
        assert backoff.next_delay() == 0.1

    def test_attempts_counts_consecutive_failures(self):
        backoff = ExponentialBackoff(0.1, 1.0, jitter=0.0)
        assert backoff.attempts == 0
        backoff.next_delay()
        backoff.next_delay()
        assert backoff.attempts == 2

    def test_jitter_stays_inside_the_cap_and_desynchronises(self):
        """Fifty cameras behind one switch must not retry in lockstep."""
        backoff = ExponentialBackoff(1.0, 2.0, factor=2.0, jitter=0.5, rng=random.Random(7))
        samples = []
        for _ in range(50):
            samples.append(backoff.next_delay())
            backoff.reset()
        assert all(0.5 <= delay <= 1.0 for delay in samples)
        assert len(set(samples)) > 40, "jitter must actually spread the retries out"

        capped = ExponentialBackoff(1.0, 2.0, factor=10.0, jitter=0.2, rng=random.Random(7))
        for _ in range(5):
            assert capped.next_delay() <= 2.0

    def test_peek_does_not_consume_an_attempt(self):
        backoff = ExponentialBackoff(0.25, 4.0, jitter=0.0)
        assert backoff.peek() == 0.25
        assert backoff.peek() == 0.25
        assert backoff.attempts == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"initial_s": 0.0},
            {"initial_s": 1.0, "cap_s": 0.5},
            {"initial_s": 1.0, "factor": 1.0},
            {"initial_s": 1.0, "jitter": 1.0},
            {"initial_s": 1.0, "jitter": -0.1},
        ],
    )
    def test_a_nonsensical_backoff_is_rejected_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            ExponentialBackoff(**kwargs)

    def test_a_long_outage_does_not_overflow_the_exponent(self):
        """A camera down for a week must keep retrying, not kill its actor thread.

        `initial * factor ** attempts` overflows a float at around attempt 1000, which a camera
        retrying every 30 seconds reaches in under nine hours. It was an OverflowError inside
        the retry path — the one code path whose whole job is to survive.
        """
        backoff = ExponentialBackoff(0.5, 30.0, factor=2.0, jitter=0.0)
        for _ in range(5_000):
            backoff.next_delay()
        assert backoff.attempts == 5_000
        assert backoff.next_delay() == 30.0
        assert backoff.peek() == 30.0


class TestFramePacing:
    """A paced source holds its rate and absorbs lateness instead of bursting."""

    def test_pacing_holds_a_steady_rate(self):
        clock = FakeClock()
        pacer = DeadlinePacer(20.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.reset()
        for _ in range(5):
            pacer.wait()
        assert clock.slept == pytest.approx([0.05] * 5)
        assert pacer.behind == 0

    def test_pacing_subtracts_the_work_already_done(self):
        """The bug this class exists to avoid: `sleep(1/fps)` yields `1/fps + work` per frame."""
        clock = FakeClock()
        pacer = DeadlinePacer(10.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.reset()

        clock.advance(0.04)  # 40 ms of decode
        pacer.wait()
        assert clock.slept == pytest.approx([0.06])

        clock.advance(0.09)  # 90 ms of decode
        pacer.wait()
        assert clock.slept[-1] == pytest.approx(0.01)

    def test_falling_behind_resets_the_deadline_instead_of_bursting(self):
        """Lateness is absorbed, not repaid as back-to-back frames nobody asked for."""
        clock = FakeClock()
        pacer = DeadlinePacer(10.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.reset()

        clock.advance(0.5)  # a 500 ms stall against a 100 ms period
        pacer.wait()
        assert clock.slept == []
        assert pacer.behind == 1

        pacer.wait()
        assert clock.slept == pytest.approx(
            [0.1]
        ), "the schedule restarts from now, not from -400ms"

    def test_zero_fps_disables_pacing(self):
        clock = FakeClock()
        pacer = DeadlinePacer(0.0, monotonic=clock.monotonic, sleep=clock.sleep)
        pacer.wait()
        pacer.wait()
        assert pacer.enabled is False
        assert clock.slept == []
