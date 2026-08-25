"""The rate limiter: a bound on *concurrent executions*, which the queue bound is not.

`scheduler.max_queue_size` says how much work may be waiting. Nothing said how much may be
running, so eight instances whose batching windows close together all entered compute at the
same instant, on one memory bus and one PCIe root complex. Triton bounds this with a rate
limiter over named resources; this is the same idea with the generality left out.

The property worth pinning is the one an operator relies on: **the peak concurrency never
exceeds the bound, and no work is dropped to achieve it.** Shedding is the queue's job, at
the edge, where the caller learns about it — a limiter that dropped work would be a second,
invisible eviction policy, which is the failure this whole project was rebuilt to remove.

Pure-layer tests only: the limiter and the config section it is built from. That the server
actually holds the bound is a server property and lives in
``tests/server/test_rate_limiting.py``.
"""

from __future__ import annotations

import threading
import time

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.repository.model_config import ModelConfig
from shipinfer.scheduling.limits import (
    RATE_LIMITERS,
    ConcurrencyRateLimiter,
    RateLimiter,
    UnlimitedRateLimiter,
    build_rate_limiter,
)


class TestRegistry:
    """A limiter is selected by name, so adding one is a file and a decorator."""

    def test_the_builtins_are_registered_under_tritons_vocabulary(self) -> None:
        assert set(RATE_LIMITERS.names()) == {"off", "concurrency"}

    def test_off_is_the_default_and_has_the_aliases_a_config_would_use(self) -> None:
        for name in ("off", "none", "unlimited"):
            assert isinstance(build_rate_limiter(name), UnlimitedRateLimiter)

    def test_an_unknown_name_names_what_was_available(self) -> None:
        with pytest.raises(ConfigurationError, match="concurrency"):
            build_rate_limiter("token_bucket", 4)

    def test_every_limiter_takes_the_same_single_argument(self) -> None:
        """The uniform signature is what lets the model config build any registered limiter
        without a branch on its name."""
        for name in RATE_LIMITERS.names():
            limiter = build_rate_limiter(name, 2 if name != "off" else 0)
            assert isinstance(limiter, RateLimiter)


class TestUnlimitedIsFree:
    """Off means off: no bound, no blocking, and nothing to give back."""

    def test_it_grants_every_request(self) -> None:
        limiter = UnlimitedRateLimiter()

        assert all(limiter.acquire(0.0) for _ in range(100))

        assert limiter.granted == 100
        assert limiter.in_flight == 0

    def test_release_is_harmless(self) -> None:
        limiter = UnlimitedRateLimiter()
        limiter.acquire()
        limiter.release()
        limiter.release()


class TestConcurrencyBound:
    """The bound, and how it behaves when it binds."""

    def test_a_bound_of_zero_is_a_configuration_error(self) -> None:
        """A model that can never run is a config mistake worth failing at start-up rather
        than a deadlock worth debugging at 3am."""
        with pytest.raises(ConfigurationError, match="off"):
            ConcurrencyRateLimiter(0)

    def test_peak_concurrency_never_exceeds_the_bound(self) -> None:
        limiter = ConcurrencyRateLimiter(3)
        inside = 0
        peak = 0
        guard = threading.Lock()
        start = threading.Event()

        def worker() -> None:
            nonlocal inside, peak
            start.wait(5)
            assert limiter.acquire(5.0)
            try:
                with guard:
                    inside += 1
                    peak = max(peak, inside)
                time.sleep(0.01)
            finally:
                with guard:
                    inside -= 1
                limiter.release()

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(10)

        assert peak <= 3
        assert limiter.granted == 12  # every one of them ran; none was dropped
        assert limiter.in_flight == 0
        assert limiter.peak_in_flight <= 3

    def test_an_uncontended_acquire_is_not_counted_as_a_wait(self) -> None:
        """`waited` is what tells an operator the limiter is actually binding; counting
        every acquire would make it useless for exactly that question."""
        limiter = ConcurrencyRateLimiter(2)

        limiter.acquire(1.0)
        limiter.release()

        assert limiter.granted == 1
        assert limiter.waited == 0
        assert limiter.mean_wait_us == 0.0

    def test_a_timed_out_acquire_says_not_yet_rather_than_raising(self) -> None:
        limiter = ConcurrencyRateLimiter(1)
        assert limiter.acquire(1.0)

        assert limiter.acquire(0.01) is False

        assert limiter.timed_out == 1
        assert limiter.waited == 1
        limiter.release()

    def test_an_unpaired_release_is_refused_rather_than_raising_the_ceiling(self) -> None:
        """A bounded semaphore turns "the pairing broke" into an error at the moment it
        broke, instead of a limiter that silently stops limiting."""
        limiter = ConcurrencyRateLimiter(1)

        with pytest.raises(ValueError):
            limiter.release()

    def test_the_context_manager_returns_the_slot_after_an_exception(self) -> None:
        limiter = ConcurrencyRateLimiter(1)

        with pytest.raises(RuntimeError), limiter.execution(1.0) as acquired:
            assert acquired
            raise RuntimeError("boom")

        assert limiter.in_flight == 0
        assert limiter.acquire(0.01)

    def test_stats_say_whether_the_bound_ever_bound(self) -> None:
        limiter = ConcurrencyRateLimiter(4)
        limiter.acquire()
        stats = limiter.stats()
        limiter.release()

        assert stats["limiter"] == "concurrency"
        assert stats["limit"] == 4
        assert stats["peak_in_flight"] == 1


class TestConfigSection:
    """``rate_limiter:`` in a model's config.yaml, and what it refuses."""

    def _config(self, **rate_limiter) -> ModelConfig:
        return ModelConfig(
            name="m",
            platform="mock",
            max_batch_size=4,
            inputs=[{"name": "x", "data_type": "FP32", "dims": [2]}],
            outputs=[{"name": "y", "data_type": "FP32", "dims": [2]}],
            rate_limiter=rate_limiter,
        )

    def test_it_is_off_by_default(self) -> None:
        config = ModelConfig(
            name="m",
            platform="mock",
            max_batch_size=4,
            inputs=[{"name": "x", "data_type": "FP32", "dims": [2]}],
            outputs=[{"name": "y", "data_type": "FP32", "dims": [2]}],
        )

        assert config.rate_limiter.kind == "off"
        assert not config.rate_limiter.enabled

    def test_a_bound_without_a_limiter_is_refused_rather_than_ignored(self) -> None:
        """A config that states a bound the server does not apply is worse than one that
        states none: it reads as protection that is not there."""
        with pytest.raises(ValueError, match="applies no bound"):
            self._config(kind="off", max_concurrent_executions=4)

    def test_a_limiter_without_a_bound_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_executions"):
            self._config(kind="concurrency")
