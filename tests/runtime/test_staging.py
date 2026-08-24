"""Pinned staging buffers must never be shared where a copy could still be reading one.

A staging buffer is reused on purpose — that is what makes pinned memory affordable and
what gives CUDA graph capture the stable address it requires. Reuse is also exactly where
it goes wrong, and the failure has no diagnostic: the DMA reads whatever the second writer
put there, and the wrong camera's frame comes back under the right camera's tag.

Two ways it was wrong here, both fixed and both pinned below:

  * one process-wide pool handed the same ``(shape, dtype)`` buffer to every model
    instance, and there is one worker thread per instance;
  * even inside one thread, a model with two inputs of the same shape staged the second
    over the first while the first's async H2D was still in flight.

These need no GPU. Not because pinning works without one — ``pin_memory=True`` raises
outright on a host with no accelerator, which is how CI found this — but because the pool
falls back to pageable memory there, and every property under test is about *identity*:
which caller gets which buffer, and whether the same one comes back. None of that depends
on the pages being locked.
"""

from __future__ import annotations

import contextlib
import threading

import pytest

from shipinfer.core.errors import DeviceError
from shipinfer.core.settings import MemorySettings
from shipinfer.runtime.memory import MemoryPool, PinnedStagingPool
from shipinfer.runtime.platform import is_available

torch = pytest.importorskip("torch")


@pytest.fixture()
def pool() -> MemoryPool:
    created = MemoryPool(MemorySettings())
    yield created
    created.close()


class TestBufferIdentity:
    """Which caller gets which buffer: never shared across owners or names, always reused
    within one.
    """

    def test_two_owners_never_share_a_buffer(self, pool: MemoryPool) -> None:
        """The cross-instance corruption. Two ship_detector instances on different GPUs both
        stage (8, 3, 640, 640) float32; if they get the same tensor, one overwrites the
        other's frames mid-DMA."""
        a = pool.staging_for("ship_detector_0_cuda:0").get(
            "images", (8, 3, 4, 4), torch.float32
        )
        b = pool.staging_for("ship_detector_1_cuda:1").get(
            "images", (8, 3, 4, 4), torch.float32
        )

        assert a is not b
        assert a.data_ptr() != b.data_ptr()

    def test_two_names_in_one_owner_never_share_a_buffer(self, pool: MemoryPool) -> None:
        """The single-threaded version of the same bug: a model with two same-shaped inputs."""
        staging = pool.staging_for("instance-0")
        images = staging.get("in:images", (2, 3, 4, 4), torch.float32)
        masks = staging.get("in:masks", (2, 3, 4, 4), torch.float32)

        assert images is not masks
        assert images.data_ptr() != masks.data_ptr()

    def test_the_same_name_is_reused(self, pool: MemoryPool) -> None:
        """Reuse within one name is the whole point — a fresh allocation per batch would be a
        synchronising cudaHostAlloc on the hot path, and the address would move under any
        captured graph."""
        staging = pool.staging_for("instance-0")
        first = staging.get("in:images", (2, 4), torch.float32)
        second = staging.get("in:images", (2, 4), torch.float32)

        assert first is second
        assert staging.stats()["hits"] == 1
        assert staging.stats()["misses"] == 1


class TestPinningFallback:
    """Pinned where a device exists, pageable where none does, and never an exception."""

    def test_buffers_are_pinned_exactly_when_pinning_is_possible(
        self, pool: MemoryPool
    ) -> None:
        """Pageable memory makes cudaMemcpyAsync silently synchronous, which is the one thing
        this class exists to prevent — so on a real device the buffer must be pinned.

        On a GPU-less host it must be pageable instead, and must not raise. That is not a
        concession to CI: with no device there is no DMA, so there is nothing for locked pages
        to accelerate, and the layers above must keep working either way (ADR-001).
        """
        staging = pool.staging_for("instance-0")
        buffer = staging.get("x", (4,), torch.float32)

        assert staging.pinned is is_available()
        assert buffer.is_pinned() is staging.pinned


class TestThreadSafety:
    """Racing lookups on one pool still yield a single allocation, not torn state."""

    def test_a_pool_is_owned_by_one_thread_but_still_locked(self) -> None:
        """One pool per owner makes contention impossible by construction; the lock is kept
        anyway so a future caller that gets the ownership wrong is merely slow, not corrupt."""
        staging = PinnedStagingPool(owner="t")
        seen: list[int] = []
        barrier = threading.Barrier(8)

        def grab() -> None:
            barrier.wait()
            seen.append(staging.get("x", (16,), torch.float32).data_ptr())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one allocation, handed to everyone: no torn state from the racing lookups.
        assert len(set(seen)) == 1
        assert staging.stats()["misses"] == 1


class TestPoolLifecycle:
    """The pool is bounded, and closing it releases every owner's buffers."""

    def test_the_pool_is_bounded(self, pool: MemoryPool) -> None:
        staging = PinnedStagingPool(owner="t", max_entries=4)
        for i in range(10):
            staging.get(f"n{i}", (4,), torch.float32)
        assert staging.stats()["entries"] <= 4

    def test_close_releases_every_owner_s_pool(self, pool: MemoryPool) -> None:
        pool.staging_for("a").get("x", (4,), torch.float32)
        pool.staging_for("b").get("x", (4,), torch.float32)
        assert any(k.startswith("staging[") for k in pool.stats())

        pool.close()
        assert not any(k.startswith("staging[") for k in pool.stats())


@contextlib.contextmanager
def _forced_capture(active: bool):
    """Make `torch.cuda.is_current_stream_capturing` report ``active``.

    The pool probes for that function with `getattr`, so this is the seam. Restored on the
    way out, and tolerant of a build that has no such attribute at all.
    """
    sentinel = object()
    previous = getattr(torch.cuda, "is_current_stream_capturing", sentinel)
    torch.cuda.is_current_stream_capturing = lambda: active
    try:
        yield
    finally:
        if previous is sentinel:
            del torch.cuda.is_current_stream_capturing
        else:
            torch.cuda.is_current_stream_capturing = previous


class TestItRefusesToAllocateMidCapture:
    """The guard that stopped a graph-capture failure from bricking an instance.

    A pinned host allocation calls into the driver, which CUDA forbids while a stream is
    capturing. Letting `torch.empty` discover that is not equivalent to refusing here: the
    failed allocation left the caching allocator with a capture still marked underway, so the
    next two capture attempts died with an internal assert and the instance never became
    ready. Observed on a four-GPU bench run — a graph-capture optimisation took the server
    down.

    Offline, because `_capturing` is a `getattr` probe a monkeypatch can force. Without these
    tests, dropping the `self.pinned and` guard or the exception suppression in `_capturing`
    leaves the suite green while every capture on a real box resumes bricking instances.
    """

    def test_a_cached_buffer_is_still_served_mid_capture(self) -> None:
        """Reuse is the whole point: a warmed pool must keep working inside a capture."""
        pool = PinnedStagingPool(owner="test")
        first = pool.get("in:images", (2, 3), torch.float32)

        with _forced_capture(True):
            again = pool.get("in:images", (2, 3), torch.float32)

        assert again is first

    def test_a_new_buffer_is_refused_mid_capture(self, monkeypatch) -> None:
        pool = PinnedStagingPool(owner="test")
        monkeypatch.setattr(pool, "pinned", True)

        with _forced_capture(True), pytest.raises(DeviceError, match="before capture began"):
            pool.get("out:embedding", (8, 512), torch.float32)

    def test_the_refusal_names_the_buffer_and_the_remedy(self, monkeypatch) -> None:
        pool = PinnedStagingPool(owner="person_embedder_0")
        monkeypatch.setattr(pool, "pinned", True)

        with _forced_capture(True), pytest.raises(DeviceError) as caught:
            pool.get("out:embedding", (8, 512), torch.float32)

        message = str(caught.value)
        assert "person_embedder_0" in message and "out:embedding" in message
        assert "warm the pool" in message, "it has to say what to do about it"

    def test_a_pageable_pool_is_unaffected(self, monkeypatch) -> None:
        """With no accelerator the buffers are ordinary memory and there is no capture to
        protect, so refusing would break the offline tier for nothing."""
        pool = PinnedStagingPool(owner="test")
        monkeypatch.setattr(pool, "pinned", False)

        with _forced_capture(True):
            assert pool.get("in:images", (2, 3), torch.float32) is not None

    def test_allocation_outside_a_capture_is_normal(self) -> None:
        pool = PinnedStagingPool(owner="test")
        with _forced_capture(False):
            assert pool.get("in:images", (4, 4), torch.float32) is not None
