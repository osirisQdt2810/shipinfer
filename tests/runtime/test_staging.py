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

These need no GPU: ``torch.empty(pin_memory=True)`` works without a device, and the
property under test is about *identity*, not about the copy.
"""

from __future__ import annotations

import threading

import pytest

from shipinfer.core.settings import MemorySettings
from shipinfer.runtime.memory import MemoryPool, PinnedStagingPool

torch = pytest.importorskip("torch")


@pytest.fixture()
def pool() -> MemoryPool:
    created = MemoryPool(MemorySettings())
    yield created
    created.close()


def test_two_owners_never_share_a_buffer(pool: MemoryPool) -> None:
    """The cross-instance corruption. Two ship_detector instances on different GPUs both
    stage (8, 3, 640, 640) float32; if they get the same tensor, one overwrites the
    other's frames mid-DMA."""
    a = pool.staging_for("ship_detector_0_cuda:0").get("images", (8, 3, 4, 4), torch.float32)
    b = pool.staging_for("ship_detector_1_cuda:1").get("images", (8, 3, 4, 4), torch.float32)

    assert a is not b
    assert a.data_ptr() != b.data_ptr()


def test_two_names_in_one_owner_never_share_a_buffer(pool: MemoryPool) -> None:
    """The single-threaded version of the same bug: a model with two same-shaped inputs."""
    staging = pool.staging_for("instance-0")
    images = staging.get("in:images", (2, 3, 4, 4), torch.float32)
    masks = staging.get("in:masks", (2, 3, 4, 4), torch.float32)

    assert images is not masks
    assert images.data_ptr() != masks.data_ptr()


def test_the_same_name_is_reused(pool: MemoryPool) -> None:
    """Reuse within one name is the whole point — a fresh allocation per batch would be a
    synchronising cudaHostAlloc on the hot path, and the address would move under any
    captured graph."""
    staging = pool.staging_for("instance-0")
    first = staging.get("in:images", (2, 4), torch.float32)
    second = staging.get("in:images", (2, 4), torch.float32)

    assert first is second
    assert staging.stats()["hits"] == 1
    assert staging.stats()["misses"] == 1


def test_buffers_are_actually_pinned(pool: MemoryPool) -> None:
    """Pageable memory makes cudaMemcpyAsync silently synchronous, which is the one thing
    this class exists to prevent."""
    assert pool.staging_for("instance-0").get("x", (4,), torch.float32).is_pinned()


def test_a_pool_is_owned_by_one_thread_but_still_locked() -> None:
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


def test_the_pool_is_bounded(pool: MemoryPool) -> None:
    staging = PinnedStagingPool(owner="t", max_entries=4)
    for i in range(10):
        staging.get(f"n{i}", (4,), torch.float32)
    assert staging.stats()["entries"] <= 4


def test_close_releases_every_owner_s_pool(pool: MemoryPool) -> None:
    pool.staging_for("a").get("x", (4,), torch.float32)
    pool.staging_for("b").get("x", (4,), torch.float32)
    assert any(k.startswith("staging[") for k in pool.stats())

    pool.close()
    assert not any(k.startswith("staging[") for k in pool.stats())
