"""The copy home goes through pinned memory, and it copies exactly what it used to.

``TorchImageOps`` ends two of its three ops with a device-to-host copy, and both used to land
in freshly allocated *pageable* memory. A copy into pageable memory never DMAs: the driver
moves it in pieces through a bounce buffer of its own, which is the difference between about
1.4 GB/s and about 10 GB/s. At this project's sizing those two sites carry the letterboxed
batch and every crop of every frame, so the bandwidth is not a rounding error.

The change must be invisible in the output, and that is what most of this module asserts:
staged and unstaged results are compared with ``assert_array_equal``, not ``assert_allclose``.
A copy that is nearly the same array is a broken copy.

**Why these are offline.** :class:`PinnedStagingPool` falls back to ordinary pageable buffers
on a host with no accelerator (``tests/runtime/test_staging.py`` pins that), so every property
here — which buffer a call takes, how the chunk loop walks the batch, that the pool is asked
once and reused, that a refusal degrades — is real on a CPU box. Two things are not, and are
faked narrowly by :func:`_staged` and the ``stream`` fixture: the constructor refuses a pool
when it is not on a device, and there is no CUDA stream to synchronise. Faking the seam
rather than the behaviour is the same technique ``test_staging.py`` uses for
``is_current_stream_capturing``. :class:`TestStagedCropOnCuda` runs the real thing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import DeviceError
from shipinfer.core.settings import ExecutionProvider
from shipinfer.runtime.memory import PinnedStagingPool
from shipinfer.runtime.ops import get_image_ops
from shipinfer.runtime.ops.base import NormalizeParams
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps
from shipinfer.runtime.ops.torch_ops import TorchImageOps

torch = pytest.importorskip("torch")

# The case table lives with the test that froze the per-box loop, because that is the module
# that owns what a crop *is*. Staging must not change any of it, so the same cases run again
# here rather than a second, weaker table being invented.
from test_torch_crop_batch import (  # noqa: E402  (after importorskip, by design)
    CASES,
    FRAME,
    IMAGENET,
    NOISE,
    PARAM_SETS,
    PIXEL,
    _random_boxes,
)

#: Two of the four normalisation cases: one BGR with unit scaling, one RGB with ImageNet
#: statistics. The staged copy is bytes, so the swap and the scale are covered by having one
#: of each rather than by the full cross product the geometry test needs.
STAGED_PARAMS = [PARAM_SETS[1], PARAM_SETS[2]]

#: ``(images, dst_size)`` for the other staged site. Sizes vary because the pool key is built
#: from the trailing shape, and a mixed-source batch is the case that resizes twice.
LETTERBOX_CASES = [
    pytest.param([FRAME], (16, 16), id="one-frame"),
    pytest.param([FRAME, NOISE], (32, 24), id="two-sources"),
    pytest.param([NOISE] * 3, (12, 20), id="three-frames"),
    pytest.param([PIXEL], (8, 8), id="one-pixel-frame"),
]

PARAMS = NormalizeParams()


class _CountingStream:
    """Stands in for the worker's CUDA stream on a host that has none.

    Counts the synchronises so a test can assert the one property the fake would otherwise
    hide: every chunk is waited on before its buffer is read again. Dropping that wait is
    silent — the buffer still holds the *previous* frame's pixels, which are plausible.
    """

    def __init__(self) -> None:
        self.syncs = 0

    def synchronize(self) -> None:
        self.syncs += 1


class _CountingEvent:
    """Stands in for ``torch.cuda.Event``: offline the "DMA" is a host memcpy that is
    complete by construction, so recording and waiting are pure bookkeeping — which is
    exactly what these counters pin: one record per chunk, one synchronise before each read.
    """

    records = 0
    syncs = 0

    def __init__(self) -> None:
        self.recorded = 0
        self.synced = 0

    def record(self, stream=None) -> None:
        self.recorded += 1
        _CountingEvent.records += 1

    def synchronize(self) -> None:
        self.synced += 1
        _CountingEvent.syncs += 1


@pytest.fixture()
def stream(monkeypatch) -> _CountingStream:
    """Make ``torch.cuda.current_stream`` and ``torch.cuda.Event`` answer, so the staged
    path can run with no device."""
    counter = _CountingStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: counter)
    monkeypatch.setattr(torch.cuda, "Event", _CountingEvent)
    _CountingEvent.records = 0
    _CountingEvent.syncs = 0
    return counter


def _staged(pool: PinnedStagingPool, **kwargs: Any) -> TorchImageOps:
    """Torch ops that will use ``pool``, on a host where the constructor would refuse it.

    The refusal is deliberate and is asserted by :class:`TestItStaysOptional`: a CPU-bound
    instance has no DMA to accelerate, so it is given no pool. Assigning past it here is what
    lets the offline tier exercise the real chunk loop, the real pool and the real copies —
    all of which are arithmetic, not hardware.
    """
    ops = TorchImageOps(**kwargs)
    ops._staging = pool
    return ops


def _pair(pool: PinnedStagingPool, **kwargs: Any) -> tuple[TorchImageOps, TorchImageOps]:
    """The same ops twice: one staged, one not. Their outputs must be indistinguishable."""
    return _staged(pool, **kwargs), TorchImageOps(**kwargs)


class TestTheStagedResultIsTheSameArray:
    """A faster copy is only a copy if it produces the identical array, bit for bit."""

    @pytest.mark.parametrize("chunked", [False, True], ids=["one-span", "multi-chunk"])
    @pytest.mark.parametrize(("image", "boxes", "dst_size"), CASES)
    @pytest.mark.parametrize("params", STAGED_PARAMS)
    def test_a_staged_crop_is_the_unstaged_crop(
        self, stream, monkeypatch, image, boxes, dst_size, params, chunked
    ) -> None:
        # #31 round 4: at the default bound every case in the table is one span, and after
        # round 3's structural rule one span never stages — so without the forced-chunk
        # variant this table compared `.cpu()` against `.cpu()` and a staged branch
        # returning zeros passed it. The multi-chunk variant shrinks the bound to one row
        # per span so the equality claim is made about the branch it names.
        if chunked:
            row = 3 * dst_size[0] * dst_size[1]
            monkeypatch.setattr(TorchImageOps, "_STAGE_CHUNK_ELEMENTS", row)
        pool = PinnedStagingPool(owner="test")
        staged, plain = _pair(pool)

        result = staged.crop_batch(image, boxes, dst_size, params)
        expected = plain.crop_batch(image, boxes, dst_size, params)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype == np.float32
        # The staged array is built by `torch.empty` and filled chunk by chunk; a backend
        # handed a non-contiguous batch gets scrambled data.
        assert result.flags["C_CONTIGUOUS"]
        if chunked and len(boxes) > 1:
            assert pool.stats()["misses"] == 2, "the pair came from the call under test"

    @pytest.mark.parametrize("chunked", [False, True], ids=["one-span", "multi-chunk"])
    @pytest.mark.parametrize(("images", "dst_size"), LETTERBOX_CASES)
    @pytest.mark.parametrize("params", STAGED_PARAMS)
    def test_a_staged_letterbox_is_the_unstaged_letterbox(
        self, stream, monkeypatch, images, dst_size, params, chunked
    ) -> None:
        if chunked:
            # One row per span (#31 round 4): letterbox stages too when a batch genuinely
            # spans chunks — the production single-frame call is one span and does not.
            monkeypatch.setattr(
                TorchImageOps, "_STAGE_CHUNK_ELEMENTS", 3 * dst_size[0] * dst_size[1]
            )
        pool = PinnedStagingPool(owner="test")
        staged, plain = _pair(pool)

        result = staged.letterbox_batch(images, dst_size, params)
        expected = plain.letterbox_batch(images, dst_size, params)

        np.testing.assert_array_equal(result.tensor, expected.tensor)
        np.testing.assert_array_equal(result.scales, expected.scales)
        np.testing.assert_array_equal(result.pads, expected.pads)
        np.testing.assert_array_equal(result.extents, expected.extents)

    def test_an_empty_batch_needs_no_buffer(self, stream) -> None:
        """`crop_batch` returns before staging on an empty box set; `_to_host` itself must
        also refuse to allocate for a zero-row tensor — driven directly, because no public
        call site can reach that branch (round 2's review caught the earlier version of
        this test exercising only the early return)."""
        pool = PinnedStagingPool(owner="test")
        ops = _staged(pool)
        out = ops.crop_batch(FRAME, np.empty((0, 4), dtype=np.float32), (8, 8), PARAMS)
        assert out.shape == (0, 3, 8, 8) and pool.stats()["misses"] == 0
        empty = ops._to_host(torch.empty((0, 3, 8, 8)), "crop")
        assert empty.shape == (0, 3, 8, 8) and empty.dtype == np.float32
        assert pool.stats()["misses"] == 0, "a zero-row copy must not allocate a buffer"
        assert _CountingEvent.records == 0


class TestTheChunkLoopCoversTheBatch:
    """A batch wider than one buffer is copied in pieces, and every piece is waited on."""

    def test_a_crop_batch_larger_than_one_chunk_matches(self, stream, monkeypatch) -> None:
        monkeypatch.setattr(TorchImageOps, "_STAGE_CHUNK_ELEMENTS", 3 * 12 * 10 * 3)
        boxes = _random_boxes(9, NOISE.shape, seed=11)
        pool = PinnedStagingPool(owner="test")
        staged, plain = _pair(pool)
        assert TorchImageOps._stage_rows((9, 3, 12, 10)) == 3, "the case must chunk"

        result = staged.crop_batch(NOISE, boxes, (12, 10), IMAGENET)

        np.testing.assert_array_equal(
            result, plain.crop_batch(NOISE, boxes, (12, 10), IMAGENET)
        )
        assert (
            _CountingEvent.records == 3 and _CountingEvent.syncs == 3
        ), "each chunk records its event once and is waited for once before its read"

    def test_letterbox_never_touches_the_pool(self, stream) -> None:
        """Round 2 of #31: letterbox is one chunk at the production shape, where the
        ping-pong cannot engage and staging only adds a serial memcpy — so it stays on the
        plain `.cpu()` path, deliberately, and a pool handed to the ops must stay cold."""
        pool = PinnedStagingPool(owner="test")
        staged_ops, plain_ops = _pair(pool)
        ours = staged_ops.letterbox_batch([FRAME], (16, 16), PARAMS)
        theirs = plain_ops.letterbox_batch([FRAME], (16, 16), PARAMS)
        np.testing.assert_array_equal(ours.tensor, theirs.tensor)
        assert pool.stats()["misses"] == 0, "letterbox is single-chunk: the structural rule"
        assert _CountingEvent.records == 0

    def test_a_single_chunk_crop_never_touches_the_pool(self, stream) -> None:
        """#31 round 3: the rule is structural, not per call site — a design-sizing
        person-reid batch (single span) takes the plain `.cpu()` path exactly as the
        letterbox frame does, and only multi-chunk work stages."""
        pool = PinnedStagingPool(owner="test")
        staged_ops, plain_ops = _pair(pool)
        boxes = _random_boxes(4, FRAME.shape)
        ours = staged_ops.crop_batch(FRAME, boxes, (8, 8), PARAMS)  # default bound: 1 span
        theirs = plain_ops.crop_batch(FRAME, boxes, (8, 8), PARAMS)
        np.testing.assert_array_equal(ours, theirs)
        assert pool.stats()["entries"] == 0
        assert _CountingEvent.records == 0


class TestTheBufferIsFixedShape:
    """One buffer per call site for the life of the worker, whatever the crowd is doing.

    Keying the pool on the true row count would look tidier and would be the defect: every
    crowd size the cameras produce becomes its own entry, the 64-entry pool starts evicting,
    and the steady state is a ``cudaHostAlloc`` per call — slower than the pageable copy this
    replaced.
    """

    def test_the_crowd_size_never_reaches_the_pool(self, stream, monkeypatch) -> None:
        monkeypatch.setattr(TorchImageOps, "_STAGE_CHUNK_ELEMENTS", 3 * 8 * 8 * 8)
        pool = PinnedStagingPool(owner="test")
        ops = _staged(pool)

        for count in (1, 5, 40):
            ops.crop_batch(NOISE, _random_boxes(count, NOISE.shape), (8, 8), PARAMS)

        # Crowds 1 and 5 are single-chunk and never touch the pool (the structural rule —
        # #31 round 3); crowd 40 spans five chunks and stages through the ping-pong pair.
        # Two entries and two allocations across three different crowds is the claim:
        # single-chunk work stays off the pool, and N never reaches the key.
        assert pool.stats()["entries"] == 2
        assert pool.stats()["misses"] == 2, "a third allocation means N reached the key"
        assert pool.stats()["hits"] == 0

    def test_the_second_buffer_exists_only_where_the_ping_pong_engages(
        self, stream, monkeypatch
    ) -> None:
        """#31 round 2: a single-chunk call must not lock a second page it never reads;
        a multi-chunk call gets the pair, and the two buffers are distinct memory."""
        monkeypatch.setattr(TorchImageOps, "_STAGE_CHUNK_ELEMENTS", 3 * 8 * 8)
        pool = PinnedStagingPool(owner="test")
        ops = _staged(pool)
        ops.crop_batch(FRAME, _random_boxes(3, FRAME.shape), (8, 8), PARAMS)  # 3 chunks
        assert pool.stats()["entries"] == 2, "multi-chunk: the ping-pong pair"
        first = pool.get("crop:a", (1, 3, 8, 8), torch.float32)
        second = pool.get("crop:b", (1, 3, 8, 8), torch.float32)
        assert first.data_ptr() != second.data_ptr()

    def test_the_result_is_not_a_view_of_the_reused_buffer(self, stream) -> None:
        """The caller owns its array. Returning a view would mean the next crop of the next
        frame silently rewrote a batch that is still on its way to a backend."""
        pool = PinnedStagingPool(owner="test")
        ops = _staged(pool)
        # Forced multi-chunk (#31 round 4): at the default bound these are one span, one
        # span never stages, and `.cpu()` cannot return a view — the assertion below would
        # be trivially true about the wrong branch.
        type(ops)._STAGE_CHUNK_ELEMENTS, _saved = 3 * 8 * 8, type(ops)._STAGE_CHUNK_ELEMENTS
        try:
            first = ops.crop_batch(FRAME, _random_boxes(3, FRAME.shape, seed=1), (8, 8), PARAMS)
            keep = first.copy()
            second = ops.crop_batch(
                NOISE, _random_boxes(3, NOISE.shape, seed=2), (8, 8), PARAMS
            )
        finally:
            type(ops)._STAGE_CHUNK_ELEMENTS = _saved
        assert pool.stats()["misses"] == 2, "both arrays came from the staged branch"

        assert not np.array_equal(first, second), "the case cannot detect aliasing"
        assert not np.shares_memory(first, second)
        np.testing.assert_array_equal(first, keep)


class TestItStaysOptional:
    """Staging is an optimisation. Everything must work without one, and nothing may break
    because a pool was offered where it cannot help."""

    def test_ops_without_a_pool_copy_as_they_always_did(self) -> None:
        ops = TorchImageOps()

        assert ops._staging is None
        assert "pinned staging" not in ops.describe()
        assert ops.crop_batch(FRAME, _random_boxes(2, FRAME.shape), (8, 8), PARAMS).shape == (
            2,
            3,
            8,
            8,
        )

    def test_a_host_bound_instance_never_touches_the_pool_it_was_given(self) -> None:
        """On CPU there is no DMA to accelerate, so the bounce would be pure cost. The
        decision is taken once in the constructor rather than per call."""
        pool = PinnedStagingPool(owner="test")
        ops = TorchImageOps(staging=pool)

        ops.crop_batch(FRAME, _random_boxes(2, FRAME.shape), (8, 8), PARAMS)
        ops.letterbox_batch([FRAME], (16, 16), PARAMS)

        assert ops._staging is None
        assert pool.stats() == {"entries": 0, "hits": 0, "misses": 0, "bytes": 0}

    def test_describe_says_when_the_copies_are_staged(self, stream) -> None:
        """The bench prints `describe()` in its table, and a number from the pageable path
        that claims to be the staged one is worse than no number."""
        assert "pinned staging" in _staged(PinnedStagingPool(owner="test")).describe()


class TestTheFactoryHandsThePoolThrough:
    """`get_image_ops` is where the pool meets an implementation, because it is what chooses
    the implementation."""

    def test_torch_is_given_the_pool_and_the_device(self, monkeypatch) -> None:
        seen: dict[str, Any] = {}

        def recorder(**kwargs: Any) -> NumpyImageOps:
            seen.update(kwargs)
            return NumpyImageOps()

        monkeypatch.setattr("shipinfer.runtime.ops.is_available", lambda: True)
        monkeypatch.setattr("shipinfer.runtime.ops.TorchImageOps", recorder)
        pool = PinnedStagingPool(owner="test")

        get_image_ops(ExecutionProvider.PYTHON, device_index=3, staging=pool)

        assert seen == {"device_index": 3, "staging": pool}

    def test_a_host_only_fallback_still_takes_a_pool_without_complaining(
        self, monkeypatch
    ) -> None:
        """A GPU-less box gets numpy ops, and the caller should not have to know that before
        deciding whether to offer a pool."""
        monkeypatch.setattr("shipinfer.runtime.ops.is_available", lambda: False)

        ops = get_image_ops(ExecutionProvider.PYTHON, staging=PinnedStagingPool(owner="test"))

        assert isinstance(ops, NumpyImageOps)


class _RefusingPool:
    """A pool that will not allocate — a capture underway, or no lockable pages left."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def get(self, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        self.calls += 1
        raise self.error


class TestARefusalDegradesInsteadOfFailing:
    """An optimisation must not be able to take a worker down.

    Both refusals are real: :class:`PinnedStagingPool` raises ``DeviceError`` rather than let
    a pinned allocation run inside a graph capture, and ``torch`` raises ``RuntimeError`` when
    the host is out of lockable pages. Neither says anything about the pixels.
    """

    @pytest.mark.parametrize(
        "error",
        [DeviceError("mid-capture"), RuntimeError("cannot allocate pinned memory")],
        ids=["mid-capture", "no-pinned-pages"],
    )
    def test_the_call_returns_the_right_array_anyway(self, stream, error) -> None:
        ops = _staged(_RefusingPool(error))
        boxes = _random_boxes(4, FRAME.shape)

        result = ops.crop_batch(FRAME, boxes, (8, 8), IMAGENET)

        np.testing.assert_array_equal(
            result, TorchImageOps().crop_batch(FRAME, boxes, (8, 8), IMAGENET)
        )

    def test_an_allocation_failure_stops_asking_but_a_capture_does_not(
        self, stream, monkeypatch
    ) -> None:
        """Two different refusals, two different lifetimes (#31 round 3): the host being
        out of lockable pages will not heal, so degrade once — a refused pool at 1000
        frames a second is a thousand log lines a second. A mid-capture refusal is
        transient by nature, so THIS call goes pageable and the next one asks again."""
        monkeypatch.setattr(TorchImageOps, "_STAGE_CHUNK_ELEMENTS", 3 * 8 * 8)
        boxes = _random_boxes(3, FRAME.shape)  # 3 chunks at the shrunken bound: staged path

        transient = _RefusingPool(DeviceError("mid-capture"))
        ops = _staged(transient)
        for _ in range(3):
            ops.crop_batch(FRAME, boxes, (8, 8), PARAMS)
        assert transient.calls == 3, "a capture refusal is retried on the next call"
        assert ops._staging is transient

        permanent = _RefusingPool(RuntimeError("cannot allocate pinned memory"))
        ops = _staged(permanent)
        for _ in range(3):
            ops.crop_batch(FRAME, boxes, (8, 8), PARAMS)
        assert permanent.calls == 1, "an allocation failure is never retried"
        assert ops._staging is None


class TestTheBufferIsBoundedByPinnedBytes:
    """The bound is page-locked host memory, which is why it is not the crop bound.

    These are the three shapes the deployment stages, and the numbers the class attribute's
    own comment quotes. A change that makes a worker hold ten times the locked memory should
    fail here rather than on the box.
    """

    @pytest.mark.parametrize(
        ("shape", "rows"),
        [((8, 3, 640, 640), 1), ((15, 3, 256, 128), 21), ((40, 3, 512, 512), 2)],
        ids=["letterbox", "person-crops", "ship-crops"],
    )
    def test_the_named_shapes_stage_as_documented(self, shape, rows: int) -> None:
        assert TorchImageOps._stage_rows(shape) == rows
        held = rows * int(np.prod(shape[1:])) * 4
        assert held <= TorchImageOps._STAGE_CHUNK_ELEMENTS * 4

    def test_stage_rows_floors_at_one(self) -> None:
        """One row over budget is still one row: refusing a 4096x4096 mask because it is
        large is a worse failure than one large buffer, which is `_crop_chunks`' rule too."""
        assert TorchImageOps._stage_rows((4, 3, 4096, 4096)) == 1


@pytest.mark.gpu
class TestStagedCropOnCuda:
    """The parts a CPU host cannot exercise: the real pinning, the real DMA, the real stream.

    The output is compared against ops with no pool at all, on the same device — so what is
    under test is the copy home and nothing else.
    """

    @pytest.mark.parametrize(("image", "boxes", "dst_size"), CASES)
    def test_a_staged_crop_is_the_unstaged_crop(self, image, boxes, dst_size) -> None:
        pool = PinnedStagingPool(owner="test:cuda:0")
        staged = TorchImageOps(device_index=0, staging=pool)
        plain = TorchImageOps(device_index=0)
        assert staged.on_device, "the gpu tier must not quietly measure the CPU path"
        assert staged._staging is pool, "the constructor gate refused a pool on a device"
        result = None
        try:
            result = staged.crop_batch(image, boxes, dst_size, IMAGENET)
            np.testing.assert_array_equal(
                result, plain.crop_batch(image, boxes, dst_size, IMAGENET)
            )
        finally:
            del result, staged, plain
            pool.clear()
            torch.cuda.empty_cache()

    def test_a_multichunk_mask_batch_stages_for_real(self) -> None:
        """The whole point, on real hardware: a mask-shaped batch — every ship its own span
        at the DEFAULT bound, no monkeypatching — goes through genuinely pinned buffers with
        the real DMA and the real `torch.cuda.Event` ordering. If that ordering were wrong,
        the result would be the *previous* chunk's pixels: plausible values, no error — the
        one failure class the offline counting fakes structurally cannot see (#31 round 4).
        """
        pool = PinnedStagingPool(owner="test:cuda:0")
        staged = TorchImageOps(device_index=0, staging=pool)
        plain = TorchImageOps(device_index=0)
        try:
            frame = np.random.default_rng(7).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
            boxes = _random_boxes(4, frame.shape, seed=11)
            ours = staged.crop_batch(frame, boxes, (640, 640), IMAGENET)  # rows=1 -> 4 spans
            theirs = plain.crop_batch(frame, boxes, (640, 640), IMAGENET)

            np.testing.assert_array_equal(ours, theirs)
            stats = pool.stats()
            assert stats["misses"] == 2, "the pair came from the call under test, not from us"
            assert stats["entries"] == 2
            # A hit, not a miss: the buffers the call created — and they are truly pinned.
            first = pool.get("crop:a", (1, 3, 640, 640), torch.float32)
            second = pool.get("crop:b", (1, 3, 640, 640), torch.float32)
            assert pool.stats()["misses"] == 2 and pool.stats()["hits"] == 2
            assert first.is_pinned() and second.is_pinned()
        finally:
            del staged, plain
            pool.clear()
            torch.cuda.empty_cache()


class TestReleaseStaging:
    def test_release_frees_the_owner_and_forgives_a_double_release(self) -> None:
        """Owner keys embed thread idents, so a stop/start cycle mints new keys; the
        release is what keeps each cycle from stranding its page-locked buffers."""
        from shipinfer.runtime.memory.pool import MemoryPool

        memory = MemoryPool()
        pool = memory.staging_for("ops:test#1:cuda:0")
        pool.get("crop:a", (2, 3, 8, 8), torch.float32)
        assert pool.stats()["entries"] == 1
        memory.release_staging("ops:test#1:cuda:0")
        assert pool.stats()["entries"] == 0, "released pools are cleared"
        assert memory.staging_for("ops:test#1:cuda:0") is not pool, "a new claim is fresh"
        memory.release_staging("ops:test#1:cuda:0")
        memory.release_staging("never-claimed")  # both must be quiet no-ops

    def test_stats_survives_a_concurrent_release(self) -> None:
        """#31 round 2: `release_staging` is the first path that removes entries, and
        `stats()` used to iterate the live dict — /v2/statistics 500'd with "dictionary
        changed size" exactly while a pipeline was stopping. 200 alternations pin the
        snapshot."""
        import threading as _threading

        from shipinfer.runtime.memory.pool import MemoryPool

        memory = MemoryPool()
        errors: list[BaseException] = []

        def churn() -> None:
            try:
                for index in range(200):
                    memory.staging_for(f"ops:hammer#{index}:cuda:0")
                    memory.release_staging(f"ops:hammer#{index - 1}:cuda:0")
            except BaseException as exc:
                errors.append(exc)

        def read() -> None:
            try:
                for _ in range(200):
                    memory.stats()
            except BaseException as exc:
                errors.append(exc)

        threads = [_threading.Thread(target=churn), _threading.Thread(target=read)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert errors == []
