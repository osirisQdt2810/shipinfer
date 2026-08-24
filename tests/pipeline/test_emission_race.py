"""The sweeper must not read a `FrameState` its owning worker is still writing — and it
must not do the reading's expensive half while holding the fleet's one lock.

`FrameState` is owned by exactly one worker for the frame's whole life (ADR-002), and the
sweeper broke that: it built a `FrameResult` under the collector's lock but the emission read
`state` afterwards, *outside* it, while the owning worker was still inside `graph.execute`.
With a 1500 ms reassembly timeout against a 5000 ms stage timeout, any stage waiting on a
backed-up model queue guarantees the overlap.

The failure was not a crash in the usual place. The reader saw an empty detection list, the
wedged stage then answered and the worker filled twelve in, and the record generator indexed
past the end of a list it had already sized. `_emit` caught that, incremented
`build_failures` and returned — but the collector had already counted the frame as reported
and `_emit` had already popped its future. So the frame vanished from the sink while
"every opened frame is reported exactly once" still read green.

**The first fix traded the race for a contention problem**, and review caught that too.
Building the *records* under the lock closed the race, but `_as_embedding` is a 2048-float
`tolist()` per object, so at ~15 000 objects/s that is ~30M conversions a second serialised
inside the one mutex every worker takes in `open`/`expect`/`deliver`/`seal` — and `sweep()`
built the whole expired list in a single hold, so one wedged instance blocked every worker.

So the division is: **capture under the lock, build outside it.** Both halves are asserted
here as properties. The previous version of this file asserted on `inspect.getsource` text
(`source.count("self._snapshot)") == 4`), which pinned the implementation rather than the
behaviour and went red for a change that improved the code.
"""

from __future__ import annotations

import numpy as np

from shipinfer.core.request import RequestContext
from shipinfer.core.settings.pipeline import ReassemblySettings
from shipinfer.pipeline.graph.detections import Detections
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.graph.state import FrameState
from shipinfer.pipeline.reassembly.collector import (
    COMPLETE,
    EVICTED,
    SHUTDOWN,
    TIMEOUT,
    FrameCollector,
    FrameResult,
)

FIELD_MAP = {"embedding": ("person_embedding",)}


class AdvancingClock:
    """A monotonic clock a test can step past the reassembly timeout.

    A *constant* clock cannot expire anything: `opened_ns` is stamped from this same clock,
    so `now - opened_ns` is zero and `is_expired` is correctly False.
    """

    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now

    def advance_past(self, settings: ReassemblySettings | None = None) -> None:
        timeout_ms = (settings or ReassemblySettings()).timeout_ms
        self.now += timeout_ms * 1_000_000 + 1


def detections(count: int) -> Detections:
    return Detections(
        boxes=np.zeros((count, 4), dtype=np.float32),
        scores=np.full((count,), 0.9, dtype=np.float32),
        class_ids=np.zeros((count,), dtype=np.int32),
        labels=("person",) * count,
    )


def state_with(count: int, *, camera: str = "cam0", frame: int = 1) -> FrameState:
    return FrameState(
        context=RequestContext(camera_id=camera, frame_id=frame),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        detections=detections(count),
    )


def embeddings(count: int, value: float) -> ObjectBatch:
    return ObjectBatch(
        name="person_embedding",
        class_name="person",
        object_indices=tuple(range(count)),
        data=np.full((count, 4), value, dtype=np.float32),
    )


class TestTheCaptureIsTakenBeforeTheWorkerCanMoveOn:
    """The race itself, asserted on what it produced: a result that changes under you.

    Driven through `seal`, which is the ordinary finish, and through `sweep`, which is the
    path the bug lived on — the sweeper finishes a frame whose worker has not returned.
    """

    def _collector(self, results: list[FrameResult]) -> FrameCollector:
        return FrameCollector(results.append)

    def test_a_sealed_frame_is_not_affected_by_later_mutation(self) -> None:
        results: list[FrameResult] = []
        collector = self._collector(results)
        state = state_with(3)
        collector.open(state, expected=("person_embedder",))
        state.attach(embeddings(3, 1.0))
        collector.deliver(state.key, "person_embedder")
        collector.seal(state.key)

        # The worker carries on: exactly what ADR-002 says cannot be observed by the emitter.
        state.set_detections(detections(12))
        state.attach(embeddings(12, 9.0))

        (result,) = results
        records = result.inputs.records(FIELD_MAP)
        assert len(records) == 3, "the emitter saw the worker's later writes"
        assert all(r.embedding == (1.0, 1.0, 1.0, 1.0) for r in records)

    def test_a_swept_frame_is_not_affected_either(self) -> None:
        """The sweeper's frame is the dangerous one — its worker is still running."""
        results: list[FrameResult] = []
        clock = AdvancingClock()
        collector = FrameCollector(results.append, clock=clock)
        state = state_with(2)
        collector.open(state, expected=("person_embedder",))
        state.attach(embeddings(2, 1.0))
        clock.advance_past()

        assert collector.sweep() == 1

        state.set_detections(detections(20))

        (result,) = results
        assert result.reason == TIMEOUT
        assert len(result.inputs.records(FIELD_MAP)) == 2

    def test_the_batches_map_is_copied_not_aliased(self) -> None:
        """`attach` after the capture must not appear in it, and neither must a `drop`."""
        results: list[FrameResult] = []
        collector = self._collector(results)
        state = state_with(2)
        collector.open(state)
        state.attach(embeddings(2, 4.0))
        collector.seal(state.key)

        state.drop("person_embedding")

        (result,) = results
        records = result.inputs.records(FIELD_MAP)
        assert all(r.embedding == (4.0, 4.0, 4.0, 4.0) for r in records)


class TestTheBuildIsNotDoneUnderTheLock:
    """The contention half. Asserted directly: the lock is free while emission runs.

    `threading.Lock.locked()` makes this a real property rather than a source-text check.
    Reaching into `_lock` is deliberate — the whole point of the finding is *where* the work
    happens relative to that specific mutex, and no public surface exposes it.
    """

    def test_the_lock_is_free_while_the_emit_callback_runs(self) -> None:
        held: list[bool] = []
        collector = FrameCollector(lambda _result: held.append(collector._lock.locked()))
        state = state_with(2)
        collector.open(state)
        collector.seal(state.key)

        assert held == [False], "emission ran inside the lock every worker contends on"

    def test_the_lock_is_free_for_every_frame_a_sweep_finishes(self) -> None:
        """`sweep()` is the case that mattered: it finished the whole expired list in one
        hold, so a single wedged instance stalled every worker for tens of ms."""
        held: list[bool] = []
        clock = AdvancingClock()
        collector = FrameCollector(
            lambda _result: held.append(collector._lock.locked()), clock=clock
        )
        for frame in range(5):
            collector.open(state_with(2, frame=frame), expected=("person_embedder",))
        clock.advance_past()

        assert collector.sweep() == 5
        assert held == [False] * 5

    def test_the_lock_is_free_during_a_drain(self) -> None:
        held: list[bool] = []
        collector = FrameCollector(lambda _result: held.append(collector._lock.locked()))
        for frame in range(3):
            collector.open(state_with(1, frame=frame), expected=("person_embedder",))

        assert collector.drain() == 3
        assert held == [False] * 3


class TestEveryFinishingPathCaptures:
    """Four ways a frame ends, and none of them may reach the emitter empty-handed.

    Asserted by driving each path rather than by counting occurrences of a call in the
    source — the previous version did the latter, and it could not have told the difference
    between a fifth path being added correctly and being added wrong.
    """

    def test_the_normal_path(self) -> None:
        results: list[FrameResult] = []
        collector = FrameCollector(results.append)
        state = state_with(1)
        collector.open(state)
        collector.seal(state.key)

        assert results[0].reason == COMPLETE
        assert results[0].inputs is not None

    def test_the_timeout_path(self) -> None:
        results: list[FrameResult] = []
        clock = AdvancingClock()
        collector = FrameCollector(results.append, clock=clock)
        collector.open(state_with(1), expected=("person_embedder",))
        clock.advance_past()
        collector.sweep()

        assert results[0].reason == TIMEOUT
        assert results[0].inputs is not None

    def test_the_shutdown_path(self) -> None:
        results: list[FrameResult] = []
        collector = FrameCollector(results.append)
        collector.open(state_with(1), expected=("person_embedder",))
        collector.drain()

        assert results[0].reason == SHUTDOWN
        assert results[0].inputs is not None

    def test_the_eviction_path(self) -> None:
        """A full buffer evicting to make room. One-deep, so the second frame forces it."""
        results: list[FrameResult] = []
        collector = FrameCollector(results.append, settings=ReassemblySettings(capacity=1))
        collector.open(state_with(1, frame=1), expected=("person_embedder",))
        collector.open(state_with(1, frame=2), expected=("person_embedder",))

        assert [r.reason for r in results] == [EVICTED]
        assert results[0].inputs is not None

    def test_a_result_constructed_directly_has_no_capture(self) -> None:
        """`None` means "nobody captured", which is how a hand-built result in a test still
        works — there, nothing else is touching the state."""
        assert FrameResult(state=state_with(1), delivered=(), missing=()).inputs is None


class TestRecordsAreInternallyConsistent:
    """One read of `detections` for the whole build, whatever happens around it."""

    def test_detections_growing_mid_build_cannot_index_past_the_end(self) -> None:
        state = state_with(0)
        records = state.objects(FIELD_MAP)

        state.set_detections(detections(12))

        assert records == (), "the call that ran when there were none returns none"
        assert len(state.objects(FIELD_MAP)) == 12, "and a later call sees the new ones"

    def test_a_record_is_produced_for_every_detection(self) -> None:
        assert len(state_with(5).objects(FIELD_MAP)) == 5

    def test_a_detection_with_no_batch_row_yields_no_fields(self) -> None:
        records = state_with(3).objects(FIELD_MAP)

        assert len(records) == 3
        assert all(record.embedding == () for record in records)
