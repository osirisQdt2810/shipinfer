"""The sweeper must not read a `FrameState` its owning worker is still writing.

`FrameState` is documented as owned by exactly one worker for the frame's whole life
(ADR-002), and the sweeper broke that: it built a `FrameResult` under the collector's lock
but the emission read `state` afterwards, *outside* it, while the owning worker was still
inside `graph.execute`. With a 1500 ms reassembly timeout against a 5000 ms stage timeout,
any stage waiting on a backed-up model queue guarantees the overlap.

The failure was not a crash in the usual place. The reader saw an empty detection list, the
wedged stage then answered and the worker filled twelve in, and the record generator indexed
past the end of a list it had already sized. `_emit` caught that, incremented
`build_failures` and returned — but the collector had already counted the frame as reported
and `_emit` had already popped its future. So the frame vanished from the sink while
"every opened frame is reported exactly once" still read green, and the caller waited on a
future nothing would ever resolve.
"""

from __future__ import annotations

import numpy as np

from shipinfer.core.request import RequestContext
from shipinfer.pipeline.graph.detections import Detections
from shipinfer.pipeline.graph.state import FrameState


def state_with(count: int) -> FrameState:
    return FrameState(
        context=RequestContext(camera_id="cam0", frame_id=1),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        detections=detections(count),
    )


def detections(count: int) -> Detections:
    return Detections(
        boxes=np.zeros((count, 4), dtype=np.float32),
        scores=np.full((count,), 0.9, dtype=np.float32),
        class_ids=np.zeros((count,), dtype=np.int32),
        labels=("person",) * count,
    )


class TestObjectsIsInternallyConsistent:
    """One read of `detections` for the whole call, whatever happens around it."""

    def test_detections_growing_mid_call_does_not_index_past_the_end(self) -> None:
        """The generator re-read `self.detections` while `fields` had been sized from an
        earlier read. Twelve detections against a zero-length `fields` is an IndexError."""
        state = state_with(0)
        records = state.objects({})

        state.set_detections(detections(12))

        assert records == (), "the call that ran when there were none returns none"
        assert len(state.objects({})) == 12, "and a later call sees the new ones"

    def test_a_record_is_produced_for_every_detection(self) -> None:
        assert len(state_with(5).objects({})) == 5

    def test_a_detection_index_beyond_the_field_list_yields_no_fields(self) -> None:
        """Defence in depth: the scatter loop guarded its index and this one did not."""
        state = state_with(3)

        records = state.objects({})

        assert len(records) == 3
        assert all(record.embedding == () for record in records)


class TestTheSnapshotIsTakenUnderTheLock:
    """The structural half: the collector captures records, the emitter does not re-read."""

    def test_frame_result_can_carry_its_objects(self) -> None:
        from shipinfer.pipeline.reassembly.collector import FrameResult

        state = state_with(2)
        result = FrameResult(state=state, delivered=(), missing=(), objects=("a", "b"))

        assert result.objects == ("a", "b")

    def test_a_result_built_without_one_says_so(self) -> None:
        """`None` means "nobody snapshotted", which is how a directly-constructed result in
        a test still works — there, nothing else is touching the state."""
        from shipinfer.pipeline.reassembly.collector import FrameResult

        assert FrameResult(state=state_with(1), delivered=(), missing=()).objects is None

    def test_the_runner_supplies_a_snapshot_to_the_collector(self) -> None:
        """Asserted on the wiring, because the race is invisible in a single-threaded test
        and a collector constructed without the snapshot silently reverts to the old path."""
        import inspect

        from shipinfer.pipeline import runner

        source = inspect.getsource(runner.PipelineRunner.__init__)
        assert "snapshot=" in source, "the collector was built without a snapshot"

    def test_every_finishing_path_snapshots(self) -> None:
        """Eviction, completion, timeout and shutdown all end a frame, and the timeout path
        is the one the sweeper takes — the only one where the race exists."""
        import inspect

        from shipinfer.pipeline.reassembly import collector

        source = inspect.getsource(collector)
        assert source.count("self._snapshot)") == 4
