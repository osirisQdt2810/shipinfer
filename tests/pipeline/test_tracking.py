"""Plane 3 inside the pipeline: one tracker per camera, in order, under threads.

Nothing here asserts a *literal* track id. ``shipvision`` hands out ids from one
process-wide counter — deliberately, so two cameras' tracklets can meet downstream without
colliding — so the id a test sees depends on how many tests ran before it. What is asserted
is the property: the same object keeps one id, two cameras never share one, and a frame that
arrives out of order changes neither.

The *wire* half of this feature — that a track id reaches the sink without disturbing the v1
payload — is pinned in ``test_schema.py`` instead, deliberately: it is a property of
:mod:`shipinfer.pipeline.schema` alone and must keep being checked on a checkout with no
tracking library, where everything in this module skips.

The tracker itself is the real one. It is pure numpy and it is the thing under test: a fake
tracker would have proved that this module can call a method, which is not the failure being
guarded against. The *models* are still fakes, as everywhere else in this tier.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, TrackingError
from shipinfer.core.request import RequestContext
from shipinfer.core.settings.pipeline import PipelineSettings, TrackingSettings
from shipinfer.pipeline.graph import (
    TRACK_IDS,
    TRACK_STATES,
    FrameState,
    PipelineGraph,
    StageStatus,
    TrackerShard,
    TrackStage,
    build_perception_graph,
    build_tracking_stage,
    tracking_available,
)
from shipinfer.pipeline.graph.detections import Detections
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.schema import ObjectRecord

#: Everything that needs a real tracker. Applied per class rather than to the module, because
#: the concurrency invariants below — mutual exclusion per camera, and *not* across cameras —
#: are properties of `TrackerShard` itself and need no tracker at all. As a module-level mark
#: they skipped in every tier that runs: the container does not install the submodule and CI
#: deliberately does not check it out (ADR-001), so the one part of Plane 3 that is a
#: threading correctness argument was tested nowhere.
needs_tracking = pytest.mark.skipif(
    not tracking_available(),
    reason="shipvision.tracking is not importable; the submodule is not checked out",
)

pytestmark = [pytest.mark.timeout(60)]

#: ``min_hits=1`` so a track is published on the frame it is born. The default is 3, which is
#: right in production — an identity that dies after two frames should never have been
#: published — and would mean every assertion here needed three warm-up frames it is not
#: about.
FAST = {"min_hits": 1}

#: The numpy reference, pinned. An unpinned build takes the fastest backend the host can
#: compile, which is right in production and wrong here for the reason ``conftest.py`` gives
#: about image ops: a test that quietly takes the compiled path on the dev box and the numpy
#: path on the runner is a test whose failures are not reproducible. Parity between the two
#: backends is ``shipvision``'s own claim and it has its own tests for it.
REFERENCE = "python"


def detections(*boxes: Sequence[float], label: str = "ship", class_id: int = 8) -> Detections:
    """A frame's detections, one row per box, all of one class."""
    array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return Detections(
        boxes=array,
        scores=np.full((array.shape[0],), 0.9, dtype=np.float32),
        class_ids=np.full((array.shape[0],), class_id, dtype=np.int32),
        labels=(label,) * array.shape[0],
    )


def state_for(camera: str, frame: int, dets: Detections) -> FrameState:
    """A frame that has already been through the detector."""
    state = FrameState(
        context=RequestContext(
            camera_id=camera, frame_id=frame, captured_ns=1_000, captured_unix_ns=2_000
        ),
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    state.set_detections(dets)
    return state


def track_ids(state: FrameState) -> dict[int, int]:
    """Detection index -> track id, read off the batch the stage attached."""
    batch = state.batches[TRACK_IDS]
    return {index: int(row[0]) for index, row in batch.scatter()}


def stage(
    *,
    options: dict | None = None,
    appearance: Sequence[str] = (),
    **settings_fields,
) -> TrackStage:
    """The stage a deployment's settings would build, via the same function production uses."""
    configured = TrackingSettings(
        enabled=True,
        backend=REFERENCE,
        options={**FAST, **(options or {})},
        **settings_fields,
    )
    return build_tracking_stage(configured, appearance=appearance)


def run(subject: TrackStage, camera: str, frame: int, dets: Detections) -> FrameState:
    """Run one frame through the stage and hand back the mutated state."""
    state = state_for(camera, frame, dets)
    outcome = subject.run(state)
    assert outcome.status is StageStatus.RAN, outcome.error
    return state


@needs_tracking
class TestATrackIdIsStableAcrossFrames:
    """The whole point of the plane: one object, one identity, for as long as it is seen."""

    def test_one_object_keeps_one_id(self):
        subject = stage()
        seen = {
            frame: track_ids(run(subject, "cam0", frame, detections([10, 10, 30, 50])))[0]
            for frame in range(1, 6)
        }
        assert len(set(seen.values())) == 1, seen

    def test_an_object_that_moves_keeps_its_id(self):
        """Association, not a coincidence of identical boxes."""
        subject = stage()
        seen = []
        for frame in range(1, 6):
            shift = float(frame) * 3.0
            state = run(subject, "cam0", frame, detections([10 + shift, 10, 30 + shift, 50]))
            seen.append(track_ids(state)[0])
        assert len(set(seen)) == 1, seen

    def test_two_objects_keep_two_distinct_ids(self):
        subject = stage()
        first = track_ids(run(subject, "cam0", 1, detections([0, 0, 20, 40], [90, 0, 110, 40])))
        assert len(set(first.values())) == 2
        second = track_ids(
            run(subject, "cam0", 2, detections([0, 0, 20, 40], [90, 0, 110, 40]))
        )
        assert second == first

    def test_the_state_reaches_the_record(self):
        """A consumer that has to trust our filtering cannot apply its own."""
        subject = stage()
        state = run(subject, "cam0", 1, detections([10, 10, 30, 50]))
        assert state.batches[TRACK_STATES].data[0, 0] == "confirmed"


@needs_tracking
class TestTwoCamerasNeverShareTrackerState:
    """Sharing one tracker does not degrade tracking; it reports an object where none was."""

    def test_each_camera_gets_its_own_tracker(self):
        subject = stage()
        run(subject, "cam0", 1, detections([10, 10, 30, 50]))
        run(subject, "cam1", 1, detections([10, 10, 30, 50]))
        shard = subject.shard
        assert set(shard.cameras) == {"cam0", "cam1"}
        assert shard.tracker_for("cam0") is not shard.tracker_for("cam1")

    def test_an_identical_box_on_two_cameras_is_two_identities(self):
        """The boxes are the same pixels; the objects are not, and nothing here may join
        them. Joining across cameras is the cross-camera tier's job and its own decision."""
        subject = stage()
        first = track_ids(run(subject, "cam0", 1, detections([10, 10, 30, 50])))[0]
        second = track_ids(run(subject, "cam1", 1, detections([10, 10, 30, 50])))[0]
        assert first != second

    def test_one_cameras_frames_never_advance_anothers_stream(self):
        """cam1 starting at frame 100 must not make cam0's frame 2 look like a replay."""
        subject = stage()
        run(subject, "cam1", 100, detections([10, 10, 30, 50]))
        outcome = subject.run(state_for("cam0", 2, detections([10, 10, 30, 50])))
        assert outcome.status is StageStatus.RAN


@needs_tracking
class TestOutOfOrderFramesDoNotCorruptIds:
    """Reassembly does not order anything, so the shard has to."""

    def test_a_replayed_frame_is_refused(self):
        subject = stage()
        run(subject, "cam0", 5, detections([10, 10, 30, 50]))
        outcome = subject.run(state_for("cam0", 5, detections([10, 10, 30, 50])))
        assert outcome.status is StageStatus.FAILED
        assert isinstance(outcome.error, TrackingError)

    def test_a_late_frame_is_refused_and_named(self):
        subject = stage()
        run(subject, "cam0", 9, detections([10, 10, 30, 50]))
        outcome = subject.run(state_for("cam0", 8, detections([10, 10, 30, 50])))
        assert outcome.status is StageStatus.FAILED
        assert "cam0" in str(outcome.error) and "8" in str(outcome.error)

    def test_a_refused_frame_carries_no_track_batch(self):
        """Not an empty one: a stage that failed must not look like a stage that ran and
        found nothing, or the event would claim complete."""
        subject = stage()
        run(subject, "cam0", 5, detections([10, 10, 30, 50]))
        late = state_for("cam0", 4, detections([10, 10, 30, 50]))
        subject.run(late)
        assert TRACK_IDS not in late.batches

    def test_the_ids_either_side_of_an_inversion_are_unchanged(self):
        """The frame that lost the race is the only thing that is lost."""
        subject = stage()
        box = [10, 10, 30, 50]
        before = track_ids(run(subject, "cam0", 1, detections(box)))[0]
        run(subject, "cam0", 2, detections(box))
        subject.run(state_for("cam0", 2, detections(box)))  # the loser of the race
        after = track_ids(run(subject, "cam0", 3, detections(box)))[0]
        assert after == before

    def test_a_refusal_is_counted_so_an_operator_can_see_the_rate(self):
        subject = stage()
        run(subject, "cam0", 4, detections([10, 10, 30, 50]))
        for frame in (1, 2, 3):
            subject.run(state_for("cam0", frame, detections([10, 10, 30, 50])))
        assert subject.shard.stats()["out_of_order"] == 3

    def test_a_reconnected_camera_may_restart_its_numbering(self):
        subject = stage()
        run(subject, "cam0", 500, detections([10, 10, 30, 50]))
        subject.shard.reset("cam0")
        outcome = subject.run(state_for("cam0", 0, detections([10, 10, 30, 50])))
        assert outcome.status is StageStatus.RAN

    def test_a_reconnected_camera_does_not_continue_its_old_identities(self):
        subject = stage()
        before = track_ids(run(subject, "cam0", 1, detections([10, 10, 30, 50])))[0]
        subject.shard.reset("cam0")
        after = track_ids(run(subject, "cam0", 2, detections([10, 10, 30, 50])))[0]
        assert after != before


@needs_tracking
class TestThreadsCannotInterleaveOneCamerasState:
    """The pipeline is multi-threaded and a tracker is not re-entrant."""

    def test_cameras_track_independently_under_concurrency(self):
        """Fifty cameras, each fed in order by its own thread, all at once. Every camera
        must come out with exactly one stable identity and no refusals."""
        subject = stage()
        cameras = [f"cam{index}" for index in range(50)]
        results: dict[str, list[int]] = {}
        failures: list[BaseException] = []
        barrier = threading.Barrier(len(cameras))

        def feed(camera: str) -> None:
            observed: list[int] = []
            barrier.wait()
            for frame in range(1, 8):
                state = state_for(camera, frame, detections([10, 10, 30, 50]))
                outcome = subject.run(state)
                if outcome.status is not StageStatus.RAN:
                    failures.append(outcome.error or RuntimeError(camera))
                    return
                observed.append(track_ids(state)[0])
            results[camera] = observed

        threads = [threading.Thread(target=feed, args=(camera,)) for camera in cameras]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        assert not failures, failures
        assert len(results) == len(cameras)
        for camera, observed in results.items():
            assert len(set(observed)) == 1, (camera, observed)
        # One id per camera, and no two cameras sharing one.
        assert len({ids[0] for ids in results.values()}) == len(cameras)

    def test_racing_frames_on_one_camera_are_serialised_not_interleaved(self):
        """Two workers with two of one camera's frames is the case that must not corrupt
        anything. Whichever loses is refused; nothing in between is possible, so every
        attempt is accounted for exactly once and the stream never goes backwards."""
        subject = stage()
        attempts = 40
        accepted: list[int] = []
        refused = 0
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def feed(frames: range) -> None:
            nonlocal refused
            barrier.wait()
            for frame in frames:
                outcome = subject.run(state_for("cam0", frame, detections([10, 10, 30, 50])))
                with lock:
                    if outcome.status is StageStatus.RAN:
                        accepted.append(frame)
                    else:
                        assert isinstance(outcome.error, TrackingError)
                        refused += 1

        threads = [
            threading.Thread(target=feed, args=(range(1, attempts + 1, 2),)),
            threading.Thread(target=feed, args=(range(2, attempts + 1, 2),)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        assert len(accepted) + refused == attempts
        assert len(set(accepted)) == len(accepted)
        assert subject.shard.stats()["out_of_order"] == refused


@needs_tracking
class TestAnEmptyFrameStillRuns:
    """Ageing is how a track dies, and a frame with nothing in it is what ages one."""

    def test_a_frame_with_no_detections_is_not_skipped(self):
        subject = stage()
        outcome = subject.run(state_for("cam0", 1, Detections.empty()))
        assert outcome.status is StageStatus.RAN
        assert outcome.rows == 0

    def test_a_departed_object_is_eventually_forgotten(self):
        subject = stage(options={"max_age": 2})
        run(subject, "cam0", 1, detections([10, 10, 30, 50]))
        assert subject.shard.stats()["tracks"] == 1
        for frame in range(2, 8):
            subject.run(state_for("cam0", frame, Detections.empty()))
        assert subject.shard.stats()["tracks"] == 0

    def test_the_stage_never_gates_on_detections_being_non_empty(self):
        """`requires` would skip the empty frames, and a skipped stage does not age."""
        assert stage().requires == ()

    def test_the_stage_is_skipped_when_the_detector_did_not_answer(self):
        """A frame with no objects and a frame with no answer are different things, and only
        one of them may age a track.

        Asserted through the planner's own predicate rather than by reading the stage's
        declarations, because the planner is what actually decides — and the plan is also
        what reassembly waits for, so a stage announced and never run is a frame that sits
        until the timeout.
        """
        graph = PipelineGraph([stage()], field_map={"track_id": (TRACK_IDS,)}, name="t")
        blind = FrameState(
            context=RequestContext(camera_id="cam0", frame_id=1),
            image=np.zeros((64, 64, 3), dtype=np.uint8),
        )
        assert graph.runnable(blind) == ()
        assert graph.runnable(state_for("cam0", 1, Detections.empty())) == ("track",)


@needs_tracking
class TestAppearanceIsOptionalNotRequired:
    """A frame holding only people has no ship embedding, and must still be tracked."""

    def test_the_embedders_are_declared_optional(self):
        subject = stage(appearance=("ship_embedding", "person_embedding"))
        assert subject.optional == ("ship_embedding", "person_embedding")
        assert subject.consumes == ("detections",)
        assert set(subject.reads) == {"detections", "ship_embedding", "person_embedding"}

    def test_a_missing_embedding_does_not_stop_the_stage(self):
        subject = stage(appearance=("ship_embedding", "person_embedding"))
        outcome = subject.run(state_for("cam0", 1, detections([10, 10, 30, 50])))
        assert outcome.status is StageStatus.RAN

    def test_an_embedding_that_landed_reaches_the_tracker(self):
        subject = stage(appearance=("ship_embedding",))
        state = state_for("cam0", 1, detections([10, 10, 30, 50]))
        state.attach(
            ObjectBatch(
                name="ship_embedding",
                class_name="ship",
                object_indices=(0,),
                data=np.ones((1, 8), dtype=np.float32),
            )
        )
        subject.run(state)
        carried = subject.shard.tracker_for("cam0").tracks[0].embedding
        assert carried is not None and carried.shape == (8,)


@needs_tracking
class TestAttribution:
    """A published track is matched back to the detection row that produced it.

    The library returns tracks, not row indices, and it is right not to — a track's box is
    the filtered estimate, and which detection fed it is the tracker's business. An event
    record is per detection, so the mapping has to be recovered here, and these are the
    properties that recovery has to have.
    """

    def test_the_id_follows_the_box_not_the_row_index(self):
        """The detector orders by score, so a row index is not stable frame to frame. An
        attribution keyed on position rather than overlap would swap two identities the
        moment two objects' confidences crossed."""
        subject = stage()
        left, right = [0, 0, 20, 40], [90, 0, 110, 40]
        first = track_ids(run(subject, "cam0", 1, detections(left, right)))
        # Same two objects, presented in the other order.
        second = track_ids(run(subject, "cam0", 2, detections(right, left)))
        assert second[0] == first[1]
        assert second[1] == first[0]

    def test_no_detection_receives_two_ids(self):
        """Greedy per-track argmax gets this wrong for two people walking together: both
        track ids land on one box, and downstream reads one object that teleported and one
        that never existed."""
        subject = stage()
        crowd = [[0, 0, 30, 60], [10, 0, 40, 60], [20, 0, 50, 60], [30, 0, 60, 60]]
        state = run(subject, "cam0", 1, detections(*crowd))
        indices = state.batches[TRACK_IDS].object_indices
        assert len(set(indices)) == len(indices)
        assert len(set(track_ids(state).values())) == len(indices)

    def test_rows_are_in_the_frames_own_order(self):
        """Two runs over one frame produce byte-identical batches, which is what makes a
        replay assertion possible at all."""
        subject = stage()
        state = run(
            subject, "cam0", 1, detections([90, 0, 110, 40], [0, 0, 20, 40], [45, 0, 65, 40])
        )
        assert list(state.batches[TRACK_IDS].object_indices) == [0, 1, 2]

    def test_the_threshold_refuses_an_answer_rather_than_forcing_one(self):
        """Set impossibly strict — nothing but an exact box may match — and the stage
        publishes no id at all rather than the least-bad one. The knob exists to say "I do
        not know", which is a thing an identity consumer must be able to be told."""
        subject = stage(attribution_iou=1.0)
        run(subject, "cam0", 1, detections([0, 0, 20, 40]))
        state = run(subject, "cam0", 2, detections([6, 0, 26, 40]))
        assert track_ids(state) == {}

    def test_an_unattributed_detection_has_no_track_id(self):
        subject = stage(attribution_iou=1.0)
        run(subject, "cam0", 1, detections([0, 0, 20, 40]))
        state = run(subject, "cam0", 2, detections([6, 0, 26, 40]))
        records = state.objects({"track_id": (TRACK_IDS,)})
        assert records[0].track_id is None


@needs_tracking
class TestTheStageIsOffUnlessAskedFor:
    """Enabling tracking changes the shape of the benchmark, so an operator opts in."""

    def test_tracking_is_disabled_by_default(self):
        assert PipelineSettings().tracking.enabled is False

    def test_a_default_graph_has_no_tracking_stage(self, pipeline_settings, ops, models):
        graph = build_perception_graph(pipeline_settings, resolve=models.__getitem__, ops=ops)
        assert "track" not in graph.stage_names
        assert "track_id" not in graph.field_map

    def test_enabling_it_appends_the_stage_last(self, pipeline_settings, ops, models):
        graph = self._tracked(pipeline_settings, ops, models)
        assert graph.stage_names[-1] == "track"

    def test_enabling_it_maps_the_two_track_fields(self, pipeline_settings, ops, models):
        graph = self._tracked(pipeline_settings, ops, models)
        assert graph.field_map["track_id"] == (TRACK_IDS,)
        assert graph.field_map["track_state"] == (TRACK_STATES,)

    def test_the_tracked_graph_validates(self, pipeline_settings, ops, models):
        """`optional` names are checked at start-up like every other edge: a graph asking
        for an appearance batch nothing produces must not start and then track blind."""
        self._tracked(pipeline_settings, ops, models).validate(models.__getitem__)

    def test_a_disabled_graph_leaves_the_field_unset_not_zero(self):
        """0 is a legitimate id in some schemes; `None` says the stage did not run."""
        assert (
            ObjectRecord(det_id="d", class_name="person", score=0.5, bbox=(0, 0, 1, 1)).track_id
            is None
        )

    def _tracked(self, pipeline_settings, ops, models):
        settings = pipeline_settings.model_copy(
            update={"tracking": TrackingSettings(enabled=True, backend=REFERENCE, options=FAST)}
        )
        return build_perception_graph(settings, resolve=models.__getitem__, ops=ops)


@needs_tracking
class TestTheTrackerIsSelectedByName:
    """A registry lookup, so adding a tracker upstream needs no edit here."""

    @pytest.mark.parametrize(
        "algorithm", ["sort", "bytetrack", "ocsort", "botsort", "deepsortv2"]
    )
    def test_every_shipped_tracker_can_be_named(self, algorithm: str):
        options = dict(FAST)
        if algorithm == "botsort":
            # The only one that reads pixels, and this stage does not hand it any — see
            # `TrackingSettings.needs_frame` for why the frame is not kept alive that long.
            options["cmc"] = "none"
        shard = TrackerShard(algorithm, options=options, backend=REFERENCE)
        assert shard.algorithm == algorithm

    def test_an_unknown_tracker_stops_the_deploy(self):
        with pytest.raises(ConfigurationError, match="nosuchtracker"):
            TrackerShard("nosuchtracker")

    def test_a_typo_in_the_options_stops_the_deploy(self):
        """Rather than surfacing identically on every frame from inside a worker thread."""
        with pytest.raises(ConfigurationError, match="max_agee"):
            TrackerShard("bytetrack", options={"max_agee": 30}, backend=REFERENCE)

    def test_the_default_is_bytetrack(self):
        assert TrackingSettings().algorithm == "bytetrack"


class _Tag:
    """The two fields `TrackerShard.update` reads off a `shipvision.types.Detections`.

    A stand-in rather than the real type, so the locking tests below run in the offline tier.
    The submodule is not checked out in CI (ADR-001) and is not installed in the test
    container, which is precisely why the per-camera lock had no test that ran anywhere.
    """

    __slots__ = ("camera_id", "frame_id")

    def __init__(self, camera_id: str, frame_id: int) -> None:
        self.camera_id = camera_id
        self.frame_id = frame_id


class _Detections:
    __slots__ = ("tag",)

    def __init__(self, tag: _Tag) -> None:
        self.tag = tag


def _shipvision_detections(camera: str, frame: int) -> _Detections:
    return _Detections(_Tag(camera, frame))


class TestThePerCameraLockActuallyExcludes:
    """The class above this one is satisfied by the high-water mark alone.

    Review proved it: replacing `_CameraShard.lock` with `contextlib.nullcontext()` left all
    42 tests green, three runs out of three, while mutating the frame-id guard reddened three.
    So the lock — which the module docstring calls the first half of the invariant — had no
    test at all, and this repository has blocked two PRs for exactly that shape.

    **Concurrency here cannot be observed through the tracker's output**, and finding that out
    is what made these tests right. Two threads racing one camera never both reach the
    tracker: the ordering guard refuses whichever checks second, so a re-entrancy detector
    reports zero overlap *with or without* the lock. That was the first attempt, and it was
    vacuous for a subtler reason than the one it replaced.

    What the lock actually guarantees is that reading the high-water mark, writing it, and
    running the tracker happen as one step. So the property is asserted from *inside* the
    critical section: the tracker checks whether the shard's lock is held while it runs. That
    is exactly what a `nullcontext` cannot satisfy.
    """

    class _LockObserver:
        """A tracker that records whether its camera's lock was held while it ran.

        Given the shard after construction, because `_CameraShard` holds the lock and the
        tracker together — reading one without the other is the race the module prevents.
        """

        def __init__(self) -> None:
            self.shard: Any = None
            self.held: list[bool] = []
            self.calls = 0
            self.pool_size = 0

        def update(self, detections, *, image=None):
            self.calls += 1
            lock = getattr(self.shard, "lock", None)
            # `locked()` exists on `threading.Lock` and on nothing else this could be, so an
            # absent method is itself the answer: whatever is guarding this is not a lock.
            self.held.append(bool(getattr(lock, "locked", lambda: False)()))
            return []

        def reset(self) -> None:
            pass

    class _SlowTracker:
        def __init__(self, hold_s: float) -> None:
            self._hold_s = hold_s
            self.pool_size = 0

        def update(self, detections, *, image=None):
            time.sleep(self._hold_s)
            return []

        def reset(self) -> None:
            pass

    def _sharded(self) -> TrackerShard:
        """A `TrackerShard` with a stand-in tracker, built without shipvision.

        `TrackerShard.__init__` refuses when the registry is absent, so the object is made
        through `__new__` and given the same fields. Reaching past the constructor is
        deliberate: the thing under test is the locking in `update`, and needing the submodule
        to test it is *why* this property went untested — the container does not install it
        and CI does not check it out.
        """
        shard = TrackerShard.__new__(TrackerShard)
        shard._algorithm = "stand-in"
        shard._options = {}
        shard._backend = None
        shard._admit = threading.Lock()
        shard._cameras = {}
        return shard

    def _seed(self, sharded: TrackerShard, camera: str, tracker):
        from shipinfer.pipeline.graph.tracking import _CameraShard

        cell = _CameraShard(tracker)
        sharded._cameras[camera] = cell
        return cell

    def test_the_cameras_lock_is_held_while_its_tracker_runs(self) -> None:
        sharded = self._sharded()
        observer = self._LockObserver()
        observer.shard = self._seed(sharded, "cam0", observer)

        sharded.update(_shipvision_detections("cam0", 1))
        sharded.update(_shipvision_detections("cam0", 2))

        assert observer.calls == 2
        assert observer.held == [True, True], (
            "the tracker ran without its camera's lock held, so the check-set-update sequence "
            "is not atomic and two workers can both pass the ordering guard"
        )

    def test_reset_takes_the_same_lock(self) -> None:
        """`reset` rewinds the high-water mark and clears the tracks. Doing that beside a
        frame in flight is the same race with a worse outcome — the frame would be tracked
        against a pool that is being emptied under it."""
        sharded = self._sharded()
        observer = self._LockObserver()
        cell = self._seed(sharded, "cam0", observer)

        held: list[bool] = []
        original = observer.reset
        observer.reset = lambda: held.append(cell.lock.locked()) or original()  # type: ignore[assignment]

        sharded.reset("cam0")

        assert held == [True]

    def test_the_lock_is_per_camera_not_shared(self) -> None:
        """The other half of the design, and the one that would silently cost throughput: a
        slow camera must not stall the other forty-nine. Fifty cameras behind one is the
        failure this project exists to prevent, one layer up."""
        sharded = self._sharded()
        self._seed(sharded, "slow", self._SlowTracker(0.4))
        self._seed(sharded, "quick", self._SlowTracker(0.0))

        started = threading.Event()

        def feed_slow() -> None:
            started.set()
            sharded.update(_shipvision_detections("slow", 1))

        thread = threading.Thread(target=feed_slow)
        thread.start()
        started.wait(5)
        time.sleep(0.05)  # the slow camera is definitely inside its critical section

        began = time.perf_counter()
        sharded.update(_shipvision_detections("quick", 1))
        elapsed = time.perf_counter() - began
        thread.join(30)

        assert elapsed < 0.2, (
            f"the quick camera waited {elapsed:.3f}s behind a slow one — the lock is shared "
            f"across cameras rather than held per camera"
        )
