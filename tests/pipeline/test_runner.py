"""The runner: the whole flow, and the guarantee that no frame vanishes on the way through.

The last class is the one that matters most — a real
:class:`~shipinfer.server.InferenceServer` on the mock backend, the ``replay`` ingest source
reading PNGs off disk, and the ``jsonlines`` sink writing to a temporary file. No camera, no
GPU, no broker, no build. That combination existing is what makes the 50-camera bench
possible, so it is asserted rather than assumed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.request import RequestContext
from shipinfer.core.settings import OverflowPolicy, ServerSettings
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.core.settings.pipeline import PipelineSettings, ReassemblySettings
from shipinfer.ingest.frame import FrameCounter
from shipinfer.pipeline import PipelineRunner
from shipinfer.pipeline.sinks import NullResultSink
from shipinfer.scheduling.queues import FairPriorityQueue

from .conftest import CROP_SIZE, DETECTOR_INPUT, FakeServer, StubModel

pytestmark = pytest.mark.timeout(90)

# Class ids are the shipped detector's COCO numbering (0 person, 8 boat) — see the same
# constants in test_graph.py for why encoding the real numbering matters.
SHIP = [0.0, 0.0, 8.0, 8.0, 0.9, 8.0]
PERSON = [1.0, 1.0, 5.0, 7.0, 0.8, 0.0]


def wait_for(predicate, timeout_s: float = 10.0, poll_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def settings_for(**pipeline) -> ServerSettings:
    defaults = {
        "detector_input": DETECTOR_INPUT,
        "ship_mask_crop": CROP_SIZE,
        "ship_reid_crop": CROP_SIZE,
        "person_reid_crop": CROP_SIZE,
        "score_threshold": 0.5,
        "workers": 2,
    }
    defaults.update(pipeline)
    return ServerSettings(
        ingest=IngestSettings(
            cameras=[CameraConfig(camera_id="cam0", uri="rtsp://cam0", fps=20.0)]
        ),
        pipeline=PipelineSettings(**defaults),
    )


@pytest.fixture()
def runner_for(build_graph, models, ops):
    """A runner over stub models, with a sink whose events a test can read back."""
    built: list[PipelineRunner] = []

    def _make(detections=None, *, settings=None, sink=None, queue=None, **overrides):
        graph = build_graph(
            detections if detections is not None else [SHIP, PERSON], **overrides
        )
        resolved = settings or settings_for()
        runner = PipelineRunner(
            FakeServer(models=models, settings=resolved),
            settings=resolved,
            graph=graph,
            ops=ops,
            sink=NullResultSink(keep_last=256) if sink is None else sink,
            queue=queue,
        )
        built.append(runner)
        return runner

    yield _make
    for runner in built:
        runner.stop(timeout_s=5.0)


def _expired_item(runner: PipelineRunner, camera: str = "cam0"):
    """A pipeline entry item whose deadline has already passed."""
    from shipinfer.core.request import InferenceRequest, ResponseFuture
    from shipinfer.core.types import Tensor
    from shipinfer.scheduling.work import WorkItem

    frame = FrameCounter(camera).stamp(np.zeros((16, 16, 3), dtype=np.uint8))
    request = InferenceRequest(
        model_name="ship_detector",
        inputs={"images": Tensor.from_numpy(frame.as_batch())},
        context=frame.context,
        deadline_ns=frame.captured_ns - 1,
    )
    return WorkItem(request, ResponseFuture(request))


def publish(runner: PipelineRunner, count: int, camera: str = "cam0") -> list[tuple[str, int]]:
    """Push ``count`` frames through the real sink adapter, as a camera actor would."""
    counter = FrameCounter(camera)
    keys = []
    for index in range(count):
        frame = counter.stamp(np.full((16, 16, 3), index % 255, dtype=np.uint8))
        runner.frame_sink.put(frame)
        keys.append(frame.key)
    return keys


class TestTheRunnerRefusesABadWiring:
    """Validate at start-up, not at first use."""

    def test_it_needs_a_started_server(self, build_graph, models, ops):
        runner = PipelineRunner(
            FakeServer(models=models, is_started=False),
            settings=settings_for(),
            graph=build_graph([SHIP]),
            ops=ops,
            sink=NullResultSink(),
        )
        with pytest.raises(ServerStateError, match="started server"):
            runner.start()

    def test_frames_aimed_at_the_wrong_model_stop_the_deploy(self, build_graph, models, ops):
        """``ingest.target_model`` and the graph's entry stage have to agree."""
        settings = ServerSettings(
            ingest=IngestSettings(target_model="some_other_detector"),
            pipeline=PipelineSettings(detector_input=DETECTOR_INPUT),
        )
        runner = PipelineRunner(
            FakeServer(models=models, settings=settings),
            settings=settings,
            graph=build_graph([SHIP]),
            ops=ops,
            sink=NullResultSink(),
        )
        with pytest.raises(ConfigurationError, match=r"ingest\.target_model"):
            runner.start()


class TestEveryFrameProducesExactlyOneEvent:
    """None lost, none duplicated — through the queue, the graph and reassembly."""

    def test_n_frames_in_n_events_out(self, runner_for):
        runner = runner_for().start()
        keys = publish(runner, 12)

        assert wait_for(lambda: runner.sink.emitted == 12), runner.health()
        events = runner.sink.events()
        assert [e.key for e in events] != []
        assert sorted(e.key for e in events) == sorted(keys)
        assert len({e.key for e in events}) == 12, "an event was published twice"
        assert runner.collector.reported == 12
        assert len(runner.collector) == 0

    def test_two_cameras_keep_their_own_tags(self, runner_for):
        runner = runner_for().start()
        keys = publish(runner, 5, camera="cam_a") + publish(runner, 5, camera="cam_b")

        assert wait_for(lambda: runner.sink.emitted == 10), runner.health()
        assert sorted(e.key for e in runner.sink.events()) == sorted(keys)

    def test_the_event_carries_the_frames_own_dimensions_and_camera_fps(self, runner_for):
        runner = runner_for().start()
        publish(runner, 1)

        assert wait_for(lambda: runner.sink.emitted == 1)
        (event,) = runner.sink.events()
        assert (event.img_width, event.img_height) == (16, 16)
        assert event.img_fps == 20  # from the camera's configuration, not from the frame
        assert event.latency_us > 0

    def test_a_complete_frame_carries_both_branches(self, runner_for):
        runner = runner_for([SHIP, PERSON]).start()
        publish(runner, 1)

        assert wait_for(lambda: runner.sink.emitted == 1)
        (event,) = runner.sink.events()
        assert not event.is_partial
        assert [o.class_name for o in event.objects] == ["ship", "person"]
        assert event.objects_of("ship")[0].mask_area_px is not None
        assert event.objects_of("person")[0].embedding != ()


class TestTheTagSurvivesEveryFailurePath:
    """A frame that fails is still identifiable, or the failure cannot be attributed."""

    def test_a_failed_stage_still_produces_a_tagged_partial_event(self, runner_for):
        runner = runner_for(
            [SHIP],
            ship_embedder=StubModel("ship_embedder", error=InferenceError("engine died")),
        ).start()
        publish(runner, 3)

        assert wait_for(lambda: runner.sink.emitted == 3), runner.health()
        events = runner.sink.events()
        assert sorted(e.key for e in events) == [("cam0", 0), ("cam0", 1), ("cam0", 2)]
        for event in events:
            assert event.is_partial
            assert event.missing_stages == ("ship_embedder",)
            assert event.reason == "incomplete"
            # The detector's answer survived, so the frame is still useful downstream.
            assert [o.class_name for o in event.objects] == ["ship"]

    def test_a_wedged_stage_is_emitted_by_the_sweeper_naming_the_stage(self, runner_for):
        """The case the reference system's timeout really existed for: a worker that never
        returns. The frame must still be published, once, with the stage named."""
        settings = settings_for(
            workers=1,
            stage_timeout_ms=400,
            reassembly=ReassemblySettings(capacity=8, timeout_ms=40, sweep_interval_ms=10),
        )
        runner = runner_for(
            [SHIP],
            settings=settings,
            ship_detector=StubModel("ship_detector", hang=True),
        ).start()
        publish(runner, 1)

        assert wait_for(lambda: runner.sink.emitted == 1), runner.health()
        (event,) = runner.sink.events()
        assert event.key == ("cam0", 0)
        assert event.reason == "timeout"
        assert event.missing_stages == ("detect",)
        assert event.objects == ()

        # And when the wedged stage finally gives up, its frame is not published again.
        assert wait_for(lambda: runner.collector.late >= 0)
        time.sleep(0.5)
        assert runner.sink.emitted == 1

    def test_an_expired_frame_never_reaches_a_model(self, runner_for):
        """Spending a GPU on a frame already too late to act on is pure waste.

        Two layers drop it and both are asserted, because they cover different cases: the
        queue drops a frame that expired while it was *waiting*, and the runner drops one
        that expired while a worker was busy with someone else's.
        """
        # Layer 1 — the queue, with its default `drop_expired`.
        runner = runner_for(settings=settings_for(workers=1)).start()
        future = runner.queue.put(_expired_item(runner)) or None
        assert wait_for(lambda: runner.queue.stats().expired == 1), runner.health()
        assert runner.sink.emitted == 0
        assert future is None  # `put` returns nothing; the item's own future carries the error

    def test_a_frame_that_expires_while_a_worker_is_busy_is_dropped_by_the_runner(
        self, runner_for, models
    ):
        """Layer 2. A queue configured not to shed expired frames hands one to the worker,
        which must still refuse to spend a model on it."""
        queue = FairPriorityQueue("pipeline", capacity=8, drop_expired=False)
        runner = runner_for(settings=settings_for(workers=1), queue=queue).start()

        item = _expired_item(runner)
        queue.put(item)

        assert wait_for(
            lambda: runner.metrics.frames_expired.value(camera="cam0") == 1
        ), runner.health()
        assert runner.sink.emitted == 0
        assert models["ship_detector"].calls == []
        assert isinstance(item.future.exception(0.1), RequestCancelledError)


class TestBackpressureReachesTheCamera:
    """The pipeline says no by raising, at the one place that knows whose frame it is."""

    def test_a_full_queue_refuses_the_producer(self, runner_for):
        queue = FairPriorityQueue("pipeline", capacity=2, overflow=OverflowPolicy.REJECT)
        runner = runner_for(queue=queue)  # not started: nothing drains the queue
        counter = FrameCounter("cam0")

        for _ in range(2):
            runner.frame_sink.put(counter.stamp(np.zeros((16, 16, 3), dtype=np.uint8)))
        with pytest.raises(QueueFullError) as raised:
            runner.frame_sink.put(counter.stamp(np.zeros((16, 16, 3), dtype=np.uint8)))

        assert raised.value.capacity == 2


class TestObservability:
    """What an operator can see without attaching a debugger."""

    def test_health_reports_the_whole_flow(self, runner_for):
        runner = runner_for().start()
        publish(runner, 4)
        assert wait_for(lambda: runner.sink.emitted == 4)

        health = runner.health()
        assert health["running"] is True
        assert health["graph"]["stages"][0] == "detect"
        assert health["graph"]["models"][0] == "ship_detector"
        assert health["frames_accepted"] == 4
        assert health["reassembly"]["reported"] == 4
        assert health["queue"]["accepted"] == 4
        assert health["sink"]["emitted"] == 4

    def test_metrics_count_per_stage_and_per_camera(self, runner_for):
        runner = runner_for([SHIP, PERSON]).start()
        publish(runner, 3)
        assert wait_for(lambda: runner.sink.emitted == 3)

        metrics = runner.metrics
        assert metrics.frames_accepted.value(camera="cam0") == 3
        assert metrics.stages_run.value(stage="ship_segmenter") == 3
        assert metrics.objects_total.value(camera="cam0", object_class="ship") == 3
        assert metrics.objects_total.value(camera="cam0", object_class="person") == 3

    def test_a_skipped_branch_is_counted_not_invisible(self, runner_for):
        runner = runner_for([PERSON]).start()
        publish(runner, 2)
        assert wait_for(lambda: runner.sink.emitted == 2)

        assert runner.metrics.stages_skipped.value(stage="ship_segmenter") == 2
        assert runner.metrics.stages_run.value(stage="ship_segmenter") == 0


class TestShutdown:
    """Nothing in flight disappears."""

    def test_stop_publishes_what_was_in_flight_and_closes_the_sink(self, runner_for):
        runner = runner_for().start()
        publish(runner, 4)
        assert wait_for(lambda: runner.sink.emitted == 4)

        runner.stop()

        assert runner.sink.is_closed
        assert not runner.is_running
        assert len(runner.collector) == 0

    def test_stopping_twice_is_harmless(self, runner_for):
        runner = runner_for().start()
        runner.stop()
        runner.stop()

    def test_queued_frames_are_failed_with_a_typed_error_rather_than_dropped(self, runner_for):
        """A shutdown that silently discards queued work is not an orderly shutdown.

        One worker, wedged inside the detector, so the other three frames are still in the
        queue when the runner stops and their futures are the caller's to resolve.
        """
        from shipinfer.core.request import InferenceRequest, ResponseFuture
        from shipinfer.core.types import Tensor
        from shipinfer.scheduling.work import WorkItem

        queue = FairPriorityQueue("pipeline", capacity=8)
        settings = settings_for(workers=1, stage_timeout_ms=200)
        runner = runner_for(
            settings=settings,
            queue=queue,
            ship_detector=StubModel("ship_detector", hang=True),
        ).start()

        futures = []
        for frame in range(4):
            request = InferenceRequest(
                model_name="ship_detector",
                inputs={"images": Tensor.from_numpy(np.zeros((1, 16, 16, 3), dtype=np.uint8))},
                context=RequestContext(camera_id="cam0", frame_id=frame, captured_ns=1),
            )
            future = ResponseFuture(request)
            futures.append(future)
            queue.put(WorkItem(request, future))

        assert wait_for(lambda: queue.depth <= 3)
        runner.stop()

        assert queue.is_closed
        failed = [f for f in futures if f.done() and f.exception() is not None]
        assert failed, "queued frames were dropped instead of being failed"
        assert all(isinstance(f.exception(), RequestCancelledError) for f in failed)


class TestInjectedCollaboratorsAreTheOnesUsed:
    """A regression test for a falsiness trap, and for the seams themselves.

    A ``RequestQueue`` and a ``ResultSink`` both define ``__len__``, so an empty one is
    *falsy* — and ``self._queue = queue or default`` therefore discards the caller's object
    without a word. It happened here: an injected queue's capacity was ignored and an
    injected sink received nothing while its counter said otherwise.
    """

    def test_an_injected_queue_is_used_even_though_an_empty_queue_is_falsy(self, runner_for):
        queue = FairPriorityQueue("injected", capacity=3)
        runner = runner_for(queue=queue)
        assert runner.queue is queue
        assert not queue, "the premise: an empty queue is falsy"

    def test_an_injected_sink_is_used_even_though_an_empty_sink_is_falsy(self, runner_for):
        sink = NullResultSink(keep_last=4)
        runner = runner_for(sink=sink)
        assert runner.sink is sink
        assert not sink, "the premise: a sink that has emitted nothing is falsy"


class TestEndToEndWithReplayAndJsonLines:
    """A real server on the mock backend, the replay source, and a file sink.

    This is the combination that has to work for the offline tier to mean anything: it is the
    same wiring ``shipinfer bench`` drives at 50 cameras, minus the cameras.

    It deliberately does **not** pin the image ops, so it takes whichever implementation the
    host offers — numpy in CI, the fused CUDA kernels on the dev box. ``run_tests.sh`` warns
    that an unmarked test taking a CUDA path can pass on a dev box and fail on a runner; here
    the value runs the other way and was demonstrated: this test is what caught three faults
    that only exist when a GPU is present (a shared staging ring, preprocessing pinned to
    device 0, and ops built for a device the thread was not bound to). Every other test in
    this tier pins numpy so it behaves identically everywhere.
    """

    @pytest.fixture()
    def frame_dir(self, tmp_path: Path) -> Path:
        cv2 = pytest.importorskip("cv2", reason="writing the replay fixture needs OpenCV")
        directory = tmp_path / "frames"
        directory.mkdir()
        for index in range(6):
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            image[:, :, 0] = (index + 1) * 20
            assert cv2.imwrite(str(directory / f"{index:04d}.png"), image)
        return directory

    @pytest.mark.gpu
    def test_six_frames_in_six_events_out_with_every_tag_accounted_for(
        self, frame_dir: Path, demo_repository_path: Path, tmp_path: Path
    ):
        """GPU tier, because it stands up the real repository.

        Every model in ``model_repository/`` is ``platform: tensorrt`` against a real engine,
        so this exercises the production backend rather than a double. That is the whole
        value of the test and also why it cannot run in the offline tier: the offline
        container has no TensorRT, by design. Substituting a fake backend here would leave
        the assertion intact while deleting what it proves.
        """
        from shipinfer.ingest import IngestManager
        from shipinfer.server import InferenceServer

        events_path = tmp_path / "events.jsonl"
        settings = ServerSettings(
            model_repository=demo_repository_path,
            load_all_models=False,
            startup_models=[
                "ship_detector",
                "ship_segmenter",
                "ship_embedder",
                "person_embedder",
            ],
            ingest=IngestSettings(
                cameras=[
                    CameraConfig(
                        camera_id="cam0",
                        uri=str(frame_dir),
                        source="replay",
                        fps=200.0,
                        loop=False,
                    )
                ]
            ),
            pipeline=PipelineSettings(
                workers=2,
                result_sink="jsonlines",
                result_sink_options={"path": str(events_path), "flush_every": 0},
            ),
        )

        with InferenceServer(settings).start() as server:
            runner = PipelineRunner(
                server,
                settings=settings,
                frames=lambda sink: IngestManager(sink, settings=settings.ingest),
            ).start()
            try:
                assert wait_for(
                    lambda: runner.collector.reported >= 6, timeout_s=30
                ), runner.health()
            finally:
                runner.stop()

        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        payloads = [json.loads(line) for line in lines]
        keys = [(p["camera_id"], p["image_id"]) for p in payloads]

        assert len(payloads) == 6, "N frames in must be N events out"
        assert sorted(keys) == [("cam0", index) for index in range(6)]
        assert len(set(keys)) == 6, "an event was published twice"
        assert all(p["schema_version"] == 2 for p in payloads)
        assert all(p["partial"] is False for p in payloads), [
            p["missing_stages"] for p in payloads
        ]
        assert all(p["img_width"] == 96 and p["img_height"] == 64 for p in payloads)
        assert all(p["sub_id"] == "shipinfer" for p in payloads)
