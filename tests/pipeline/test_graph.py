"""The DAG: what runs, what does not, what it frees, and what it refuses to start with.

The two claims worth the most here are the ones the previous generation could not make.
*Conditional execution* is asserted as an absence of **calls**, not as output that happens to
look right — output can be right for the wrong reason and an empty call list cannot. And
*liveness* is asserted in bytes, because a frame held in reassembly with its pixels and its
crops attached is the difference between a 1024-frame bound and tens of gigabytes.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, InferenceError
from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.pipeline.graph import (
    Cardinality,
    CropSpec,
    CropStage,
    DetectStage,
    ObjectStage,
    PipelineGraph,
    StageStatus,
)
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT

from .conftest import (
    CROP_SIZE,
    DETECTOR_INPUT,
    EMBEDDING_DIM,
    FakeArtifact,
    FakeConfig,
    StubModel,
)

pytestmark = pytest.mark.timeout(30)

# Detections come back in **descending score** order, so the scores here are distinct and
# the order they impose is stated in every assertion that depends on it. A test that assumed
# input order would be asserting on an ordering the decoder does not promise.
# The trailing column is a class id, and the ids are the *shipped* detector's own COCO
# numbering (0 person, 8 boat) — the same table as
# ``PipelineSettings.class_labels``. They read 0=ship, 1=person while the detector was a
# stand-in; against the real engine that labelled every person "ship" and every boat
# "unknown", with every shape check still passing. Encoding the real numbering here is what
# makes these tests able to fail on that mismatch.
SHIP = [0.0, 0.0, 8.0, 8.0, 0.9, 8.0]
PERSON = [1.0, 1.0, 5.0, 7.0, 0.8, 0.0]
PERSON_WEAKER = [2.0, 2.0, 6.0, 7.0, 0.7, 0.0]


class Recorder:
    """A :class:`StageObserver` that remembers everything it was told.

    ``planned`` is called repeatedly and idempotently by design, so this keeps the *set* of
    announced stages rather than the call sequence — a test asserting on the sequence would
    be asserting on how often the graph re-plans, which is not a property worth pinning.
    """

    def __init__(self) -> None:
        self.planned_seen: list[str] = []
        self.outcomes: list = []

    def planned(self, stages) -> None:
        for stage in stages:
            if stage not in self.planned_seen:
                self.planned_seen.append(stage)

    def finished(self, outcome) -> None:
        self.outcomes.append(outcome)

    def status(self, stage: str) -> StageStatus | None:
        return next((o.status for o in self.outcomes if o.stage == stage), None)


@pytest.fixture()
def recorder() -> Recorder:
    return Recorder()


class TestConditionalExecution:
    """A branch whose crop set is empty is not merely cheap — it is never called."""

    def test_a_frame_with_only_people_never_reaches_the_ship_segmenter(
        self, build_graph, models, make_state, recorder
    ):
        graph = build_graph([PERSON])
        state = make_state()

        graph.execute(state, recorder)

        assert models["ship_segmenter"].calls == [], "the heaviest model in the DAG ran"
        assert models["ship_embedder"].calls == []
        assert len(models["person_embedder"].calls) == 1
        assert recorder.status("ship_segmenter") is StageStatus.SKIPPED
        assert recorder.status("person_embedder") is StageStatus.RAN

    def test_a_frame_with_only_ships_never_reaches_the_person_embedder(
        self, build_graph, models, make_state, recorder
    ):
        graph = build_graph([SHIP])
        state = make_state()

        graph.execute(state, recorder)

        assert models["person_embedder"].calls == []
        assert len(models["ship_segmenter"].calls) == 1
        assert len(models["ship_embedder"].calls) == 1

    def test_a_frame_with_no_detections_runs_nothing_downstream(
        self, build_graph, models, make_state, recorder
    ):
        """A quiet camera costs one detector call and nothing else."""
        graph = build_graph([[0.0, 0.0, 4.0, 4.0, 0.1, 0.0]])  # below the 0.5 threshold
        state = make_state()

        graph.execute(state, recorder)

        assert len(models["ship_detector"].calls) == 1
        assert all(not model.calls for name, model in models.items() if name != "ship_detector")
        assert state.detections.is_empty
        assert recorder.status("crop") is StageStatus.SKIPPED

    def test_only_stages_that_will_run_are_announced(self, build_graph, make_state, recorder):
        """Reassembly waits for exactly this set, so an over-announcement is a stuck frame."""
        graph = build_graph([PERSON])

        graph.execute(make_state(), recorder)

        assert set(recorder.planned_seen) == {"detect", "crop", "person_embedder"}


class TestTheFanOut:
    """One frame, N crops — batched in one call, and every row traceable to its detection."""

    def test_three_people_become_one_call_with_three_rows(self, tiny_graph, models, make_state):
        graph = tiny_graph([PERSON, PERSON, PERSON])
        state = make_state()

        graph.execute(state, _NullObserver())

        assert models["person_embedder"].batch_sizes == [3], "a per-image loop, not a batch"
        assert state.batches["person_embedding"].object_indices == (0, 1, 2)

    def test_crops_carry_the_detection_index_of_every_row(self, build_graph, make_state):
        """Mixed classes: each batch holds its own class's indices into the score order."""
        graph = build_graph([PERSON, SHIP])
        state = make_state()

        graph.execute(state, _NullObserver())

        assert state.detections.labels == ("ship", "person")  # 0.9 before 0.8
        assert state.batches["ship_embedding"].object_indices == (0,)
        assert state.batches["person_embedding"].object_indices == (1,)

    def test_embeddings_land_on_the_object_that_produced_them(self, build_graph, make_state):
        """The failure this guards has no symptom until a tracker starts swapping identities.

        Three objects of two classes, so the two per-object batches interleave in the record
        list: the ship is record 0 and the people are records 1 and 2, while inside the person
        batch they are rows 0 and 1. A scatter that used the row instead of the detection index
        would attach the second person's embedding to the ship and pass every shape check.
        """
        graph = build_graph([PERSON, PERSON_WEAKER, SHIP])
        state = make_state()
        graph.execute(state, _NullObserver())

        records = graph.objects(state)
        assert [r.class_name for r in records] == ["ship", "person", "person"]
        # The stub embeds row i as the constant i, so a shifted scatter is visible.
        assert records[0].embedding == (0.0,) * EMBEDDING_DIM  # ship batch, row 0
        assert records[1].embedding == (0.0,) * EMBEDDING_DIM  # person batch, row 0
        assert records[2].embedding == (1.0,) * EMBEDDING_DIM  # person batch, row 1
        # No stage fills identity any more: the gallery query that turns a ship embedding
        # into an id is stateful and left the graph. The field survives on the record and
        # serialises as null, so a consumer's schema does not change when it comes back.
        assert all(r.ship_id is None for r in records)

    def test_a_model_returning_the_wrong_row_count_fails_the_stage(
        self, build_graph, make_state, recorder
    ):
        """``(N-1, 512)`` from a batch of N is the shape a naive consumer misattributes."""

        def short(request):
            rows = next(iter(request.inputs.values())).batch_size
            return {
                "embedding": Tensor.from_numpy(
                    np.zeros((rows - 1, EMBEDDING_DIM), dtype=np.float32)
                )
            }

        graph = build_graph(
            [PERSON, PERSON], person_embedder=StubModel("person_embedder", short)
        )
        state = make_state()

        graph.execute(state, recorder)

        outcome = next(o for o in recorder.outcomes if o.stage == "person_embedder")
        assert outcome.status is StageStatus.FAILED
        assert "row" in str(outcome.error)


class TestTheTagSurvives:
    """``(camera_id, frame_id)`` rides through untouched — the ADR-002 invariant."""

    def test_every_stage_request_carries_the_frames_tag(self, build_graph, models, make_state):
        graph = build_graph([SHIP, PERSON])
        state = make_state(camera="quay_west", frame=4321)

        graph.execute(state, _NullObserver())

        for model in models.values():
            assert model.calls, f"{model.name} never ran"
            for request in model.calls:
                assert request.context.key == ("quay_west", 4321)

    def test_the_tag_survives_a_stage_that_raises(self, build_graph, make_state, recorder):
        graph = build_graph(
            [SHIP],
            ship_embedder=StubModel(
                "ship_embedder", error=InferenceError("the engine fell over")
            ),
        )
        state = make_state(camera="cam9", frame=7)

        graph.execute(state, recorder)

        assert state.key == ("cam9", 7)
        assert [r.det_id for r in graph.objects(state)] == ["cam9_7_0"]

    def test_det_ids_are_derivable_from_the_tag(self, build_graph, make_state):
        graph = build_graph([SHIP, PERSON])
        state = make_state(camera="cam0", frame=12)
        graph.execute(state, _NullObserver())

        assert [r.det_id for r in graph.objects(state)] == ["cam0_12_0", "cam0_12_1"]
        assert [r.class_name for r in graph.objects(state)] == ["ship", "person"]


class TestOneFailedBranchDoesNotCostTheOther:
    """A person embedder that raises must not lose the frame its ship identity."""

    def test_the_other_branch_still_completes(self, build_graph, models, make_state, recorder):
        graph = build_graph(
            [SHIP, PERSON],
            person_embedder=StubModel("person_embedder", error=InferenceError("boom")),
        )
        state = make_state()

        graph.execute(state, recorder)

        assert recorder.status("person_embedder") is StageStatus.FAILED
        assert recorder.status("ship_embedder") is StageStatus.RAN
        records = graph.objects(state)
        assert records[0].embedding != (), "the ship branch lost its embedding"
        assert records[1].embedding == (), "a failed stage leaves the field unset, not zeroed"

    def test_a_failed_detector_skips_everything_and_keeps_the_tag(
        self, build_graph, make_state, recorder
    ):
        graph = build_graph(
            [SHIP], ship_detector=StubModel("ship_detector", error=InferenceError("no engine"))
        )
        state = make_state(camera="cam3", frame=88)

        graph.execute(state, recorder)

        assert recorder.status("detect") is StageStatus.FAILED
        assert all(
            recorder.status(name) is StageStatus.SKIPPED
            for name in graph.stage_names
            if name != "detect"
        )
        assert state.key == ("cam3", 88)
        assert graph.objects(state) == ()


class TestLiveness:
    """The frame is megabytes and the results are kilobytes; reassembly may only hold the
    latter."""

    def test_the_image_is_released_once_the_last_pixel_reader_is_done(
        self, build_graph, make_state
    ):
        graph = build_graph([SHIP])
        state = make_state()
        assert state.image.nbytes > 0

        graph.execute(state, _NullObserver())

        assert state.image_released
        assert FRAME_INPUT not in state.available()

    def test_crop_tensors_are_dropped_once_their_consumer_has_run(
        self, build_graph, make_state
    ):
        graph = build_graph([SHIP, PERSON])
        state = make_state()

        graph.execute(state, _NullObserver())

        assert "ship_mask_crops" not in state.batches
        assert "ship_reid_crops" not in state.batches
        assert "person_crops" not in state.batches
        # What the event needs is exactly what survives.
        assert set(state.batches) == {
            "ship_mask_area",
            "ship_embedding",
            "person_embedding",
        }

    def test_what_reassembly_holds_is_kilobytes_not_megabytes(self, build_graph, make_state):
        graph = build_graph([SHIP, PERSON])
        state = make_state()
        before = state.footprint_bytes()

        graph.execute(state, _NullObserver())

        assert state.footprint_bytes() < before
        # Two embeddings plus four scalars at this test's sizes; the assertion that matters
        # is the ratio, which at production sizes is 6 MB -> 30 KB.
        assert state.footprint_bytes() <= 2 * EMBEDDING_DIM * 4 + 64

    def test_a_frame_whose_detector_failed_gives_its_pixels_back_immediately(
        self, build_graph, make_state
    ):
        """Otherwise a failed frame holds 6 MB until the reassembly timeout."""
        graph = build_graph(
            [SHIP], ship_detector=StubModel("ship_detector", error=InferenceError("boom"))
        )
        state = make_state()

        graph.execute(state, _NullObserver())

        assert state.image_released


class TestValidationRefusesABrokenGraph:
    """A mis-wired graph is a configuration mistake and must stop a deploy."""

    def test_a_stage_consuming_a_name_nobody_produces_is_refused(self, ops, models):
        graph = PipelineGraph(
            [
                ObjectStage(
                    "person_embedder",
                    "person_embedder",
                    resolve=models.__getitem__,
                    source="crops_that_do_not_exist",
                    input_name="crops",
                    outputs={"embedding": "person_embedding"},
                )
            ],
            field_map={"embedding": ("person_embedding",)},
        )
        with pytest.raises(ConfigurationError, match="no earlier stage produces"):
            graph.validate(models.__getitem__)

    def test_a_per_frame_stage_may_not_read_a_per_object_batch(self, ops, models):
        """The mistake that attaches camera B's thirtieth embedding to camera A's second."""
        confused = DetectStage(
            "second_detect",
            "ship_detector",
            resolve=models.__getitem__,
            ops=ops,
            dst_size=DETECTOR_INPUT,
        )
        confused.consumes = ("person_crops",)
        confused.requires = ()
        graph = PipelineGraph(
            [
                DetectStage(
                    "detect",
                    "ship_detector",
                    resolve=models.__getitem__,
                    ops=ops,
                    dst_size=DETECTOR_INPUT,
                ),
                CropStage(
                    "crop", ops=ops, crops=[CropSpec("person_crops", "person", CROP_SIZE)]
                ),
                confused,
            ],
            field_map={},
        )
        with pytest.raises(ConfigurationError, match="cardinality"):
            graph.validate(models.__getitem__)

    def test_a_crop_size_the_model_does_not_declare_is_refused(self, ops, models):
        """A 512x512 crop handed to a 256x128 embedder: a valid name and the wrong tensor."""
        models["person_embedder"] = StubModel(
            "person_embedder",
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(
                        TensorSpec(name="crops", dtype=DataType.FP32, shape=(3, 256, 128)),
                    )
                )
            ),
        )
        graph = PipelineGraph(
            [
                DetectStage(
                    "detect",
                    "ship_detector",
                    resolve=models.__getitem__,
                    ops=ops,
                    dst_size=DETECTOR_INPUT,
                ),
                CropStage(
                    "crop", ops=ops, crops=[CropSpec("person_crops", "person", (512, 512))]
                ),
                ObjectStage(
                    "person_embedder",
                    "person_embedder",
                    resolve=models.__getitem__,
                    source="person_crops",
                    input_name="crops",
                    outputs={"embedding": "person_embedding"},
                    row_shape=(3, 512, 512),
                ),
            ],
            field_map={"embedding": ("person_embedding",)},
        )
        with pytest.raises(ConfigurationError, match="feeds rows of"):
            graph.validate(models.__getitem__)

    def test_an_edge_between_two_models_must_agree_on_shape(self, ops, models):
        """One model emits 8 floats into another that declares 512 — refused at start-up.

        The edge is built explicitly instead of through ``build_perception_graph`` because
        the production DAG no longer has a model-to-model edge: vessel identity is a gallery
        query, which is stateful, so it left the graph together with the ``ship_recognizer``
        stage. The guard that edge exercised is still real and still worth a test, so the
        test now supplies its own edge rather than asserting through a stage that is gone.
        """
        models["ship_embedder"] = StubModel(
            "ship_embedder",
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(
                        TensorSpec(name="crops", dtype=DataType.FP32, shape=(3, *CROP_SIZE)),
                    ),
                    output_specs=(
                        TensorSpec(name="embedding", dtype=DataType.FP32, shape=(8,)),
                    ),
                )
            ),
        )
        models["ship_matcher"] = StubModel(
            "ship_matcher",
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(
                        TensorSpec(name="embedding", dtype=DataType.FP32, shape=(512,)),
                    )
                )
            ),
        )
        graph = PipelineGraph(
            [
                DetectStage(
                    "detect",
                    "ship_detector",
                    resolve=models.__getitem__,
                    ops=ops,
                    dst_size=DETECTOR_INPUT,
                ),
                CropStage("crop", ops=ops, crops=[CropSpec("ship_crops", "ship", CROP_SIZE)]),
                ObjectStage(
                    "ship_embedder",
                    "ship_embedder",
                    resolve=models.__getitem__,
                    source="ship_crops",
                    input_name="crops",
                    outputs={"embedding": "ship_embedding"},
                    row_shape=(3, *CROP_SIZE),
                ),
                ObjectStage(
                    "ship_matcher",
                    "ship_matcher",
                    resolve=models.__getitem__,
                    source="ship_embedding",
                    input_name="embedding",
                    outputs={"score": "ship_score"},
                ),
            ],
            field_map={},
        )
        with pytest.raises(ConfigurationError, match="does not accept"):
            graph.validate(models.__getitem__)

    def test_a_stage_requiring_what_it_does_not_consume_is_refused(self, ops, models):
        stage = CropStage(
            "crop", ops=ops, crops=[CropSpec("person_crops", "person", CROP_SIZE)]
        )
        stage.requires = (DETECTIONS, "something_else")
        with pytest.raises(ConfigurationError, match="which it does not consume"):
            PipelineGraph([stage], field_map={}).validate(models.__getitem__)

    def test_an_event_field_no_stage_produces_is_refused(self, ops, models):
        graph = PipelineGraph(
            [
                DetectStage(
                    "detect",
                    "ship_detector",
                    resolve=models.__getitem__,
                    ops=ops,
                    dst_size=DETECTOR_INPUT,
                )
            ],
            field_map={"embedding": ("nobody_produces_this",)},
        )
        with pytest.raises(ConfigurationError, match="the event reads"):
            graph.validate(models.__getitem__)

    def test_two_stages_with_one_name_are_refused(self, ops, models):
        def stage(name):
            return DetectStage(
                name,
                "ship_detector",
                resolve=models.__getitem__,
                ops=ops,
                dst_size=DETECTOR_INPUT,
            )

        with pytest.raises(ConfigurationError, match="twice"):
            PipelineGraph([stage("detect"), stage("detect")])

    def test_the_production_graph_validates_against_the_demo_repository(
        self, demo_repository_path, ops
    ):
        """The shipped repository and the shipped graph must agree, or the demo is a lie."""
        from shipinfer.core.settings.pipeline import PipelineSettings
        from shipinfer.pipeline.graph import build_perception_graph
        from shipinfer.repository import ModelRepository

        repository = ModelRepository.load(demo_repository_path)

        class Loaded:
            def __init__(self, name):
                self.name = name
                self.artifact = repository.resolve(name)
                self.is_ready = True

            def infer(self, request):  # pragma: no cover - validation never infers
                raise AssertionError("validation must not run inference")

        graph = build_perception_graph(PipelineSettings(), resolve=Loaded, ops=ops)
        graph.validate(Loaded)
        assert graph.models() == (
            "ship_detector",
            "ship_segmenter",
            "ship_embedder",
            "person_embedder",
        )


class TestCardinalityIsDeclared:
    """The fan-out is a declaration, not a comment."""

    def test_only_the_crop_stage_changes_cardinality(self, build_graph):
        graph = build_graph([SHIP])
        expanders = [s.name for s in graph.stages if s.expands]
        assert expanders == ["crop"]

    def test_per_object_stages_say_so(self, build_graph):
        graph = build_graph([SHIP])
        assert graph.stage("detect").cardinality is Cardinality.PER_FRAME
        assert graph.stage("ship_embedder").cardinality is Cardinality.PER_OBJECT


class _NullObserver:
    """Runs the graph without recording anything."""

    def planned(self, stages) -> None:
        return None

    def finished(self, outcome) -> None:
        return None
