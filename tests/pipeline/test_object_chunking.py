"""A frame with more objects than the engine's batch must still be embedded.

A per-object stage submits one request per *frame*, and a frame holds however many objects
the detector found. The sizing this project targets is 10-20 people per frame; 25 was
observed on the benchmark's own footage. A TensorRT plan is built at a fixed batch, so the
whole frame arrived as one request the model could never accept::

    InferenceError: assembled batch of 25 rows exceeds max_batch_size 16

Every crop in that frame was lost and the stage reported a failure. Bounding the batch
*across* requests — which the queue now does — cannot help here, because the single request
is already over the limit. The split has to happen where the request is built.
"""

from __future__ import annotations

import numpy as np

from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.pipeline.graph.objects import ObjectBatch, ObjectStage, _chunks

from .conftest import EMBEDDING_DIM, FakeArtifact, FakeConfig, StubModel


class TestChunkRanges:
    def test_a_batch_within_the_limit_is_one_call(self) -> None:
        assert _chunks(8, 16) == [(0, 8)]

    def test_exactly_the_limit_is_still_one_call(self) -> None:
        assert _chunks(16, 16) == [(0, 16)]

    def test_a_crowded_frame_is_split(self) -> None:
        """The observed failure: 25 crops against a plan built at 16."""
        assert _chunks(25, 16) == [(0, 16), (16, 25)]

    def test_every_row_is_covered_exactly_once(self) -> None:
        for count in (1, 7, 16, 17, 25, 64, 100):
            ranges = _chunks(count, 16)
            covered = [i for start, stop in ranges for i in range(start, stop)]
            assert covered == list(range(count)), f"count={count} ranges={ranges}"

    def test_no_chunk_exceeds_the_limit(self) -> None:
        for count in (1, 17, 25, 100):
            assert all(stop - start <= 16 for start, stop in _chunks(count, 16))

    def test_a_model_with_no_declared_limit_is_one_call(self) -> None:
        """`max_batch_size` absent means the artefact does not constrain us. Inventing a
        limit would split requests for no reason and cost a launch per chunk."""
        assert _chunks(100, None) == [(0, 100)]
        assert _chunks(100, 0) == [(0, 100)]

    def test_an_empty_batch_still_makes_one_call(self) -> None:
        """So the response's shape is checked as usual rather than silently skipped."""
        assert _chunks(0, 16) == [(0, 0)]


class TestTheStageSplitsAndReassembles:
    """Through the real stage, against a model that declares a batch of 16."""

    def _model(self) -> StubModel:
        def respond(request):
            rows = next(iter(request.inputs.values())).batch_size
            assert rows <= 16, f"the model was handed {rows} rows"
            data = np.arange(rows, dtype=np.float32).reshape(rows, 1) * np.ones(
                (1, EMBEDDING_DIM), dtype=np.float32
            )
            return {"embedding": Tensor.from_numpy(data)}

        return StubModel(
            "person_embedder",
            respond,
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(
                        TensorSpec(name="crops", dtype=DataType.FP32, shape=(3, 4, 4)),
                    ),
                    output_specs=(
                        TensorSpec(
                            name="embedding", dtype=DataType.FP32, shape=(EMBEDDING_DIM,)
                        ),
                    ),
                    max_batch_size=16,
                )
            ),
        )

    def _stage(self, model: StubModel) -> ObjectStage:
        return ObjectStage(
            "person_embedder",
            "person_embedder",
            resolve=lambda _: model,
            source="person_crops",
            input_name="crops",
            outputs={"embedding": "person_embedding"},
        )

    def _state_with(self, make_state, crops: int):
        state = make_state()
        state.attach(
            ObjectBatch(
                name="person_crops",
                class_name="person",
                object_indices=tuple(range(crops)),
                data=np.zeros((crops, 3, 4, 4), dtype=np.float32),
            )
        )
        return state

    def _sizes(self, model: StubModel) -> list[int]:
        return [next(iter(call.inputs.values())).batch_size for call in model.calls]

    def test_a_frame_of_25_crops_reaches_a_batch_16_model(self, make_state) -> None:
        model = self._model()
        state = self._state_with(make_state, 25)

        outcome = self._stage(model).run(state)

        assert outcome.status.name == "RAN", outcome
        assert self._sizes(model) == [16, 9], "split at the engine's limit"
        result = state.batch("person_embedding")
        assert result.data.shape[0] == 25, "every crop comes back"
        assert result.object_indices == tuple(range(25))

    def test_the_rows_come_back_in_order(self, make_state) -> None:
        """The stub embeds row *i* of each chunk as the constant *i*, so a concatenation
        that reordered or overwrote a chunk is visible in the values."""
        model = self._model()
        state = self._state_with(make_state, 25)

        self._stage(model).run(state)

        embeddings = state.batch("person_embedding").data[:, 0]
        expected = list(range(16)) + list(range(9))
        assert embeddings.tolist() == [float(v) for v in expected]

    def test_a_frame_within_the_limit_still_makes_one_call(self, make_state) -> None:
        """Splitting costs a launch per chunk, so it must not happen when it need not."""
        model = self._model()
        state = self._state_with(make_state, 12)

        self._stage(model).run(state)

        assert self._sizes(model) == [12]
