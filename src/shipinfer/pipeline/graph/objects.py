"""Per-object tensors, and the stages that consume them.

This module is where the pipeline's fan-out lives, and the reason it is a type rather than a
loop. One frame yields a variable number of crops — 2 on a corridor camera, 30 on a busy
quay — so a per-object stage's row count is a property of the *frame*, not of the graph. The
previous generation carried that variability in a shared buffer keyed by nothing, and a
crowded camera's crops evicted a quiet camera's before they finished the pipeline
(``references/bitbucket-subfaceid/docs/flow.md``).

:class:`ObjectBatch` makes two invariants unbreakable:

* **every row knows which detection it came from**, so an embedding cannot be attached to
  the wrong object — a failure with no symptom until a tracker starts swapping identities;
* **a model that returns the wrong number of rows is an error**, not a silent misalignment.
  ``(N, 512)`` from a batch of N-1 crops is exactly the shape a reasonable consumer accepts
  and misattributes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from shipinfer.core.errors import InferenceError, ValidationError
from shipinfer.core.types import Tensor
from shipinfer.pipeline.graph.stage import Cardinality, ModelStage, Servable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.pipeline.graph.state import FrameState

__all__ = ["ObjectBatch", "ObjectStage"]


@dataclass(frozen=True, slots=True)
class ObjectBatch:
    """``N`` rows, each belonging to one detection of one frame.

    Covers both halves of a per-object stage: a crop tensor going *into* a model
    (``(N, 3, h, w)``) and an embedding, mask summary or identity coming *out* (``(N, 512)``,
    ``(N, 1)``). They are the same thing — a batch-major array plus the detection index of
    every row — and giving them one type is what lets a stage consume either without caring
    which produced it.
    """

    #: The name this batch is known by in the frame's state, e.g. ``"ship_reid_crops"``.
    name: str
    #: Which detection class these rows belong to, for provenance and for the event builder.
    class_name: str
    #: ``object_indices[row]`` is the index of the detection that produced ``data[row]``.
    object_indices: tuple[int, ...]
    #: ``(N, ...)`` — batch-major, always, even at N = 1.
    data: np.ndarray
    #: The source boxes for a crop batch, ``(N, 4)``. Empty for a model output.
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))

    def __post_init__(self) -> None:
        rows = int(self.data.shape[0]) if self.data.ndim else 0
        if rows != len(self.object_indices):
            raise ValidationError(
                f"object batch {self.name!r} has {rows} row(s) for "
                f"{len(self.object_indices)} object(s); a model that returns a different "
                "number of rows than it was given would misattribute every result after "
                "the first missing one"
            )

    @classmethod
    def empty(cls, name: str, class_name: str, row_shape: tuple[int, ...] = ()) -> ObjectBatch:
        """A batch with no rows — a frame with no objects of this class.

        Zero rows rather than a missing entry, for the reason
        :meth:`shipinfer.server.ensemble.EnsembleModel._collect_outputs` gives: "no ships in
        this frame" and "the ship branch raised" must not look the same to a consumer.
        """
        return cls(
            name=name,
            class_name=class_name,
            object_indices=(),
            data=np.zeros((0, *row_shape), dtype=np.float32),
        )

    @property
    def count(self) -> int:
        return len(self.object_indices)

    @property
    def is_empty(self) -> bool:
        return not self.object_indices

    @property
    def row_shape(self) -> tuple[int, ...]:
        """The shape of one row — what a model's input spec is checked against."""
        return tuple(int(d) for d in self.data.shape[1:])

    def scatter(self, array: np.ndarray | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(detection_index, row)`` for this batch, or for ``array`` alongside it.

        The inverse of the fan-out, and the only sanctioned way to get from a model's
        batched output back to individual objects.

        Raises:
            InferenceError: ``array`` has a different number of rows than this batch has
                objects. Naming it here, at the seam, is the difference between an error and
                fifteen embeddings quietly shifted by one.
        """
        source = self.data if array is None else array
        rows = int(source.shape[0]) if source.ndim else 0
        if rows != self.count:
            raise InferenceError(
                f"object batch {self.name!r}: got {rows} row(s) for {self.count} object(s)"
            )
        for row, index in enumerate(self.object_indices):
            yield index, source[row]

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return (
            f"<ObjectBatch {self.name} class={self.class_name} n={self.count} "
            f"row={self.row_shape}>"
        )


class ObjectStage(ModelStage):
    """A model applied to every object of one frame at once.

    Segmentation, re-identification embedding and gallery recognition are all this stage: a
    per-object array in, a per-object array out, with the detection indices carried across
    unchanged. What differs between them is the model, the tensor names and the crop size,
    which is configuration — so there is one class rather than three near-identical ones.

    It runs only when its source batch is **non-empty**. That is not an optimisation
    detail: the ship segmenter is the heaviest model in the DAG and running it on every
    frame would cost more than everything else combined. A frame with only people must never
    reach it, and ``requires`` is how the planner knows that before the stage is called.

    Args:
        source: the batch this stage reads — a crop set, or another stage's output.
        outputs: output name -> the batch name to store it under. A stage may forward
            several, and each becomes its own per-object batch so the event builder can pick
            them up by name. The keys are the *model's* output names unless ``combine`` is
            given, in which case they are the names it produces.
        row_shape: the per-row shape this stage feeds, when it is fixed by configuration (a
            crop size). ``None`` for a stage fed by another stage's output, whose shape the
            graph checks against the producing model's declared output instead.
        combine: fold the engine's raw outputs into the quantities this stage forwards, for
            an artefact whose outputs are not already those quantities. The standing example
            is segmentation: a YOLO segmentation engine emits detections *and* a bank of mask
            prototypes, and one mask per crop is the two multiplied together — which a
            per-output reducer cannot express, because it never sees both. See
            :class:`~shipinfer.pipeline.graph.masks.InstanceMaskArea`.
    """

    cardinality: ClassVar[Cardinality] = Cardinality.PER_OBJECT

    def __init__(
        self,
        name: str,
        model: str,
        *,
        resolve: Callable[[str], Servable],
        source: str,
        input_name: str | None = None,
        outputs: Mapping[str, str],
        timeout_s: float = 5.0,
        row_shape: tuple[int, ...] | None = None,
        combine: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        if not outputs:
            raise ValidationError(f"stage {name!r} must forward at least one model output")
        super().__init__(
            name,
            model,
            resolve=resolve,
            input_name=input_name,
            timeout_s=timeout_s,
            consumes=(source,),
            requires=(source,),
            produces=tuple(outputs.values()),
        )
        self._source = source
        self._outputs = dict(outputs)
        self._row_shape = row_shape
        self._combine = combine

    @property
    def source(self) -> str:
        return self._source

    @property
    def outputs(self) -> Mapping[str, str]:
        return self._outputs

    @property
    def expected_row_shape(self) -> tuple[int, ...] | None:
        return self._row_shape

    def produced_output_name(self, batch_name: str) -> str | None:
        """Which model output this stage stores under ``batch_name``.

        The graph needs it to type-check an edge between two model stages: the recogniser's
        input has to agree with the embedder's output, and only the producing stage knows
        which of its outputs became which name.
        """
        return next((out for out, name in self._outputs.items() if name == batch_name), None)

    def _do_run(self, state: FrameState) -> int:
        batch = state.batch(self._source)
        response = self._infer(state, {self.input_name: Tensor.from_numpy(batch.data)})
        arrays = self._quantities(response.outputs)
        for output_name, batch_name in self._outputs.items():
            array = arrays.get(output_name)
            if array is None:
                raise InferenceError(
                    f"stage {self.name!r}: model {self.model_name!r} returned no output "
                    f"{output_name!r} (got: {sorted(arrays)})"
                )
            state.attach(
                ObjectBatch(
                    name=batch_name,
                    class_name=batch.class_name,
                    object_indices=batch.object_indices,
                    data=np.asarray(array),
                )
            )
        return batch.count

    def _quantities(self, outputs: Mapping[str, Tensor]) -> Mapping[str, np.ndarray]:
        """The arrays this stage forwards, raw or combined.

        Without a combiner only the forwarded outputs are materialised: a segmentation engine
        emits 3 MB of prototypes per row, and converting an output nothing reads would pay for
        it on every frame.
        """
        if self._combine is None:
            return {
                name: tensor.numpy()
                for name, tensor in outputs.items()
                if name in self._outputs
            }
        return self._combine({name: tensor.numpy() for name, tensor in outputs.items()})
