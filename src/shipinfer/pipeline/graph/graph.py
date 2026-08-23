"""The DAG: which stages exist, which of them will run on this frame, and in what order.

What this adds over :class:`shipinfer.server.ensemble.EnsembleModel` — which already
executes a validated DAG of models and is the right tool for exposing one as a single
addressable model — is the three things an ensemble's fixed-size tensors cannot express:

* **cardinality.** One frame becomes a variable number of crops. The ensemble pads to a
  declared maximum; here the row count is the frame's, and
  :class:`~shipinfer.pipeline.graph.objects.ObjectBatch` carries the detection index of
  every row so a result cannot be attached to the wrong object.
* **liveness.** A 1080p frame is 6 MB and a ship crop batch is tens of megabytes, and
  reassembly holds a frame until every stage answers. The graph knows each name's last
  consumer, so it frees the frame and the crops as soon as nothing will read them again.
* **planning.** The reassembly collector has to know which stages to wait for *before* they
  answer, and on a frame with no ships that set is smaller. :meth:`PipelineGraph.runnable`
  is the one predicate that decides it, and the same predicate drives execution — so the
  plan and the run cannot disagree.

Everything tensor-level is *not* reimplemented: stage validation compares
:class:`~shipinfer.core.types.TensorSpec` objects from the same model configs the ensemble
reads, in the same way, for the same reason (a mis-wired graph must stop a deploy).

**Where vessel identity went.** There is no ``ship_recognizer`` stage, and that is deliberate
rather than unfinished. Identity is a **gallery query over the ship embedding**, not a second
network: `shipvision.reid` already carries bounded galleries with the same-camera exclusion
protocol and CMC/mAP evaluation, and running an identity *model* would mean training one to
answer a question a nearest-neighbour search over the embedder's output already answers.

It also puts the step on the right side of the system's main division. Recognition against a
gallery is **stateful** — the gallery is the state — so it belongs with tracking in the
stateful plane, not in the stateless GPU pool this graph drives. A stage was there while the
model was a stand-in; pointing the repository at real engines is what made the mismatch
visible, since a ResNet embedder cannot answer "which vessel is this".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings.pipeline import PipelineSettings
from shipinfer.pipeline.graph.crop import CropSpec, CropStage
from shipinfer.pipeline.graph.detect import DetectStage
from shipinfer.pipeline.graph.detections import DecodeParams
from shipinfer.pipeline.graph.objects import ObjectStage, mask_area
from shipinfer.pipeline.graph.stage import (
    ModelStage,
    PipelineStage,
    Servable,
    StageOutcome,
    StageStatus,
)
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT, FrameState, field_map_names
from shipinfer.pipeline.schema import ObjectRecord
from shipinfer.runtime.ops import ImageOps

__all__ = [
    "DEFAULT_RECORD_FIELDS",
    "PipelineGraph",
    "StageObserver",
    "build_perception_graph",
]

#: Which per-object batch fills which :class:`~shipinfer.pipeline.schema.ObjectRecord` field.
#: Two candidates per field is normal — a ship's embedding comes from the ship embedder and a
#: person's from the person embedder, and a batch only ever holds rows of its own class, so
#: they cannot collide. This table is also the *retain set*: a name mentioned here is never
#: freed early, because the emitted event reads it.
DEFAULT_RECORD_FIELDS: Mapping[str, tuple[str, ...]] = {
    "embedding": ("ship_embedding", "person_embedding"),
    "ship_id": ("ship_id",),
    "similarity": ("ship_similarity",),
    "mask_area_px": ("ship_mask_area",),
}


class StageObserver(Protocol):
    """What the graph tells its caller while a frame is in flight.

    Two callbacks, and the split is the point. :meth:`planned` fires as soon as a stage is
    *known* to run — its inputs exist and are non-empty — which is what lets reassembly wait
    for exactly the right set on this frame rather than for the whole DAG. :meth:`finished`
    fires once per stage with what happened, including skips and failures, so nothing about
    a frame's fate is inferred.
    """

    def planned(self, stages: Sequence[str]) -> None:
        """These stages will run. Called repeatedly and idempotently."""

    def finished(self, outcome: StageOutcome) -> None:
        """One stage is done — ran, was skipped, or raised."""


class PipelineGraph:
    """An ordered list of stages, validated once and executed per frame.

    Ordered rather than topologically sorted at run time: the order is a declaration by the
    person who wrote the graph, and validation refuses one whose declaration is not already a
    valid topological order. Sorting it for them would hide the mistake and make the
    execution order depend on dict iteration.
    """

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        field_map: Mapping[str, tuple[str, ...]] | None = None,
        name: str = "perception",
    ) -> None:
        if not stages:
            raise ConfigurationError("a pipeline graph needs at least one stage")
        self.name = name
        self._stages = tuple(stages)
        self._field_map = dict(field_map if field_map is not None else DEFAULT_RECORD_FIELDS)
        duplicates = sorted(
            {
                s.name
                for s in self._stages
                if sum(1 for o in self._stages if o.name == s.name) > 1
            }
        )
        if duplicates:
            raise ConfigurationError(f"graph {name!r} declares stage(s) twice: {duplicates}")
        self._retain = field_map_names(self._field_map) | {DETECTIONS}
        self._dead_after = self._compute_liveness()

    # -- structure ---------------------------------------------------------------------

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self._stages)

    @property
    def field_map(self) -> Mapping[str, tuple[str, ...]]:
        return self._field_map

    @property
    def entry_model(self) -> str | None:
        """The model the first stage drives — what an ingest frame is aimed at."""
        return self._stages[0].model_name

    def models(self) -> tuple[str, ...]:
        """Every model this graph needs loaded, in stage order, without duplicates."""
        seen: list[str] = []
        for stage in self._stages:
            model = stage.model_name
            if model is not None and model not in seen:
                seen.append(model)
        return tuple(seen)

    def stage(self, name: str) -> PipelineStage:
        for stage in self._stages:
            if stage.name == name:
                return stage
        raise ConfigurationError(f"graph {self.name!r} has no stage {name!r}")

    def _compute_liveness(self) -> tuple[frozenset[str], ...]:
        """For each stage index, the names nothing after it will read.

        The whole reason this exists: without it, a frame sitting in reassembly holds its
        pixels and its crop tensors — 6 MB and tens of megabytes — until every stage answers,
        and 1024 of those is not a bound worth having.
        """
        last: dict[str, int] = {}
        for index, stage in enumerate(self._stages):
            for name in stage.consumes:
                last[name] = index
        return tuple(
            frozenset(
                name
                for name, final in last.items()
                if final == index and name not in self._retain
            )
            for index in range(len(self._stages))
        )

    # -- planning ----------------------------------------------------------------------

    def runnable(
        self, state: FrameState, done: Sequence[str] | set[str] = ()
    ) -> tuple[str, ...]:
        """Stages whose inputs are present and non-empty right now.

        Deliberately *not* speculative. A stage is announced only when its inputs already
        exist, never on the assumption that an earlier stage will produce something
        non-empty — because a frame with three ships and no people makes that assumption
        false, and reassembly would then wait for a person embedder that is never called
        until the frame timed out.
        """
        finished = set(done)
        available = state.available()
        non_empty = state.non_empty()
        return tuple(
            stage.name
            for stage in self._stages
            if stage.name not in finished
            and all(name in available for name in stage.consumes)
            and all(name in non_empty for name in stage.requires)
        )

    # -- execution ---------------------------------------------------------------------

    def execute(self, state: FrameState, observer: StageObserver) -> tuple[StageOutcome, ...]:
        """Run every stage that can run, in declared order, and report each one.

        A stage that raises does **not** end the frame: its outputs simply never become
        available, so its branch is skipped and every other branch continues. A person
        embedder that fails must not cost the frame its ship identity, and the emitted event
        names what was lost rather than pretending the frame was complete.
        """
        outcomes: list[StageOutcome] = []
        done: set[str] = set()
        for index, stage in enumerate(self._stages):
            ready = self.runnable(state, done)
            if ready:
                observer.planned(ready)
            if stage.name in ready:
                outcome = stage.run(state)
            else:
                outcome = StageOutcome(stage.name, StageStatus.SKIPPED)
            done.add(stage.name)
            outcomes.append(outcome)
            observer.finished(outcome)
            # Released after a skip as well as after a run: `dead_after` is keyed on the
            # *last consumer*, so once this stage is past, nothing later reads those names
            # whether it ran or not — and a frame whose detector failed should give its
            # 6 MB back immediately rather than at the reassembly timeout.
            self._release(state, index)
        return tuple(outcomes)

    def _release(self, state: FrameState, index: int) -> None:
        for name in self._dead_after[index]:
            if name == FRAME_INPUT:
                state.release_image()
            else:
                state.drop(name)

    def objects(self, state: FrameState) -> tuple[ObjectRecord, ...]:
        """The frame's detections as event records, filled from whatever landed."""
        return state.objects(self._field_map)

    # -- validation --------------------------------------------------------------------

    def validate(self, resolve: Callable[[str], Servable]) -> None:
        """Refuse a graph that cannot work, at start-up, naming both ends of the fault.

        Five checks, in the order a reader would ask them:

        1. every consumed name is produced by an **earlier** stage (or is the frame itself);
        2. ``requires`` is a subset of ``consumes`` — a stage cannot condition on something
           it does not read;
        3. cardinality is preserved by every stage except the declared fan-out, so a
           per-frame stage can never be handed a per-object batch;
        4. each stage's own tensor agrees with its model's declared input (this is where a
           512x512 crop fed to a 256x128 embedder dies);
        5. an edge between two model stages agrees on dtype and shape — the recogniser's
           input against the embedder's output.

        Raises:
            ConfigurationError: naming the stage, the tensor and both shapes. A mis-wired
                graph is a configuration mistake and should stop a deploy, not produce a
                ``KeyError`` on the thousandth frame from inside a worker thread.
        """
        produced: dict[str, PipelineStage] = {}
        for stage in self._stages:
            where = f"graph {self.name!r} stage {stage.name!r}"
            if not set(stage.requires) <= set(stage.consumes):
                raise ConfigurationError(
                    f"{where}: requires {sorted(set(stage.requires) - set(stage.consumes))} "
                    f"which it does not consume"
                )
            for name in stage.consumes:
                if name == FRAME_INPUT:
                    continue
                source = produced.get(name)
                if source is None:
                    raise ConfigurationError(
                        f"{where}: consumes {name!r}, which no earlier stage produces "
                        f"(available: {sorted(produced) or ['image']})"
                    )
                if not stage.expands and source.cardinality is not stage.cardinality:
                    raise ConfigurationError(
                        f"{where}: is {stage.cardinality.value} but reads {name!r}, which "
                        f"{source.name!r} produces as {source.cardinality.value}. Only the "
                        f"declared fan-out stage may change cardinality — otherwise one "
                        f"camera's rows end up attached to another's objects."
                    )
            if isinstance(stage, ModelStage):
                stage.validate(resolve)
                self._validate_edge(stage, produced, resolve)
            for name in stage.produces:
                produced[name] = stage

        unproduced = sorted(name for name in self._retain if name not in produced)
        if unproduced:
            raise ConfigurationError(
                f"graph {self.name!r}: the event reads {unproduced}, which no stage produces"
            )

    def _validate_edge(
        self,
        stage: ModelStage,
        produced: Mapping[str, PipelineStage],
        resolve: Callable[[str], Servable],
    ) -> None:
        """Type-check a stage fed by another model stage's output.

        Only reachable for a stage whose row shape is not fixed by configuration — the
        recogniser, whose input is whatever the embedder emits. Names alone are not enough
        here for exactly the reason the ensemble's validator says: a 256-d embedding fed to a
        model expecting 512 has a valid tensor name and is still wrong.
        """
        if stage.expected_row_shape is not None or len(stage.consumes) != 1:
            return
        producer = produced.get(stage.consumes[0])
        if producer is None or producer.model_name is None:
            return
        output_name = producer.produced_output_name(stage.consumes[0])
        if output_name is None:
            return
        source_config = getattr(resolve(producer.model_name), "artifact", None)
        target_config = getattr(resolve(stage.model_name or ""), "artifact", None)
        if source_config is None or target_config is None:
            return
        source = next(
            (s for s in source_config.config.output_specs if s.name == output_name), None
        )
        target = next(
            (s for s in target_config.config.input_specs if s.name == stage.input_name), None
        )
        if source is None or target is None:
            return
        if source.dtype is not target.dtype or not target.matches(source.shape):
            raise ConfigurationError(
                f"graph {self.name!r} stage {stage.name!r}: input {target.describe()} does "
                f"not accept {producer.name!r}'s output {source.describe()}"
            )

    def __len__(self) -> int:
        return len(self._stages)

    def __repr__(self) -> str:
        return f"<PipelineGraph {self.name} {' -> '.join(self.stage_names)}>"


def build_perception_graph(
    settings: PipelineSettings,
    *,
    resolve: Callable[[str], Servable],
    ops: ImageOps,
) -> PipelineGraph:
    """The ship + person DAG from ``references/.../new-system-architecture.md``::

        frame -> detect -+- ship   -> segment -> embed -> recognise
                         +- person -> embed

    The shape is the specification's, not an invention: detect, segment, embed and recognise
    are stateless and therefore poolable across all 16 GPUs, while tracking is stateful and
    lives in another service behind the message bus. Segmentation is conditional because it
    is the heaviest model in the DAG — running it on every frame would cost more than
    everything else combined, and running it only where a ship was detected makes it a
    minority of the load.

    Model names come from ``settings.model_overrides`` when present, so a deployment can A/B
    a retrained detector by editing settings rather than this function.
    """
    timeout_s = settings.stage_timeout_ms / 1000.0

    def model(default: str) -> str:
        return settings.model_overrides.get(default, default)

    ship_mask_crops = "ship_mask_crops"
    ship_reid_crops = "ship_reid_crops"
    person_crops = "person_crops"

    stages: list[PipelineStage] = [
        DetectStage(
            "detect",
            model("ship_detector"),
            resolve=resolve,
            ops=ops,
            input_name="images",
            dst_size=settings.detector_input,
            decode=DecodeParams(
                class_labels=dict(settings.class_labels),
                score_threshold=settings.score_threshold,
                max_detections=settings.max_detections,
            ),
            timeout_s=timeout_s,
        ),
        CropStage(
            "crop",
            ops=ops,
            crops=[
                CropSpec(ship_mask_crops, "ship", settings.ship_mask_crop),
                CropSpec(ship_reid_crops, "ship", settings.ship_reid_crop),
                CropSpec(person_crops, "person", settings.person_reid_crop),
            ],
        ),
        ObjectStage(
            "ship_segmenter",
            model("ship_segmenter"),
            resolve=resolve,
            source=ship_mask_crops,
            outputs={"masks": "ship_mask_area"},
            # Reduced where it is produced: fifteen 512x512 float masks are 15 MB, and
            # reassembly would hold every one of them until the frame completed.
            reducers={"masks": mask_area},
            row_shape=(3, *settings.ship_mask_crop),
            timeout_s=timeout_s,
        ),
        ObjectStage(
            "ship_embedder",
            model("ship_embedder"),
            resolve=resolve,
            source=ship_reid_crops,
            outputs={"embedding": "ship_embedding"},
            row_shape=(3, *settings.ship_reid_crop),
            timeout_s=timeout_s,
        ),
        ObjectStage(
            "person_embedder",
            model("person_embedder"),
            resolve=resolve,
            source=person_crops,
            outputs={"embedding": "person_embedding"},
            row_shape=(3, *settings.person_reid_crop),
            timeout_s=timeout_s,
        ),
    ]
    return PipelineGraph(stages, name="ship_person_perception")
