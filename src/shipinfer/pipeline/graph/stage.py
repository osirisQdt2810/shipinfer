"""One step of the perception DAG, and the two kinds of step there are.

The distinction that shapes this whole package is **cardinality**. A detector consumes one
frame and produces one result; an embedder consumes N crops from one frame and produces N
vectors. The previous generation expressed that difference with a loop and a shared buffer,
and the shared buffer is the bug this project exists to fix. Here it is in the type: a stage
declares :class:`Cardinality`, a per-object stage moves an
:class:`~shipinfer.pipeline.graph.objects.ObjectBatch`, and the batch carries the detection
index of every row so a result can never be attached to the wrong object.

**Why not just use an ensemble.** :class:`shipinfer.server.ensemble.EnsembleModel` already
executes a validated DAG of models and it is the right tool for a caller that wants the DAG
as *one addressable model*, with fixed tensor shapes throughout. It cannot express what this
layer needs: a variable number of crops per frame (its tensors are
fixed-size and padded), the identity of each object across stages, or reassembly and
emission. So this package sits above it and drives the models directly; nothing here
re-implements the ensemble's tensor-level DAG validation, and
:class:`~shipinfer.pipeline.graph.graph.PipelineGraph` reuses the same
:class:`~shipinfer.core.types.TensorSpec` checks rather than writing a second set.
"""

from __future__ import annotations

import abc
import enum
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from shipinfer.core.errors import ConfigurationError, RequestTimeoutError
from shipinfer.core.request import InferenceRequest, InferenceResponse, ResponseFuture
from shipinfer.core.types import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

    from shipinfer.pipeline.graph.state import FrameState

__all__ = [
    "Cardinality",
    "ModelStage",
    "PipelineStage",
    "Servable",
    "StageOutcome",
    "StageStatus",
]


class Cardinality(enum.Enum):
    """How many rows a stage's output has, relative to its input.

    ``PER_FRAME`` is one row per frame — detection, or anything that looks at the whole
    image. ``PER_OBJECT`` is one row per detected object, so its row count varies frame to
    frame and camera to camera. Mixing the two up is the mistake that attaches camera B's
    thirtieth embedding to camera A's second person, and
    :meth:`shipinfer.pipeline.graph.graph.PipelineGraph.validate` refuses a graph that does.
    """

    PER_FRAME = "per_frame"
    PER_OBJECT = "per_object"


class StageStatus(enum.Enum):
    """What happened to one stage on one frame."""

    RAN = "ran"
    #: Its branch was empty — no ships in the frame, so no segmentation. Not a failure, and
    #: the reason the heaviest model in the DAG is affordable at all.
    SKIPPED = "skipped"
    FAILED = "failed"


class StageOutcome:
    """The result of running one stage on one frame.

    A plain class with ``__slots__`` rather than a dataclass: one of these exists per stage
    per frame — six thousand a second at fleet scale — and this is the smallest shape that
    still carries what the collector and the metrics need.
    """

    __slots__ = ("elapsed_us", "error", "rows", "stage", "status")

    def __init__(
        self,
        stage: str,
        status: StageStatus,
        *,
        rows: int = 0,
        elapsed_us: int = 0,
        error: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.status = status
        self.rows = rows
        self.elapsed_us = elapsed_us
        self.error = error

    @property
    def ran(self) -> bool:
        return self.status is StageStatus.RAN

    def __repr__(self) -> str:
        detail = f" error={self.error!r}" if self.error is not None else f" rows={self.rows}"
        return f"<StageOutcome {self.stage} {self.status.value}{detail}>"


class Servable(Protocol):
    """The slice of a loaded model a stage needs.

    Both :class:`shipinfer.server.model.Model` and
    :class:`shipinfer.server.ensemble.EnsembleModel` satisfy it, so a stage can be pointed
    at either without knowing which — an ensemble step inside a pipeline stage is a valid
    thing to want, and this is what makes it free.
    """

    name: str

    def infer(self, request: InferenceRequest) -> ResponseFuture: ...

    @property
    def is_ready(self) -> bool: ...


class PipelineStage(abc.ABC):
    """One step of the DAG: what it reads, what it writes, and when it may run.

    Three declarations drive both execution and validation, which is what keeps the planner
    and the runner from disagreeing about whether a stage will run:

    * :attr:`consumes` — names that must **exist** in the frame's state.
    * :attr:`requires` — names that must exist **and be non-empty**. This is the conditional
      execution the demo ensemble spells ``condition:``, and the reason a frame with only
      people never reaches the ship segmenter.
    * :attr:`produces` — names this stage adds.
    """

    #: Registered stage name; unique within a graph and used as the reassembly key's stage
    #: label, so it appears verbatim in a partial event's ``missing_stages``.
    name: ClassVar[str] = "abstract"
    #: The cardinality of what this stage **produces**.
    cardinality: ClassVar[Cardinality] = Cardinality.PER_FRAME
    #: True for the one stage where cardinality changes — the crop step, one frame in and N
    #: objects out. Every other stage must preserve it, and
    #: :meth:`shipinfer.pipeline.graph.graph.PipelineGraph.validate` checks that it does.
    expands: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        *,
        consumes: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
        produces: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.consumes = consumes
        self.requires = requires
        self.produces = produces

    # -- execution ---------------------------------------------------------------------

    def run(self, state: FrameState) -> StageOutcome:
        """Execute this stage against ``state``, catching whatever it raises.

        A template method, so timing and error containment are written once. Containment is
        deliberate and it is *not* swallowing: the error is carried on the outcome, counted,
        and turned into a named missing stage in the emitted event. One failed branch must
        not lose the other branch's results — a person embedder that raises should not cost
        the frame its ship identity.
        """
        started = time.monotonic_ns()
        try:
            rows = self._do_run(state)
        except Exception as exc:
            return StageOutcome(
                self.name,
                StageStatus.FAILED,
                elapsed_us=(time.monotonic_ns() - started) // 1000,
                error=exc,
            )
        return StageOutcome(
            self.name,
            StageStatus.RAN,
            rows=rows,
            elapsed_us=(time.monotonic_ns() - started) // 1000,
        )

    @abc.abstractmethod
    def _do_run(self, state: FrameState) -> int:
        """Do the work and mutate ``state``. Returns the number of rows produced."""

    # -- validation --------------------------------------------------------------------

    @property
    def model_name(self) -> str | None:
        """The model this stage drives, or ``None`` for a host-side transform."""
        return None

    @property
    def expected_row_shape(self) -> tuple[int, ...] | None:
        """The per-row shape this stage will feed its model, if it knows.

        ``None`` means "whatever an earlier stage produced", and the graph then checks the
        edge against the producing model's declared output instead.
        """
        return None

    def validate(self, resolve: Callable[[str], Servable]) -> None:
        """Check anything only this stage can check. Called once, at start-up."""

    def produced_output_name(self, batch_name: str) -> str | None:
        """Which model output this stage stores under ``batch_name``, if it drives a model."""
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class ModelStage(PipelineStage):
    """A stage that submits one request to a loaded model and waits for it.

    Waiting is not the throughput problem it looks like. Each stage's *model* batches across
    every frame in flight, so while frame A is embedding, frame B is detecting; the
    concurrency comes from the worker pool, not from this call. That is the same reasoning
    :class:`shipinfer.server.ensemble.EnsembleModel` documents for its sequential steps, and
    keeping the two consistent means one mental model for the whole server.

    Args:
        model: the loaded model, or a callable resolving its name — a stage is constructed
            before the server has necessarily loaded anything.
        timeout_s: how long to wait for the model. Bounded so a wedged instance costs one
            frame and one worker for this long rather than forever; exceeding it raises
            :class:`~shipinfer.core.errors.RequestTimeoutError`, which becomes a named
            missing stage in the emitted event.
    """

    def __init__(
        self,
        name: str,
        model: str,
        *,
        resolve: Callable[[str], Servable],
        input_name: str | None = None,
        timeout_s: float = 5.0,
        consumes: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
        produces: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name, consumes=consumes, requires=requires, produces=produces)
        if timeout_s <= 0:
            raise ConfigurationError(f"stage {name!r}: timeout_s must be > 0")
        self._model = model
        self._resolve = resolve
        # Resolved lazily, from the model, when the caller did not name it. See the property.
        self._input_name = input_name
        self._timeout_s = timeout_s

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def input_name(self) -> str:
        """The tensor this stage feeds, taken from the model when not given explicitly.

        A tensor's name belongs to the artefact, not to this layer. Hard-coding a convention
        here — the first version said ``"crops"`` — makes the graph refuse any real engine
        whose ONNX happened to call its input something else, and every export tool has its
        own habit: ultralytics emits ``images``, torch emits whatever the ``input_names``
        argument said. Refusing a valid engine over a naming preference is the wrong failure.

        Resolution order: what the caller said, then the model's single declared input, then
        its first declared input, then ``"images"``. It never raises. A name that cannot be
        determined is not a useful failure here — this property is read on the hot path, once
        per batch — and :meth:`validate` already refuses a wrong one at start-up with both the
        expected and the available names printed, which is where that failure belongs.

        ``"images"`` as the last resort is the overwhelmingly common export name (ultralytics
        and every torch export that passes ``input_names=["images"]``), so it is the guess most
        likely to be right and the least likely to hide a real mistake.
        """
        if self._input_name is not None:
            return self._input_name
        specs = self._declared_inputs()
        if specs:
            self._input_name = next(iter(specs))
        else:
            # No resolvable config — a model that is not loaded yet, or a stand-in in a test.
            # Guess, and let validate() be the one to complain if the guess is wrong.
            self._input_name = "images"
        return self._input_name

    def _declared_inputs(self) -> dict[str, Any] | None:
        """The model's declared input specs by name, or None if it cannot be resolved."""
        return self._declared("input_specs")

    def _declared_outputs(self) -> dict[str, Any] | None:
        """The model's declared output specs by name, or None if it cannot be resolved.

        The mirror of :meth:`_declared_inputs`, and there for the same reason: an output's
        name belongs to the artefact too, so a stage that has to name one asks the model
        rather than assuming an export convention.
        """
        return self._declared("output_specs")

    def _declared(self, attribute: str) -> dict[str, Any] | None:
        try:
            model = self._resolve(self._model)
        except Exception:  # a model that is not loaded yet is not an error here
            return None
        artifact = getattr(model, "artifact", None)
        if artifact is None:
            return None
        return {spec.name: spec for spec in getattr(artifact.config, attribute)}

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def _infer(self, state: FrameState, inputs: Mapping[str, Tensor]) -> InferenceResponse:
        """Submit one request for this frame and block until the model answers.

        The request carries the frame's :class:`~shipinfer.core.request.RequestContext`
        **unchanged**. That is the ADR-002 invariant, and it is what makes every reorder
        between here and the response safe: batching, spillover to another GPU and
        out-of-order completion are all fine because reassembly keys on the tag rather than
        on arrival order.
        """
        request = InferenceRequest(
            model_name=self._model,
            inputs=dict(inputs),
            context=state.context,
            priority=state.priority,
            deadline_ns=state.deadline_ns,
        )
        future = self._resolve(self._model).infer(request)
        try:
            return future.result(self._timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RequestTimeoutError(
                f"stage {self.name!r}: model {self._model!r} did not answer within "
                f"{self._timeout_s:g}s for {state.key}"
            ) from exc

    def validate(self, resolve: Callable[[str], Servable]) -> None:
        """Resolve the model and check the tensor this stage will feed it.

        The dtype and per-row shape are checked against the model's *declared* input spec,
        which catches the mistake the ensemble's docstring names: a crop sized for a 512x512
        segmenter handed to a 256x128 embedder has a perfectly valid tensor name and is
        still wrong. Checking it here makes it a start-up failure with both ends named,
        instead of a validation error from inside a worker thread on the thousandth frame.
        """
        model = resolve(self._model)
        config = getattr(model, "artifact", None)
        if config is None:
            return
        specs = {spec.name: spec for spec in config.config.input_specs}
        spec = specs.get(self.input_name)
        if spec is None:
            raise ConfigurationError(
                f"stage {self.name!r}: model {self._model!r} declares no input "
                f"{self.input_name!r} (has: {sorted(specs)})"
            )
        row_shape = self.expected_row_shape
        if row_shape is not None and not spec.matches(row_shape):
            raise ConfigurationError(
                f"stage {self.name!r}: model {self._model!r} expects {spec.describe()} "
                f"but this stage feeds rows of {row_shape}"
            )
