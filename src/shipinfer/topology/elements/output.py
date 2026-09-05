"""The chain's last element: one frame's metadata becomes one published event.

It assembles a :class:`~shipinfer.core.events.PerceptionEvent` from what the chain filed and
hands it to a :class:`~shipinfer.topology.sinks.base.ResultSink`. Which element fills which
field is one table, in ``docs/design/event-schema.md``.

* **A key nobody filed is a ``None`` field**, never a zero and never an omission.
* **A sink that refuses is counted, not raised**: the runner fails an item's future on any
  exception, so raising when a broker went away would stop the walk over a frame whose boxes,
  vectors and ids were all good.
* **No ``classes:`` here**: a crop element selects rows because it pays a GPU per row; an
  event that published only the ships would leave the people out of the one message that
  says what was in the frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.events import ObjectRecord, PerceptionEvent, as_embedding
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.elements.detections import (
    MISSING_STAGES,
    TRACK_ROWS,
    Detections,
    per_row,
)
from shipinfer.topology.registry import registry_for
from shipinfer.topology.sinks import RESULT_SINKS, ResultSink

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.core.metrics import Counter

__all__ = [
    "JsonLinesOutput",
    "NullOutput",
    "SinkOutput",
]

#: Element params that are this element's own; everything else in ``params:`` is handed to the
#: sink's constructor. Kept as a set rather than popped one by one so the split is readable and
#: so a new element-level key cannot be forgotten in one of the two places.
_ELEMENT_PARAMS = frozenset({"source_id"})


class _OutputMetrics:
    """The element's metric handles, resolved once at ``open``.

    Same shape and the same reason as ``_TrackMetrics`` in ``elements/track.py``: at a thousand
    frames a second a metric looked up by string per frame is a hash and a dict probe nobody
    needs to pay for, and ``context.metrics is None`` gets one answer — a null object — rather
    than an ``if`` on the per-frame path.
    """

    __slots__ = ("element", "emitted", "failed")

    def __init__(self, registry: Any, element: str) -> None:
        self.element = element
        counter = getattr(registry, "counter", None)
        if counter is None:
            self.emitted: Counter | None = None
            self.failed: Counter | None = None
            return
        self.emitted = registry.counter(
            "shipinfer_output_events_total",
            "Perception events this element handed to its sink and the sink accepted, per "
            "camera. The number that says the chain is producing; a camera missing from it is "
            "a camera whose frames are not reaching anybody.",
        )
        self.failed = registry.counter(
            "shipinfer_output_events_dropped_total",
            "Events a sink refused, per camera. A broker whose DNS stopped resolving, a full "
            "disk, a closed sink. Counted rather than raised: the frame was good and the walk "
            "must not stop, but publish loss that nobody counts is publish loss nobody sees.",
        )

    def event_emitted(self, camera_id: str) -> None:
        if self.emitted is not None:
            self.emitted.inc(camera=camera_id)

    def event_dropped(self, camera_id: str, count: int = 1) -> None:
        if self.failed is not None:
            self.failed.inc(count, camera=camera_id)


class SinkOutput(Element):
    """Assemble the event and hand it to one sink, for one chain, in one process.

    One element per node (``Topology.from_spec``), so a chain with two ``output`` slots gets
    two sinks and neither is shared. The sink itself is built at ``open`` and closed at
    ``close``: a broker connection is a resource with a lifetime, not a module global.

    ``source_id`` defaults to ``shard-<n>`` so two shards publishing the same camera are
    distinguishable at the consumer without either being told who it is.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT

    #: Read one entry per detection row (``_records``), so the loader refuses a chain whose
    #: producer of any of them files a model's raw response under it instead.
    reads_per_row: ClassVar[tuple[str, ...]] = ("vectors", "identities", "masks")
    #: Metadata first, host pixels second. A sink serialises, so it can never read device
    #: memory — declaring ``nv12@gpu`` here is how a chain grows a silent per-frame download
    #: (arch.md §8). ``test_chain.py::TestRefusals::test_a_serialising_sink_behind_a_device_decoder_is_refused``
    #: is what proves the loader refuses that pairing.
    #: ``produces`` stays empty, which is what makes this a sink.
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu", "bgr@cpu")
    #: The name this element resolves in :data:`~shipinfer.topology.sinks.RESULT_SINKS`.
    sink_name: ClassVar[str] = ""

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        declared = self.params.get("source_id")
        if declared is not None and not str(declared).strip():
            raise ConfigurationError(
                f"output element {self.name!r}: `params: source_id:` must not be empty; "
                "it is the event's `sub_id`, and a consumer joins on it"
            )
        self._declared_source_id = None if declared is None else str(declared).strip()
        self._sink_params = {
            key: value for key, value in self.params.items() if key not in _ELEMENT_PARAMS
        }
        self._sink: ResultSink | None = None
        self._source_id = "shard-0"
        self._metrics = _OutputMetrics(None, name)

    # -- lifecycle ---------------------------------------------------------------------

    def _do_open(self, context: ElementContext) -> None:
        """Build the sink now, so a bad path or an absent broker client stops the deploy.

        Raises:
            ConfigurationError: a param the sink's constructor does not accept, or one it
                rejects. Both at start-up rather than on the first frame, which is the whole
                point of building it here (CONVENTIONS 2.6).
            BackendUnavailableError: the sink's client library is not installed — the Kafka
                sink's refusal, which carries the ``pip install`` that fixes it.
        """
        self._source_id = self._declared_source_id or f"shard-{context.shard_id}"
        try:
            self._sink = RESULT_SINKS.create(self.sink_name, **self._sink_params)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"output element {self.name!r}: `params:` {sorted(self._sink_params)} were "
                f"refused by the {self.sink_name!r} sink ({exc})"
            ) from exc
        self._metrics = _OutputMetrics(context.metrics, self.name)

    @property
    def sink(self) -> ResultSink | None:
        """The sink this element is publishing through, or ``None`` before ``open``.

        Read-only and deliberately nullable: the lifecycle is the interesting part, and a
        caller that gets ``None`` is being told the element is not open rather than handed
        a half-built object. Exists because the alternative was a dozen assertions reaching
        through ``_sink`` -- a private attribute whose value changes on ``close`` -- which
        couples them to an implementation detail the class is free to change.
        """
        return self._sink

    def _do_close(self) -> None:
        sink, self._sink = self._sink, None
        if sink is not None:
            # `close` flushes and is idempotent, and it is what turns a buffered jsonlines
            # file into a readable one. A chain that stopped without it would lose up to
            # `flush_every` events with no error anywhere.
            sink.close()

    # -- one frame ---------------------------------------------------------------------
    # doc: long the two-key precedence and the None-vs-zero rule are the reasons this

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        """Publish one frame's event, and consume the item.

        Returns:
            ``None`` — the item is *consumed*, which is what a sink does, and not a failure.

        Raises:
            ValidationError: the chain filed a per-object key this element cannot attribute
                to detection rows. A mis-wired chain, not a bad frame: the alternative is
                publishing an event with a whole field silently empty.
            ServerStateError: called before :meth:`Element.open`.
        """
        assert self._sink is not None  # `process` refuses before `open`
        event = self.event(item)
        published = self._sink.emit(event)
        if published:
            self._metrics.event_emitted(event.camera_id)
        else:
            self._metrics.event_dropped(event.camera_id)
        # doc: long the after-accept path charges a loss to a different camera than this send
        # Refusals the transport reported *after* it accepted the message, for the frames they
        # belong to rather than for this one. `emit` cannot carry them: a broker answers for
        # frame n while frame n+k is being produced, so folding them into this call's bool
        # would charge one camera's loss to another camera's frame
        # (`topology/sinks/base.py::drain_delivery_failures`). A synchronous sink returns ().
        for camera_id, _frame_id in self._sink.drain_delivery_failures():
            self._metrics.event_dropped(camera_id)
        # `None` because the item is *consumed*, not because anything failed -- the
        # distinction `Element.process` documents and the reason this is not typed `-> None`.
        return None

    # -- assembly ----------------------------------------------------------------------

    def event(self, item: ChainItem) -> PerceptionEvent:
        """One item's metadata as a publishable event. Public so a test can read it.

        Raises:
            ValidationError: a per-object key cannot be attributed to detection rows
                (:func:`~shipinfer.topology.elements.detections.per_row`).
        """
        detections = item.meta.get("detections")
        if not isinstance(detections, Detections):
            # Not a refusal. A frame the detector never answered for still deserves an event —
            # it carries `detect` in `missing_stages`, and a consumer that stopped seeing a
            # camera would otherwise have to guess whether it went quiet or went down.
            detections = Detections.empty()
        height, width = item.meta.get("frame_hw", (0, 0))
        return PerceptionEvent.build(
            camera_id=item.context.camera_id,
            frame_id=item.context.frame_id,
            source_id=self._source_id,
            objects=self._records(item, detections),
            width=int(width),
            height=int(height),
            captured_ns=item.context.captured_ns,
            captured_unix_ns=item.context.captured_unix_ns,
            fps=float(item.meta.get("fps", 0.0)),
            missing_stages=tuple(item.meta.get(MISSING_STAGES, ())),
        )

    def _records(self, item: ChainItem, detections: Detections) -> tuple[ObjectRecord, ...]:
        """One record per detection row, filled from whatever the chain filed."""
        count = len(detections)
        what = f"output element {self.name!r}"
        vectors = per_row(item.meta.get("vectors"), count, what=what, key="vectors")
        identities = per_row(item.meta.get("identities"), count, what=what, key="identities")
        masks = per_row(item.meta.get("masks"), count, what=what, key="masks")
        tracks = self._tracks_by_row(item, count)
        camera_id = item.context.camera_id
        frame_id = item.context.frame_id
        boxes = detections.boxes
        scores = detections.scores
        # One conversion for the frame rather than four numpy calls per detection. This is
        # the emission path -- ~15 000 objects a second at the documented sizing -- and it is
        # the same argument `as_embedding` makes, applied to the box it sits next to.
        box_rows = (
            np.asarray(boxes, dtype=float).reshape(count, -1)[:, :4].tolist() if count else []
        )
        return tuple(
            ObjectRecord(
                det_id=f"{camera_id}_{frame_id}_{index}",
                class_name=detections.labels[index],
                score=float(scores[index]),
                bbox=(
                    box_rows[index][0],
                    box_rows[index][1],
                    box_rows[index][2],
                    box_rows[index][3],
                ),
                embedding=_embedding(None if vectors is None else vectors[index]),
                mask_area_px=_mask_area(None if masks is None else masks[index]),
                ship_id=_identity(None if identities is None else identities[index]),
                similarity=_similarity(None if identities is None else identities[index]),
                track_id=tracks[index][0],
                track_state=tracks[index][1],
                global_id=tracks[index][2],
            )
            for index in range(count)
        )

    # doc: long three keys read as one answer in step; the refusal depends on the pairing

    def _tracks_by_row(
        self, item: ChainItem, count: int
    ) -> list[tuple[int | None, str | None, int | None]]:
        """``(track_id, track_state, global_id)`` per detection row.

        The three travel together because they are one answer read through one attribution:
        ``meta["tracks"]`` is the tracker's own list, ``meta["track_rows"]`` says which
        detection each of those tracks came from (``elements/track.py``), and
        ``meta["global_ids"]`` is the cross-camera tier's answer *aligned with the tracks*
        (``elements/mtmc.py``). Reading them separately is how one of the three ends up on a
        different row from the other two.

        Raises:
            ValidationError: ``meta["tracks"]`` is present without ``meta["track_rows"]``.
                Those tracks cannot be put on an object, so publishing the frame anyway would
                mean every ``track_id`` silently ``null`` on a chain that is tracking — the
                failure this element's docstring says a ``None`` must never stand for.
        """
        empty: list[tuple[int | None, str | None, int | None]] = [(None, None, None)] * count
        tracks = item.meta.get("tracks")
        if not tracks:
            return empty
        rows = item.meta.get(TRACK_ROWS)
        if rows is None:
            raise ValidationError(
                f"output element {self.name!r} was handed meta['tracks'] with no "
                f"meta[{TRACK_ROWS!r}] to place them on. A track's box is a filtered estimate, "
                "so only the element that ran the tracker can say which detection it came "
                "from; a `track` element files both (`topology/elements/track.py`)"
            )
        if len(rows) != len(tracks):
            raise ValidationError(
                f"output element {self.name!r}: meta[{TRACK_ROWS!r}] has {len(rows)} entries "
                f"for {len(tracks)} tracks; they are one list read in step"
            )
        global_ids = item.meta.get("global_ids")
        if global_ids is not None and len(global_ids) != len(tracks):
            raise ValidationError(
                f"output element {self.name!r}: meta['global_ids'] has {len(global_ids)} "
                f"entries for {len(tracks)} tracks. The cross-camera tier answers one id per "
                "track, aligned with them (`topology/elements/mtmc.py`)"
            )
        # A copy, not another name for `empty`: the early returns above hand `empty` back
        # as the frame's answer, and a list that is mutated here must not be that one.
        placed = list(empty)
        for position, row in enumerate(rows):
            index = int(row)
            if not 0 <= index < count:
                # -1 is ordinary: a track the attribution could not match to any detection
                # above the threshold. Dropped rather than forced onto its nearest box.
                continue
            track = tracks[position]
            placed[index] = (
                _as_int(getattr(track, "track_id", None)),
                _as_state(getattr(track, "state", None)),
                None if global_ids is None else _as_int(global_ids[position]),
            )
        return placed


# doc: long the coercion accepts four shapes and each has a caller that produces it


def _embedding(vector: Any) -> tuple[float, ...]:
    """One appearance vector as the wire's JSON array. ``()`` when nothing was filed.

    ``()`` rather than ``None`` because :class:`~shipinfer.core.events.ObjectRecord` already
    says an empty embedding is "the embedder did not run for this object", and v1 consumers
    read ``body_feature_vec`` as a list of lists.

    The conversion itself is :func:`shipinfer.core.events.as_embedding` and is deliberately
    not written here: ``tuple(float(v) for v in vector)`` is a per-element Python loop on the
    emission path — ~30 M ``float()`` calls a second at the documented load, paid even with
    the ``null`` sink — and it has been written and removed twice already. This function is
    the *absence* rule and nothing else.
    """
    return () if vector is None else as_embedding(vector)


def _mask_area(entry: Any) -> float | None:
    """One object's foreground area in crop pixels, or ``None`` where no segmenter ran.

    A segment element scatters a one-wide row per detection
    (:class:`~shipinfer.topology.elements.masks.InstanceMaskArea`), so this unwraps that row.
    ``None`` and not ``0.0``: ``0.0`` is the answer for a crop the engine found nothing in,
    and "not segmented" must stay distinguishable from "segmented, and empty".
    """
    if entry is None:
        return None
    value = entry[0] if isinstance(entry, Sequence) and not isinstance(entry, str) else entry
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else None
    return None if value is None else float(value)


def _identity(entry: Any) -> int | None:
    """The gallery id from a ``meta["identities"]`` entry, or ``None``.

    An entry is ``(identity, similarity)`` — what a ``recognize`` element files per row — and
    ``None`` where the gallery had no answer. ``None`` is never ``0``: ``0`` is a legitimate
    gallery id, which is why the field is optional rather than sentinelled.
    """
    identity = entry[0] if isinstance(entry, Sequence) and not isinstance(entry, str) else entry
    return _as_int(identity)


def _similarity(entry: Any) -> float | None:
    """The cosine similarity behind an identity, or ``None`` when there is none."""
    if not isinstance(entry, Sequence) or isinstance(entry, str) or len(entry) < 2:
        return None
    return None if entry[1] is None else float(entry[1])


def _as_int(value: Any) -> int | None:
    """An id as a plain ``int``, or ``None``. Never ``0`` standing in for absence.

    ``int()`` and not a cast that trusts the source: a numpy ``int64`` serialises through
    ``json`` as a failure, not as a number, and both the tracker and the cross-camera assigner
    hand back numpy scalars.
    """
    return None if value is None else int(value)


def _as_state(value: Any) -> str | None:
    """A track's lifecycle state as a plain ``str``, or ``None``.

    ``str()`` for the same reason as :func:`_as_int`: a numpy ``str_`` is a ``str`` subclass
    that compares and serialises correctly right up until a consumer type-checks it.
    """
    return None if value is None else str(value)


@registry_for(ElementKind.OUTPUT).register("jsonlines", "jsonl", "file")
class JsonLinesOutput(SinkOutput):
    """One JSON object per line, to a file or to stdout.

    The implementation that makes a whole chain runnable and checkable with no broker: a test
    walks frames through it and reads the file back. ``params: {path: "-"}`` writes to stdout,
    which is what a container with no volume wants.
    """

    sink_name: ClassVar[str] = "jsonlines"


# doc: long the none/null alias inversion has a YAML argument behind it


#: ``none`` is the canonical name and ``null`` is the alias, which is the opposite of the
#: sink's own registration and is not an inconsistency worth removing: YAML reads a bare
#: ``impl: null`` as the *null literal*, so a chain file that spelled it that way is refused
#: by the schema with "Input should be a valid string" and nothing about the mistake. The name
#: an operator can actually type is the one this registry leads with.
@registry_for(ElementKind.OUTPUT).register("none", "null", "count")
class NullOutput(SinkOutput):
    """Counts events and publishes nowhere — the default, and the measurement harness.

    A chain should start without a broker: a deployment that has not decided where results go
    is better off producing none loudly than failing to boot. It is also how a bench measures
    the *chain* rather than the sink, since a throughput number that includes a JSON encoder
    is not a throughput number for the chain.
    """

    sink_name: ClassVar[str] = "null"


# doc: long the lazy registration is the layering promise this module makes

# `kafka` is registered LAZILY, and it is the one implementation here that has to be. Its
# module builds a sink whose constructor imports `confluent_kafka` -- the one heavy dependency
# `topology` is allowed to name (`check_layers.py`, `topology/sinks/__init__.py`) -- and it
# also has behaviour the other two do not: a broker's verdict arrives *after* it accepts a
# message, so the element drains the late refusals. None of that is worth importing on a host
# whose chain names `jsonlines`, and `Topology.from_spec` builds only the implementations a
# chain actually declares. The dotted path is resolved on first use
# (`core/registry.py::RegistryEntry.resolve`).
registry_for(ElementKind.OUTPUT).register_lazy(
    "kafka",
    "shipinfer.topology.elements.output_kafka:KafkaOutput",
    "broker",
    description="Publish one message per frame onto a Kafka topic",
)
