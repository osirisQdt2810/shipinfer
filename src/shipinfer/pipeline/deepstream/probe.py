"""Reading one batch's metadata out of the graph, and turning it into events.

THE ONE PLACE ``pyds`` IS TOUCHED, AND IT IS AN ARGUMENT
--------------------------------------------------------
``pyds`` ships with the DeepStream SDK and exists on no other machine. If it were imported at
module scope, everything in this package — the config generation, the geometry, the event
mapping — would be unreachable from the offline tier, and the only way to find out that a
bounding box had been published in the wrong coordinate space would be to run fifty cameras on
a DeepStream host. So :func:`walk_batch` takes the module as a **parameter**, reads the linked
lists into plain values, and stops there. Everything after it is ordinary Python over
:class:`FrameView` / :class:`ObjectView`, and every rule in it is pinned by a test that has
never seen a GPU.

WHAT THE GRAPH KNOWS THAT WE HAVE TO UNDO
-----------------------------------------
Three things, and each has been a silent bug in a DeepStream deployment somewhere:

* **Boxes are in the muxed frame's pixels.** ``nvstreammux`` scaled a 4K camera into
  1920x1080 before the detector saw it, so ``rect_params`` is in that space and a consumer
  told "source pixels" reads every box at half size. :class:`FrameGeometry` undoes it.
* **``object_id`` is unsigned, and "no track" is not zero.** It is
  ``UNTRACKED_OBJECT_ID`` — 2^64-1 — which read as an integer is a track id nobody will ever
  see again, published once per object.
* **``frame_num`` restarts at 0 when a source reconnects.** ``(camera, frame)`` is the tag this
  whole system keys reassembly and tracking on (ADR-002), so a reused pair hands downstream a
  key it has already seen. :class:`FrameNumbering` is the fix, per camera and monotonic.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from shipinfer.pipeline.graph.detections import UNKNOWN_LABEL
from shipinfer.pipeline.schema import ObjectRecord, PerceptionEvent

__all__ = [
    "UNTRACKED_OBJECT_ID",
    "FrameGeometry",
    "FrameNumbering",
    "FrameView",
    "ObjectView",
    "build_event",
    "walk_batch",
]

#: What ``NvDsObjectMeta.object_id`` holds when no tracker has claimed the object —
#: ``UNTRACKED_OBJECT_ID`` in ``nvds_tracker_meta.h``. Read as a number it is 18446744073709551615,
#: which is why it must be turned into ``None`` exactly once, here.
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """How ``nvstreammux`` mapped one camera's frame into the batch, so a box can be unmapped.

    ``padded`` follows ``nvstreammux``'s ``enable-padding``: off, the frame is stretched to the
    muxed extent and the inverse is one scale per axis; on, it is letterboxed — scaled by the
    smaller ratio and centre-padded — and the inverse has to subtract the pad before dividing.
    Getting this wrong is not a crash: it is boxes that are plausibly placed and consistently
    wrong, on the aspect ratios that differ from the mux's.
    """

    mux_width: int
    mux_height: int
    source_width: int
    source_height: int
    padded: bool = False

    def to_source(
        self, left: float, top: float, width: float, height: float
    ) -> tuple[float, float, float, float]:
        """``(left, top, w, h)`` in muxed pixels -> ``(x1, y1, x2, y2)`` in source pixels.

        The shape changes as well as the scale, deliberately: DeepStream's ``rect_params`` is
        an origin plus extents and :attr:`~shipinfer.pipeline.schema.ObjectRecord.bbox` is a
        corner pair. A conversion that kept the shape and changed only the scale would publish
        ``(x1, y1, w, h)`` into a field every consumer reads as corners, and the boxes would
        look almost right.
        """
        x1, y1, x2, y2 = left, top, left + width, top + height
        if self.source_width <= 0 or self.source_height <= 0:
            # The frame meta did not say how big the source was. Scaling by an unknown is worse
            # than not scaling, so the muxed coordinates travel as they are and the event says
            # width/height 0 — visibly missing rather than quietly halved.
            return (x1, y1, x2, y2)
        if self.padded:
            scale = min(
                self.mux_width / self.source_width, self.mux_height / self.source_height
            )
            pad_x = (self.mux_width - self.source_width * scale) / 2.0
            pad_y = (self.mux_height - self.source_height * scale) / 2.0
            x1, x2 = (x1 - pad_x) / scale, (x2 - pad_x) / scale
            y1, y2 = (y1 - pad_y) / scale, (y2 - pad_y) / scale
        else:
            scale_x = self.source_width / self.mux_width
            scale_y = self.source_height / self.mux_height
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
        return (
            _clamp(x1, self.source_width),
            _clamp(y1, self.source_height),
            _clamp(x2, self.source_width),
            _clamp(y2, self.source_height),
        )


@dataclass(frozen=True, slots=True)
class ObjectView:
    """One object as the graph reported it. ``rect`` is ``(left, top, w, h)`` in muxed pixels."""

    class_id: int
    confidence: float
    rect: tuple[float, float, float, float]
    object_id: int
    #: The secondary GIE's output tensor for this object, empty when no secondary ran on it —
    #: which is a *skip*, not a missing stage: the embedder ran and this object was below its
    #: minimum size or of a class it does not operate on. The two are different facts and the
    #: event distinguishes them (`missing_stages` names a stage that never ran at all).
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class FrameView:
    """One frame of the batch, flattened out of the metadata's linked lists."""

    #: The muxer's sink pad this frame arrived on — how a frame is attributed to a camera.
    pad_index: int
    frame_num: int
    source_width: int
    source_height: int
    #: Wall-clock capture time in nanoseconds, from the source's RTCP sender reports. 0 when
    #: the source has none (a file, or an RTSP camera before its first sender report).
    ntp_timestamp_ns: int
    objects: tuple[ObjectView, ...] = ()


def walk_batch(pyds: Any, batch_meta: Any) -> list[FrameView]:
    """Flatten one ``NvDsBatchMeta`` into plain values. The only ``pyds``-aware function here.

    Args:
        pyds: the DeepStream Python bindings module, passed in rather than imported so a fake
            can drive this walk in the offline tier.
        batch_meta: what ``pyds.gst_buffer_get_nvds_batch_meta`` returned for this buffer.

    The ``StopIteration`` guards are not decoration and not a Python idiom used wrongly:
    ``pyds``' ``cast`` and its ``.next`` raise it at the end of a list, which is why every
    DeepStream sample is written this way. What matters is that hitting one **ends the walk
    without discarding what was already collected** — a truncated batch is still every frame
    before the truncation, and dropping them would lose real detections to a list-walking
    accident.
    """
    frames: list[FrameView] = []
    node = batch_meta.frame_meta_list
    while node is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(node.data)
        except StopIteration:
            break
        frames.append(
            FrameView(
                pad_index=int(frame_meta.pad_index),
                frame_num=int(frame_meta.frame_num),
                source_width=int(getattr(frame_meta, "source_frame_width", 0) or 0),
                source_height=int(getattr(frame_meta, "source_frame_height", 0) or 0),
                ntp_timestamp_ns=int(getattr(frame_meta, "ntp_timestamp", 0) or 0),
                objects=_objects(pyds, frame_meta),
            )
        )
        try:
            node = node.next
        except StopIteration:
            break
    return frames


def _objects(pyds: Any, frame_meta: Any) -> tuple[ObjectView, ...]:
    objects: list[ObjectView] = []
    node = frame_meta.obj_meta_list
    while node is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(node.data)
        except StopIteration:
            break
        rect = obj_meta.rect_params
        # `confidence` is the detector's. The tracker publishes a box on a frame the detector
        # missed and sets confidence to -1 there, keeping its own in `tracker_confidence`; a
        # negative score in an event would be a consumer's threshold silently inverted.
        confidence = float(obj_meta.confidence)
        if confidence < 0.0:
            confidence = float(getattr(obj_meta, "tracker_confidence", 0.0) or 0.0)
        objects.append(
            ObjectView(
                class_id=int(obj_meta.class_id),
                confidence=confidence,
                rect=(
                    float(rect.left),
                    float(rect.top),
                    float(rect.width),
                    float(rect.height),
                ),
                object_id=int(obj_meta.object_id),
                embedding=_embedding(pyds, obj_meta),
            )
        )
        try:
            node = node.next
        except StopIteration:
            break
    return tuple(objects)


def _embedding(pyds: Any, obj_meta: Any) -> tuple[float, ...]:
    """The first output tensor a secondary GIE attached to this object, as floats.

    ``output-tensor-meta=1`` puts an ``NvDsInferTensorMeta`` on the object's user-meta list;
    the layers inside it are the network's outputs and the buffer is device-mapped host memory,
    so it is read through the pointer rather than copied element by element — 2048 floats at
    15 000 objects a second is not a per-element Python call.

    The *first* tensor wins, and that is safe only because
    :func:`~shipinfer.pipeline.deepstream.configs.write_configs` refuses two secondaries that
    claim the same label. An object therefore carries at most one.
    """
    node = getattr(obj_meta, "obj_user_meta_list", None)
    while node is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(node.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            values = _first_output_layer(pyds, tensor_meta)
            if values is not None:
                return values
        try:
            node = node.next
        except StopIteration:
            break
    return ()


def _first_output_layer(pyds: Any, tensor_meta: Any) -> tuple[float, ...] | None:
    for index in range(int(tensor_meta.num_output_layers)):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, index)
        if getattr(layer, "isInput", False):
            continue
        count = int(layer.dims.numElements)
        if count <= 0:
            continue
        pointer = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
        # `astype` copies, and the copy is the point: the buffer belongs to the graph and is
        # recycled the moment this probe returns, so a view into it would be re-filled by the
        # next batch while the event was still being serialised.
        return tuple(np.ctypeslib.as_array(pointer, shape=(count,)).astype(float).tolist())
    return None


class FrameNumbering:
    """Per-camera frame ids that keep going up across a source reconnect.

    ``nvurisrcbin`` restarts ``frame_num`` at 0 when it reopens a camera, and ``(camera_id,
    frame_id)`` is the tag everything downstream keys on (ADR-002) — reassembly, the tracker,
    a consumer's own de-duplication. Re-issuing a pair means handing them a key they have
    already seen, which is a wrong join rather than a lost frame.

    Pure and per process: a restarted *shard* starts its numbering again, which is what
    :attr:`~shipinfer.core.settings.ingest.CameraConfig.first_frame_id` exists for on the
    ingest path.
    """

    __slots__ = ("_last", "_offset")

    def __init__(self) -> None:
        self._last: dict[str, int] = {}
        self._offset: dict[str, int] = {}

    def next(self, camera_id: str, frame_num: int) -> int:
        """The id to publish for ``camera_id``'s ``frame_num``. Strictly increasing per camera."""
        issued = frame_num + self._offset.get(camera_id, 0)
        last = self._last.get(camera_id)
        if last is not None and issued <= last:
            self._offset[camera_id] = last + 1 - frame_num
            issued = last + 1
        self._last[camera_id] = issued
        return issued


def build_event(
    view: FrameView,
    *,
    camera_id: str,
    source_id: str,
    labels: Mapping[int, str],
    geometry: FrameGeometry,
    fps: float,
    frame_id: int,
    epoch_offset_ns: int,
    missing_stages: Sequence[str] = (),
) -> PerceptionEvent:
    """One frame's metadata as the event every topology publishes.

    Args:
        labels: ``pipeline.class_labels`` — the same map the Python DAG branches on. An id it
            does not mention becomes ``"unknown"`` rather than a raw index, because an index is
            a property of the checkpoint and means nothing to a consumer.
        epoch_offset_ns: ``time.time_ns() - time.monotonic_ns()``, read **once** by the caller.
            The graph stamps wall-clock capture time (NTP) and
            :meth:`~shipinfer.pipeline.schema.PerceptionEvent.build` measures latency from the
            monotonic clock, so the two have to be joined somewhere; doing it with an offset
            sampled once per process rather than a clock read per frame keeps the latency
            comparable across frames and costs nothing on the probe thread.
        missing_stages: the stages this topology does not run at all. Non-empty on purpose in
            PR1: the segmenter and the recogniser are absent, and an event that did not say so
            would be a partial frame published as a complete one.

    The track id is coarser than the Python plane's on purpose. DeepStream reports an
    ``object_id`` and nothing about the track's lifecycle, so a tracked object gets
    ``track_state="tracked"`` — where :class:`~shipinfer.pipeline.graph.tracking.TrackerShard`
    would say ``tentative`` / ``confirmed``. Publishing a state we did not measure would be
    worse than publishing a coarse one; ``None`` means no tracker claimed the object.
    """
    captured_unix_ns = view.ntp_timestamp_ns
    records = tuple(
        ObjectRecord(
            det_id=f"{camera_id}_{frame_id}_{index}",
            class_name=labels.get(obj.class_id, UNKNOWN_LABEL),
            score=obj.confidence,
            bbox=geometry.to_source(*obj.rect),
            embedding=obj.embedding,
            track_id=None if obj.object_id == UNTRACKED_OBJECT_ID else obj.object_id,
            track_state=None if obj.object_id == UNTRACKED_OBJECT_ID else "tracked",
        )
        for index, obj in enumerate(view.objects)
    )
    # A missing NTP stamp must not become a zero that reads as a measured zero (#32
    # round 4): with `attach-sys-ts=false` a file source NEVER stamps, so every event
    # would publish latency_us=0 on the axis this project exists to optimise, and the
    # bench comparing topologies would read "no latency" as a result. When the source
    # gave no capture time, the probe's own receipt is the capture time — the latency
    # then measures probe→emit, and `extra.capture_origin` says which clock it was so a
    # consumer can tell the two apart instead of trusting one field with two meanings.
    if captured_unix_ns:
        origin = "source"
        captured_ns = captured_unix_ns - epoch_offset_ns
    else:
        origin = "probe"
        captured_ns = time.monotonic_ns()
        captured_unix_ns = captured_ns + epoch_offset_ns
    event = PerceptionEvent.build(
        camera_id=camera_id,
        frame_id=frame_id,
        source_id=source_id,
        objects=records,
        width=geometry.source_width,
        height=geometry.source_height,
        fps=fps,
        captured_ns=captured_ns,
        captured_unix_ns=captured_unix_ns,
        missing_stages=missing_stages,
    )
    event.extra["capture_origin"] = origin
    return event


def _clamp(value: float, limit: int) -> float:
    return float(min(max(value, 0.0), float(limit)))
