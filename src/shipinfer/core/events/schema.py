"""The perception event: one frame's finished answer, as the wire and the file see it.

Schema v4: per-object parallel arrays (``*_bbox_vec``, ``*_track_id_vec``,
``*_global_id_vec``, ``*_feature_vec``) split by class, plus frame identity (``camera_id``,
``image_id``, ``sub_id``), geometry (``img_width/height/fps``) and ``missing_stages`` — a partial frame
says so instead of reading as an empty complete one. Why every v1 key keeps its name, type
and people-only meaning (a deployed ``motservice`` must need no rebuild):
``docs/design/event-schema.md``. Stdlib only, by construction and by test
(``TestTheSchemaIsPortable``): a consumer may copy this module out wholesale.
``PerceptionEvent.build`` is the one constructor; ``pipeline/schema.py`` re-exports for the
old spelling.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MESSAGE_TYPE",
    "SCHEMA_VERSION",
    "ObjectRecord",
    "PerceptionEvent",
]

#: The ``type`` field every consumer switches on. Unchanged from v1 on purpose: the value is
#: part of the contract, and a new one would be routed nowhere by a deployed consumer.
MESSAGE_TYPE = "Det2MOT"

#: 1 was ``DetectionMOTFrameData``. 2 adds ships, timing and completeness; 3 adds the
#: track id and its state; 4 adds the cross-camera ``global_id``. Every step is additive, and
#: the number is bumped rather than left alone precisely so a consumer can branch on it
#: instead of probing for a key.
SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    """One detected object and everything the DAG learned about it.

    ``frozen`` and ``slots`` because there are 15 000 of these a second: this is the
    highest-cardinality value in the system, and a per-object dict would cost more than the
    inference that produced it.

    A field is ``None`` when the stage that fills it did not run — a person has no
    ``ship_id`` and never will, and a ship whose recogniser timed out has none *yet*. Both
    are distinguishable from ``0``, which is a legitimate gallery id.
    """

    #: ``<camera>_<frame>_<index>``: unique across the fleet and derivable by a consumer, so
    #: a tracker can key on it without a side channel.
    det_id: str
    #: The label from ``pipeline.class_labels`` — ``"ship"``, ``"person"``, or whatever a
    #: deployment configured. Never a raw class index: the index is a property of the
    #: checkpoint and changes when the model is retrained.
    class_name: str
    score: float
    #: ``(x1, y1, x2, y2)`` in the **source frame's** pixels, letterbox already undone.
    bbox: tuple[float, float, float, float]
    #: Appearance embedding, empty when the embedder did not run for this object.
    embedding: tuple[float, ...] = ()
    #: Gallery identity, ships only.
    ship_id: int | None = None
    #: Cosine similarity behind :attr:`ship_id`, so a consumer can apply its own threshold
    #: instead of trusting ours.
    similarity: float | None = None
    #: Mask area in pixels. The mask *itself* is deliberately not published: a 512x512
    #: float mask is 1 MB and this bus carries metadata, not pixels — the architecture doc
    #: is explicit that frames stay in shared memory and Kafka gets the small results.
    mask_area_px: float | None = None
    #: Single-camera track identity, from Plane 3 — process-unique so two cameras' tracklets
    #: meet in the cross-camera tier without colliding. ``None`` = tracking off, the track not
    #: yet confirmed (an identity that dies after two frames never existed downstream), or the
    #: frame lost the ordering race; distinct from id 0, which the counter never issues.
    track_id: int | None = None
    #: That track's lifecycle state, as the tracker reported it. Carried rather than assumed
    #: because a consumer that has to trust our filtering cannot apply its own.
    track_state: str | None = None
    #: Fleet-wide identity for this object's tracklet, from the cross-camera tier (v4). The
    #: same object seen by two cameras carries the same ``global_id`` and two different
    #: ``track_id``\ s, which is the whole distinction between the two fields: one is unique
    #: within a camera's timeline, the other across the deployment.
    #:
    #: ``None`` when cross-camera association is off, when the frame's instant closed without
    #: this camera (``mtmc`` in :attr:`PerceptionEvent.missing_stages`), or when this object
    #: had no published track for an id to attach to. Distinguishable from ``0``, which the
    #: assigner never issues.
    global_id: int | None = None

    @property
    def bbox_list(self) -> list[float]:
        """The box in the wire format's shape (a JSON array, as in v1)."""
        return [float(v) for v in self.bbox]


@dataclass(frozen=True, slots=True)
class PerceptionEvent:
    """One frame's perception result, ready to serialise.

    Immutable because it is handed to a sink that may batch, retry or fan it out, and a
    value that can be mutated after being queued for publication is a race with no symptom
    beyond a wrong number in a dashboard.
    """

    camera_id: str
    #: The frame id from the ingest tag. Named ``image_id`` on the wire, as in v1.
    frame_id: int
    #: Which perception process produced this — v1's ``sub_id``.
    source_id: str
    objects: tuple[ObjectRecord, ...] = ()
    img_width: int = 0
    img_height: int = 0
    img_fps: int = 0
    #: Wall-clock capture time, for a human and for a consumer joining across processes.
    captured_unix_ns: int = 0
    emitted_unix_ns: int = 0
    #: Capture to emission. Measured from the monotonic clock, which is why it is carried
    #: rather than derived from the two wall-clock stamps: NTP stepping the clock must not
    #: turn into a negative latency in a dashboard.
    latency_us: int = 0
    #: Stages that never delivered. Empty for a complete frame.
    missing_stages: tuple[str, ...] = ()
    #: Why this event was emitted, in the collector's own five words: ``complete``,
    #: ``incomplete``, ``timeout``, ``shutdown``, ``evicted``, passed through verbatim by
    #: ``pipeline/runner.py``. It used to say ``failed``, which nothing emits -- and the C++
    #: port read this line and wrote it for two of the five (fixed with P5-A).
    reason: str = "complete"
    schema_version: int = SCHEMA_VERSION
    type: str = MESSAGE_TYPE
    #: Free-form additions a deployment needs and the schema should not grow a field for.
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        camera_id: str,
        frame_id: int,
        source_id: str,
        objects: Sequence[ObjectRecord],
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        captured_ns: int = 0,
        captured_unix_ns: int = 0,
        missing_stages: Sequence[str] = (),
        reason: str = "complete",
    ) -> PerceptionEvent:
        """Stamp an event with both clocks read at the moment of emission.

        ``captured_ns`` is monotonic and ``captured_unix_ns`` is wall clock, exactly as the
        ingest tag carries them, and the latency is computed from the monotonic pair only.
        """
        now_ns = time.monotonic_ns()
        return cls(
            camera_id=camera_id,
            frame_id=frame_id,
            source_id=source_id,
            objects=tuple(objects),
            img_width=int(width),
            img_height=int(height),
            img_fps=round(fps),
            captured_unix_ns=captured_unix_ns,
            emitted_unix_ns=time.time_ns(),
            latency_us=max(0, (now_ns - captured_ns) // 1000) if captured_ns else 0,
            missing_stages=tuple(missing_stages),
            reason=reason,
        )

    # -- views -------------------------------------------------------------------------

    @property
    def key(self) -> tuple[str, int]:
        """``(camera_id, frame_id)`` — the tag this event must carry unchanged (ADR-002)."""
        return (self.camera_id, self.frame_id)

    @property
    def is_partial(self) -> bool:
        return bool(self.missing_stages)

    def objects_of(self, class_name: str) -> tuple[ObjectRecord, ...]:
        """Every object of one class, in detection order."""
        return tuple(o for o in self.objects if o.class_name == class_name)

    def as_det2mot(self) -> dict[str, Any]:
        """Exactly the v1 ``Det2MOT`` payload: people only, v1 key names, v1 types.

        Byte-compatible with ``DetectionMOTFrameData::serialize`` so a deployed
        ``motservice`` consumes it unchanged. Nothing new appears here — a v1 consumer that
        validates its input strictly must see a v1 message.
        """
        people = self.objects_of("person")
        return {
            "sub_id": self.source_id,
            "det_id_vec": [o.det_id for o in people],
            "camera_id": self.camera_id,
            "image_id": self.frame_id,
            "det_body_score_vec": [o.score for o in people],
            "body_bbox_vec": [o.bbox_list for o in people],
            "body_feature_vec": [o.embedding for o in people],
            "img_width": self.img_width,
            "img_height": self.img_height,
            "img_fps": self.img_fps,
        }

    def as_dict(self) -> dict[str, Any]:
        """The current payload: every v1 key, plus ships, timing, completeness and tracklets."""
        ships = self.objects_of("ship")
        people = self.objects_of("person")
        payload = self.as_det2mot()
        payload.update(
            {
                "type": self.type,
                "schema_version": self.schema_version,
                # Ships in the same parallel-array idiom as the person fields above, so a
                # consumer written against v1 reads v2 with the same helpers.
                "ship_det_id_vec": [o.det_id for o in ships],
                "det_ship_score_vec": [o.score for o in ships],
                "ship_bbox_vec": [o.bbox_list for o in ships],
                "ship_feature_vec": [o.embedding for o in ships],
                "ship_id_vec": [o.ship_id for o in ships],
                "ship_similarity_vec": [o.similarity for o in ships],
                "ship_mask_area_vec": [o.mask_area_px for o in ships],
                # Tracklets (v3). People keep the unprefixed ``body_`` idiom the v1 person
                # arrays use and ships keep the ``ship_`` one, so a consumer that already
                # walks one set of parallel arrays walks these with the same helper. A null
                # entry means this object has no published identity — see ObjectRecord.
                "body_track_id_vec": [o.track_id for o in people],
                "body_track_state_vec": [o.track_state for o in people],
                "ship_track_id_vec": [o.track_id for o in ships],
                "ship_track_state_vec": [o.track_state for o in ships],
                # Cross-camera identity (v4), one entry per object, beside the per-camera
                # track id rather than replacing it: a consumer that joins two cameras reads
                # these, and one that only follows a single camera keeps reading the pair
                # above. A null entry means this object has no global identity — see
                # ObjectRecord.global_id for the three ways that happens.
                "body_global_id_vec": [o.global_id for o in people],
                "ship_global_id_vec": [o.global_id for o in ships],
                "captured_unix_ns": self.captured_unix_ns,
                "emitted_unix_ns": self.emitted_unix_ns,
                "latency_us": self.latency_us,
                # A frame with no people and a frame whose embedder timed out look
                # identical in v1. These two keys are the difference.
                "partial": self.is_partial,
                "missing_stages": list(self.missing_stages),
                "reason": self.reason,
            }
        )
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def to_json(self) -> str:
        """One line of JSON. ``separators`` because 1000 of these a second is 1000 x the
        whitespace, and every byte crosses a broker."""
        return json.dumps(self.as_dict(), separators=(",", ":"))

    def __repr__(self) -> str:
        state = f" partial missing={list(self.missing_stages)}" if self.is_partial else ""
        return (
            f"<PerceptionEvent cam={self.camera_id} frame={self.frame_id} "
            f"objects={len(self.objects)}{state}>"
        )
