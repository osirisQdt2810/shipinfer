"""The event this system publishes — the old Kafka contract, extended for ships.

Lives in ``core`` because both generations of the pipeline build one: ``pipeline/graph/``
assembles it from a frame's stage outputs, and the ``output`` element under
``topology/elements/`` will assemble it from a chain item's metadata. ``topology`` may import
``core`` and nothing else, so a value shared by the two has exactly one legal home
(``shipinfer.pipeline.schema`` re-exports it for the coexistence arch.md §9 describes).

The downstream services already exist: ``motservice`` consumes per-frame detections and
``mtmcservice`` consumes tracklets (``references/bitbucket-subfaceid``). Replacing their
input format would mean rewriting both, so the rule from
``docs/new-system-architecture.md`` is *giữ contract cũ, mở rộng schema cho ship* — keep the
old contract, extend the schema for ships.

**What v1 was.** ``DetectionMOTFrameData`` (``KafkaData/DetectionMOTFrameData.h``) serialises
one message type, ``Det2MOT``, as parallel arrays::

    sub_id, det_id_vec, camera_id, image_id, det_body_score_vec,
    body_bbox_vec, body_feature_vec, img_width, img_height, img_fps

**What v2 adds**, and how it stays compatible:

* every v1 key keeps its name, its type and its meaning, and still carries **people only**,
  so a running ``motservice`` needs no change and no rebuild;
* ships get their own parallel arrays in the same idiom (``ship_bbox_vec``,
  ``ship_feature_vec``, ``ship_id_vec``, ...), because that is how the existing consumer
  code is written and a new consumer should not have to learn a second style;
* ``schema_version`` is explicit, so a consumer can branch on the number instead of
  guessing from the presence of a key;
* completeness is explicit (``partial``, ``missing_stages``). A frame that lost its
  embedder is *not* the same event as a frame with no people in it, and v1 could not tell
  those apart.

**What v3 adds** is the tracklet. ``motservice`` was named for the step that was missing:
this pipeline detected and embedded and then handed the result to a separate service to
associate. With Plane 3 running in-process the identity is already known when the event is
built, so it travels with the object — ``body_track_id_vec`` beside ``body_bbox_vec``,
``ship_track_id_vec`` beside ``ship_bbox_vec``, in the same parallel-array idiom as
everything else. Purely additive: ``as_det2mot`` is untouched, so a deployed ``motservice``
that ignores the new keys and does its own association keeps working, and one that reads them
can stop.

People are **not** duplicated into a generic object array. Two representations of a 512-d
embedding would double the largest field in the message — at 15 people per frame and 1000
frames a second that is the difference between 150 MB/s and 300 MB/s of JSON — and it would
leave two places for a consumer to disagree with itself.
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
#: track id and its state. Every step is additive, and the number is bumped rather than
#: left alone precisely so a consumer can branch on it instead of probing for a key.
SCHEMA_VERSION = 3


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
    #: Single-camera track identity, from Plane 3. Process-unique rather than per-camera, so
    #: two cameras' tracklets can meet in the cross-camera tier without colliding; the camera
    #: is in :attr:`PerceptionEvent.camera_id`, where it can be read.
    #:
    #: ``None`` when tracking is off, when the tracker did not publish this object yet (a
    #: track is withheld until it has earned confirmation — publishing one that dies after two
    #: frames hands downstream an identity that never existed), or when the frame lost the
    #: ordering race. Distinguishable from a track id of 0, which the counter never issues.
    track_id: int | None = None
    #: That track's lifecycle state, as the tracker reported it. Carried rather than assumed
    #: because a consumer that has to trust our filtering cannot apply its own.
    track_state: str | None = None

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
    #: Why this event was emitted: ``complete``, ``timeout``, ``failed`` or ``shutdown``.
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
