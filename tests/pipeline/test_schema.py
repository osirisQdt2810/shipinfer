"""The emitted event: the old contract kept, and the new fields added beside it.

``motservice`` and ``mtmcservice`` are deployed and consume ``Det2MOT`` as
``KafkaData/DetectionMOTFrameData.h`` serialises it. The rule from the architecture doc is to
keep that contract and extend the schema for ships, so the first class here asserts the v1
payload key for key, and the rest assert that everything new is *additive*.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from shipinfer.pipeline.schema import (
    MESSAGE_TYPE,
    SCHEMA_VERSION,
    ObjectRecord,
    PerceptionEvent,
)

pytestmark = pytest.mark.timeout(30)

#: Exactly the keys ``DetectionMOTFrameData::serialize`` writes.
V1_KEYS = {
    "sub_id",
    "det_id_vec",
    "camera_id",
    "image_id",
    "det_body_score_vec",
    "body_bbox_vec",
    "body_feature_vec",
    "img_width",
    "img_height",
    "img_fps",
}


def person(index: int, *, score: float = 0.9) -> ObjectRecord:
    return ObjectRecord(
        det_id=f"cam0_1_{index}",
        class_name="person",
        score=score,
        bbox=(float(index), 1.0, 2.0, 3.0),
        embedding=(0.1, 0.2),
    )


def ship(index: int) -> ObjectRecord:
    return ObjectRecord(
        det_id=f"cam0_1_{index}",
        class_name="ship",
        score=0.8,
        bbox=(10.0, 11.0, 12.0, 13.0),
        embedding=(0.3, 0.4),
        ship_id=42,
        similarity=0.91,
        mask_area_px=1234.0,
    )


def event(*objects: ObjectRecord, **kwargs) -> PerceptionEvent:
    defaults = {
        "camera_id": "cam0",
        "frame_id": 1,
        "source_id": "perception-0",
        "width": 1920,
        "height": 1080,
        "fps": 20.0,
        "captured_ns": 1,
        "captured_unix_ns": 2,
    }
    defaults.update(kwargs)
    return PerceptionEvent.build(objects=list(objects), **defaults)


class TestTheV1ContractIsUnchanged:
    """A deployed consumer must read this message without being rebuilt."""

    def test_the_legacy_payload_has_exactly_the_v1_keys(self):
        assert set(event(person(0), ship(1)).as_det2mot()) == V1_KEYS

    def test_the_legacy_payload_carries_people_only(self):
        payload = event(person(0), ship(1), person(2)).as_det2mot()
        assert payload["det_id_vec"] == ["cam0_1_0", "cam0_1_2"]
        assert len(payload["body_bbox_vec"]) == 2
        assert payload["image_id"] == 1
        assert payload["sub_id"] == "perception-0"
        assert payload["img_fps"] == 20

    def test_the_parallel_arrays_stay_aligned(self):
        """The v1 format is parallel arrays; a consumer indexes them together."""
        payload = event(person(0, score=0.9), person(1, score=0.5)).as_det2mot()
        assert payload["det_id_vec"] == ["cam0_1_0", "cam0_1_1"]
        assert payload["det_body_score_vec"] == [0.9, 0.5]
        assert payload["body_bbox_vec"] == [[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0]]
        # Tuples, not lists. `ObjectRecord.embedding` is already a tuple of floats and
        # `json.dumps` encodes a tuple as an array, so copying each one into a list per
        # object bought nothing — at 2048 floats and ~15 000 objects/s it was a second
        # full copy of every embedding on the emission path. The wire form is unchanged,
        # which is what `TestTheWireFormIsUnchangedByTheTupleChange` below pins.
        assert payload["body_feature_vec"] == [(0.1, 0.2), (0.1, 0.2)]

    def test_the_v2_payload_still_contains_every_v1_key_with_the_same_value(self):
        subject = event(person(0), ship(1))
        legacy = subject.as_det2mot()
        current = subject.as_dict()
        assert V1_KEYS.issubset(current)
        assert all(current[key] == value for key, value in legacy.items())

    def test_the_message_type_is_unchanged(self):
        """A new type value would be routed nowhere by a deployed consumer."""
        assert MESSAGE_TYPE == "Det2MOT"
        assert event().as_dict()["type"] == "Det2MOT"


class TestShipsAreAnExtension:
    """Same idiom, new names — a consumer written against v1 reads v2 with its own helpers."""

    def test_ships_get_their_own_parallel_arrays(self):
        payload = event(person(0), ship(1)).as_dict()
        assert payload["ship_det_id_vec"] == ["cam0_1_1"]
        assert payload["det_ship_score_vec"] == [0.8]
        assert payload["ship_bbox_vec"] == [[10.0, 11.0, 12.0, 13.0]]
        assert payload["ship_feature_vec"] == [(0.3, 0.4)]
        assert payload["ship_id_vec"] == [42]
        assert payload["ship_similarity_vec"] == [0.91]
        assert payload["ship_mask_area_vec"] == [1234.0]

    def test_a_person_never_appears_in_a_ship_array(self):
        payload = event(person(0), person(1), ship(2)).as_dict()
        assert len(payload["ship_bbox_vec"]) == 1
        assert len(payload["body_bbox_vec"]) == 2

    def test_an_embedding_is_carried_once_not_twice(self):
        """Duplicating a 512-d embedding would double the largest field in the message."""
        assert event(person(0)).to_json().count("0.1,0.2") == 1

    def test_the_schema_version_is_explicit(self):
        """So a consumer branches on a number rather than on the presence of a key.

        Pinned to a literal, not to the constant: the number *is* the contract, and a test
        that read the constant would agree with any value the source happened to hold.
        """
        assert SCHEMA_VERSION == 3
        assert event().as_dict()["schema_version"] == 3

    def test_masks_are_summarised_not_published(self):
        """This bus carries metadata; a 512x512 float mask is 1 MB and stays out of it."""
        payload = event(ship(0)).as_dict()
        assert payload["ship_mask_area_vec"] == [1234.0]
        assert not any("mask" in key and "area" not in key for key in payload)


class TestTrackletsAreAnExtension:
    """v3 carries the identity Plane 3 assigned, in the same idiom and beside the old keys.

    ``motservice`` was named for the step that used to be missing: this pipeline detected and
    embedded, and a separate service associated. With tracking in-process the identity is
    already known when the event is built — so it travels with the object, and a deployed
    consumer that ignores the new keys and does its own association keeps working unchanged.
    """

    def tracked(self, **fields) -> PerceptionEvent:
        return event(
            replace(person(0), **fields),
            replace(ship(1), **fields),
        )

    def test_the_legacy_payload_gains_no_key(self):
        payload = self.tracked(track_id=7, track_state="confirmed").as_det2mot()
        assert set(payload) == V1_KEYS

    def test_the_legacy_payload_is_identical_tracked_or_not(self):
        assert (
            self.tracked().as_det2mot()
            == self.tracked(track_id=7, track_state="confirmed").as_det2mot()
        )

    def test_tracks_follow_the_existing_parallel_array_idiom(self):
        payload = self.tracked(track_id=7, track_state="confirmed").as_dict()
        assert payload["body_track_id_vec"] == [7]
        assert payload["ship_track_id_vec"] == [7]
        assert payload["body_track_state_vec"] == ["confirmed"]
        assert payload["ship_track_state_vec"] == ["confirmed"]

    def test_the_arrays_stay_aligned_with_the_boxes_they_describe(self):
        payload = self.tracked(track_id=7).as_dict()
        assert len(payload["body_track_id_vec"]) == len(payload["body_bbox_vec"])
        assert len(payload["ship_track_id_vec"]) == len(payload["ship_bbox_vec"])

    def test_an_untracked_object_serialises_as_null_not_zero(self):
        """0 is a legitimate track id; ``null`` is "the stage did not run for this object"."""
        payload = self.tracked().as_dict()
        assert payload["body_track_id_vec"] == [None]
        assert payload["body_track_state_vec"] == [None]
        assert json.loads(self.tracked().to_json())["ship_track_id_vec"] == [None]


class TestCompletenessIsExplicit:
    """v1 could not tell "no people in this frame" from "the embedder never answered"."""

    def test_a_complete_event_says_so(self):
        payload = event(person(0)).as_dict()
        assert payload["partial"] is False
        assert payload["missing_stages"] == []
        assert payload["reason"] == "complete"

    def test_a_partial_event_names_what_is_missing(self):
        payload = event(
            person(0), missing_stages=("person_embedder",), reason="timeout"
        ).as_dict()
        assert payload["partial"] is True
        assert payload["missing_stages"] == ["person_embedder"]
        assert payload["reason"] == "timeout"

    def test_an_unset_field_is_none_not_zero(self):
        """0 is a legitimate gallery id, so "no answer" must be distinguishable from it."""
        record = ObjectRecord(det_id="d", class_name="ship", score=0.5, bbox=(0, 0, 1, 1))
        assert record.ship_id is None
        assert event(record).as_dict()["ship_id_vec"] == [None]


class TestTimestampsAndLatency:
    """Latency comes from the monotonic clock; wall clock is for humans."""

    def test_latency_is_derived_from_the_monotonic_capture_stamp(self):
        subject = event(captured_ns=1)
        assert subject.latency_us > 0
        assert subject.captured_unix_ns == 2
        assert subject.emitted_unix_ns > 0

    def test_a_frame_with_no_monotonic_stamp_reports_no_latency(self):
        """Rather than a nonsense number derived from an epoch."""
        assert event(captured_ns=0).latency_us == 0

    def test_latency_is_never_negative(self):
        """NTP stepping the wall clock must not produce a negative number in a dashboard."""
        assert event(captured_ns=2**62).latency_us == 0


class TestTheWireFormIsUnchangedByTheTupleChange:
    """The contract is the bytes on the wire, not the Python type in the dict.

    `body_feature_vec` and `ship_feature_vec` hold tuples now rather than freshly copied
    lists. That is invisible to every consumer — `json.dumps` writes an array either way —
    and it removes a per-object copy of a 2048-float embedding from the emission path.
    """

    def test_an_embedding_encodes_as_a_json_array(self) -> None:
        subject = event(person(0), ship(1))

        decoded = json.loads(subject.to_json())

        assert decoded["body_feature_vec"] == [[0.1, 0.2]]
        assert decoded["ship_feature_vec"] == [[0.3, 0.4]]

    def test_the_v1_payload_encodes_identically(self) -> None:
        """`as_det2mot` is the v1 contract an existing consumer is written against."""
        payload = event(person(0)).as_det2mot()

        assert json.dumps(payload["body_feature_vec"]) == "[[0.1, 0.2]]"


class TestSerialisation:
    """One line of JSON per frame, because a thousand a second cross a broker."""

    def test_to_json_is_one_line_and_round_trips(self):
        """JSON has no tuple, so the embeddings come back as lists — which is the point:
        the payload holds tuples and the wire holds arrays, and a consumer sees no change."""
        subject = event(person(0), ship(1))
        line = subject.to_json()
        assert "\n" not in line

        decoded = json.loads(line)
        payload = subject.as_dict()
        assert decoded.keys() == payload.keys()
        for key, value in payload.items():
            expected = json.loads(json.dumps(value))
            assert decoded[key] == expected, key

    def test_it_has_no_wasted_whitespace(self):
        assert ", " not in event(person(0)).to_json()

    def test_the_key_is_the_frame_tag(self):
        assert event().key == ("cam0", 1)

    def test_objects_of_selects_one_class(self):
        subject = event(person(0), ship(1), person(2))
        assert len(subject.objects_of("person")) == 2
        assert len(subject.objects_of("ship")) == 1
        assert subject.objects_of("buoy") == ()
