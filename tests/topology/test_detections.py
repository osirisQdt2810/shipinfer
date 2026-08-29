"""The decode's arithmetic, asserted on values rather than on shapes.

This PR's premise is "three mismatches the stubs were hiding, each of which passed its
tests". Shipping the replacement arithmetic with the same property -- passes its tests,
asserts no number -- would reproduce that failure rather than close it, so every test here
names the number it expects.

Each of these was green before this file existed:

- reordering the un-letterbox so the scale is applied before the pad is subtracted, or
  spelling the slice ``boxes[:, [0, 2]]`` (a copy) instead of ``boxes[:, 0::2]`` (a view),
  which makes every published box wrong by the pad -- a mistake the source comment records
  having been made once already;
- ignoring ``count``, so a detector reporting 3 filled rows of 300 yields 297 crops of
  undefined memory;
- the ``max_detections`` cap and ``UNKNOWN_LABEL``, both entirely unexercised.

It sits under ``tests/topology/`` because the module does: ``track`` consumes decoded
detections, so the decode moved into the pure layer the elements live in. The last class here
is the other half of that move -- ``pipeline/graph/detections.py`` is now a re-export, and a
copy instead of a re-export would break every ``isinstance`` that crosses the boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.topology.elements.detections import (
    UNKNOWN_LABEL,
    DecodeParams,
    Detections,
    decode_detections,
    parse_classes,
    per_row,
)

LABELS = {0: "person", 8: "ship"}


def row(x1: float, y1: float, x2: float, y2: float, score: float, cls: int) -> list[float]:
    """One detector row in the layout the decoder documents."""
    return [x1, y1, x2, y2, score, float(cls)]


def mixed() -> Detections:
    """Four rows, ship/person/ship/person — a selection no slice expresses."""
    return Detections(
        boxes=np.array(
            [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]],
            dtype=np.float32,
        ),
        scores=np.full(4, 0.9, dtype=np.float32),
        class_ids=np.array([8, 0, 8, 0], dtype=np.int32),
        labels=("ship", "person", "ship", "person"),
    )


def params(**overrides) -> DecodeParams:
    kwargs = {"class_labels": LABELS, "score_threshold": 0.25, "max_detections": 100}
    kwargs.update(overrides)
    return DecodeParams(**kwargs)


class TestTheLetterboxIsUndone:
    """The forward transform is ``resized = source * scale + pad``, so the inverse is
    ``source = (resized - pad) / scale``. Applying them the other way round is the mistake."""

    def test_a_box_round_trips_through_a_known_letterbox(self) -> None:
        scale, pad_x, pad_y = 0.5, 20.0, 60.0
        source = (100.0, 200.0, 300.0, 400.0)
        letterboxed = [
            value * scale + (pad_x if index % 2 == 0 else pad_y)
            for index, value in enumerate(source)
        ]

        result = decode_detections(
            np.array([row(*letterboxed, 0.9, 8)], dtype=np.float32),
            params=params(),
            scale=scale,
            pad=(pad_x, pad_y),
        )

        assert result.boxes[0] == pytest.approx(source, abs=1e-4)

    def test_the_pad_is_subtracted_before_the_scale_is_divided_out(self) -> None:
        """Swapping the order gives ``x/scale - pad`` instead of ``(x - pad)/scale``. At
        scale 0.5 and pad 20 those differ by 20 pixels, which is a whole small object."""
        result = decode_detections(
            np.array([row(120.0, 120.0, 220.0, 220.0, 0.9, 8)], dtype=np.float32),
            params=params(),
            scale=0.5,
            pad=(20.0, 20.0),
        )

        assert result.boxes[0][0] == pytest.approx(200.0), "= (120 - 20) / 0.5"
        assert result.boxes[0][0] != pytest.approx(220.0), "= 120 / 0.5 - 20, the swap"

    def test_no_letterbox_leaves_the_box_alone(self) -> None:
        result = decode_detections(
            np.array([row(10.0, 20.0, 30.0, 40.0, 0.9, 8)], dtype=np.float32),
            params=params(),
        )

        assert result.boxes[0] == pytest.approx((10.0, 20.0, 30.0, 40.0))

    def test_boxes_are_clipped_to_the_frame(self) -> None:
        """A letterboxed coordinate can land outside the source frame, and a negative box
        crops to an empty array downstream."""
        result = decode_detections(
            np.array([row(-50.0, -50.0, 9999.0, 9999.0, 0.9, 8)], dtype=np.float32),
            params=params(),
            frame_hw=(480, 640),
        )

        assert result.boxes[0] == pytest.approx((0.0, 0.0, 640.0, 480.0))


class TestCountTruncatesBeforeAnythingElse:
    """A padded output's trailing rows are undefined, not zero."""

    def test_only_the_reported_rows_are_read(self) -> None:
        rows = np.array(
            [row(0, 0, 10, 10, 0.9, 8), row(0, 0, 10, 10, 0.8, 0)]
            + [row(1e9, 1e9, 2e9, 2e9, 0.99, 3)] * 8,
            dtype=np.float32,
        )

        result = decode_detections(rows, params=params(), count=2)

        assert len(result) == 2
        assert result.labels == ("ship", "person"), "the undefined rows never appeared"

    def test_a_count_larger_than_the_array_is_harmless(self) -> None:
        rows = np.array([row(0, 0, 10, 10, 0.9, 8)], dtype=np.float32)
        assert len(decode_detections(rows, params=params(), count=300)) == 1

    def test_a_count_of_zero_yields_nothing(self) -> None:
        rows = np.array([row(0, 0, 10, 10, 0.9, 8)], dtype=np.float32)
        assert len(decode_detections(rows, params=params(), count=0)) == 0

    def test_no_count_reads_every_row(self) -> None:
        rows = np.array([row(0, 0, 10, 10, 0.9, 8)] * 3, dtype=np.float32)
        assert len(decode_detections(rows, params=params())) == 3


class TestThresholdAndCap:
    def test_rows_below_the_threshold_are_discarded_and_counted(self) -> None:
        rows = np.array(
            [row(0, 0, 10, 10, 0.9, 8), row(0, 0, 10, 10, 0.1, 8)], dtype=np.float32
        )

        result = decode_detections(rows, params=params(score_threshold=0.25))

        assert len(result) == 1
        assert result.discarded == 1, "the row below threshold, counted rather than dropped"

    def test_the_cap_keeps_the_strongest(self) -> None:
        """Never exercised before. Keeping an arbitrary three of ten is a different system
        from keeping the best three."""
        rows = np.array(
            [row(0, 0, 10, 10, 0.30 + index / 100, 8) for index in range(10)],
            dtype=np.float32,
        )

        result = decode_detections(rows, params=params(max_detections=3))

        assert len(result) == 3
        assert sorted(result.scores.tolist(), reverse=True) == pytest.approx(
            [0.39, 0.38, 0.37], abs=1e-6
        )

    def test_results_come_back_in_descending_score_order(self) -> None:
        rows = np.array(
            [row(0, 0, 10, 10, 0.5, 8), row(0, 0, 10, 10, 0.9, 0)], dtype=np.float32
        )

        result = decode_detections(rows, params=params())

        assert result.scores.tolist() == pytest.approx([0.9, 0.5])
        assert result.labels == ("person", "ship")


class TestLabels:
    def test_a_known_class_gets_its_label(self) -> None:
        rows = np.array([row(0, 0, 10, 10, 0.9, 8)], dtype=np.float32)
        assert decode_detections(rows, params=params()).labels == ("ship",)

    def test_an_unmapped_class_is_visible_rather_than_dropped(self) -> None:
        """A detector emitting class 4 against a two-class table is a configuration
        mismatch, and it has to show up in the event instead of vanishing."""
        rows = np.array([row(0, 0, 10, 10, 0.9, 4)], dtype=np.float32)

        result = decode_detections(rows, params=params())

        assert result.labels == (UNKNOWN_LABEL,)
        assert result.class_ids.tolist() == [4], "the raw id survives for diagnosis"


class TestTheShapeContract:
    def test_a_wrong_row_width_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="x1,y1,x2,y2,score,class"):
            decode_detections(np.zeros((3, 5), dtype=np.float32), params=params())

    def test_an_empty_output_is_empty_detections(self) -> None:
        assert len(decode_detections(np.zeros((0, 6), dtype=np.float32), params=params())) == 0


class TestTheLabelIsTheIdReadThroughTheTable:
    """A row's label is its id looked up in the chain file's own table, and nothing else.

    A caller that carries a label beside an id — a chain step reading ``when: class == ship``,
    a test staging detections for a tracker — has two things that can disagree, and only one
    of them the decode ever produced. So the invariant is pinned on the table itself: whatever
    the deployment declared, decoding an id yields the label declared *for that id*, and an id
    nobody declared yields :data:`UNKNOWN_LABEL` rather than borrowing a real class's number.
    """

    def test_every_declared_id_decodes_to_its_own_label(self) -> None:
        assert DecodeParams().class_labels == LABELS, "the fixture is the shipped table"

        for class_id, label in LABELS.items():
            rows = np.array([row(0, 0, 10, 10, 0.9, class_id)], dtype=np.float32)

            assert decode_detections(rows, params=params()).labels == (label,)

    def test_an_id_the_table_has_no_entry_for_is_not_given_a_real_class_name(self) -> None:
        """``0`` is a real class, so an unknown id must not be resolved to one."""
        absent = max(LABELS) + 1

        rows = np.array([row(0, 0, 10, 10, 0.9, absent)], dtype=np.float32)
        decoded = decode_detections(rows, params=params())

        assert decoded.labels == (UNKNOWN_LABEL,)
        assert decoded.labels[0] not in LABELS.values()


class TestTheOldImportPathStillResolves:
    """The shim ``pipeline/graph/detections.py`` re-exports; it does not redefine.

    The counting-simulation graph builds a ``Detections`` and hands it to a stage, and phase
    C's chain will hand one the other way. Two classes with identical fields would satisfy no
    ``isinstance`` across that boundary and no ``is`` comparison in a test, and the symptom
    would be a tracker that silently sees no detections rather than an import error.
    """

    def test_it_is_the_same_class_object_not_a_copy(self) -> None:
        from shipinfer.pipeline.graph import detections as old_home

        assert old_home.Detections is Detections
        assert old_home.decode_detections is decode_detections
        assert old_home.UNKNOWN_LABEL == UNKNOWN_LABEL


# -- row selection, shared by every element that has a `classes:` --------------------------


class TestParseClasses:
    """One parser, because ``track`` and every crop element read the same key.

    They had a copy each, differing only in the kind word inside the message. Two copies of a
    row-selection rule is two places for "an absent key is not an empty list" to drift, and the
    drift has no symptom at run time: the element covers every row instead of none, or none
    instead of every.
    """

    def test_an_absent_key_selects_everything(self) -> None:
        assert parse_classes(None, "track element 'tracker'") is None

    def test_an_empty_list_is_not_an_absent_key(self) -> None:
        """``()`` means "select nothing" and ``None`` means "select everything". Conflating
        them is how a typo doubles a GPU bill in silence."""
        assert parse_classes([], "track element 'tracker'") == ()

    def test_labels_come_back_as_a_tuple_of_strings(self) -> None:
        assert parse_classes(["ship", "person"], "x") == ("ship", "person")

    def test_a_yaml_number_is_still_a_label(self) -> None:
        """``classes: [8]`` is a chain file saying a label that happens to look like an id;
        it is compared against ``Detections.labels``, which are strings."""
        assert parse_classes([8], "x") == ("8",)

    def test_a_bare_string_is_refused_rather_than_iterated(self) -> None:
        """``classes: ship`` would otherwise select the rows labelled ``s``, ``h``, ``i``,
        ``p`` — that is, none of them, and without a word said."""
        with pytest.raises(ConfigurationError, match="must be a list of detection labels"):
            parse_classes("ship", "embed element 'embed_ship'")

    def test_the_refusal_names_the_element_that_carries_the_key(self) -> None:
        with pytest.raises(ConfigurationError, match="embed element 'embed_ship'"):
            parse_classes(7, "embed element 'embed_ship'")


class TestSelectingRowsByClass:
    def test_several_classes_at_once_in_score_order(self) -> None:
        assert mixed().indices_of_any(("ship", "person")) == (0, 1, 2, 3)

    def test_one_class_is_the_non_contiguous_subset(self) -> None:
        """The ship rows are 0 and 2, which no slice expresses — the reason a selection by
        label and a selection by position cannot be allowed to agree by accident."""
        assert mixed().indices_of_any(("ship",)) == (0, 2)

    def test_it_agrees_with_the_single_class_form(self) -> None:
        assert mixed().indices_of_any(("person",)) == mixed().indices_of("person")

    def test_a_label_no_row_carries_selects_nothing(self) -> None:
        assert mixed().indices_of_any(("dinghy",)) == ()

    def test_the_match_is_case_sensitive(self) -> None:
        """Stated as a test rather than as a comment: a chain file writing ``classes: [Ship]``
        against a detector emitting ``ship`` embeds nothing, permanently and silently, and the
        only place that rule is decided is here."""
        assert mixed().indices_of_any(("Ship",)) == ()

    def test_selecting_nothing_selects_nothing(self) -> None:
        assert mixed().indices_of_any(()) == ()


class TestGatheringTheBoxesOfASelection:
    """What a crop element hands a kernel: one contiguous ``(K, 4)`` array, in row order."""

    def test_a_subset_is_the_rows_in_the_order_they_were_asked_for(self) -> None:
        detections = mixed()

        gathered = detections.boxes_at((0, 2))

        assert np.array_equal(gathered, detections.boxes[[0, 2]])
        assert gathered.flags["C_CONTIGUOUS"]

    def test_every_row_is_a_flat_copy_never_the_live_array(self) -> None:
        """The whole-frame path answers with a copy: the caller hands it to an arbitrary
        ``ImageOps`` (a fused kernel takes a device pointer) and may hold it past the frame,
        so mutating the answer must never write through to the detections."""
        detections = mixed()

        answered = detections.boxes_at(range(len(detections)))

        assert answered is not detections.boxes
        assert np.array_equal(answered, detections.boxes)
        assert answered.flags["C_CONTIGUOUS"]
        answered[0, 0] = -1.0
        assert detections.boxes[0, 0] != -1.0

    def test_a_full_length_reordering_is_permuted_not_handed_through(self) -> None:
        """The pass-through is a value test, not a length test.

        ``(3, 2, 1, 0)`` is as long as the frame and is not the frame's order, so a fast path that
        counted rows would answer it with the boxes unpermuted — every crop cut from the wrong
        detection, at full row count and full contiguity, with nothing to see until a tracker
        starts swapping identities. No caller asks for this order today; the point is that the
        method cannot be made to lie by one that does.
        """
        detections = mixed()

        gathered = detections.boxes_at((3, 2, 1, 0))

        assert np.array_equal(gathered, detections.boxes[[3, 2, 1, 0]])
        assert not np.array_equal(gathered, detections.boxes), "the fixture is not symmetric"

    def test_an_index_past_the_end_is_a_typed_refusal(self) -> None:
        """A selection built against a different frame, refused in this module's vocabulary.

        ``range(1, 5)`` is as long as a four-row frame and is not a selection *of* it. The
        value-not-length pass-through already declines to answer it with the boxes unpermuted;
        what is left is numpy's bare ``IndexError`` out of a public method on a value type,
        which CONVENTIONS Part 1 does not allow and which tells an operator nothing about
        which frame or which row.
        """
        detections = mixed()

        with pytest.raises(ValidationError) as raised:
            detections.boxes_at(range(1, 5))

        assert "4" in str(raised.value), "the frame's row count"
        assert "[4]" in str(raised.value), "and the index it has no row for"

    def test_a_selection_numpy_refused_for_another_reason_names_the_whole_selection(
        self,
    ) -> None:
        """The recovery scan looks for out-of-range integers, and that is not the only way a
        gather fails. ``(True, False)`` is read by numpy as a boolean *mask*, and a mask of the
        wrong length raises — while the scan finds nothing out of range, because ``bool`` is a
        subclass of ``int`` and both values index a four-row frame perfectly well. The message
        would then be "detection indices [] name no row of a 4-row frame", which names no index
        at all and is the one diagnostic an operator cannot act on.

        Unreachable from ``indices_of`` / ``indices_of_any`` / ``range``, so this is about the
        message and not about the failure; a public method on a value type does not get to
        report a diagnostic it can see is empty.
        """
        detections = mixed()

        with pytest.raises(ValidationError) as raised:
            detections.boxes_at((True, False))

        message = str(raised.value)
        assert "indices []" not in message, "an empty index list names nothing"
        assert "True" in message, "the whole selection, since no single index is at fault"

    def test_a_valid_selection_is_never_charged_for_the_guard(self) -> None:
        """The bounds are read off the failure, not checked before the gather, so the ordinary
        frame does one pass and not two. Pinned as a property rather than a timing: every
        in-range selection still answers, including the negative index numpy accepts."""
        detections = mixed()

        assert np.array_equal(detections.boxes_at((-1,)), detections.boxes[[3]])

    def test_selecting_nothing_is_an_empty_box_array_not_an_error(self) -> None:
        assert mixed().boxes_at(()).shape == (0, 4)

    def test_the_single_class_form_gathers_the_same_rows(self) -> None:
        detections = mixed()

        boxes, indices = detections.boxes_of("ship")

        assert indices == (0, 2)
        assert np.array_equal(boxes, detections.boxes_at(indices))

    def test_a_class_the_frame_has_none_of_is_an_empty_pair(self) -> None:
        boxes, indices = mixed().boxes_of("dinghy")

        assert indices == ()
        assert boxes.shape == (0, 4)


class TestPerRow:
    """The one rule for reading a per-object metadata key, in both legal shapes.

    Exercised through both readers elsewhere — ``track`` for ``meta["vectors"]`` and
    ``output`` for those plus ``meta["identities"]`` — and pinned here on its own because it
    is the rule, not either element's use of it. Two copies of this would be two places for
    "a mapping that covers no row is an off-by-N" to drift, and the drift has no symptom:
    both readers fail by quietly attaching nothing.
    """

    def test_an_absent_key_is_none_not_an_empty_row_set(self) -> None:
        """ "Nothing was filed" and "nothing was found" are different, so they answer differently."""
        assert per_row(None, 3, what="e", key="vectors") is None

    def test_a_mapping_becomes_one_entry_per_row_with_gaps(self) -> None:
        rows = per_row({1: "b"}, 3, what="e", key="vectors")

        assert rows == [None, "b", None]

    def test_a_sequence_of_the_right_length_is_handed_through_uncopied(self) -> None:
        """A numpy array stays the same array: this runs per frame, per key."""
        array = np.zeros((3, 2), dtype=np.float32)

        assert per_row(array, 3, what="e", key="vectors") is array

    def test_a_sequence_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="2 rows for 3 detections"):
            per_row([1, 2], 3, what="e", key="vectors")

    def test_a_mapping_covering_no_row_is_an_off_by_n(self) -> None:
        with pytest.raises(ValidationError, match="names row 7 and the frame has 3"):
            per_row({7: "x"}, 3, what="e", key="vectors")

    def test_a_mapping_covering_some_rows_stays_legal(self) -> None:
        """Only the person rows are embedded when only a person re-ID model ran."""
        assert per_row({0: "a"}, 2, what="e", key="vectors") == ["a", None]

    def test_an_empty_mapping_is_a_stage_with_nothing_to_say(self) -> None:
        """Zero keys index nothing because there was nothing to index — not the same thing."""
        assert per_row({}, 2, what="e", key="vectors") == [None, None]

    def test_one_stray_key_refuses_the_frame_even_beside_valid_ones(self) -> None:
        """The reviewer's scenario: {0: a, 5: b} on a 3-row frame. The old rule tolerated
        it when ANY key was in range, silently dropping row 5's value — half-attributed
        metadata reported as a healthy chain."""
        with pytest.raises(ValidationError, match="names row 5 and the frame has 3"):
            per_row({0: "a", 5: "b"}, 3, what="e", key="vectors")

    def test_coercible_keys_are_still_not_indices(self) -> None:
        """`"3"`, `3.0` and `True` all survive int() — a coerced key is a guessed
        attribution, so each is refused by type, not converted."""
        for bad in ("3", 3.0, True):
            with pytest.raises(ValidationError, match="not detection row indices"):
                per_row({bad: "v"}, 4, what="e", key="vectors")

    def test_a_negative_key_is_refused_not_wrapped(self) -> None:
        with pytest.raises(ValidationError, match="never negative"):
            per_row({-1: "v"}, 4, what="e", key="vectors")

    def test_keys_that_are_not_indices_are_refused(self) -> None:
        """A model's raw ``{output_name: Tensor}`` is not an attribution."""
        with pytest.raises(ValidationError, match="not detection row indices"):
            per_row({"embedding": [1.0]}, 2, what="e", key="vectors")

    def test_the_refusal_names_the_key_that_carries_it(self) -> None:
        """The message has to point at the wiring, not at the rule."""
        with pytest.raises(ValidationError, match="identities"):
            per_row(object(), 2, what="output element 'output'", key="identities")


class TestAStringIsNotOneEntryPerRow:
    """``len("abc") == 3`` makes a string pass the length check for a three-row frame.

    It then lands one *character* per detection, which is a published field that is wrong
    rather than absent — the failure this whole rule exists to make impossible.
    """

    def test_a_string_is_refused_rather_than_split_into_characters(self) -> None:
        with pytest.raises(ValidationError, match="str"):
            per_row("abc", 3, what="output element 'output'", key="identities")

    def test_bytes_too(self) -> None:
        with pytest.raises(ValidationError, match="bytes"):
            per_row(b"abc", 3, what="output element 'output'", key="identities")
