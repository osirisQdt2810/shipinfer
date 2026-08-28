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

from shipinfer.core.errors import ValidationError
from shipinfer.topology.elements.detections import (
    UNKNOWN_LABEL,
    DecodeParams,
    Detections,
    decode_detections,
)

LABELS = {0: "person", 8: "ship"}


def row(x1: float, y1: float, x2: float, y2: float, score: float, cls: int) -> list[float]:
    """One detector row in the layout the decoder documents."""
    return [x1, y1, x2, y2, score, float(cls)]


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


class TestTheMockDetectorAgreesWithTheDecode:
    """``invented_detections`` files a class id its own label maps to.

    The mocks exist so the elements behind a detector can be tested offline against the type a
    real one produces, and an id that contradicts its label is a type no decode could ever
    emit: it filed ``class_ids=[0]`` under ``"ship"`` while the shipped table maps 0 to
    ``person`` and 8 to ``ship``. The first thing a tracker test does is copy it, and a
    class-conditional chain step (``when: class == ship``) reads the id.
    """

    def test_the_ship_mock_files_the_ship_id(self) -> None:
        from shipinfer.topology.elements.mock import invented_detections

        invented = invented_detections("ship")

        assert invented.labels == ("ship",)
        assert invented.class_ids.tolist() == [8]

    def test_the_id_it_files_decodes_back_to_the_label_it_claims(self) -> None:
        """The property, rather than the number: run the id through the real decode."""
        from shipinfer.topology.elements.mock import invented_detections

        for label in ("ship", "person"):
            invented = invented_detections(label)
            rows = np.array(
                [row(0, 0, 10, 10, 0.9, int(invented.class_ids[0]))], dtype=np.float32
            )

            assert decode_detections(rows, params=params()).labels == (label,)

    def test_a_label_the_table_has_no_id_for_is_not_filed_as_person(self) -> None:
        """``0`` is a real class, so an invented one must not borrow its number."""
        from shipinfer.topology.elements.mock import invented_detections

        invented = invented_detections("dinghy")

        assert invented.class_ids.tolist() == [-1]
        assert invented.labels == ("dinghy",)


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
