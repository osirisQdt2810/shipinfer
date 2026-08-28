"""The one reader of ``meta["vectors"]``: every shape it takes and every one it refuses.

:func:`~shipinfer.topology.elements._vectors.rows_by_index` exists because the convention had
two owners that disagreed at the edges — ``recognize`` refused ``{"3": v}`` and ``track``
coerced it, ``recognize`` refused a mapping when *any* key was out of range and ``track`` only
when *no* key was in range — and C8's scatter-back was about to be the third. A rule with one
home needs one test file, and this is it: the element tests above check what an element *does*
with the rows, this checks what a row *is*.

No element, no gallery, no ``shipvision`` — the reader is numpy and
:mod:`shipinfer.core.errors`, so every test here runs on any host, with or without the
submodule. That is not an accident of this file; it is the layering rule the module was
extracted to keep.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ValidationError
from shipinfer.topology.elements._vectors import rows_by_index

#: The prefix a caller passes so a refusal names the element and the frame. Every message
#: this module raises has to start with it, which the last test in this file asserts over
#: the whole refusal table rather than one message at a time.
WHO = "recognize element 'ship_id' on ('cam-A', 184102)"


class Sized:
    """A ``detections`` stand-in that is only ``len()``, which is the whole dependency.

    One method, and that is the evidence: the reader asks the frame's detections exactly one
    question, so an element may hand it C3's real ``Detections`` or a decoder's own value and
    the reader neither knows nor cares.
    """

    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count


class Unsized:
    """A ``detections`` value with no ``__len__`` — treated as absent, not refused.

    A chain may legitimately embed with no decoding detector in front of it, and a value the
    reader cannot size is not its key to validate. What this must **not** do is turn the
    negative-key rule off with it.
    """


def vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


A = vec(1.0, 0.0)
B = vec(0.0, 1.0)


class TestTheTwoShapesItAccepts:
    """A per-row array and an index → vector mapping, and nothing else."""

    def test_a_per_row_array_is_row_i_for_detection_i(self) -> None:
        """The shape an embedder that embedded the whole frame produces."""
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        rows = list(rows_by_index(vectors, Sized(2), who=WHO))

        assert [index for index, _ in rows] == [0, 1]
        assert np.array_equal(rows[0][1], A)
        assert np.array_equal(rows[1][1], B)

    def test_a_sequence_of_vectors_is_the_same_shape_spelled_in_python(self) -> None:
        """A list of lists is what a pure-Python embedder or a test hands over."""
        rows = list(rows_by_index([[1.0, 0.0], [0.0, 1.0]], Sized(2), who=WHO))

        assert [index for index, _ in rows] == [0, 1]
        assert rows[0][1].dtype == np.float32, "coerced once, here, not by each caller"

    def test_a_mapping_keeps_the_row_a_branch_embedder_embedded(self) -> None:
        """The shape ``embed_ship`` produces: rows 0 and 2 of a four-row frame.

        The pairs come back in ascending row order and carry the *detector's* indices, which
        is the alignment that keeps a vector attached to the object it came from. A reader
        that renumbered them would put row 2's embedding on row 1.
        """
        rows = list(rows_by_index({2: B, 0: A}, Sized(4), who=WHO))

        assert [index for index, _ in rows] == [0, 2]
        assert np.array_equal(rows[0][1], A)
        assert np.array_equal(rows[1][1], B)

    def test_an_empty_mapping_is_a_frame_with_nothing_to_embed(self) -> None:
        """Not a refusal: a ship embedder meeting a frame of people selects no rows.

        The *missing* key is the wiring failure, and it belongs to the caller — only the
        element knows whether ``vectors`` may be absent where it stands.
        """
        assert list(rows_by_index({}, Sized(3), who=WHO)) == []

    def test_numpy_integer_keys_come_back_as_plain_ints(self) -> None:
        """So two branches' mappings are keyed in one key space, not two.

        An ``np.int64(3)`` from one branch and an ``int`` 3 from another have to name the
        same row once the runner's fan-in unions them. They hash the same today, but the
        pairs this reader yields are the ones a caller files, so the coercion happens here.
        """
        rows = list(rows_by_index({np.int64(1): B, np.int32(0): A}, Sized(2), who=WHO))

        assert [type(index) for index, _ in rows] == [int, int]
        assert [index for index, _ in rows] == [0, 1]

    def test_a_float64_row_is_narrowed_and_a_float32_row_is_not_copied(self) -> None:
        """One coercion, and it costs nothing on the shape production actually files.

        ``np.asarray`` returns a *view* for a float32 row, so the per-row cost on the hot
        path is a type check; a float64 row is narrowed once here rather than by whichever
        library the caller hands it to.
        """
        wide = np.array([[1.0, 0.0]], dtype=np.float64)
        narrow = np.array([[1.0, 0.0]], dtype=np.float32)

        _, widened = next(iter(rows_by_index(wide, Sized(1), who=WHO)))
        assert widened.dtype == np.float32
        _, row = next(iter(rows_by_index(narrow, Sized(1), who=WHO)))
        assert np.shares_memory(row, narrow), "no copy of a row that was already float32"


class TestTheKeyRule:
    """Which mapping keys are detection row indices, and which are somebody's mistake."""

    def test_a_raw_model_response_is_refused_with_the_scatter_back_named(self) -> None:
        """``PoolEmbed`` files ``response.outputs`` verbatim: ``{tensor_name: Tensor}``.

        The most important refusal in this module, because guessing which of that tensor's
        rows belongs to which detection is exactly how an embedding lands on the wrong
        object — and the failure has no symptom until a tracker starts swapping identities.
        """
        with pytest.raises(ValidationError) as caught:
            rows_by_index({"embeddings": np.zeros((3, 2), np.float32)}, Sized(3), who=WHO)

        message = str(caught.value)
        assert "embeddings" in message
        assert "pool" in message and "C8" in message

    @pytest.mark.parametrize(
        ("key", "why"),
        [
            ("3", "a row index that went through JSON is still a string"),
            (3.0, "a row index that went through a divide is still a float"),
            (2.5, "and a float that is not integral has no row at all"),
            (True, "`True` is an `int` in Python, and a flag is not a row"),
            (np.bool_(True), "numpy's bool is the same mistake wearing numpy's clothes"),
            (None, "a key nobody meant to write"),
            ((0, 1), "a tuple key is a different indexing scheme entirely"),
        ],
        ids=["str", "float", "fractional", "bool", "np-bool", "none", "tuple"],
    )
    def test_a_key_that_is_not_an_integral_index_is_refused(self, key: Any, why: str) -> None:
        """Coercion is what made the two old readers disagree; there is none here.

        ``track`` used ``int(key)`` and so accepted ``"3"`` and ``3.0`` where ``recognize``
        refused them, which meant one embedder could feed a chain whose ``track`` accepted
        the frame and whose ``recognize`` refused it, over the same metadata key.
        """
        with pytest.raises(ValidationError, match="not detection row"):
            rows_by_index({key: A}, Sized(4), who=WHO)

    def test_one_bad_key_refuses_the_frame_even_beside_good_ones(self) -> None:
        """Every key, not the first one — a scatter-back is wrong as a whole or not at all."""
        with pytest.raises(ValidationError, match="not detection row"):
            rows_by_index({0: A, "1": B}, Sized(4), who=WHO)


class TestTheRangeRule:
    """Which rows a key may name, with and without a frame to check it against."""

    def test_a_negative_key_is_refused_even_with_no_detections_to_check_against(self) -> None:
        """The rule that holds when there is nothing to bound the key with.

        ``detections`` is legitimately absent — a fixed-crop source, a test — and then no
        upper bound is knowable. A *negative* index is not knowable-or-not: there is no
        index space in which ``-1`` names a detection row, and the one thing a consumer is
        likely to do with it is index a list, which silently attaches the vector to the last
        row of the frame.
        """
        with pytest.raises(ValidationError, match="never negative"):
            rows_by_index({-1: A}, None, who=WHO)

    def test_an_unsized_detections_value_does_not_turn_the_negative_rule_off(self) -> None:
        """The same, through the other door: a value the reader cannot size is not a licence."""
        with pytest.raises(ValidationError, match="never negative"):
            rows_by_index({-1: A}, Unsized(), who=WHO)

    def test_a_key_above_the_frame_is_refused_even_when_another_key_is_valid(self) -> None:
        """The rule the two old readers disagreed about, and the reason for this direction.

        ``track`` refused a mapping only when **no** key named a row, so ``{0: v, 99: v}`` on
        a two-row frame was accepted: row 0 attributed, row 99 dropped without a word. A
        scatter-back that is off by all-but-one row then reads exactly like a partial
        embedder, which is a legitimate state. Refusing on *any* out-of-range key separates
        the two, and costs the legitimate case nothing — a partial embedder's keys are all
        real rows.
        """
        with pytest.raises(ValidationError, match="names row 99"):
            rows_by_index({0: A, 99: B}, Sized(2), who=WHO)

    def test_partial_coverage_of_a_real_frame_stays_legal(self) -> None:
        """The case the strict rule must not break: only the ship rows were embedded."""
        rows = list(rows_by_index({0: A, 3: B}, Sized(4), who=WHO))

        assert [index for index, _ in rows] == [0, 3]

    def test_a_mapping_with_no_detections_is_bounded_only_from_below(self) -> None:
        """Unbounded above, on purpose: with no frame to check against there is no bound.

        Stated as a test rather than left implicit, because "``{999999: v}`` is accepted"
        looks like a hole until the alternative is written down — refusing it would mean a
        chain cannot embed without a decoding detector in front of it.
        """
        rows = list(rows_by_index({999999: A}, None, who=WHO))

        assert [index for index, _ in rows] == [999999]

    def test_a_frame_with_no_detections_names_no_rows_at_all(self) -> None:
        """Count zero is a real count, and the message does not offer ``rows 0..-1``."""
        with pytest.raises(ValidationError) as caught:
            rows_by_index({0: A}, Sized(0), who=WHO)

        message = str(caught.value)
        assert "0 detection(s)" in message
        assert "0..-1" not in message


class TestThePerRowShapeRule:
    """What an ``(N, d)`` array has to be, and what happens when N is not the frame's."""

    def test_a_positional_array_that_disagrees_with_the_frame_is_refused(self) -> None:
        """Two vectors for four detections is not "the first two"; it is unaligned."""
        with pytest.raises(ValidationError, match=r"2 vector\(s\) for 4 detection"):
            rows_by_index(np.zeros((2, 2), np.float32), Sized(4), who=WHO)

    def test_a_three_dimensional_array_is_not_one_row_per_detection(self) -> None:
        with pytest.raises(ValidationError, match="3-d array"):
            rows_by_index(np.zeros((2, 2, 2), np.float32), Sized(2), who=WHO)

    @pytest.mark.parametrize(
        "vectors",
        [42, "abc", {1, 2}, None, object()],
        ids=["int", "str", "set", "none", "object"],
    )
    def test_anything_that_is_neither_shape_is_refused_by_type(self, vectors: Any) -> None:
        """Including ``str``, which is a ``Sequence`` and would otherwise be a row of chars."""
        with pytest.raises(ValidationError, match="it needs an"):
            rows_by_index(vectors, Sized(2), who=WHO)

    @pytest.mark.parametrize(
        ("value", "form"),
        [
            (np.zeros((1, 0), np.float32), "array"),
            ([[]], "sequence"),
            ([[[1.0, 0.0]]], "sequence"),
            ({0: np.zeros((0,), np.float32)}, "mapping"),
            ({0: np.zeros((2, 2), np.float32)}, "mapping"),
        ],
        ids=[
            "empty-array-row",
            "empty-seq-row",
            "nested-seq-row",
            "empty-map-row",
            "2d-map-row",
        ],
    )
    def test_a_row_that_is_not_a_non_empty_vector_is_refused_by_row(
        self, value: Any, form: str
    ) -> None:
        """An embedding is a ``(d,)``, in every form — and the refusal names which row.

        Checked in all three forms because the per-row coercion used to happen in two of
        them and not the third, so a zero-width ``(N, 0)`` array reached the gallery and was
        refused there instead, with a message about a query width rather than about a row.
        """
        with pytest.raises(ValidationError, match="row 0"):
            rows_by_index(value, Sized(1), who=WHO)


class TestWhenTheRefusalHappens:
    """Before the first pair, always — the callers act on each pair as it arrives."""

    def test_a_refusal_arrives_from_the_call_and_not_from_the_first_iteration(self) -> None:
        """``GalleryRecognize`` queries a gallery per pair and ``track`` builds a detection.

        A generator that validated lazily would let half a frame's queries be made and then
        raise, and neither caller can file an honest answer for half a frame. Written as a
        test because "returns an iterator" is exactly the signature that usually means the
        opposite.
        """
        with pytest.raises(ValidationError):
            rows_by_index({0: A, 99: B}, Sized(2), who=WHO)  # not consumed, and still raises

    def test_a_refused_mapping_never_reads_an_embedder_s_arrays(self) -> None:
        """Key rule first, values second: a frame that is refused touches no payload."""

        class Explodes(dict):  # type: ignore[type-arg]
            def __getitem__(self, key: Any) -> Any:
                raise AssertionError("a refused mapping must not have its values read")

        with pytest.raises(ValidationError, match="names row 99"):
            rows_by_index(Explodes({0: None, 99: None}), Sized(2), who=WHO)


class TestEveryRefusalNamesTheCaller:
    """``who`` is the whole reason a shared helper can still produce an actionable message."""

    @pytest.mark.parametrize(
        ("vectors", "detections"),
        [
            ({"embeddings": np.zeros((1, 2), np.float32)}, Sized(1)),
            ({"3": A}, Sized(4)),
            ({-1: A}, None),
            ({0: A, 99: B}, Sized(2)),
            (np.zeros((2, 2, 2), np.float32), Sized(2)),
            (np.zeros((2, 2), np.float32), Sized(4)),
            (42, Sized(1)),
            ([[]], Sized(1)),
        ],
        ids=["raw", "str-key", "negative", "out-of-range", "3d", "count", "type", "row"],
    )
    def test_the_message_starts_with_who_asked(self, vectors: Any, detections: Any) -> None:
        """Every one of them, because a helper cannot know which element is standing there.

        Parametrised over the whole refusal table rather than checked once: a rule added
        later without a prefix is a message that names neither the element nor the camera,
        and on a fifty-camera chain that is not actionable.
        """
        with pytest.raises(ValidationError) as caught:
            rows_by_index(vectors, detections, who=WHO)

        assert str(caught.value).startswith(WHO), str(caught.value)
