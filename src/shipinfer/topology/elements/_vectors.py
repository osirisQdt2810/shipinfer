"""The rule for reading ``meta["vectors"]``, written once so it can stop being written twice.

``recognize`` queries a gallery per row and ``track`` attaches an appearance to a track, and
written twice the two disagreed at the edges. ``recognize`` is the first caller and today the
**only** one: ``track._embeddings`` keeps its own laxer copy (``{"3": v}`` coerced, an
out-of-range key tolerated) until TRACK-VECTORS swaps it — stricter owes tests, in C8b's file.

``vectors`` is either a mapping ``{row index: vector}`` — what a *branch* embedder produces,
since only the original index says which row a vector came from — or a per-row ``(N, d)``
array whose row *i* is detection *i*. Keys must be integral (``int``/numpy); ``bool``, floats
and strings are refused, which is also how the raw ``{tensor_name: Tensor}`` a ``pool``
embedder files gets caught. A negative key is refused always. When the count is known every
key must name a row — the looser rule silently drops the out-of-range half of a mis-scattered
frame. An empty mapping is legal; a *missing* key is the caller's. Pure: numpy, ``core.errors``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from shipinfer.core.errors import ValidationError

__all__ = ["rows_by_index"]


def rows_by_index(
    vectors: Any, detections: Any, *, who: str
) -> Iterator[tuple[int, np.ndarray]]:
    """``(detection row index, vector)`` pairs, from either shape an embedder may file.

    Args:
        vectors: ``meta["vectors"]`` verbatim — a mapping of row index to vector, or an
            ``(N, d)`` array or sequence with one row per detection.
        detections: ``meta["detections"]``, or ``None``. Only ``len()`` is read off it, so
            the whole dependency on C3's
            :class:`~shipinfer.topology.elements.detections.Detections` is that it is sized;
            a value that is not sized is treated as absent rather than refused, because it
            is not this reader's key to validate.
        who: who is asking and about what, as a message prefix — for example
            ``"recognize element 'ship_id' on ('cam-A', 184102)"``. Every refusal below
            starts with it, because a chain has several elements and several cameras and a
            refusal that names neither is not actionable.

    Returns:
        An iterator over the pairs, in ascending row order.

        **Every refusal is made before the first pair is yielded.** The callers query a
        gallery or build a tracker input per pair, and a refusal that arrived halfway
        through would leave half a frame's work done and the other half undone — a partial
        frame is the one state neither caller can file an honest answer for.

    Raises:
        ValidationError: any shape or key this reader cannot attribute to a detection row.
            One refusal per rule, each naming the value it saw; the module docstring is the
            rule they enforce.
    """
    count = _row_count(detections)
    if isinstance(vectors, Mapping):
        return iter(_from_mapping(vectors, count, who=who))
    if isinstance(vectors, np.ndarray):
        if vectors.ndim != 2:
            raise ValidationError(
                f"{who}: `vectors` is a {vectors.ndim}-d array and needs (N, d), one row "
                "per detection"
            )
        rows = tuple(
            (index, _as_vector(vectors[index], who, index)) for index in range(vectors.shape[0])
        )
    elif isinstance(vectors, Sequence) and not isinstance(vectors, (str, bytes)):
        rows = tuple(
            (index, _as_vector(value, who, index)) for index, value in enumerate(vectors)
        )
    else:
        raise ValidationError(
            f"{who}: `vectors` is a {type(vectors).__name__}; it needs an (N, d) array, a "
            "sequence of vectors, or a mapping from row index to vector"
        )

    if count is not None and len(rows) != count:
        raise ValidationError(
            f"{who}: {len(rows)} vector(s) for {count} detection(s). A per-row array must "
            "be one row per detection; an embedder that ran on a *subset* of the rows — a "
            "`when:` branch — must file a mapping from row index to vector, or every value "
            "downstream lands on the wrong object"
        )
    return iter(rows)


def _from_mapping(
    vectors: Mapping[Any, Any], count: int | None, *, who: str
) -> tuple[tuple[int, np.ndarray], ...]:
    """The mapping shape, with the key rule applied to every key before any row is read.

    Key type first and the range second, deliberately: a raw ``{tensor_name: Tensor}``
    response fails the type rule, and its message is the one that names the missing
    scatter-back rather than one about a row index nobody wrote.
    """
    keys = list(vectors)
    if not all(_is_row_index(key) for key in keys):
        shown = ", ".join(sorted(repr(key) for key in keys)[:4])
        raise ValidationError(
            f"{who}: `vectors` is a mapping keyed by {shown}, which are not detection row "
            "indices. That is a model response filed verbatim by a `pool` embedder, and "
            "scattering its output rows back to the detections that produced them is the "
            "embed element's job (phase C8). One vector per detection row is what this "
            "needs: an (N, d) array, or a mapping keyed by row index"
        )

    # `(row, original key)` and not `(row, value)`: the value is read only after every key
    # has cleared the rule, so a frame that is refused never touches an embedder's arrays.
    pairs = sorted(((int(key), key) for key in keys), key=lambda pair: pair[0])
    for index, _ in pairs:
        if index < 0:
            raise ValidationError(
                f"{who}: `vectors` names row {index}, and a detection row index is never "
                "negative. A negative key attaches the vector to the last row of the frame "
                "instead of refusing, which is the mis-attribution this reader exists to "
                "make loud"
            )
        if count is not None and index >= count:
            span = f" (rows 0..{count - 1})" if count else ""
            raise ValidationError(
                f"{who}: `vectors` names row {index} and the frame has {count} "
                f"detection(s){span}. Covering *some* rows is fine — only the ship rows "
                "are embedded when only a ship embedder ran — but naming a row that does "
                "not exist is an off-by-N scatter-back"
            )
    return tuple((index, _as_vector(vectors[key], who, index)) for index, key in pairs)


def _is_row_index(key: Any) -> bool:
    """Whether one mapping key is an integral detection row index.

    ``bool`` and ``np.bool_`` are excluded explicitly rather than left to the isinstance
    chain: ``True`` really is an ``int`` in Python, and ``np.bool_`` is not integral in
    numpy 2 but was closer to it in numpy 1 — an exclusion that depends on which numpy is
    installed is not a rule.
    """
    if isinstance(key, (bool, np.bool_)):
        return False
    return isinstance(key, (int, np.integer))


def _row_count(detections: Any) -> int | None:
    """How many rows the frame's detections hold, or ``None`` when the item carries none."""
    if detections is None:
        return None
    try:
        return len(detections)
    except TypeError:
        return None


def _as_vector(value: Any, who: str, index: int) -> np.ndarray:
    """One row as a 1-D float32 array, or a refusal naming the row.

    ``np.asarray`` and not a copy: a float32 row passes through as a view, and the callers
    hand the result to a library that normalises into its own buffer, so a second copy here
    would be per-row work on the hot path for nothing.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValidationError(
            f"{who}: the vector for row {index} has shape {array.shape}, and an embedding "
            "is a non-empty (d,)"
        )
    return array
