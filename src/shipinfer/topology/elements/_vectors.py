"""One reader for ``meta["vectors"]``, so one convention means one thing.

``meta["vectors"]`` is the key an ``embed`` element files and that every element downstream
of one has to attribute back to detection rows. Two elements read it today —
:class:`~shipinfer.topology.elements.recognize.GalleryRecognize` queries a gallery per row,
:class:`~shipinfer.topology.elements.track.ShipvisionTrack` attaches an appearance to a
track — and C8's scatter-back will be the third. Written twice it was already read twice
differently: one accepted ``{"3": v}`` and ``{3.0: v}`` and the other refused them, one
refused a mapping only when *no* key named a row and the other when *any* key did not. Two
readers of one key that disagree at the edges mean a chain file's ``vectors`` means
something different depending on which element is standing there, and the symptom is a frame
that one element accepts and the next refuses.

So the rule lives here, once, and the callers own only what they do with the rows.
:meth:`~shipinfer.topology.elements.track.ShipvisionTrack._embeddings` is repointed at this
module in the rebase over the C8a slice, which is already rewriting it; until that lands
this module has one caller and the second is named rather than assumed.

**The rule, written down.** ``vectors`` is one of exactly two shapes:

* a **mapping** ``{row index: vector}`` — the shape a *branch* embedder produces, because
  ``embed_ship`` embeds the ship rows of a frame that also holds people and only the
  original index says which row a vector came from;
* a **per-row** ``(N, d)`` array or sequence, whose row *i* is detection *i* — the shape an
  embedder that embedded the whole frame produces.

and the edges resolve like this:

* **Keys are integral row indices**: ``int`` or ``numpy`` integer. ``bool`` and ``np.bool_``
  are refused by name — ``True`` is an ``int`` in Python and a flag is not a row. Floats and
  strings are refused too: ``3.0`` is an index that went through a divide, ``"3"`` is one
  that went through JSON, and the raw ``{tensor_name: Tensor}`` a ``pool`` embedder files
  straight from its response arrives through exactly that door. Coercing them would make
  this reader the place a scatter-back's type error is silently repaired.
* **A negative key is refused always**, including when the detection count is unknown. There
  is no index space in which ``-1`` names a detection row, and the one thing a consumer is
  likely to do with it — index a list — attaches the vector to the *last* row instead.
* **When the count is known, every key must name a row.** Any key outside
  ``range(count)`` refuses the frame. The looser rule (refuse only when *no* key is in
  range) accepts ``{0: v, 99: v}`` on a two-row frame: row 0 is attributed, row 99 is
  dropped without a word, and a scatter-back that is off by all-but-one row reads exactly
  like a partial embedder. The legitimate partial case is untouched, because a partial
  embedder's keys are all real rows.
* **An empty mapping is legal.** It is "this embedder selected no rows in this frame", which
  is what an ordinary frame of people looks like to a ship embedder. A *missing* ``vectors``
  key is a different thing — the element is in the wrong place — and belongs to the caller,
  which is the only one that knows whether it may be absent.
* **The count is a cross-check, not a requirement.** A chain may legitimately embed with no
  decoding detector ahead of it (a fixed-crop source, a test), so ``detections`` may be
  ``None`` and then only the negative-key rule applies. The caller is the one that knows
  whether its own settings turn the detections into a requirement.

Pure by construction: numpy and :mod:`shipinfer.core.errors`, nothing else. It registers
nothing, so it is not imported by the package ``__init__`` — the callers import it directly.
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
