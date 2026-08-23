"""Choosing the batch size to actually execute."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["choose_batch_size"]


def choose_batch_size(size: int, preferred: Sequence[int], maximum: int) -> int:
    """Largest preferred size that fits in ``size``, else ``size``, capped at ``maximum``.

    A TensorRT engine built with optimisation profiles for {8, 16, 32} handles a batch of
    31 by falling back to a slower path or padding internally. Asking for the profiled 16
    is usually better than asking for an unprofiled 31 — but only when the remainder gets
    picked up by the next batch, which the queue guarantees because it never blocks on a
    partial drain.
    """
    capped = min(size, maximum)
    fits = [p for p in preferred if p <= capped]
    return max(fits) if fits else capped
