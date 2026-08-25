"""Which batch sizes are worth capturing a graph for — and where that answer came from.

A captured graph records addresses *and* extents, so one graph serves exactly one batch
size. That makes the capture set a scheduling question rather than a runtime one: the sizes
worth capturing are the sizes the batcher will actually emit, and nothing else.

The harm is the *missing* size, not the extra one. ``GraphCache.should_capture`` is a lazy
membership filter — a graph is captured the first time a batch of that size arrives — so an
entry of 16 on a model whose ceiling is 8 is inert: never captured, because a batch of 16
never comes. But a model with ``max_batch_size: 12`` and ``preferred_batch_sizes: [3, 6]``
emits batches of 3 and 6, neither of which is in the default ``[1, 2, 4, 8, 16, 32]``; CUDA
graphs are enabled, every capture is declined, and ``shipinfer_cuda_graph_replays_total``
sits at zero with no log line saying why, because declining is a correct outcome by design.
An earlier version of this paragraph justified the module with the oversized case, which
cannot happen here, and omitted the missing one, which can.

Triton derives the same set from the same place (``tensorrt_backend/src/instance_state.cc``
3683-3700), and the rule it uses is reproduced here verbatim in behaviour:

    Graphs are most likely to help for small batch sizes so by default build for batch
    sizes 1, 2, 3, 4, 6, 8, 12, 16, max_batch_size. If preferred batch size is specified,
    then the batch sizes will be 1, preferred batch sizes, max_batch_size.

This module holds no accelerator code and imports nothing from the project on purpose: the
derivation is arithmetic over two integers a config file already states, so it is testable
on a machine with no driver — which is where the config mistake it prevents is made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shipinfer.core.errors import ConfigurationError

__all__ = ["GraphSpec", "derive_graph_batch_sizes", "resolve_graph_spec"]

#: Triton's default ladder, before it is clamped to the model's ``max_batch_size``. Small
#: sizes dominate because that is where launch overhead is a large share of wall time; a
#: batch of 64 spends long enough in the kernels that replaying the launch saves little.
_LADDER: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """A capture set together with the reason it looks like that.

    The ``source``/``reason`` pair is not decoration. Graph capture fails quietly by design
    — a model that cannot be captured falls back to the ordinary launch path and stays
    correct — so the only way an operator learns that the captured sizes describe a batcher
    that no longer exists is if the server says which sizes it chose and why.
    """

    batch_sizes: tuple[int, ...]
    #: ``derived`` | ``model`` (the model's ``parameters.graph_spec``) | ``settings``
    #: (``execution.cuda_graph_batch_sizes``, set explicitly by the operator).
    #: Where the set came from, **as prose for the log line** — a config path, a settings key,
    #: "derived: no size in [...] fits", "(filtered to this model's window)". Deliberately a
    #: free-form string and not an enum: the caller composes it from what actually happened,
    #: and an enum would force those four-and-counting situations into a word each and lose
    #: the part an operator reads. An earlier draft annotated this as a ``Literal`` of three
    #: values that a comment listed; the code emitted a fourth, and the server emits none of
    #: them. If a machine-readable kind is ever needed, add a second field rather than
    #: narrowing this one.
    source: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_sizes": list(self.batch_sizes),
            "source": self.source,
            "reason": self.reason,
        }


def derive_graph_batch_sizes(
    max_batch_size: int, preferred: Sequence[int] = ()
) -> tuple[int, ...]:
    """The batch sizes this model's batching window can actually produce.

    Args:
        max_batch_size: the window's ceiling — ``effective_max_batch_size``, so 1 when
            server-side batching is off and there is exactly one size to capture.
        preferred: the batcher's ``preferred_batch_sizes``. Non-empty means the batcher
            stops early at those sizes, so they are the ones a request will arrive with.

    Returns:
        A sorted, deduplicated tuple, every entry within ``[1, max_batch_size]``.

    Raises:
        ConfigurationError: if ``max_batch_size`` is below 1. There is no such batcher, and
            silently treating it as 1 would hide a config that means something else.
    """
    if max_batch_size < 1:
        raise ConfigurationError(
            f"max_batch_size must be >= 1 to derive a capture set, got {max_batch_size}"
        )
    # 1 joins any preferred list: the batcher emits a batch of one whenever a request
    # arrives into an empty queue, which at low frame rates is *every* request.
    sizes = {1, max_batch_size, *preferred} if preferred else {max_batch_size, *_LADDER}
    return tuple(sorted(size for size in sizes if 1 <= size <= max_batch_size))


def resolve_graph_spec(
    *,
    max_batch_size: int,
    preferred: Sequence[int] = (),
    override: Sequence[int] | None = None,
    override_source: str = "",
) -> GraphSpec:
    """Pick the capture set, preferring an explicit one and explaining either way.

    Args:
        max_batch_size: as :func:`derive_graph_batch_sizes`.
        preferred: as :func:`derive_graph_batch_sizes`.
        override: an explicit list, or ``None`` to derive. An override still has to fit
            inside the window: a captured graph for a size the batcher cannot emit is dead
            weight, and stating one is a mistake worth failing on rather than clamping,
            because clamping would leave the config saying something the server ignores.
        override_source: what to report as :attr:`GraphSpec.source` when ``override`` is
            used — the name of the key the operator actually wrote.

    Raises:
        ConfigurationError: for an override that is empty, non-integral, or outside the window.
    """
    if override is None:
        derived = derive_graph_batch_sizes(max_batch_size, preferred)
        reason = (
            f"derived from the batching window (max_batch_size={max_batch_size}, "
            f"preferred={list(preferred)})"
            if preferred
            else f"derived from the batching window (max_batch_size={max_batch_size}, "
            f"no preferred sizes, so Triton's ladder clamped to it)"
        )
        return GraphSpec(batch_sizes=derived, source="derived", reason=reason)

    sizes = list(override)
    if not sizes:
        raise ConfigurationError(
            "an explicit graph spec must name at least one batch size; remove the key to "
            "derive the capture set from the batching window instead"
        )
    if any(not isinstance(size, int) or isinstance(size, bool) for size in sizes):
        raise ConfigurationError(
            f"an explicit graph spec must be a list of integers, got {sizes!r}"
        )
    outside = sorted({size for size in sizes if size < 1 or size > max_batch_size})
    if outside:
        raise ConfigurationError(
            f"explicit graph spec names batch size(s) {outside} outside the batching "
            f"window [1, {max_batch_size}]; those graphs could never be replayed"
        )
    return GraphSpec(
        batch_sizes=tuple(sorted(set(sizes))),
        source=override_source or "explicit",
        reason=f"set explicitly by {override_source or 'the operator'}",
    )
