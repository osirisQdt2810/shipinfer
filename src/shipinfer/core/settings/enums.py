"""Small closed vocabularies shared by several settings sections."""

from __future__ import annotations

import enum

__all__ = ["ExecutionProvider", "OverflowPolicy"]


class OverflowPolicy(str, enum.Enum):
    """What a full queue does.

    ``REJECT`` is the default: dropping a frame at the edge, loudly and countably, beats
    half-processing it and evicting somebody else's work three stages later. That silent
    eviction is the exact bug this system was rebuilt to remove.
    """

    REJECT = "reject"  # raise QueueFullError back to the caller
    BLOCK = "block"  # block the producer until space frees, bounded by timeout
    DROP_OLDEST = "drop_oldest"  # sacrifice the stalest request of the greediest camera


class ExecutionProvider(str, enum.Enum):
    """Which implementation of the hot path to use.

    ``AUTO`` prefers the compiled ``shipinfer._C`` extension and falls back to Python.
    Pinning to ``PYTHON`` is how a test proves the two implementations agree; pinning to
    ``NATIVE`` is how a deployment refuses to start if the fast path is missing, rather
    than quietly running 5x slower.
    """

    AUTO = "auto"
    NATIVE = "native"
    PYTHON = "python"
