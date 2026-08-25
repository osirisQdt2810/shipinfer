"""Small closed vocabularies shared by several settings sections."""

from __future__ import annotations

import enum

__all__ = ["ExecutionProvider", "ModelControlMode", "OverflowPolicy"]


class OverflowPolicy(str, enum.Enum):
    """What a full queue does.

    ``REJECT`` is the default: dropping a frame at the edge, loudly and countably, beats
    half-processing it and evicting somebody else's work three stages later. That silent
    eviction is the exact bug this system was rebuilt to remove.
    """

    REJECT = "reject"  # raise QueueFullError back to the caller
    BLOCK = "block"  # block the producer until space frees, bounded by timeout
    DROP_OLDEST = "drop_oldest"  # sacrifice the stalest request of the greediest camera


class ModelControlMode(str, enum.Enum):
    """Who decides which models are loaded, and when — Triton's ``--model-control-mode``.

    ``NONE`` is the default and matches what this server has always done: the set of
    models is fixed at start-up and the control endpoints refuse. That is the right
    default for a fixed six-model pipeline, because a control plane that can unload the
    detector is a control plane that can take the deployment down by accident.

    ``EXPLICIT`` is for a repository that grows: start-up loads only ``startup_models``
    (possibly none) and everything else arrives through ``/v2/repository/models/*/load``.
    Fifty cameras over six models fits in memory; a hundred models does not, and loading
    all of them to serve three is how a box runs out of VRAM at start-up.

    Triton's third mode, ``POLL``, is deliberately absent: re-scanning the repository on a
    timer means a half-written config file can be loaded, and the failure it produces
    surfaces minutes later with nothing pointing at the edit that caused it.
    """

    NONE = "none"
    EXPLICIT = "explicit"


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
