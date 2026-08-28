"""Environment knobs, read at access time.

``envs.SHIPINFER_INGEST_BACKEND`` *is* the parsed value — str, int, float or bool — so no
call site casts. Add a knob by adding one entry to :data:`environment_variables`.

Unset or blank takes the default. Set but malformed raises
:class:`~shipinfer.core.errors.ConfigurationError` naming the variable and, for a closed
vocabulary, the options: a server that ignores an operator's explicit instruction and starts
anyway is worse than one that refuses to start.

Not here, deliberately: the ``SHIPINFER_<SECTION>__<FIELD>`` settings tree, owned by
pydantic-settings (:mod:`shipinfer.core.settings`), and ``SHIPINFER_LOG_LEVEL`` /
``SHIPINFER_LOG_SINK``, read in :mod:`shipinfer.core.logging`, which sits below this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from shipinfer.core.errors import ConfigurationError

__all__ = ["INGEST_BACKENDS", "describe", "environment_variables", "is_set"]

_TRUE = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE = frozenset({"0", "false", "no", "off", "n", "f"})

#: Video sources selectable from the environment. A literal, not a read of
#: ``ingest.SOURCES``: that package sits above this one. ``tests/ingest/test_envs.py``
#: asserts the two agree, so the duplication cannot drift.
INGEST_BACKENDS: tuple[str, ...] = ("gstreamer", "pyav", "replay")


def _raw(name: str) -> str | None:
    """The variable's value, or ``None`` when unset or blank."""
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def is_set(name: str) -> bool:
    """Whether the operator said anything, as opposed to taking the default."""
    return _raw(name) is not None


def _refuse(name: str, raw: str, expected: str) -> ConfigurationError:
    return ConfigurationError(f"{name}={raw!r} is invalid: expected {expected}")


def _str(name: str, default: str = "") -> str:
    return _raw(name) or default


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise _refuse(name, raw, f"a boolean, one of {sorted(_TRUE | _FALSE)}")


def _int(name: str, default: int, *, positive: bool = True) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise _refuse(name, raw, "an integer") from exc
    if positive and parsed <= 0:
        raise _refuse(name, raw, "a positive integer")
    return parsed


def _float(name: str, default: float, *, positive: bool = True) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise _refuse(name, raw, "a number") from exc
    if positive and parsed <= 0.0:
        raise _refuse(name, raw, "a positive number")
    return parsed


def _choice(name: str, default: str, options: Sequence[str]) -> str:
    raw = _raw(name)
    if raw is None:
        return default
    if raw not in options:
        raise _refuse(name, raw, f"one of {sorted(options)}")
    return raw


environment_variables: dict[str, Callable[[], Any]] = {
    # ── ingest ──
    # Which source implementation a camera that names none gets: `gstreamer` is the
    # production path (NVDEC), `pyav` the portable fallback, `replay` reads a file or frame
    # directory and is what makes the 50-camera stress test runnable with no camera.
    "SHIPINFER_INGEST_BACKEND": lambda: _choice(
        "SHIPINFER_INGEST_BACKEND", "gstreamer", INGEST_BACKENDS
    ),
    # Off isolates a decoder bug, or a driver with no video engine; the source falls back to
    # software decode and says so.
    "SHIPINFER_INGEST_HWACCEL": lambda: _bool("SHIPINFER_INGEST_HWACCEL", True),
    # TCP by default: UDP loses packets under load, and the decode artefacts that follow
    # look exactly like a model regression.
    "SHIPINFER_INGEST_RTSP_TRANSPORT": lambda: _choice(
        "SHIPINFER_INGEST_RTSP_TRANSPORT", "tcp", ("tcp", "udp", "auto")
    ),
    # Jitter buffer, milliseconds. A direct latency cost, so not the 1000 ms of the previous
    # generation; raise it only for a camera whose network needs it.
    "SHIPINFER_INGEST_LATENCY_MS": lambda: _int("SHIPINFER_INGEST_LATENCY_MS", 200),
    # Without a connect deadline an unreachable camera blocks its own actor thread forever
    # and reports healthy.
    "SHIPINFER_INGEST_OPEN_TIMEOUT_S": lambda: _float("SHIPINFER_INGEST_OPEN_TIMEOUT_S", 10.0),
    # Also what lets a camera actor notice a stop: the loop checks the stop event once per
    # read.
    "SHIPINFER_INGEST_READ_TIMEOUT_S": lambda: _float("SHIPINFER_INGEST_READ_TIMEOUT_S", 5.0),
    # Force a GStreamer decoder element instead of probing the registry — an escape hatch
    # for a box whose probe picks a decoder that is installed but broken.
    "SHIPINFER_GST_DECODER": lambda: _str("SHIPINFER_GST_DECODER"),
    # Frames the appsink may hold before dropping the oldest. Deliberately tiny: a deep
    # decoder queue turns a throughput problem into a latency problem and hides it.
    "SHIPINFER_GST_APPSINK_MAX_BUFFERS": lambda: _int("SHIPINFER_GST_APPSINK_MAX_BUFFERS", 2),
    # ── engine ──
    # Override CUDA graph capture for every model; empty leaves each model's
    # `execution.cuda_graphs` alone, which is the normal case. The blunt instrument answers
    # "is the graph path what is hurting?" in one restart. Triton decides per model too.
    "SHIPINFER_CUDA_GRAPHS": lambda: _choice("SHIPINFER_CUDA_GRAPHS", "", ("on", "off")),
    # ── profiling (heavy; never in production) ──
    # Directory for torch-profiler traces; empty disables profiling. Bounded by
    # SHIPINFER_PROFILE_STEPS: an unbounded profiler at 1000 fps fills a disk in minutes and
    # writes a trace too large to open, which is a failed measurement.
    "SHIPINFER_PROFILE_DIR": lambda: _str("SHIPINFER_PROFILE_DIR"),
    # Batches captured once the directory is set. Eight shows a steady-state pattern and
    # still opens.
    "SHIPINFER_PROFILE_STEPS": lambda: _int("SHIPINFER_PROFILE_STEPS", 8),
    # Split compute_us into h2d/execute/d2h with timed CUDA events. Opt-in because the
    # synchronise needed to read them serialises the overlap the numbers describe.
    "SHIPINFER_PROFILE_PHASES": lambda: _bool("SHIPINFER_PROFILE_PHASES", False),
}


def __getattr__(name: str) -> Any:
    # PEP 562: resolved at access, not at import, so a test's monkeypatch.setenv is seen.
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*__all__, *environment_variables]


def describe() -> list[tuple[str, str, bool]]:
    """``(name, resolved value, was it set)`` for every knob — what ``doctor`` prints.

    Resolved values rather than raw strings: it is the only way an operator can tell a
    working override from a typo.
    """
    return [(name, str(read()), is_set(name)) for name, read in environment_variables.items()]
