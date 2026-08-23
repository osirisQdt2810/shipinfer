"""The one place process environment variables are read.

Scattered ``os.getenv`` calls are how a deployment ends up running the wrong code path in
silence: a typo'd variable name falls back to the default, everything starts, and the
operator's evidence that hardware decode is on is a shell history entry rather than the
program's behaviour. Declaring each variable once — name, type, default, meaning — buys
three things:

* a **typed** parse, so ``SHIPINFER_INGEST_BACKEND=gstremaer`` fails loudly at start-up
  with the valid options listed, instead of silently selecting the default;
* :func:`describe`, so ``shipinfer doctor`` can print what the process actually resolved;
* one grep-able inventory of the process's environment contract.

**Deliberately not here.** Two families of variable are read elsewhere, and both are
one-way dependencies that this module must not invert:

* ``SHIPINFER_*`` settings-tree overrides (``SHIPINFER_SCHEDULER__PLACEMENT_POLICY``, ...)
  are pydantic-settings' job — see :mod:`shipinfer.core.settings`. Duplicating them here
  would create two owners of the same key.
* ``SHIPINFER_LOG_LEVEL`` / ``SHIPINFER_LOG_SINK`` are read in
  :mod:`shipinfer.core.logging`, because ``core`` sits below this module in the layering
  and may not import upward (ADR-001).

This module imports nothing but the standard library and
:mod:`shipinfer.core.errors`: no torch, no cv2, and no side effects at import time.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "ALL",
    "GST_APPSINK_MAX_BUFFERS",
    "GST_DECODER_OVERRIDE",
    "INGEST_BACKEND",
    "INGEST_BACKENDS",
    "INGEST_HWACCEL",
    "INGEST_LATENCY_MS",
    "INGEST_OPEN_TIMEOUT_S",
    "INGEST_READ_TIMEOUT_S",
    "INGEST_RTSP_TRANSPORT",
    "EnvVar",
    "describe",
]

T = TypeVar("T")

_TRUE = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE = frozenset({"0", "false", "no", "off", "n", "f"})


@dataclass(frozen=True, slots=True)
class EnvVar(Generic[T]):
    """One environment variable: its name, its parser, its default and why it exists.

    Read with :meth:`get`. A malformed value raises
    :class:`~shipinfer.core.errors.ConfigurationError` rather than falling back to the
    default: a server that ignores an operator's explicit instruction and starts anyway is
    worse than one that refuses to start.
    """

    name: str
    default: T
    parse: Callable[[str], T]
    doc: str
    #: Populated for closed vocabularies, so an error message can list the alternatives.
    choices: tuple[str, ...] = ()

    def get(self) -> T:
        """The parsed value, or the default when the variable is unset or empty.

        Raises:
            ConfigurationError: if the variable is set to something unparseable. The
                message names the variable and, for a closed vocabulary, every valid
                option — the two things a `KeyError` deep in a factory never says.
        """
        raw = os.environ.get(self.name)
        if raw is None or not raw.strip():
            return self.default
        try:
            return self.parse(raw.strip())
        except (ConfigurationError, TypeError, ValueError) as exc:
            # Re-wrapped rather than re-raised so the message always names the variable: a
            # parser knows what it expected but not which key carried the wrong value.
            raise ConfigurationError(f"{self.name}={raw.strip()!r} is invalid: {exc}") from exc

    def is_set(self) -> bool:
        """Whether the operator said anything, as opposed to taking the default."""
        raw = os.environ.get(self.name)
        return raw is not None and bool(raw.strip())

    def __str__(self) -> str:
        return f"{self.name}={self.get()!r}{'' if self.is_set() else ' (default)'}"


def _bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigurationError(
        f"expected a boolean, got {value!r}; use one of {sorted(_TRUE | _FALSE)}"
    )


def _choice(options: Sequence[str]) -> Callable[[str], str]:
    def parse(value: str) -> str:
        if value not in options:
            raise ConfigurationError(f"not a valid choice; valid options: {sorted(options)}")
        return value

    return parse


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ConfigurationError(f"expected a positive number, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ConfigurationError(f"expected a positive integer, got {value!r}")
    return parsed


#: The video-source implementations selectable from the environment. Kept as a literal
#: rather than read from :data:`shipinfer.ingest.SOURCES`, because this module sits below
#: ``ingest`` and importing upward would be a cycle. ``tests/ingest/test_envs.py`` asserts
#: the two agree, so the duplication cannot drift.
INGEST_BACKENDS: tuple[str, ...] = ("gstreamer", "pyav", "replay")

INGEST_BACKEND: EnvVar[str] = EnvVar(
    name="SHIPINFER_INGEST_BACKEND",
    default="gstreamer",
    parse=_choice(INGEST_BACKENDS),
    doc=(
        "Which video source implementation to use when a camera does not name one. "
        "`gstreamer` is the production path (NVDEC via nvv4l2decoder); `pyav` is the "
        "portable fallback; `replay` reads a local file or frame directory and is what "
        "makes the 50-camera stress test runnable with no camera."
    ),
    choices=INGEST_BACKENDS,
)

INGEST_HWACCEL: EnvVar[bool] = EnvVar(
    name="SHIPINFER_INGEST_HWACCEL",
    default=True,
    parse=_bool,
    doc=(
        "Prefer hardware decode (NVDEC) where the backend supports it. Turn it off to "
        "isolate a decoder bug or to run on a host whose driver has no video engine; the "
        "source then falls back to software decode and says so in the log."
    ),
)

INGEST_RTSP_TRANSPORT: EnvVar[str] = EnvVar(
    name="SHIPINFER_INGEST_RTSP_TRANSPORT",
    default="tcp",
    parse=_choice(("tcp", "udp", "auto")),
    doc=(
        "RTSP lower transport. TCP by default: UDP loses packets under load and the "
        "resulting decode artefacts look exactly like a model regression."
    ),
    choices=("tcp", "udp", "auto"),
)

INGEST_LATENCY_MS: EnvVar[int] = EnvVar(
    name="SHIPINFER_INGEST_LATENCY_MS",
    default=200,
    parse=_positive_int,
    doc=(
        "Jitter buffer the source keeps, in milliseconds. It is a direct latency cost, so "
        "the 1000 ms of the previous generation is not a default here; raise it only if a "
        "camera's network genuinely needs it."
    ),
)

INGEST_OPEN_TIMEOUT_S: EnvVar[float] = EnvVar(
    name="SHIPINFER_INGEST_OPEN_TIMEOUT_S",
    default=10.0,
    parse=_positive_float,
    doc=(
        "How long a connection attempt may take before it counts as a failure and the "
        "reconnect backoff advances. Without it an unreachable camera blocks its own "
        "actor thread forever and reports as healthy."
    ),
)

INGEST_READ_TIMEOUT_S: EnvVar[float] = EnvVar(
    name="SHIPINFER_INGEST_READ_TIMEOUT_S",
    default=5.0,
    parse=_positive_float,
    doc=(
        "How long a single frame read may block. This is also what lets a camera actor "
        "notice a stop request: the loop checks the stop event once per read."
    ),
)

PROFILE_DIR: EnvVar[str] = EnvVar(
    name="SHIPINFER_PROFILE_DIR",
    default="",
    parse=str,
    doc=(
        "Directory for torch-profiler Chrome traces. Empty disables profiling entirely. "
        "Heavy: never set this in production. Bounded by SHIPINFER_PROFILE_STEPS, because "
        "an unbounded profiler at 1000 frames a second fills a disk in minutes and produces "
        "a trace too large to open — which is a failed measurement, not a thorough one."
    ),
)

PROFILE_STEPS: EnvVar[int] = EnvVar(
    name="SHIPINFER_PROFILE_STEPS",
    default=8,
    parse=int,
    doc=(
        "How many batches the torch profiler captures once SHIPINFER_PROFILE_DIR is set. "
        "Eight is enough to see a steady-state pattern and small enough to open."
    ),
)

PROFILE_PHASES: EnvVar[bool] = EnvVar(
    name="SHIPINFER_PROFILE_PHASES",
    default=False,
    parse=_bool,
    doc=(
        "Split compute_us into h2d_us / execute_us / d2h_us with timed CUDA events, and "
        "report the device-idle fraction. Answers 'is the GPU idle while we copy?' as a "
        "Prometheus histogram rather than a trace someone has to open. Opt-in because "
        "Event(enable_timing=True) plus the synchronise needed to read it serialises the very "
        "overlap the numbers are meant to inform: measuring this changes it."
    ),
)

GST_DECODER_OVERRIDE: EnvVar[str] = EnvVar(
    name="SHIPINFER_GST_DECODER",
    default="",
    parse=str,
    doc=(
        "Force a specific GStreamer decoder element (e.g. `nvh264dec`) instead of probing "
        "the plugin registry. An escape hatch for a box where the probe picks a decoder "
        "that is installed but broken."
    ),
)

GST_APPSINK_MAX_BUFFERS: EnvVar[int] = EnvVar(
    name="SHIPINFER_GST_APPSINK_MAX_BUFFERS",
    default=2,
    parse=_positive_int,
    doc=(
        "Frames the appsink may hold before it drops the oldest. Deliberately tiny: a deep "
        "decoder queue converts a throughput problem into a latency problem and hides it."
    ),
)

#: Every variable this module owns, for :func:`describe` and for the drift test.
ALL: tuple[EnvVar[object], ...] = (
    INGEST_BACKEND,
    INGEST_HWACCEL,
    INGEST_RTSP_TRANSPORT,
    INGEST_LATENCY_MS,
    INGEST_OPEN_TIMEOUT_S,
    INGEST_READ_TIMEOUT_S,
    GST_DECODER_OVERRIDE,
    GST_APPSINK_MAX_BUFFERS,
)


def describe() -> list[tuple[str, str, bool, str]]:
    """``(name, resolved value, was it set, doc)`` for every variable declared here.

    What ``shipinfer doctor`` prints. Showing *resolved* values rather than raw strings is
    the point: it is the only way an operator can tell a working override from a typo.
    """
    return [(var.name, str(var.get()), var.is_set(), var.doc) for var in ALL]
