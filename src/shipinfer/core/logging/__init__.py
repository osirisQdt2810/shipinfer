"""Logging: a logger factory, structured context fields, and pluggable sinks.

Two rules hold everywhere in this codebase:

1. **Library code configures nothing at import time.** Only :func:`configure` touches
   handlers, and only an entry point calls it. Anything else steals control from whoever
   embeds this package.
2. **Every logger is named** ``shipinfer.<area>``, so an operator can silence one
   subsystem without an edit — ``logging.getLogger("shipinfer.scheduling").setLevel(...)``.

Sinks are a registry (:data:`SINKS`) rather than an if/elif, so
``configure(sink="async")`` and ``configure(sink=AsyncSink(RotatingFileSink(path)))`` are
both first-class, and a deployment that needs syslog adds a file instead of a branch.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from shipinfer.core.logging.base import LogSink
from shipinfer.core.logging.context import CONTEXT_FIELDS, log_context
from shipinfer.core.logging.formatters import ContextFormatter, JsonFormatter
from shipinfer.core.logging.registry import SINKS
from shipinfer.core.logging.sinks import AsyncSink, NullSink, RotatingFileSink, StreamSink

__all__ = [
    "CONTEXT_FIELDS",
    "LOG_LEVEL_ENV",
    "LOG_SINK_ENV",
    "SINKS",
    "AsyncSink",
    "ContextFormatter",
    "JsonFormatter",
    "LogSink",
    "NullSink",
    "RotatingFileSink",
    "StreamSink",
    "configure",
    "get_logger",
    "log_context",
    "shutdown",
]

LOG_LEVEL_ENV = "SHIPINFER_LOG_LEVEL"
LOG_SINK_ENV = "SHIPINFER_LOG_SINK"

_ROOT = "shipinfer"
_active_sink: LogSink | None = None


def get_logger(area: str) -> logging.Logger:
    """Return the logger for ``area`` (e.g. ``"scheduling.dispatcher"``)."""
    return logging.getLogger(area if area.startswith(_ROOT) else f"{_ROOT}.{area}")


def configure(
    level: str | int | None = None,
    *,
    sink: LogSink | str | None = None,
    force: bool = False,
    **sink_kwargs: Any,
) -> LogSink:
    """Attach one sink to the ``shipinfer`` logger. Idempotent unless ``force``.

    Args:
        level: level name or number; defaults to ``$SHIPINFER_LOG_LEVEL`` then ``INFO``.
        sink: a :class:`LogSink`, or a registered name (``"stream"``, ``"async"``,
            ``"file"``, ``"null"``); defaults to ``$SHIPINFER_LOG_SINK`` then ``"stream"``.
        force: reconfigure even if this has already run (tests, ``--log-level``).
        **sink_kwargs: forwarded to the sink's constructor when ``sink`` is a name.

    Returns:
        The active sink, so a caller can :meth:`LogSink.close` it on shutdown.
    """
    global _active_sink
    if _active_sink is not None and not force:
        return _active_sink

    shutdown()

    resolved_level = level if level is not None else os.environ.get(LOG_LEVEL_ENV, "INFO")
    if isinstance(sink, LogSink):
        active = sink
    else:
        name = sink or os.environ.get(LOG_SINK_ENV) or "stream"
        active = SINKS.create(name, **sink_kwargs)

    logger = logging.getLogger(_ROOT)
    logger.addHandler(active.build())
    logger.setLevel(resolved_level)
    # This package is a library first: never let records escape to the root logger and get
    # printed a second time by whatever embeds us.
    logger.propagate = False

    _active_sink = active
    return active


def shutdown() -> None:
    """Detach and close the active sink. Safe to call when nothing is configured."""
    global _active_sink
    logger = logging.getLogger(_ROOT)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if _active_sink is not None:
        _active_sink.close()
        _active_sink = None
