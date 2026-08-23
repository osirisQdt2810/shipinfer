"""Write to a stream (stderr by default)."""

from __future__ import annotations

import logging
import sys
from typing import IO, Any

from shipinfer.core.logging.base import LogSink
from shipinfer.core.logging.formatters import (
    DEFAULT_DATEFMT,
    DEFAULT_FORMAT,
    ContextFormatter,
    JsonFormatter,
)
from shipinfer.core.logging.registry import SINKS

__all__ = ["StreamSink"]


@SINKS.register("stream", "stderr", "console")
class StreamSink(LogSink):
    """Plain synchronous stream handler.

    Defaults to **stderr** so stdout stays clean for piped output — a CLI that prints a
    JSON result to stdout must not have log lines interleaved into it.
    """

    name = "stream"

    def __init__(self, stream: IO[Any] | None = None, *, json: bool = False) -> None:
        self._stream = stream or sys.stderr
        self._json = json

    def build(self) -> logging.Handler:
        handler = logging.StreamHandler(self._stream)
        handler.setFormatter(
            JsonFormatter() if self._json else ContextFormatter(DEFAULT_FORMAT, DEFAULT_DATEFMT)
        )
        return handler

    def describe(self) -> str:
        return f"stream({'json' if self._json else 'text'})"
