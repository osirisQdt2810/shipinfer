"""Write to a size-rotated file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from shipinfer.core.logging.base import LogSink
from shipinfer.core.logging.formatters import (
    DEFAULT_DATEFMT,
    DEFAULT_FORMAT,
    ContextFormatter,
    JsonFormatter,
)
from shipinfer.core.logging.registry import SINKS

__all__ = ["RotatingFileSink"]


@SINKS.register("file", "rotating_file")
class RotatingFileSink(LogSink):
    """Rotating file handler, JSON by default.

    A service that runs for weeks needs bounded log files, and it needs them machine
    readable — so unlike :class:`StreamSink` this one defaults to JSON.
    """

    name = "file"

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 64 * 1024 * 1024,
        backup_count: int = 5,
        json: bool = True,
    ) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._json = json

    def build(self) -> logging.Handler:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            self._path,
            maxBytes=self._max_bytes,
            backupCount=self._backup_count,
            encoding="utf-8",
            delay=True,
        )
        handler.setFormatter(
            JsonFormatter() if self._json else ContextFormatter(DEFAULT_FORMAT, DEFAULT_DATEFMT)
        )
        return handler

    def describe(self) -> str:
        return f"file({self._path})"
