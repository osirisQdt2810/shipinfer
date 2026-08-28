"""Formatters: human-readable with appended context, and line-delimited JSON."""

from __future__ import annotations

import json
import logging
from typing import Any

from shipinfer.core.logging.context import CONTEXT_FIELDS

__all__ = ["DEFAULT_DATEFMT", "DEFAULT_FORMAT", "ContextFormatter", "JsonFormatter"]

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(module)-22s %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

#: Attributes :class:`logging.LogRecord` always has — anything else on a record came from
#: an ``extra=`` and is therefore payload.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class ContextFormatter(logging.Formatter):
    """Appends structured extras as ``key=value`` instead of interpolating them.

    Keeping them out of the message means lines stay grep-able and a missing field never
    raises mid-log — a formatter that can throw takes the server down with it.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = [
            f"{field}={getattr(record, field)}"
            for field in CONTEXT_FIELDS
            if getattr(record, field, None) is not None
        ]
        return f"{base} [{' '.join(extras)}]" if extras else base


class JsonFormatter(logging.Formatter):
    """One JSON object per line — what a log shipper wants from a 24/7 service."""

    def __init__(self, *, service: str = "shipinfer") -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "service": self._service,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED and not key.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))
