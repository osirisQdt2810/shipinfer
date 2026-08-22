"""Discard everything."""

from __future__ import annotations

import logging

from shipinfer.core.logging.base import LogSink
from shipinfer.core.logging.registry import SINKS

__all__ = ["NullSink"]


@SINKS.register("null", "none", "discard")
class NullSink(LogSink):
    """Drop every record.

    Not a curiosity: a benchmark run measures the scheduler, not the terminal, and even a
    formatted-and-discarded record costs microseconds at 15k requests/s.
    """

    name = "null"

    def build(self) -> logging.Handler:
        return logging.NullHandler()
