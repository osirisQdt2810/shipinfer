"""The log-sink contract.

A sink builds a :class:`logging.Handler`. That indirection buys two things: sinks are
selectable by name from config, and a sink can wrap another one — which is exactly how
:class:`~shipinfer.core.logging.sinks.asynchronous.AsyncSink` turns any sink non-blocking.
"""

from __future__ import annotations

import abc
import logging
from typing import ClassVar

__all__ = ["LogSink"]


class LogSink(abc.ABC):
    """Builds one configured :class:`logging.Handler`.

    Subclasses live one-per-file in :mod:`shipinfer.core.logging.sinks` and register
    themselves with :data:`shipinfer.core.logging.registry.SINKS`.
    """

    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def build(self) -> logging.Handler:
        """Create the handler. Called once, by :func:`shipinfer.core.logging.configure`."""

    def close(self) -> None:
        """Release anything the sink owns beyond the handler (threads, file descriptors).

        The default is a no-op; :class:`AsyncSink` overrides it to stop its listener
        thread, and a server that exits without calling it drops buffered records.
        """

    def describe(self) -> str:
        return type(self).__name__
