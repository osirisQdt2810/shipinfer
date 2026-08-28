"""One JSON object per trace, per line — the sink that makes the six stamps readable.

A line is Triton's trace shape: an id, the model, and a ``timestamps`` array of
``{"name": ..., "ns": ...}``. A trace from this server therefore reads in the same tools
and diffs against a Triton one without a translation table.

The buffering decision is the only interesting one, and it is the same trade
``topology.sinks.jsonlines`` makes: flushing per line is a write syscall — on many
filesystems a metadata update too — at whatever rate the sampler lets through, which is
enough to make the *instrument* the bottleneck. So writes go through Python's buffer and are
flushed every ``flush_every`` records, on :meth:`flush`, and on close. A ``SIGKILL`` loses up
to that many lines, which is the right trade for a diagnostic stream and the wrong one for a
ledger. This is a diagnostic stream.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, ClassVar, TextIO

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.tracing.base import RequestTrace, TraceSink
from shipinfer.core.tracing.registry import TRACE_SINKS

__all__ = ["JsonLinesTraceSink"]


@TRACE_SINKS.register(
    "jsonlines", "jsonl", "file", description="One JSON trace per line, to a file or stdout"
)
class JsonLinesTraceSink(TraceSink):
    """Appends ``trace.as_dict()`` plus a newline, once per sampled request.

    Args:
        path: where to write. ``"-"`` means stdout, which is what a container with no volume
            wants. A missing parent directory is created rather than failing the start-up:
            refusing to serve because a trace directory is absent is a worse outage than the
            missing telemetry.
        flush_every: records between flushes. 0 flushes every record, which is what a test
            reading the file while the server still runs needs.
        append: keep an existing file's contents. False truncates, for a repeated local run.
        rate: trace one request in ``rate``. Passed through to :class:`TraceSink`.

    Thread safety is a lock around the write. Every model instance's worker thread records
    concurrently, and ``write`` on a buffered text stream is not atomic with respect to a
    line — interleaving would corrupt every record in the file rather than one.
    """

    name: ClassVar[str] = "jsonlines"

    def __init__(
        self,
        path: str | Path = "-",
        *,
        flush_every: int = 64,
        append: bool = True,
        rate: int = 1,
    ) -> None:
        super().__init__(rate=rate)
        if flush_every < 0:
            raise ConfigurationError(f"flush_every must be >= 0, got {flush_every}")
        self.path = Path(path) if str(path) != "-" else None
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._since_flush = 0
        self._stream: TextIO
        if self.path is None:
            self._stream = sys.stdout
            self._owns_stream = False
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a" if append else "w", encoding="utf-8")
            self._owns_stream = True

    def _do_record(self, trace: RequestTrace) -> None:
        line = json.dumps(trace.as_dict(), separators=(",", ":"))
        with self._lock:
            self._stream.write(line)
            self._stream.write("\n")
            self._since_flush += 1
            if self.flush_every == 0 or self._since_flush >= self.flush_every:
                self._stream.flush()
                self._since_flush = 0

    def flush(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.flush()
                self._since_flush = 0

    def _do_close(self) -> None:
        if self._owns_stream and not self._stream.closed:
            self._stream.close()

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "path": str(self.path) if self.path else "-"}
