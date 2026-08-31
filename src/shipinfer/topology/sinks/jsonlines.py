"""One JSON object per line, to a file or to stdout.

The sink that makes the whole pipeline testable end to end with no broker: a test runs the
replay source through a chain into this, then reads the file back and asserts that N frames
in produced N events out with every tag accounted for. It is also the right sink for
an offline evaluation run, where the consumer is a notebook rather than a service.

The buffering is the only interesting decision. ``flush()`` per line would be a write
syscall — and on many filesystems a metadata update — a thousand times a second, which is
enough to make the *sink* the pipeline's bottleneck. So writes go through Python's buffer and
are flushed every ``flush_every`` events, on :meth:`flush`, and on close. The cost is that a
process killed with ``SIGKILL`` loses up to that many lines, which is the correct trade for a
diagnostic stream and the wrong one for a ledger — and this is a diagnostic stream.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, ClassVar, TextIO

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.events.schema import PerceptionEvent
from shipinfer.topology.sinks.base import ResultSink
from shipinfer.topology.sinks.registry import RESULT_SINKS

__all__ = ["JsonLinesResultSink"]


@RESULT_SINKS.register("jsonlines", "jsonl", "file")
class JsonLinesResultSink(ResultSink):
    """Appends ``event.to_json()`` plus a newline, once per frame.

    Args:
        path: where to write. ``"-"`` means stdout, which is what a container without a
            volume wants. A parent directory is created if it does not exist, because
            failing to start over a missing directory is a worse deploy experience than
            creating it.
        flush_every: events between flushes. 0 flushes every event, which is what a test
            asserting on file content while the pipeline is still running needs.
        append: keep an existing file's contents. False truncates, which is what a repeated
            local run wants.

    Thread safety is a lock around the write. Several workers and the sweeper thread emit
    concurrently, and interleaved partial lines would corrupt every record in the file
    rather than one — ``write`` on a buffered text stream is not atomic with respect to a
    line.
    """

    name: ClassVar[str] = "jsonlines"

    def __init__(
        self, path: str | Path = "-", *, flush_every: int = 64, append: bool = True
    ) -> None:
        super().__init__()
        if flush_every < 0:
            raise ConfigurationError("flush_every must be >= 0")
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

    def _do_emit(self, event: PerceptionEvent) -> None:
        line = event.to_json()
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
