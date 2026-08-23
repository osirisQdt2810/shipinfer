"""Make any sink non-blocking by moving its I/O to a background thread."""

from __future__ import annotations

import logging
import queue
from logging.handlers import QueueHandler, QueueListener

from shipinfer.core.logging.base import LogSink
from shipinfer.core.logging.registry import SINKS

__all__ = ["AsyncSink"]


@SINKS.register("async", "async_stream", "queue")
class AsyncSink(LogSink):
    """Wrap another sink behind a bounded queue and a listener thread.

    This is the sink a production deployment should use, and the reason is measurable: a
    synchronous ``StreamHandler`` performs a blocking ``write()`` **while holding the
    handler lock**, on the thread that called ``logger.info``. On an inference worker that
    thread is the one feeding a GPU, so a slow terminal, a full pipe or a laggy log volume
    becomes GPU idle time.

    Here the calling thread only does ``queue.put_nowait`` — microseconds, no I/O, no
    contention with the writer. The listener thread does the formatting and the write.

    The queue is **bounded** on purpose. An unbounded log queue under a burst is a memory
    leak that ends in the OOM killer; dropping the oldest record and counting the loss is
    the honest failure mode.
    """

    name = "async"

    def __init__(
        self,
        inner: LogSink | None = None,
        *,
        max_queue: int = 8192,
        drop_on_full: bool = True,
    ) -> None:
        # Imported here rather than at module scope: sinks/__init__ imports this module,
        # so a top-level import of a sibling sink would be circular.
        from shipinfer.core.logging.sinks.stream import StreamSink

        self._inner = inner or StreamSink()
        self._queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=max_queue)
        self._drop_on_full = drop_on_full
        self._listener: QueueListener | None = None
        self.dropped = 0

    def build(self) -> logging.Handler:
        target = self._inner.build()
        self._listener = QueueListener(self._queue, target, respect_handler_level=True)
        self._listener.daemon = True
        self._listener.start()
        return _BoundedQueueHandler(self)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._inner.close()

    def describe(self) -> str:
        return f"async({self._inner.describe()}, dropped={self.dropped})"


class _BoundedQueueHandler(QueueHandler):
    """Never blocks the emitting thread; counts what it had to drop."""

    def __init__(self, sink: AsyncSink) -> None:
        super().__init__(sink._queue)
        self._sink = sink

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            if not self._sink._drop_on_full:
                self.queue.put(record)  # explicit opt-in to backpressure
                return
            self._sink.dropped += 1
