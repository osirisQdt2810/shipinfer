"""Bridge TensorRT's ``ILogger`` into this project's logging."""

from __future__ import annotations

import threading
from typing import Any

from shipinfer.core.logging import get_logger

__all__ = ["build_trt_logger"]

_LOG = get_logger("backends.tensorrt")

#: One ILogger for the process. TensorRT keeps a global logger and warns that it is
#: *ignoring* every later one, so building a fresh subclass per backend created N Python
#: objects that TensorRT would never call, and N chances for a callback to cross a thread
#: boundary during a concurrent load. Cached on (verbose,) because that is the only thing
#: that changes behaviour.
_LOGGERS: dict[bool, Any] = {}
_LOGGER_LOCK = threading.Lock()


def build_trt_logger(trt: Any, verbose: bool = False) -> Any:
    """A ``trt.ILogger`` that forwards to ``shipinfer.backends.tensorrt``.

    Constructed lazily against the imported module rather than declared at class scope,
    because subclassing ``trt.ILogger`` requires TensorRT to be importable — and this
    package must be importable without it.

    Severity mapping is deliberate: TensorRT emits a great deal of INFO during engine
    deserialisation (tactic selection, memory pools) that is noise in a server log but
    exactly what you want when an engine will not load. So INFO lands at DEBUG unless
    ``verbose``, while WARNING and above are always surfaced.
    """

    class _Logger(trt.ILogger):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            trt.ILogger.__init__(self)

        def log(self, severity: Any, msg: str) -> None:
            if severity == trt.ILogger.INTERNAL_ERROR:
                _LOG.error("TensorRT internal error: %s", msg)
            elif severity == trt.ILogger.ERROR:
                _LOG.error("TensorRT: %s", msg)
            elif severity == trt.ILogger.WARNING:
                _LOG.warning("TensorRT: %s", msg)
            elif verbose:
                _LOG.info("TensorRT: %s", msg)
            else:
                _LOG.debug("TensorRT: %s", msg)

    with _LOGGER_LOCK:
        existing = _LOGGERS.get(verbose)
        if existing is None:
            existing = _Logger()
            _LOGGERS[verbose] = existing
        return existing
