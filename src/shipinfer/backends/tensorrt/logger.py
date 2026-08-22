"""Bridge TensorRT's ``ILogger`` into this project's logging."""

from __future__ import annotations

from typing import Any

from shipinfer.core.logging import get_logger

__all__ = ["build_trt_logger"]

_LOG = get_logger("backends.tensorrt")


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

    return _Logger()
