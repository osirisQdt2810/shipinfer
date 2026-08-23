"""The structured fields every log line in this system may carry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["CONTEXT_FIELDS", "log_context"]

#: Fields worth having on a line in a distributed perception system. Appended only when
#: present, so an ordinary line stays short.
CONTEXT_FIELDS: tuple[str, ...] = (
    "model",
    "version",
    "device",
    "instance",
    "camera_id",
    "frame_id",
    "request_id",
    "batch_size",
    "queue_depth",
)


def log_context(**fields: Any) -> Mapping[str, Any]:
    """Build the ``extra=`` mapping for a structured log call.

    Exists so call sites read ``logger.info("...", extra=log_context(model=m, device=d))``
    and a typo in a field name raises here instead of being silently dropped by
    :mod:`logging` — which is what makes structured fields trustworthy enough to alert on.
    """
    unknown = set(fields) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"unknown log context field(s): {sorted(unknown)}")
    return fields
