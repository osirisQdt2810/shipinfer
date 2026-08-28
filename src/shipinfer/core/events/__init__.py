"""The perception event this system publishes, and the conversion its fields need.

One value type, kept in ``core`` for the reason every other value type is: the two layers
that build a :class:`~shipinfer.core.events.schema.PerceptionEvent` sit on opposite sides of
the accelerator seam. ``pipeline/`` — the previous generation's DAG — builds one per frame,
and so does the ``output`` element under ``topology/``, which may import ``core`` and
nothing else (``scripts/hooks/check_layers.py``). A shared value that lived in either of
them would make the other import a layer it must not.

It is pure by construction: stdlib, dataclasses and numpy, no transport and nothing that
touches a device. Where an event *goes* is a sink's business (``topology/sinks/``), and how
it is assembled is the business of whatever holds the frame.
"""

from __future__ import annotations

from shipinfer.core.events.convert import as_embedding
from shipinfer.core.events.schema import (
    MESSAGE_TYPE,
    SCHEMA_VERSION,
    ObjectRecord,
    PerceptionEvent,
)

__all__ = [
    "MESSAGE_TYPE",
    "SCHEMA_VERSION",
    "ObjectRecord",
    "PerceptionEvent",
    "as_embedding",
]
