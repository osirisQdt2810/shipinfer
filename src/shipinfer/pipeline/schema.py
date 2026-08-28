"""Re-export of the perception event, which now lives in ``core``.

The type moved to :mod:`shipinfer.core.events.schema` ahead of the ``output`` element that
will build one under ``topology/elements/``: that layer may import ``core`` and nothing else
(``scripts/hooks/check_layers.py``), and both it and this package build the same event. This
module stays because arch.md §9 says ``pipeline/`` remains the working application until the
chain has replaced it, and ~30 modules plus their tests name it here — a shim is cheaper and
more honest than a rename spread across two generations of the same code.

Nothing is redefined below: a second definition would be two contracts with one name, which
is the failure this project has already paid for once elsewhere. New code should import
:mod:`shipinfer.core.events` directly.
"""

from __future__ import annotations

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
]
