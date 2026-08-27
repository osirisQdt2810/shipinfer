"""``shipinfer backends|policies|queues|runners`` — what is pluggable, and what is plugged in."""

from __future__ import annotations

from typing import Any

from shipinfer.cli.common import print_table
from shipinfer.core.registry import Registry

__all__ = ["list_backends", "list_policies", "list_queues", "list_runners"]


def _render(title: str, registry: Registry[Any]) -> int:
    rows = [[name, description or "-"] for name, description in registry.describe()]
    print_table(title, ["name", "description"], rows)
    return 0


def list_backends() -> int:
    """Every registered execution runtime, whether or not it is installed."""
    from shipinfer.backends import BACKENDS

    return _render("Backends", BACKENDS)


def list_policies() -> int:
    """Every registered placement policy."""
    from shipinfer.scheduling.policies import POLICIES

    return _render("Placement policies", POLICIES)


def list_queues() -> int:
    """Every registered request queue."""
    from shipinfer.scheduling.queues import QUEUES

    return _render("Request queues", QUEUES)


def list_runners() -> int:
    """Every registered runner — the names `shipinfer run --runner` accepts.

    "Topologies" used to be listed here and meant *placement*. A topology is the chain now
    (arch.md section 1) and it is a file, not a registry entry; what an operator picks by name
    is the runner.
    """
    from shipinfer.runners import RUNNERS

    return _render("Runners", RUNNERS)
