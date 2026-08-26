"""``shipinfer backends|policies|queues|topologies`` — what is pluggable, and what is plugged in."""

from __future__ import annotations

from typing import Any

from shipinfer.cli.common import print_table
from shipinfer.core.registry import Registry

__all__ = ["list_backends", "list_policies", "list_queues", "list_topologies"]


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


def list_topologies() -> int:
    """Every registered process topology — the names `shipinfer fleet --topology` accepts."""
    from shipinfer.server.topology import TOPOLOGIES

    return _render("Topologies", TOPOLOGIES)
