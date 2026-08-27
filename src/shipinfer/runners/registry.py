"""The runner registry, and the one function that builds a runner from a name.

Three runners execute the same chain (arch.md §1) and an operator picks one by name, so this
is a registry for the same reason the queues, the policies and the element kinds are: adding
``fleet`` must be a new file and a decorator, never an ``if kind == "fleet"`` in a launcher
that already knows about two others.

Registration is **eager**. A runner module imports ``core``, ``topology`` and ``scheduling``
— all pure — and reaches the engine only through the
:class:`~shipinfer.topology.base.ModelResolver` it is handed, so importing this package
costs nothing on a host with no accelerator. :meth:`~shipinfer.core.registry.Registry.
register_lazy` is there for the day a runner cannot honour that (the ``deepstream``
compiler, whose import may need ``pyds``).
"""

from __future__ import annotations

from typing import Any

from shipinfer.core.registry import Registry
from shipinfer.core.settings import ServerSettings
from shipinfer.runners.base import Runner
from shipinfer.topology import Topology

__all__ = ["RUNNERS", "build_runner"]

#: Every runner implementation, by the name a settings tree or a CLI flag uses.
RUNNERS: Registry[Runner] = Registry("runner", Runner)


def build_runner(
    name: str,
    topology: Topology,
    settings: ServerSettings | None = None,
    **options: Any,
) -> Runner:
    """Build the named runner over ``topology``.

    The single door, so that the name in the settings tree is resolved in exactly one place
    and every caller gets the same error text for a typo.

    Args:
        name: a registered runner name or alias (``inprocess``, ``single``, ...).
        topology: the validated chain to run.
        settings: the deployment settings; the runner's default applies when omitted.
        options: implementation-specific keywords — ``shard_id``, ``device``, ``models``, and
            whatever the implementation adds (``workers``, ``queue``).

    Raises:
        ConfigurationError: no runner is registered under that name; the message lists the
            ones that are.
    """
    return RUNNERS.create(name, topology, settings, **options)
