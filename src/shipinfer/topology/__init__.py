"""The chain: elements, caps, and the YAML loader that refuses a broken one.

The first two of arch.md's three concepts (§1) live here — **element** and **topology** —
and the third, the runner, deliberately does not: how a chain executes (in one process, as
a fleet of shards, or compiled into a GStreamer graph) is a separate package, so that one
chain definition serves all three.

This package imports **only** ``shipinfer.core``. Not ``scheduling``, not ``runtime``, not
the engine. Everything an element needs from the outside arrives through the
:class:`~shipinfer.topology.base.ElementContext` handed to ``open()``, which is what makes a
chain loadable and testable on a machine with no driver — the property the whole offline
tier rests on.

Typical use::

    from shipinfer.topology import load_topology

    chain = load_topology("topology/ship_person.yaml")
    print(chain.describe())          # the resolved wiring, with negotiated caps

``shipinfer.topology.elements`` is imported here for its registration side effect: the
registries must be populated before a chain file naming ``impl: mock`` can be loaded. That
import is safe to make unconditionally, because every element module loads its runtime
inside ``_do_open`` — see that package's docstring for the rule.
"""

from __future__ import annotations

from shipinfer.topology.base import (
    MODEL_KINDS,
    ChainItem,
    Element,
    ElementContext,
    ElementKind,
    ModelResolver,
)
from shipinfer.topology.caps import ANY, LOCATIONS, Caps, negotiate, parse_caps
from shipinfer.topology.chain import (
    ChainSpec,
    Condition,
    Edge,
    ElementNode,
    ElementSpec,
    Topology,
    load_topology,
)

# Imported for the side effect: this is what puts `mock` in the registries.
from shipinfer.topology.elements import mock as mock
from shipinfer.topology.registry import (
    ELEMENTS,
    ElementRegistry,
    create_element,
    describe_elements,
    registry_for,
)

__all__ = [
    "ANY",
    "ELEMENTS",
    "LOCATIONS",
    "MODEL_KINDS",
    "Caps",
    "ChainItem",
    "ChainSpec",
    "Condition",
    "Edge",
    "Element",
    "ElementContext",
    "ElementKind",
    "ElementNode",
    "ElementRegistry",
    "ElementSpec",
    "ModelResolver",
    "Topology",
    "create_element",
    "describe_elements",
    "load_topology",
    "mock",
    "negotiate",
    "parse_caps",
    "registry_for",
]
