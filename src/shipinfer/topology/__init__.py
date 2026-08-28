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

# Imported so that `import shipinfer.topology` covers it: `bridge` is the one module in this
# package that names `shipvision`, and the enforcement of "it is never imported at module
# scope" is the subprocess assertion in `tests/test_architecture.py`, which imports *this*
# package. A module that assertion cannot reach is a module the laziness is not enforced on.
# Free to import: `bridge` imports `functools`, `types` and `shipinfer.core.errors`, and its
# four loaders each `from shipvision import ...` inside the function body.
from shipinfer.topology import bridge
from shipinfer.topology.barrier import InstantBarrier, WaiterBudget
from shipinfer.topology.base import (
    CameraGroup,
    ChainItem,
    Element,
    ElementContext,
    ElementKind,
    ImageOpsLike,
    LetterboxLike,
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

# Imported for the side effect only: this is what puts `mock` and `pool` in the registries.
# Not re-exported. An implementation is reached through its registry, by the name a chain file
# uses -- `create_element(ElementKind.DETECT, "mock", ...)` -- never by importing the class,
# which is what keeps the registry the seam rather than a lookup table beside one.
from shipinfer.topology.elements import mock, pool
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
    "CameraGroup",
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
    "ImageOpsLike",
    "InstantBarrier",
    "LetterboxLike",
    "ModelResolver",
    "Topology",
    "WaiterBudget",
    "create_element",
    "describe_elements",
    "load_topology",
    "negotiate",
    "parse_caps",
    "registry_for",
]
