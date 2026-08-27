"""Topologies: how a deployment is laid out into processes. See :mod:`.base`."""

from __future__ import annotations

from shipinfer.core.errors import ConfigurationError
from shipinfer.server.topology.base import TOPOLOGIES, TOPOLOGY_ENV, Topology
from shipinfer.server.topology.deepstream import DeepStreamTopology
from shipinfer.server.topology.fleet import FleetTopology
from shipinfer.server.topology.service import ServiceTopology

__all__ = [
    "TOPOLOGIES",
    "TOPOLOGY_ENV",
    "DeepStreamTopology",
    "FleetTopology",
    "ServiceTopology",
    "Topology",
    "build_topology",
]


def build_topology(kind: str) -> Topology:
    """The topology named in the settings, or a refusal that lists what there is.

    A typo here is a deployment that never starts, so the message carries the known names
    rather than only the unknown one.
    """
    if kind not in TOPOLOGIES:
        known = ", ".join(sorted(TOPOLOGIES.names()))
        raise ConfigurationError(f"unknown topology {kind!r}; known topologies: {known}")
    return TOPOLOGIES.create(kind)
