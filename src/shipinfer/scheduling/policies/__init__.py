"""Placement policies — one class per file, selected through a registry.

Importing this package registers every built-in policy. A new policy is a new module here
plus ``@POLICIES.register("name")``; nothing else in the tree changes. Third-party policies
register the same way from their own package, as long as that package is imported before
the server starts.
"""

from shipinfer.scheduling.policies.base import Placeable, PlacementPolicy
from shipinfer.scheduling.policies.join_shortest_queue import JoinShortestQueuePolicy
from shipinfer.scheduling.policies.locality_spillover import LocalityAwareSpilloverPolicy
from shipinfer.scheduling.policies.power_of_two import PowerOfTwoChoicesPolicy
from shipinfer.scheduling.policies.registry import POLICIES, build_policy
from shipinfer.scheduling.policies.round_robin import RoundRobinPolicy
from shipinfer.scheduling.policies.sequence_affinity import SequenceAffinityPolicy

__all__ = [
    "POLICIES",
    "JoinShortestQueuePolicy",
    "LocalityAwareSpilloverPolicy",
    "Placeable",
    "PlacementPolicy",
    "PowerOfTwoChoicesPolicy",
    "RoundRobinPolicy",
    "SequenceAffinityPolicy",
    "build_policy",
]
