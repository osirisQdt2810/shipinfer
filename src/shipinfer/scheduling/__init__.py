"""Scheduling: the work item, the queues, the batchers, the policies, the dispatcher.

Pure Python, and that is not incidental. The load-balancing and fairness behaviour is the
part of this system most worth having tests for, and tests that need sixteen GPUs get
written once and then never run again.

Three extension families, each a sub-package with its own registry:

* :mod:`~shipinfer.scheduling.queues` — ordering, fairness and overflow (:data:`QUEUES`)
* :mod:`~shipinfer.scheduling.batching` — pack/scatter (:data:`BATCHERS`)
* :mod:`~shipinfer.scheduling.policies` — instance placement (:data:`POLICIES`)

The compiled ``shipinfer._C`` extension implements the queue and batcher contracts with a
lock-free ring and a fused staging copy; parity tests keep the two honest.
"""

from shipinfer.scheduling.batching import (
    BATCHERS,
    AssembledBatch,
    Batcher,
    StackingBatcher,
    choose_batch_size,
)
from shipinfer.scheduling.dispatcher import Dispatcher, DispatchResult
from shipinfer.scheduling.policies import (
    POLICIES,
    JoinShortestQueuePolicy,
    LocalityAwareSpilloverPolicy,
    Placeable,
    PlacementPolicy,
    PowerOfTwoChoicesPolicy,
    RoundRobinPolicy,
    SequenceAffinityPolicy,
    build_policy,
)
from shipinfer.scheduling.queues import (
    QUEUES,
    BatchWindow,
    FairPriorityQueue,
    FifoQueue,
    QueueStats,
    RequestQueue,
)
from shipinfer.scheduling.work import WorkItem, summarise_fairness

__all__ = [
    "BATCHERS",
    "POLICIES",
    "QUEUES",
    "AssembledBatch",
    "BatchWindow",
    "Batcher",
    "DispatchResult",
    "Dispatcher",
    "FairPriorityQueue",
    "FifoQueue",
    "JoinShortestQueuePolicy",
    "LocalityAwareSpilloverPolicy",
    "Placeable",
    "PlacementPolicy",
    "PowerOfTwoChoicesPolicy",
    "QueueStats",
    "RequestQueue",
    "RoundRobinPolicy",
    "SequenceAffinityPolicy",
    "StackingBatcher",
    "WorkItem",
    "build_policy",
    "choose_batch_size",
    "summarise_fairness",
]
