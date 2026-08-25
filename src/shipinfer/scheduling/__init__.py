"""Scheduling: the work item, the queues, the batchers, the policies, the dispatcher.

Pure Python, and that is not incidental. The load-balancing and fairness behaviour is the
part of this system most worth having tests for, and tests that need sixteen GPUs get
written once and then never run again.

Four extension families, each a sub-package with its own registry:

* :mod:`~shipinfer.scheduling.queues` — ordering, fairness and overflow (:data:`QUEUES`)
* :mod:`~shipinfer.scheduling.batching` — pack/scatter (:data:`BATCHERS`)
* :mod:`~shipinfer.scheduling.policies` — instance placement (:data:`POLICIES`)
* :mod:`~shipinfer.scheduling.limits` — concurrent-execution bounds (:data:`RATE_LIMITERS`)

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
from shipinfer.scheduling.limits import (
    RATE_LIMITERS,
    ConcurrencyRateLimiter,
    RateLimiter,
    UnlimitedRateLimiter,
    build_rate_limiter,
)
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
    "RATE_LIMITERS",
    "AssembledBatch",
    "BatchWindow",
    "Batcher",
    "ConcurrencyRateLimiter",
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
    "RateLimiter",
    "RequestQueue",
    "RoundRobinPolicy",
    "SequenceAffinityPolicy",
    "StackingBatcher",
    "UnlimitedRateLimiter",
    "WorkItem",
    "build_policy",
    "build_rate_limiter",
    "choose_batch_size",
    "summarise_fairness",
]
