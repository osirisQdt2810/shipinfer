"""Scheduling: the work item, the queues, the batchers, the policies, the dispatcher.

Pure Python by design: the fairness and balancing logic is the part most worth testing,
and it must be testable with no GPU. Four extension families, each with a registry:

* :mod:`~shipinfer.scheduling.queues` — ordering, fairness and overflow (:data:`QUEUES`)
* :mod:`~shipinfer.scheduling.batching` — pack/scatter (:data:`BATCHERS`)
* :mod:`~shipinfer.scheduling.policies` — instance placement (:data:`POLICIES`)
* :mod:`~shipinfer.scheduling.limits` — concurrent-execution bounds (:data:`RATE_LIMITERS`)

``shipinfer._C`` implements the queue and batcher contracts natively; parity tests keep
the two honest.
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
