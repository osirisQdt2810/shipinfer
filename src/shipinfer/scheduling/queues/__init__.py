"""Request queues — one implementation per file, selected through :data:`QUEUES`.

``fair`` is the default and the one this system needs; ``fifo`` is the baseline it is
measured against. A deployment picks with ``scheduler.queue_type`` in the settings tree.
"""

from shipinfer.scheduling.queues.base import BatchWindow, QueueStats, RequestQueue
from shipinfer.scheduling.queues.fair import FairPriorityQueue
from shipinfer.scheduling.queues.fifo import FifoQueue
from shipinfer.scheduling.queues.lanes import Lane
from shipinfer.scheduling.queues.registry import QUEUES

__all__ = [
    "QUEUES",
    "BatchWindow",
    "FairPriorityQueue",
    "FifoQueue",
    "Lane",
    "QueueStats",
    "RequestQueue",
]
