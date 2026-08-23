"""Batch assembly and scatter — one implementation per file, selected via :data:`BATCHERS`."""

from shipinfer.scheduling.batching.base import AssembledBatch, Batcher
from shipinfer.scheduling.batching.registry import BATCHERS
from shipinfer.scheduling.batching.sizing import choose_batch_size
from shipinfer.scheduling.batching.stacking import StackingBatcher

__all__ = [
    "BATCHERS",
    "AssembledBatch",
    "Batcher",
    "StackingBatcher",
    "choose_batch_size",
]
