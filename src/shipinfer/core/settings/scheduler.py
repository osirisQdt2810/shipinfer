"""Dispatch, batching and backpressure — the behaviour this project exists to own."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shipinfer.core.settings.enums import OverflowPolicy

__all__ = ["SchedulerSettings"]


class SchedulerSettings(BaseModel):
    """How work is placed, batched and shed."""

    model_config = ConfigDict(extra="forbid")

    #: A name registered in :data:`shipinfer.scheduling.policies.POLICIES`. A plain string
    #: rather than an enum so a policy shipped by a third-party package is selectable from
    #: config without editing this file — and so ``core`` never imports ``scheduling``.
    placement_policy: str = "locality_spillover"
    #: Constructor keyword arguments for that policy, e.g. ``{spill_threshold: 8}``.
    #: Validated by the policy's own ``__init__``, so a typo fails at start-up.
    placement_policy_options: dict[str, Any] = Field(default_factory=dict)

    #: Per-instance queue capacity. Small on purpose: a deep queue converts a throughput
    #: problem into a latency problem and then hides it.
    max_queue_size: int = Field(default=64, ge=1)
    overflow_policy: OverflowPolicy = OverflowPolicy.REJECT
    enqueue_block_timeout_ms: int = Field(default=50, ge=0)

    #: Round-robin across camera ids inside a queue so one busy camera cannot starve the
    #: rest — the concrete fix for the shared-buffer eviction bug in the previous system.
    fair_queueing: bool = True
    #: Drop a request whose deadline has already passed instead of executing it.
    drop_expired_requests: bool = True
    #: Smoothing factor for the per-instance latency EWMA used by load-aware policies.
    latency_ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    #: Retry a full queue on the next-shortest instance before giving up. This is what
    #: turns a policy's guess into a delivery guarantee under a burst.
    spill_on_full: bool = True
