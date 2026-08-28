"""Re-export shim: the per-thread image ops moved to :mod:`shipinfer.runtime.ops.thread_local`.

The class is an :class:`~shipinfer.runtime.ops.ImageOps` decorator with no knowledge of the
pipeline, and it grew two more callers outside this package —
:mod:`shipinfer.cli.commands.run` and :mod:`shipinfer.cli.shard`, which hand it to a chain
:class:`~shipinfer.runners.base.Runner`. Sitting under ``pipeline`` it could not be reached
from either without ``cli`` importing the application layer to fix an accelerator-seam
problem, so it now lives in the layer that owns the seam.

This module stays because ``pipeline`` imports it by this name in two places
(:mod:`shipinfer.pipeline.runner`, :mod:`shipinfer.pipeline.graph`) and both re-export it. It
re-exports the *same objects*, so ``isinstance`` and ``is`` agree across the two paths.
"""

from __future__ import annotations

from shipinfer.runtime.ops.thread_local import ThreadLocalImageOps, staging_owner

__all__ = ["ThreadLocalImageOps", "staging_owner"]
