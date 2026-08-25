"""Errors raised by the stateful tracking plane."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = ["TrackingError"]


class TrackingError(ShipInferError):
    """A tracker was asked for something that would corrupt its per-camera state.

    Its own class because the operator's response is different from every other failure in
    the pipeline. A timed-out stage is a slow GPU and a validation error is a wrong config;
    this one means two of a camera's frames reached its tracker out of order, which is a
    *scheduling* fact — and the fix is at the queue, not at the model.

    Raised rather than absorbed. A tracker fed a frame it has already passed double-ages
    every track and double-counts the hit that promotes one, so a replayed frame quietly
    changes which identities exist downstream; a named failure on one frame is cheaper than
    an identity that never existed.
    """
