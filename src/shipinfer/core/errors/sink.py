"""Errors raised on the publish side of the pipeline — the last hop out of the process.

Separate from :mod:`shipinfer.core.errors.inference` because the operator response is
different: an inference error points at a model or a GPU, this one points at the bus and at
whoever owns the topic.
"""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = ["SinkDeliveryError"]


class SinkDeliveryError(ShipInferError):
    """A sink accepted an event for sending and the transport then failed to deliver it.

    The distinction this type exists to keep is between *queued* and *published*. An
    asynchronous producer returns as soon as the message is in its own buffer, so a refused
    topic or a denied ACL surfaces later, on a delivery callback, and is invisible to the
    call that queued it. Raising this from the sink's emit hook is what turns that late
    verdict back into a counted, per-frame publish failure instead of a green dashboard.
    """
