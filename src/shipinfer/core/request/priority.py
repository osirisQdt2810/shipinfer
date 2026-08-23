"""Scheduling priority classes."""

from __future__ import annotations

import enum

__all__ = ["Priority"]


class Priority(enum.IntEnum):
    """Lower value == served first.

    ``TRACKING_CRITICAL`` exists for the one thing a generic inference server cannot
    express: a frame from a camera whose tracker is about to lose a target is worth more
    than a frame from an idle camera, even though both are "just a detection request".
    That is the customisation this project was built to make possible (ADR-005).
    """

    TRACKING_CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    BACKGROUND = 3
