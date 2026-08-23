"""The root of the error tree."""

from __future__ import annotations

__all__ = ["ShipInferError"]


class ShipInferError(Exception):
    """Base class for everything this package raises deliberately.

    One rule makes the whole tree worth having: **nothing in this codebase returns an
    empty result to mean "something went wrong"**. A dropped frame, a saturated queue and
    a dead GPU are three different operational events, and an empty list tells the
    operator which of them happened: none.
    """
