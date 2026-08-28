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

    @classmethod
    def parse(cls, value: object) -> Priority:
        """Resolve a band from whatever an operator or a client wrote.

        One vocabulary, two doors. ``priority: tracking_critical`` under ``ingest.cameras``
        and ``POST /streams {"priority": "tracking_critical"}`` are the same request, and
        until this method they were not: the HTTP door had matched band *names* since it
        existed, while the configuration door was a bare ``Priority`` annotation, so
        pydantic's ``IntEnum`` coercion took the numbers and nothing else. The name that
        this file's own neighbours told an operator to write — ``core/settings/ingest.py``,
        ``launch/control.py``, ``runners/inprocess.py`` all spell it ``tracking_critical`` —
        was a start-up ``ValidationError``. Owning the rule in ``core`` rather than at
        either door is what makes the two agree, including on the wording of the refusal.

        Accepted: a :class:`Priority`; a band *name* in any case (``tracking_critical``,
        ``TRACKING_CRITICAL``, ``Tracking_Critical``); an ``int`` in ``0..3``; and the
        numeric *string* spellings such as ``"0"``, because pydantic coerced those before
        this method existed and quietly narrowing what a running deployment's config file
        may contain is not a fix.

        Refused, deliberately: a hyphenated name — ``tracking-critical`` is not what any
        generated stub, document or Python member is spelled — an out-of-range number,
        ``None``, and every ``bool``. The last is the one worth the branch: ``priority: no``
        in YAML is ``False``, which is ``0``, which is :attr:`TRACKING_CRITICAL`. An
        operator who meant "off" would be handed the *highest* lane on the deployment, and
        that falsy-zero trap is the exact failure ADR-005 keeps paying for.

        Args:
            value: typed ``object`` rather than ``int | str | Priority``, because both
                callers are pydantic ``mode="before"`` validators and are handed whatever
                was in the document; deciding what is a band is this method's job, not the
                annotation's.

        Returns:
            The band ``value`` names or numbers.

        Raises:
            ValueError: ``value`` is not a band. The message names the value and lists the
                bands, and reads identically at both doors because it is written once,
                here. A plain ``ValueError`` and not a
                :mod:`shipinfer.core.errors` type on purpose: both callers are pydantic
                validators, and pydantic wraps a ``ValueError`` into the
                ``ValidationError`` an operator — or a 422 — actually reads, while any
                other exception escapes as a 500.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, bool):
            if isinstance(value, int):
                try:
                    return cls(value)
                except ValueError:
                    pass
            elif isinstance(value, str):
                try:
                    return cls[value.upper()]
                except KeyError:
                    pass
                try:
                    return cls(int(value))
                except ValueError:
                    pass
        # The numbers are accepted above but unlisted here: the names are the vocabulary
        # both doors take, and `POST /streams` refuses the integers outright (ADR-005), so
        # a message offering them would be a document disagreeing with itself.
        bands = ", ".join(band.name.lower() for band in cls)
        raise ValueError(
            f"{value!r} is not a priority band; expected one of {bands} (any case)"
        )
