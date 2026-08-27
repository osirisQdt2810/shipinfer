"""Caps: what an element hands over, and what the next one will take (arch.md §8).

A cap is two words — ``<format>@<location>``, as in ``nv12@gpu`` — and that is the whole
type system of the chain. Two halves, because the two questions an adjacent pair of
elements has to agree on are different in kind:

* **format** is an open vocabulary: ``nv12``, ``bgr``, ``tensor``, ``meta``, and whatever a
  future element needs. Closing it would mean editing this file to add a decoder, which is
  exactly the switch-statement shape the registries exist to avoid.
* **location** is closed to ``gpu`` and ``cpu``. There are two memories, that is a fact
  about the hardware rather than about our vocabulary, and it is the half that costs
  milliseconds when it is wrong.

**A wildcard is a declaration, not a converter.** ``*@*`` on an output sink means "I will
take whatever arrives". It never *bridges* two concrete caps: ``nv12@gpu`` does not match
``nv12@cpu``, and no amount of wildcarding elsewhere in the chain makes it. That is the rule
arch.md §8 asks for — "a chain that would silently download to CPU refuses to load instead"
— and it is the reason this module has no ``convert`` anything in it. At 1000 frames/s an
implicit device-to-host copy is not a fallback, it is a 3 GB/s tax that looks like a working
deployment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shipinfer.core.errors import CapsSyntaxError

__all__ = ["ANY", "LOCATIONS", "Caps", "negotiate", "parse_caps"]

#: The wildcard token, usable for either half.
ANY = "*"

#: Where data can live. Closed, unlike the format vocabulary — see the module docstring.
LOCATIONS = frozenset({"gpu", "cpu"})

#: An open token: lowercase, starts with a letter. Lowercase because caps appear in YAML and
#: in log lines, and ``NV12@GPU`` matching ``nv12@gpu`` would mean two spellings of every
#: cap in every error message.
_FORMAT = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class Caps:
    """One format in one memory. Frozen and hashable, so an edge can be keyed on it.

    Args:
        format: an open lowercase token, or ``"*"``.
        location: ``"gpu"``, ``"cpu"`` or ``"*"``.

    Raises:
        CapsSyntaxError: either half is not a legal token. Validated in ``__post_init__``
            rather than only in :meth:`parse`, so a cap built in code cannot skip the check
            that a cap read from YAML gets.
    """

    format: str
    location: str

    def __post_init__(self) -> None:
        if self.format != ANY and not _FORMAT.match(self.format):
            raise CapsSyntaxError(
                f"cap format {self.format!r} must be a lowercase token like 'nv12' or '*'"
            )
        if self.location != ANY and self.location not in LOCATIONS:
            raise CapsSyntaxError(
                f"cap location {self.location!r} must be one of " f"{sorted(LOCATIONS)} or '*'"
            )

    @classmethod
    def parse(cls, text: str) -> Caps:
        """Parse ``"nv12@gpu"``.

        Both halves are mandatory. ``"nv12"`` is refused rather than defaulted to a
        location, because the whole value of this type is that the memory an element reads
        from is stated and not assumed.

        Raises:
            CapsSyntaxError: no ``@``, an empty half, or an illegal token.
        """
        raw = text.strip()
        format_, sep, location = raw.partition("@")
        if not sep or not format_ or not location:
            raise CapsSyntaxError(
                f"cap {text!r} must be '<format>@<location>', for example 'nv12@gpu'"
            )
        return cls(format_.strip().lower(), location.strip().lower())

    @property
    def is_wildcard(self) -> bool:
        """Whether either half is unconstrained."""
        return self.format == ANY or self.location == ANY

    def matches(self, other: Caps) -> bool:
        """Whether data described by ``self`` can be handed to something accepting ``other``.

        Symmetric, and true only when *both* halves agree: equal, or one of them a
        wildcard. No pair of concrete locations ever matches across ``gpu``/``cpu``.
        """
        return _half_matches(self.format, other.format) and _half_matches(
            self.location, other.location
        )

    def resolve(self, other: Caps) -> Caps:
        """The concrete cap two matching declarations settle on.

        Each half takes whichever side is concrete; a half both sides left wildcard stays a
        wildcard, which is honest — nothing in the chain has said what will flow there.

        Raises:
            CapsSyntaxError: the two do not match. A caller that has not checked
                :meth:`matches` first is asking for a negotiated cap that does not exist,
                and returning one of the two arbitrarily would put a wrong cap on an edge.
        """
        if not self.matches(other):
            raise CapsSyntaxError(f"cannot resolve {self} against {other}: they do not match")
        return Caps(
            other.format if self.format == ANY else self.format,
            other.location if self.location == ANY else self.location,
        )

    def __str__(self) -> str:
        return f"{self.format}@{self.location}"


def _half_matches(left: str, right: str) -> bool:
    """One half of two caps agrees: the same token, or either side unconstrained."""
    return left == right or ANY in (left, right)


def parse_caps(items: Iterable[str]) -> tuple[Caps, ...]:
    """Parse a declared cap list, keeping its order.

    Order is preference order — see :func:`negotiate` — so this never sorts or dedupes.
    """
    return tuple(Caps.parse(item) for item in items)


def negotiate(produced: Sequence[Caps], accepted: Sequence[Caps]) -> Caps | None:
    """The cap an edge will carry, or ``None`` when the two sides share none.

    Preference is **declaration order, producer first**: the first entry in ``produced``
    that any entry of ``accepted`` will take wins. So a detector that lists
    ``nv12@gpu, bgr@cpu`` says "device-resident if my producer can, host memory if it
    cannot", and the loader picks the device path without anyone configuring a preference.

    ``None`` here is a *no*, not a failure signal: the only caller is the chain loader,
    which turns it into a
    :class:`~shipinfer.core.errors.CapsMismatchError` naming both elements. Keeping the
    error out of this function is what lets it be used to *ask* the question — a future
    convert-insertion pass needs the answer, not an exception.
    """
    for candidate in produced:
        for target in accepted:
            if candidate.matches(target):
                return candidate.resolve(target)
    return None
