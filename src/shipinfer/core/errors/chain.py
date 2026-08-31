"""Chain failures — everything wrong with a *declared element chain* (arch.md §1, §8).

All raised at **load time**, before a frame exists: a mis-wired topology should stop a
deploy, not surface as a camera that produces no detections. They subclass
:class:`~shipinfer.core.errors.config.ConfigurationError`.

Each error carries the *names* involved as plain strings — ``core`` sits below
``topology`` and must stay importable on its own. Not to be confused with
:mod:`shipinfer.core.errors.launch`, the shard-wire failures of the older
topology-as-placement classes.
"""

from __future__ import annotations

from collections.abc import Sequence

from shipinfer.core.errors.config import ConfigurationError

__all__ = [
    "CapsMismatchError",
    "CapsSyntaxError",
    "ChainCycleError",
    "ChainSpecError",
    "ChainStructureError",
    "ConditionSyntaxError",
    "TopologyError",
    "UnknownElementError",
    "UnknownElementImplError",
    "UnknownElementKindError",
]


class TopologyError(ConfigurationError):
    """Base for every chain-definition failure, so a loader can catch one thing."""


class ChainSpecError(TopologyError):
    """The file is not a chain spec: unreadable, not a mapping, unknown key, wrong type.

    Always carries the source path in the message. A pydantic validation report with no
    file name is nearly useless once a deployment holds more than one topology.
    """


class CapsSyntaxError(TopologyError):
    """A cap string is not ``<format>@<location>``.

    Its own type rather than a bare ``ValueError`` because the fix is textual and local —
    ``nv12@vram`` is a typo for ``nv12@gpu`` — while every other error in this module means
    the *chain* is wrong.
    """


class ConditionSyntaxError(TopologyError):
    """A ``when:`` expression is not ``<field> == <value>`` or ``<field> != <value>``."""


class UnknownElementKindError(TopologyError):
    """A chain slot, or an explicit ``kind:``, does not name one of the eight kinds.

    The schema keeps ``kind`` a plain string so
    :meth:`~shipinfer.topology.base.ElementKind.parse` raises this for both spellings of
    the same typo (``sement:`` and ``kind: sement``) — one mistake, one error type.

    ``slot`` is whatever was misspelled — the slot name or the ``kind:`` value.
    """

    def __init__(self, slot: str, known: Sequence[str]) -> None:
        super().__init__(
            f"{slot!r} does not name an element kind; use one of {sorted(known)} as the "
            "slot name, or add an explicit `kind:`"
        )
        self.slot = slot
        self.known = tuple(known)


class UnknownElementImplError(TopologyError):
    """No implementation of this kind is registered under this name.

    Lists the implementations of *that kind only*. This is the reason the registries are
    per kind (ADR: see :mod:`shipinfer.topology.registry`): a chain that asks for a
    misspelled detector wants the four detector names, not all twenty element names.
    """

    def __init__(self, kind: str, impl: str, available: Sequence[str]) -> None:
        super().__init__(
            f"no {kind} element implementation named {impl!r}; "
            f"available: {sorted(available)}"
        )
        self.kind = kind
        self.impl = impl
        self.available = tuple(available)


class UnknownElementError(TopologyError):
    """One element refers to another that the chain does not declare.

    Names the referrer as well as the missing name: ``after: segement`` on a nine-element
    chain is a two-second fix once you know which line to look at.
    """

    def __init__(self, referrer: str, missing: str, known: Sequence[str]) -> None:
        super().__init__(
            f"element {referrer!r} refers to {missing!r}, which the chain does not "
            f"declare; declared: {sorted(known)}"
        )
        self.referrer = referrer
        self.missing = missing
        self.known = tuple(known)


class ChainCycleError(TopologyError):
    """The ``after`` edges form a cycle, so no execution order exists.

    Carries the cycle as a **path**, in the direction data would flow, and prints it that
    way: ``detect -> track -> detect``. A set of names ("the chain has a cycle through
    ['detect', 'track', 'output']") is the least actionable message a loader can produce,
    and a set built from "everything Kahn's algorithm could not place" also drags in every
    element that merely hangs off the cycle.
    """

    def __init__(self, cycle: Sequence[str]) -> None:
        path = " -> ".join([*cycle, cycle[0]]) if cycle else "<unknown>"
        super().__init__(f"the chain has a cycle: {path}")
        self.cycle = tuple(cycle)


class ChainStructureError(TopologyError):
    """The chain is a DAG but not a runnable one.

    A root that is not a decode element, no output element, an output with a successor,
    a model element with no ``model:``, a root or fan-in whose caps cannot be pinned
    down — all start-up refusals rather than silent no-ops. Distinct from
    :class:`CapsMismatchError` (two elements disagreeing): this is one element's place
    in the chain.
    """


class CapsMismatchError(TopologyError):
    """Two elements that can hand data to each other agree on no format/location pair.

    The message names the fix and refuses the tempting alternative — an implicit
    device-to-host copy so ``nv12@gpu`` can feed ``bgr@cpu`` — per arch.md §8.

    Args:
        produced: the producer's output caps **as resolved** (wildcards filled in),
            not the declared ``*@*``.
        skipped: the ``when:`` element whose *bypass* created the checked hand-over,
            if any.
    """

    def __init__(
        self,
        producer: str,
        produced: Sequence[str],
        consumer: str,
        accepted: Sequence[str],
        *,
        skipped: str | None = None,
    ) -> None:
        detour = "" if skipped is None else f"when {skipped!r} is skipped by its `when:`, "
        super().__init__(
            f"{detour}{producer!r} produces {list(produced)} but {consumer!r} accepts "
            f"{list(accepted)}: no cap in common. Declare a matching cap on one side or "
            "spell an explicit convert element — this loader will not download a frame to "
            "host memory implicitly (arch.md §8)."
        )
        self.producer = producer
        self.produced = tuple(produced)
        self.consumer = consumer
        self.accepted = tuple(accepted)
        self.skipped = skipped
