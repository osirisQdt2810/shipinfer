"""Chain failures — everything wrong with a *declared element chain* (arch.md §1, §8).

Every error here is raised at **load time**, before a frame exists. That is the whole point
of having a declarative chain: a mis-wired topology should stop a deploy, not surface at
3 a.m. as a camera that produces no detections. So they subclass
:class:`~shipinfer.core.errors.config.ConfigurationError` — the layer above treats them like
a bad ``config.yaml``, which is what they are.

Each one carries the *names* involved, as strings. Deliberately strings and never
:class:`~shipinfer.topology.caps.Caps` or :class:`~shipinfer.topology.base.ElementKind`
objects: ``core`` sits below ``topology`` and must stay importable on its own, so the error
vocabulary cannot depend on the vocabulary of the thing that raises it.

Not to be confused with :mod:`shipinfer.core.errors.launch` next door, which names the
*shard-wire* failures of the older topology-as-placement classes (rings, peers, shards).
The word "topology" means the chain from here on (arch.md §1); that module was renamed to
``launch`` for the same reason, so the two vocabularies no longer share a name.
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

    Raised when neither an explicit ``kind:`` nor the slot name resolves: ``sement:`` is a
    typo, and inferring nothing from it and running a chain with a missing stage would be
    far worse than refusing to load.

    **One mistake, one error type.** A misspelled kind is the same mistake whether it is
    written as the slot name (``sement:``) or as an explicit value (``kind: sement``), so
    the schema keeps ``kind`` a plain string and
    :meth:`~shipinfer.topology.base.ElementKind.parse` raises this for both. Declaring the
    field as the enum would answer the second spelling with a pydantic ``ChainSpecError``
    and the first with this, which is two vocabularies for one typo.

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

    A chain whose root is not a decode element has nothing to read frames with; one with no
    output element computes results nobody emits; an output element with a successor is not
    the end of anything; a model element with no ``model:`` has nothing to run; a root or a
    fan-in whose caps cannot be pinned down leaves the loader unable to say what flows on an
    edge. All of them are start-up refusals rather than silent no-ops.

    Distinct from :class:`CapsMismatchError`, which is about *two* elements disagreeing.
    This one is about *one* element's place in the chain.
    """


class CapsMismatchError(TopologyError):
    """Two elements that can hand data to each other agree on no format/location pair.

    The message names the fix on purpose. The tempting alternative — quietly inserting a
    device-to-host copy so ``nv12@gpu`` can feed ``bgr@cpu`` — is precisely the failure
    arch.md §8 refuses: a 1000 fps chain that silently downloads every frame to host memory
    looks like a working deployment and performs like a broken one.

    Args:
        producer: the element the data comes from.
        produced: its output caps, **as resolved** — a wildcard ``produces`` is filled in
            from what arrives at the element before this check runs, so the message names
            the cap that would really flow rather than the ``*@*`` the class declared.
        consumer: the element the data goes to.
        accepted: its declared input caps.
        skipped: the ``when:`` element whose *bypass* is being refused, if any. A
            conditional element is skipped for items its condition rejects, and the item
            then travels straight from its predecessor to its successor; that hand-over is
            checked too, and it is worth saying which element being absent creates it.
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
