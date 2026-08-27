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

Not to be confused with :mod:`shipinfer.core.errors.topology` next door, which names the
*shard-wire* failures of the older topology-as-placement classes (rings, peers, shards).
The word "topology" means the chain from here on (arch.md §1); that module keeps its name
until the phase that dissolves ``server/topology`` renames both together.
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
    """A chain slot does not name one of the eight element kinds.

    Raised when neither an explicit ``kind:`` nor the slot name resolves: ``sement:`` is a
    typo, and inferring nothing from it and running a chain with a missing stage would be
    far worse than refusing to load.
    """

    def __init__(self, slot: str, known: Sequence[str]) -> None:
        super().__init__(
            f"element slot {slot!r} does not name an element kind; "
            f"add an explicit `kind:` or use one of {sorted(known)}"
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

    Carries the members, because "the chain has a cycle" without them is the least
    actionable message a loader can produce.
    """

    def __init__(self, cycle: Sequence[str]) -> None:
        super().__init__(f"the chain has a cycle through {sorted(cycle)}")
        self.cycle = tuple(cycle)


class ChainStructureError(TopologyError):
    """The chain is a DAG but not a runnable one.

    A chain with no decode root has nothing to read frames with; one with no output sink
    computes results nobody emits; a model element with no ``model:`` has nothing to run.
    All three are start-up refusals rather than silent no-ops.
    """


class CapsMismatchError(TopologyError):
    """Two adjacent elements agree on no format/location pair.

    The message names the fix on purpose. The tempting alternative — quietly inserting a
    device-to-host copy so ``nv12@gpu`` can feed ``bgr@cpu`` — is precisely the failure
    arch.md §8 refuses: a 1000 fps chain that silently downloads every frame to host memory
    looks like a working deployment and performs like a broken one.
    """

    def __init__(
        self,
        producer: str,
        produced: Sequence[str],
        consumer: str,
        accepted: Sequence[str],
    ) -> None:
        super().__init__(
            f"{producer!r} produces {list(produced)} but {consumer!r} accepts "
            f"{list(accepted)}: no cap in common. Declare a matching cap on one side or "
            "spell an explicit convert element — this loader will not download a frame to "
            "host memory implicitly (arch.md §8)."
        )
        self.producer = producer
        self.produced = tuple(produced)
        self.consumer = consumer
        self.accepted = tuple(accepted)
