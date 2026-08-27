"""One registry per element kind, and the factory the chain loader calls.

**Why per kind and not one flat registry** — three reasons, all of them things that went
wrong when this was sketched flat:

1. **Implementation names repeat across kinds, legitimately.** ``pool`` is the default for
   all four model kinds and ``shipvision`` is both a tracker and an MTMC implementation
   (arch.md §1). :class:`~shipinfer.core.registry.Registry` refuses a duplicate name — it
   is right to — so a flat registry would force ``detect-pool``, ``segment-pool``,
   ``embed-pool``, i.e. the kind encoded in the name anyway, but by convention instead of
   by type.
2. **The error message stays readable.** A misspelled detector should be answered with the
   four detector implementations, not with all twenty element implementations sorted
   alphabetically.
3. **The kind check has somewhere to live.** ``@DETECTORS.register("pool")`` on a class
   whose ``kind`` is ``segment`` is caught at import time, which is the difference between
   a start-up refusal and a chain that runs the wrong model. The check is repeated in
   :func:`create_element` because a lazy registration has no class to check at registration
   time, and the loader trusts ``node.kind`` for its structure rules.

Registration is **eager**, following ``ingest/registry.py``: an element module imports
nothing heavier than ``core`` at module scope and loads its runtime inside ``_do_open``. So
``ELEMENTS`` can list ``gstreamer-gpu`` on a host with no GStreamer and still fail usefully,
at ``open()``, naming the package to install. :meth:`ElementRegistry.register_lazy` is for
the one case that cannot honour that — a module whose *import* is impossible without the
runtime (``pyds``) — and it is overridden here to say where its kind check went.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from shipinfer.core.errors import ConfigurationError, UnknownElementImplError
from shipinfer.core.registry import Registry
from shipinfer.topology.base import Element, ElementKind

__all__ = [
    "ELEMENTS",
    "ElementRegistry",
    "create_element",
    "describe_elements",
    "registry_for",
]


class ElementRegistry(Registry[Element]):
    """The implementations of one element kind.

    Args:
        kind: the kind every registration must declare. Used verbatim in error messages, so
            "unknown detect element 'poool'" reads the way an operator thinks about it.
    """

    def __init__(self, kind: ElementKind) -> None:
        super().__init__(f"{kind.value} element", Element)
        self.element_kind = kind

    def register(
        self, name: str, *aliases: str, description: str = ""
    ) -> Callable[[type[Element]], type[Element]]:
        """Class decorator, with the kind checked before anything is recorded.

        Raises:
            ConfigurationError: the class declares a different :attr:`Element.kind`, or
                none at all. Checked *before* delegating, so a rejected class does not
                stay half-registered in the map. :meth:`register_lazy` cannot check this
                early; see its docstring.
        """
        inner = super().register(name, *aliases, description=description)

        def decorator(cls: type[Element]) -> type[Element]:
            _check_kind(cls, self, name)
            result = inner(cls)
            # One source of truth for the registered name: the chain file says `impl:
            # pool`, the log line says `impl=pool`, and neither is retyped by hand.
            result.impl = name
            return result

        return decorator

    def register_lazy(
        self, name: str, target: str, *aliases: str, description: str = ""
    ) -> None:
        """Register ``"module:ClassName"`` — with the kind checked at *creation*, not here.

        The eager :meth:`register` can check :attr:`Element.kind` immediately, because it
        holds the class. This one cannot: the whole point is that the module is not imported
        yet, and importing it to check the kind would defeat the registration style it
        exists for (``pyds``, whose import needs DeepStream present).

        So the check moves one step later, to :func:`create_element` — the single place
        every element in every chain is built. A lazily registered class of the wrong kind
        is refused when a chain first asks for it, with the same
        :class:`~shipinfer.core.errors.ConfigurationError`, rather than never. Overridden
        only to say so: without this docstring the inherited method looks like it inherits
        the guarantee too, and it does not.
        """
        super().register_lazy(name, target, *aliases, description=description)


def _check_kind(cls: type[Element], registry: ElementRegistry, name: str) -> None:
    """Refuse a class whose :attr:`Element.kind` is not the registry's kind.

    Both kinds are in the message. "wrong kind" without them sends the reader looking at the
    registration when the mistake is as often in the class, or the other way round.

    Raises:
        ConfigurationError: the class declares a different kind, or none at all.
    """
    declared = getattr(cls, "kind", None)
    if declared is not registry.element_kind:
        raise ConfigurationError(
            f"{cls.__name__} declares kind {declared!r} and cannot be registered as a "
            f"{registry.element_kind.value} element (as {name!r}); set "
            f"`kind = ElementKind.{registry.element_kind.name}` or register it under its "
            "own kind"
        )


#: Every kind's registry, created once. A ``MappingProxyType`` because the eight kinds are
#: the closed vocabulary of :class:`~shipinfer.topology.base.ElementKind`: a ninth registry
#: appearing at runtime would mean a kind exists that the loader cannot validate.
ELEMENTS: Mapping[ElementKind, ElementRegistry] = MappingProxyType(
    {kind: ElementRegistry(kind) for kind in ElementKind}
)


def registry_for(kind: ElementKind | str) -> ElementRegistry:
    """The registry of one kind, by enum or by name.

    Raises:
        UnknownElementKindError: the string does not name a kind.
    """
    resolved = kind if isinstance(kind, ElementKind) else ElementKind.parse(kind)
    return ELEMENTS[resolved]


def create_element(
    kind: ElementKind | str,
    impl: str,
    name: str,
    params: Mapping[str, Any] | None = None,
) -> Element:
    """Build one element: ``create_element("detect", "pool", "detect", {...})``.

    **The one place every element is built**, which is why the kind check is repeated here
    rather than left to :meth:`ElementRegistry.register`. A lazy registration has no class
    to check at registration time, so without this a
    ``register_lazy("kafka", "...:MockTrack")`` on the output registry would hand the chain
    loader a tracker labelled as an output sink — and the loader's structure rules, which
    read ``node.kind``, would agree that the chain ends properly. Checking at creation
    covers eager and lazy registrations with one rule.

    Args:
        kind: which registry to look in.
        impl: the registered implementation name, or one of its aliases.
        name: the chain slot this instance fills.
        params: the slot's ``params:`` block, passed through untouched.

    Raises:
        UnknownElementKindError: ``kind`` does not name a kind.
        UnknownElementImplError: no implementation of that kind under that name; the message
            lists the ones there are.
        ConfigurationError: the registered class declares a different
            :attr:`~shipinfer.topology.base.Element.kind`, or the lazy target cannot be
            imported.
    """
    registry = registry_for(kind)
    if impl not in registry:
        raise UnknownElementImplError(registry.element_kind.value, impl, registry.names())
    registered = registry.canonical(impl)
    cls = registry.get(impl)
    _check_kind(cls, registry, registered)
    # `register` sets this for an eager registration; a lazy one has nothing to set it on
    # until now. Assigning here rather than only asserting keeps the promise that
    # `element.impl` is the name the chain file used, whichever style registered it.
    if cls.impl != registered:
        cls.impl = registered
    return cls(name, params)


def describe_elements() -> dict[str, list[tuple[str, str]]]:
    """``{kind: [(impl, one-line description), ...]}`` — what a ``--list`` flag prints."""
    return {kind.value: registry.describe() for kind, registry in ELEMENTS.items()}
