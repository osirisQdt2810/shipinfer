"""``Element.needs_model`` and what ``open()`` actually requires, checked against each other.

:attr:`~shipinfer.topology.base.Element.needs_model` is a *declaration*: the class says
whether :meth:`~shipinfer.topology.base.Element.open` resolves it against
``ElementContext.models``. Two readers trust it without asking the element again — the
in-process runner's expiry gate, and ``shipinfer run``, which builds an
:class:`~shipinfer.engine.InferenceServer` for a chain that carries one and none for a chain
of mocks.

A declaration with readers and no test is a declaration that drifts. The failure it drifts
into is not loud: an element that answers ``False`` and then reaches for
``ElementContext.models`` fails at ``open()`` on the deployment that had no pool built for
it, and an element that answers ``True`` and never uses one makes every chain containing it
load a repository it does not need. So this walks *every registered implementation of every
kind* and checks the declaration against the behaviour, which is the only way a new element
added next month is covered without anybody remembering to add it here.

Note what is deliberately **not** checked here: whether a slot must name a ``model:``. That
is a separate ClassVar with a separate owner, and the two diverge — an ``nvinfer`` element
will need the artefact named and will still never touch this process's pool.

Offline by construction: element constructors are required to be hardware-free
(:class:`~shipinfer.topology.base.Element`), and opening one with an empty context is what
the loader already does to validate a chain on a laptop.
"""

from __future__ import annotations

import pytest

import shipinfer.topology.elements  # noqa: F401  -- imported for its registrations
from shipinfer.core.errors import BackendUnavailableError, ConfigurationError
from shipinfer.topology.base import Element, ElementContext, ElementKind
from shipinfer.topology.registry import ELEMENTS, create_element


def _ships_with_the_package(kind: ElementKind, impl: str) -> bool:
    """Whether this registration is one of ours rather than a test's.

    The element registries are process-wide and other test modules register doubles into
    them at import time -- ``tests/runners/test_inprocess.py`` alone contributes a detector
    that fails on a nominated frame and a recogniser whose ``_do_open`` raises on purpose.
    Those are not counter-examples to the invariant; they are deliberate misbehaviour, and
    walking them would make this file's result depend on collection order.

    So the filter is the defining module, not a name list: everything under ``shipinfer.`` is
    in scope, including an element some future package registers from outside
    ``topology/elements/``, and nothing a test defines is.
    """
    return ELEMENTS[kind].get(impl).__module__.startswith("shipinfer.")


#: Every implementation the package registers, as ``(kind, impl)``. Read from the registries
#: rather than listed here, so a new element is covered the moment it registers -- a
#: hand-written list would pass forever while the thing it was written to protect went
#: untested.
EVERY_IMPLEMENTATION = [
    (kind, impl)
    for kind, registry in ELEMENTS.items()
    for impl in sorted(registry.names())
    if _ships_with_the_package(kind, impl)
]


def _open_with_no_pool(kind: ElementKind, impl: str) -> Element:
    """Build one element and open it against a context that carries no model pool.

    ``model="probe_model"`` is passed to every one of them, including the kinds that have no
    model: a ``pool`` element refuses a missing ``model:`` *before* it looks at the pool, so
    without a name the four pool implementations would all fail this test for the wrong
    reason and the invariant under test would never be reached.
    """
    element = create_element(kind, impl, "probe", model="probe_model")
    element.open(ElementContext())
    return element


@pytest.mark.parametrize(
    ("kind", "impl"),
    EVERY_IMPLEMENTATION,
    ids=[f"{k.value}:{i}" for k, i in EVERY_IMPLEMENTATION],
)
def test_needs_model_predicts_whether_open_demands_a_pool(kind: ElementKind, impl: str) -> None:
    """The no-drift invariant, in both directions.

    ``True`` means ``open()`` must refuse an :class:`ElementContext` with no ``models=``, and
    refuse it *there* rather than on the first frame -- a chain that starts and then fails
    per-frame is the failure the eager resolve in ``elements/pool.py`` exists to convert into
    a start-up refusal. ``False`` means it must **never refuse for want of a pool** -- because
    ``shipinfer run`` builds none for a chain whose elements all answer ``False``.

    "Never for want of a pool" and not "always opens", and the difference is the whole reason
    this branch is written the way it is. An element may legitimately refuse an empty context
    for a reason of its own: ``track: {impl: shipvision}`` needs ``3rdparty/shipvision``, which
    CI deliberately does not check out, and ``elements/__init__.py`` states that a host lacking
    an element's runtime should still *list* the implementation and fail at ``open()`` naming
    the package to install. Asserting ``is_open`` unconditionally would make this file demand
    the opposite of that -- and would go red on the one checkout the offline tier exists to
    protect. So the refusal is allowed and then *read*: it must not be the pool's.

    :class:`~shipinfer.core.errors.BackendUnavailableError` is allowed for the same reason and
    read the same way. It is the typed refusal for "the runtime is not installed on this host"
    (``output: {impl: kafka}`` without ``confluent_kafka`` is the one that raises it today),
    and by construction it is never about a model pool -- which is the only thing this test
    is entitled to have an opinion about.
    """
    declared = ELEMENTS[kind].get(impl).needs_model
    element = None
    try:
        if declared:
            with pytest.raises(ConfigurationError, match="needs a model pool"):
                _open_with_no_pool(kind, impl)
            return
        try:
            element = _open_with_no_pool(kind, impl)
        except BackendUnavailableError:
            return
        except ConfigurationError as exc:
            assert "model pool" not in str(exc), (
                f"{kind.value}:{impl} declares needs_model=False and refused an empty "
                f"context for want of one: {exc}"
            )
            return
        assert element.is_open, f"{kind.value}:{impl} declares needs_model=False"
    finally:
        if element is not None and element.is_open:
            element.close()


def test_the_walk_covers_the_pool_implementations_and_the_mocks() -> None:
    """A guard on the guard: an empty or lopsided parametrisation would pass silently.

    The registries are populated by import side effect, so a refactor that moved
    ``elements/pool.py`` out of ``elements/__init__.py`` would leave this file green over a
    registry that no longer contains the class it exists to check.
    """
    declared_true = {
        (kind.value, impl)
        for kind, impl in EVERY_IMPLEMENTATION
        if ELEMENTS[kind].get(impl).needs_model
    }

    assert declared_true == {
        ("detect", "pool"),
        ("segment", "pool"),
        ("embed", "pool"),
        ("recognize", "pool"),
    }, "the four model kinds' `pool` implementations are the only elements that resolve one"
    assert len(EVERY_IMPLEMENTATION) > len(declared_true), "no `needs_model=False` element ran"
