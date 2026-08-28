"""The one place ``topology`` names ``shipvision``, and it names it inside a function.

``3rdparty/shipvision`` is a submodule (ADR-010) and **CI deliberately does not check it
out**, which is how the promise that a machine with no build still runs stays true
(``.claude/CLAUDE.md``). Every element that runs a tracking, cross-camera or re-identification
algorithm needs it; none of them may pay for it at import time, because
:meth:`~shipinfer.topology.chain.Topology.from_spec` instantiates every element in a chain to
read its declared caps, and that has to work on a laptop.

So the import is **function-scope, here, and nowhere else**. Three things fall out of that,
and each one is the reason this module exists rather than a ``try``/``except`` at the top of
every element module:

1. **One message, not five.** A host without the submodule gets the same sentence — with the
   command that fixes it — whether the missing piece is the tracker, the MTMC clusterer or
   the gallery. The wording is the one ``pipeline/graph/tracking.py`` already uses, because an
   operator who has seen it once should recognise it.
2. **One thing to grep.** "Where does the server touch shipvision?" has a file for an answer.
   The module-scope ``try: from shipvision.mot import ...`` style used in ``pipeline/`` works,
   but it puts a name-per-symbol rebinding block at the top of each module and makes the
   answer a search.
3. **The laziness is checkable.** ``scripts/hooks/check_layers.py`` walks the AST and counts a
   function-scope import exactly like a module-scope one, so a ``FORBIDDEN_EXTERNAL`` row
   naming ``shipvision`` would ban the very thing this design requires. The half that *can*
   tell them apart is the subprocess assertion in ``tests/test_architecture.py``: it imports
   ``shipinfer.topology`` and refuses the run if ``shipvision`` came with it. That test is the
   enforcement point for this module, and it is why the loaders below must never be called at
   import time.

Registration stays eager, as ``topology/elements/__init__.py`` requires: an element module
imports this one at module scope (it is pure), lists itself in a registry, and calls a loader
from inside ``_do_open``. A host that lacks the submodule therefore still *lists* the
implementation and fails at ``open()`` naming the fix, rather than at load with "unknown
element" — two different problems with two different fixes.
"""

from __future__ import annotations

import functools
from types import ModuleType

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "load_errors",
    "load_mot",
    "load_mtmc",
    "load_reid",
    "load_types",
    "shipvision_available",
]

#: How to get the submodule, verbatim from ``pipeline/graph/tracking.py`` — the sentence an
#: operator may already have seen once. A refusal that does not carry the command is a
#: refusal an operator has to go and look up.
_FIX = (
    "Check out the submodule and install it — "
    "`git submodule update --init 3rdparty/shipvision && "
    "pip install -e 3rdparty/shipvision`"
)


def _unavailable(subpackage: str, exc: BaseException) -> ConfigurationError:
    """The one refusal, so all four loaders say the same thing.

    Args:
        subpackage: the dotted name that could not be imported. Named explicitly because
            ``shipvision`` present but ``shipvision.reid`` missing is a *different* problem
            from no submodule at all — a stale checkout rather than an absent one — and the
            operator cannot tell them apart from "shipvision is missing".
        exc: the original :class:`ImportError`, interpolated rather than swallowed. It is what
            distinguishes "no such module" from "the compiled extension is for another Python".

    Returns:
        The error, for the caller to ``raise ... from exc``. Returned rather than raised so the
        exception chain is built at the site that has the context, which is what puts the
        original traceback under the message.
    """
    return ConfigurationError(f"{subpackage} cannot be imported ({exc}). {_FIX}")


@functools.cache
def load_mot() -> ModuleType:
    """``shipvision.mot`` — the multi-object trackers and their association helpers.

    Memoised with :func:`functools.cache`, which caches the *success* and not the failure:
    a host that installs the submodule while the process is running gets the module on the
    next call rather than a cached refusal, and a host that never does pays one failed import
    per call instead of holding a stale exception. ``load_mot.cache_clear()`` is how a test
    resets it.

    Raises:
        ConfigurationError: the submodule is not checked out or not installed. The message
            carries the command that fixes it.
    """
    try:
        from shipvision import mot
    except ImportError as exc:
        raise _unavailable("shipvision.mot", exc) from exc
    return mot


@functools.cache
def load_mtmc() -> ModuleType:
    """``shipvision.mtmc`` — cross-camera association over per-camera tracklets.

    Raises:
        ConfigurationError: as :func:`load_mot`.
    """
    try:
        from shipvision import mtmc
    except ImportError as exc:
        raise _unavailable("shipvision.mtmc", exc) from exc
    return mtmc


@functools.cache
def load_reid() -> ModuleType:
    """``shipvision.reid`` — the bounded galleries behind an identity query.

    Raises:
        ConfigurationError: as :func:`load_mot`.
    """
    try:
        from shipvision import reid
    except ImportError as exc:
        raise _unavailable("shipvision.reid", exc) from exc
    return reid


@functools.cache
def load_errors() -> ModuleType:
    """``shipvision.errors`` — the library's own typed refusals.

    Needed by an element that has to **catch** one rather than let it fail a frame. The
    standing case is the cross-camera tier: ``GlobalIdAssigner`` raises ``TrackingError`` for a
    track with no appearance vector, which is a per-frame data condition (an embedder that was
    spilled, a crop that produced nothing) and not a fault the frame should die of. Catching it
    by name needs the class, and the class needs the submodule, so it comes through the same
    door as everything else.

    ``shipvision.errors`` is a leaf module that imports nothing of the library, so this is the
    cheapest of the loaders — but it is still function-scope, for the reason the module
    docstring gives.

    Raises:
        ConfigurationError: as :func:`load_mot`.
    """
    try:
        from shipvision import errors
    except ImportError as exc:
        raise _unavailable("shipvision.errors", exc) from exc
    return errors


@functools.cache
def load_types() -> ModuleType:
    """``shipvision.types`` — ``Detection``, ``Detections``, ``FrameTag`` and ``iou_matrix``.

    The vocabulary the other three speak, so an element that builds a ``Detections`` to hand
    to a tracker needs this one as well as :func:`load_mot`.

    Raises:
        ConfigurationError: as :func:`load_mot`.
    """
    try:
        from shipvision import types
    except ImportError as exc:
        raise _unavailable("shipvision.types", exc) from exc
    return types


def shipvision_available() -> bool:
    """Whether ``shipvision`` can be imported on this host at all.

    For the callers that need to *decide* rather than to fail: a health report, a
    ``--list`` that marks an implementation unusable, a test that skips. An element must
    **not** use this to fall back silently — a chain that names ``impl: shipvision`` and
    quietly ran without a tracker is exactly the empty-result-means-failure this codebase
    refuses (ADR-005). Call the loader and let it raise.

    Asks for :func:`load_types`, which every element needs and which imports the top-level
    package on the way, so a ``True`` here still leaves a specific subpackage free to fail
    with its own message — a checkout stale enough to be missing ``shipvision.reid`` is a
    real state and reporting it as "shipvision is fine" would be worse than useless.
    """
    try:
        load_types()
    except ConfigurationError:
        return False
    return True
