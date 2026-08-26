"""Loader for the fused-kernel extension, ``shipvision._C``.

The split this module guards is the project's central performance decision: **Python owns
the control plane, C++/CUDA owns the data plane** (ADR-007). Configuration, lifecycle,
repository scanning and the HTTP surface are Python, because they run once and clarity is
worth more than nanoseconds. The pre/post-processing kernels run 15 000 times a second, and
those live in `shipvision <https://github.com/osirisQdt2810/shipvision>`_ — a separate
repository, vendored here as the submodule ``3rdparty/shipvision``. The four per-module
repositories that preceded it (imgproc, mot, reid, mtmc) were merged into it; this loader
still named the old one, so ``provider: native`` could never resolve and the kernel
boundary was dead code (ADR-012 supersedes ADR-010).

It is separate because it is a *library*: a fused preprocessing kernel is useful to anything
that feeds a vision model, not only to this server. It has its own CI, its own release
cadence and its own reviewer, and this project pins a commit of it like any dependency.

The extension is optional. Every native component has a Python counterpart implementing
the same contract, so a machine without a build still runs — just slower. Which one is
used is decided by ``execution.provider``:

* ``auto``   — native if importable, else Python (the default)
* ``native`` — native or refuse to start, so a production deploy cannot silently regress
* ``python`` — Python always, which is how the parity tests pin both sides
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.core.settings import ExecutionProvider

__all__ = [
    "is_native_available",
    "native_module",
    "native_version",
    "require_native",
    "resolve_provider",
]

_LOG = get_logger("runtime.native")


@functools.lru_cache(maxsize=1)
def native_module() -> ModuleType | None:
    """Import ``shipvision._C`` once, or ``None`` if its kernels are unavailable.

    Two ways to be unavailable and both are ordinary: the package is not installed, or it
    is installed but its extension was never built on this host. The distinction is logged
    because the fixes differ — ``pip install -e 3rdparty/shipvision`` against
    ``python 3rdparty/shipvision/build.py``.

    Cached because a failed import is not free and this is asked on every component
    construction.
    """
    try:
        # `import_module("shipvision._C")`, not `import shipvision` followed by
        # `shipvision._C`. The second only works when something has *already* imported the
        # submodule, because a submodule becomes an attribute of its package as a side effect
        # of being imported — and `shipvision/__init__.py` deliberately imports nothing eagerly
        # so that `import shipvision` stays free. So the kernels were reachable only when some
        # unrelated import happened to have pulled them in first, and reported "not installed"
        # otherwise, on a machine where they were built and working.
        kernels = importlib.import_module("shipvision._C")
    except ImportError as exc:
        # The package may be absent, or present as pure Python with no build. Both are
        # ordinary, and both arrive here as ImportError.
        _LOG.debug("shipvision kernels are not installed: %s", exc)
        return None

    if not _reports_devices(kernels):
        _LOG.debug(
            "shipvision kernels are installed but report no usable device "
            "(platform=%s); build them with `python 3rdparty/shipvision/build.py`",
            _describe(kernels, "platform"),
        )
        return None

    _LOG.info(
        # `__version__`, not `version` — the extension has only ever set the attribute. The
        # banner degraded to "shipvision ?" because `_describe` swallowed the AttributeError,
        # which is exactly why fixing `native_version()` alone left this behind: a guard that
        # turns a wrong name into a question mark hides the wrong name.
        "fused kernels loaded: shipvision %s (%s, %s device(s))",
        _version_of(kernels),
        _describe(kernels, "platform"),
        _describe(kernels, "device_count"),
    )
    module: ModuleType = kernels
    return module


def _version_of(module: ModuleType) -> str:
    """The extension's version string, or ``"?"``.

    A free function taking the module rather than a call to :func:`native_version`, because
    the banner runs *inside* ``native_module()`` and that is `lru_cache`d — the cache is
    populated only after the call returns, so re-entering it would recurse until the stack
    ran out. Cheap to write the other way and impossible to get wrong.
    """
    version = getattr(module, "__version__", None)
    return "?" if version is None else str(version)


def _describe(module: ModuleType, name: str) -> str:
    """One of the extension's introspection calls, or ``"?"``.

    Guarded because these are the extension's own API and an older build may not carry all
    of them — and a loader that raises while explaining why it could not load is worse than
    one that says less.
    """
    probe = getattr(module, name, None)
    if probe is not None and not callable(probe):
        # `platform` is a string attribute, not a call. Calling it raised TypeError, which the
        # except below turned into "?" — the same trap as `version()`: a guard that converts a
        # wrong access into a question mark hides the wrong access.
        return str(probe)
    if probe is None:
        return "?"
    try:
        return str(probe())
    except Exception:  # pragma: no cover - diagnostics must not raise
        return "?"


def _reports_devices(module: ModuleType) -> bool:
    """Does this build have GPU kernels and a device to run them on?

    ``cuda_available`` is the name the extension binds. It used to read ``is_available``, which
    it never defined — and because a missing probe was treated as "assume usable" here, that
    disagreement produced no error at all on this side. It produced one two files away, in
    `runtime/ops/native_ops.py`, where the same wrong name was called directly.

    A build with no probe is *not* usable: `NativeImageOps` refuses it, so saying "usable" here
    would have the banner, `health()` and `resolve_provider(AUTO)` all report the fast path
    while every instance construction failed over to torch — the regression the provider
    machinery exists to prevent. One answer, given once.
    """
    probe = getattr(module, "cuda_available", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # pragma: no cover
        return False


def is_native_available() -> bool:
    return native_module() is not None


def native_version() -> str | None:
    """The extension's version, or None when there is no extension.

    `__version__`, not `version()`. The extension has only ever set the attribute
    (`csrc/bindings/module.cpp` — `m.attr("__version__")`), and the call was unreachable for
    as long as nothing in `shipinfer` put `shipvision._C` into `sys.modules`. The moment the
    tracking stage imported `shipvision.tracking` at module scope, it became reachable and
    took out `InferenceServer.start()` on every host where the submodule is built — which is
    the production container, and not CI, because CI deliberately does not check the
    submodule out (ADR-001).

    `getattr` with a default rather than a bare attribute read: a version is a diagnostic, and
    an extension that does not announce one is a fact to report, not a reason to refuse to
    start.
    """
    module = native_module()
    if module is None:
        return None
    version = _version_of(module)
    return None if version == "?" else version


def require_native() -> ModuleType:
    """The extension, or a message that says how to get it.

    Raises:
        ConfigurationError: naming the build command, because "ModuleNotFoundError:
            shipinfer._C" on its own tells an operator nothing actionable.
    """
    module = native_module()
    if module is None:
        raise ConfigurationError(
            "the fused kernels are unavailable. Fetch and build the submodule:\n"
            "    git submodule update --init 3rdparty/shipvision\n"
            "    pip install -e 3rdparty/shipvision\n"
            "    python 3rdparty/shipvision/build.py\n"
            "(needs CMake >= 3.22 and the CUDA toolkit, or --hip for ROCm). "
            "Or set execution.provider=python to run the pure-Python data plane."
        )
    return module


def resolve_provider(requested: ExecutionProvider) -> ExecutionProvider:
    """Turn ``auto`` into a concrete choice, and validate an explicit one.

    Raises:
        ConfigurationError: if ``native`` was demanded and is not available. Failing at
            start-up is the point: a deployment that asked for the fast path and silently
            got the slow one is a performance regression nobody will notice for a month.
    """
    if requested is ExecutionProvider.PYTHON:
        return ExecutionProvider.PYTHON
    if requested is ExecutionProvider.NATIVE:
        require_native()
        return ExecutionProvider.NATIVE
    return ExecutionProvider.NATIVE if is_native_available() else ExecutionProvider.PYTHON
