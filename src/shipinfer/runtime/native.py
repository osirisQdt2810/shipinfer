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
        import shipvision

        kernels = shipvision._C
    except (ImportError, AttributeError) as exc:
        # AttributeError too: `shipvision` imports fine as pure Python while its compiled
        # extension is absent, which is the normal state on a machine with no build.
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
        "fused kernels loaded: shipvision %s (%s, %s device(s))",
        _describe(kernels, "version"),
        _describe(kernels, "platform"),
        _describe(kernels, "device_count"),
    )
    module: ModuleType = kernels
    return module


def _describe(module: ModuleType, name: str) -> str:
    """One of the extension's introspection calls, or ``"?"``.

    Guarded because these are the extension's own API and an older build may not carry all
    of them — and a loader that raises while explaining why it could not load is worse than
    one that says less.
    """
    probe = getattr(module, name, None)
    if probe is None:
        return "?"
    try:
        return str(probe())
    except Exception:  # pragma: no cover - diagnostics must not raise
        return "?"


def _reports_devices(module: ModuleType) -> bool:
    probe = getattr(module, "is_available", None)
    if probe is None:
        return True  # an extension that does not say assumes usable; ops will tell us
    try:
        return bool(probe())
    except Exception:  # pragma: no cover
        return False


def is_native_available() -> bool:
    return native_module() is not None


def native_version() -> str | None:
    module = native_module()
    return module.version() if module is not None else None


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
