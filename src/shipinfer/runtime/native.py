"""Loader for the compiled data-plane extension, ``shipinfer._C``.

The split this module guards is the project's central performance decision: **Python owns
the control plane, C++/CUDA owns the data plane** (ADR-007). Configuration, lifecycle,
repository scanning and the HTTP surface are Python, because they run once and clarity is
worth more than nanoseconds. The queue, the batch staging copy and the pre/post-processing
kernels run 15 000 times a second, and those live in ``native/`` behind pybind11.

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
    """Import ``shipinfer._C`` once, or return ``None`` if it was never built.

    Cached because a failed import is not free and this is asked on every component
    construction.
    """
    try:
        from shipinfer import _C  # type: ignore[attr-defined]
    except ImportError as exc:
        _LOG.debug("native extension unavailable: %s", exc)
        return None
    native: ModuleType = _C
    _LOG.info(
        "native extension loaded: shipinfer._C %s (cuda=%s)",
        getattr(native, "__version__", "?"),
        getattr(native, "cuda_available", lambda: False)(),
    )
    return native


def is_native_available() -> bool:
    return native_module() is not None


def native_version() -> str | None:
    module = native_module()
    return getattr(module, "__version__", None) if module else None


def require_native() -> ModuleType:
    """The extension, or a message that says how to get it.

    Raises:
        ConfigurationError: naming the build command, because "ModuleNotFoundError:
            shipinfer._C" on its own tells an operator nothing actionable.
    """
    module = native_module()
    if module is None:
        raise ConfigurationError(
            "the native extension shipinfer._C is not built. Build it with "
            "`python scripts/build_native.py` (needs CMake >= 3.22 and, for GPU kernels, "
            "the CUDA toolkit or ROCm), or set execution.provider=python to run the "
            "pure-Python data plane."
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
