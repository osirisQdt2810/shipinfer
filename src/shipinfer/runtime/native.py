"""Loader for the fused-kernel extension, ``shipinfer_imgproc``.

The split this module guards is the project's central performance decision: **Python owns
the control plane, C++/CUDA owns the data plane** (ADR-007). Configuration, lifecycle,
repository scanning and the HTTP surface are Python, because they run once and clarity is
worth more than nanoseconds. The pre/post-processing kernels run 15 000 times a second, and
those live in `shipinfer-imgproc <https://github.com/osirisQdt2810/shipinfer-imgproc>`_ —
a separate repository, vendored here as the submodule ``3rdparty/shipinfer-imgproc``.

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
    """Import ``shipinfer_imgproc`` once, or ``None`` if its kernels are unavailable.

    Two ways to be unavailable and both are ordinary: the package is not installed, or it
    is installed but its extension was never built on this host. The distinction is logged
    because the fixes differ — ``pip install -e 3rdparty/shipinfer-imgproc`` against
    ``python 3rdparty/shipinfer-imgproc/build.py``.

    Cached because a failed import is not free and this is asked on every component
    construction.
    """
    try:
        import shipinfer_imgproc
    except ImportError as exc:
        _LOG.debug("shipinfer-imgproc is not installed: %s", exc)
        return None

    if not shipinfer_imgproc.is_available():
        _LOG.debug(
            "shipinfer-imgproc %s is installed but has no usable kernels "
            "(platform=%s, devices=%d); build them with "
            "`python 3rdparty/shipinfer-imgproc/build.py`",
            shipinfer_imgproc.version(),
            shipinfer_imgproc.platform(),
            shipinfer_imgproc.device_count(),
        )
        return None

    _LOG.info(
        "fused kernels loaded: shipinfer-imgproc %s (%s, %d device(s))",
        shipinfer_imgproc.version(),
        shipinfer_imgproc.platform(),
        shipinfer_imgproc.device_count(),
    )
    module: ModuleType = shipinfer_imgproc
    return module


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
            "    git submodule update --init 3rdparty/shipinfer-imgproc\n"
            "    pip install -e 3rdparty/shipinfer-imgproc\n"
            "    python 3rdparty/shipinfer-imgproc/build.py\n"
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
