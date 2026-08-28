"""Low-level driver providers — the substrate for the ``custom`` reference implementations.

**Not the default path.** :mod:`shipinfer.runtime.platform` uses ``torch.cuda`` for
everything the server actually does. This package exists so the hand-written allocator and
graph cache have something to stand on, and so there is a readable place to see what a
driver call actually looks like (ADR-003).

Probe order is by ``priority``: cuda-python, then torch, then null.
``SHIPINFER_CUDA_PROVIDER`` forces one.
"""

from __future__ import annotations

import os
import threading

from shipinfer.core.errors import DeviceError
from shipinfer.core.logging import LOG
from shipinfer.runtime.providers.base import (
    CudaProvider,
    DevicePtr,
    HostPtr,
    RawDeviceProperties,
    StreamHandle,
)
from shipinfer.runtime.providers.cuda_python import CudaPythonProvider
from shipinfer.runtime.providers.null import NullCudaProvider
from shipinfer.runtime.providers.registry import PROVIDERS
from shipinfer.runtime.providers.torch_provider import TorchCudaProvider

__all__ = [
    "PROVIDERS",
    "PROVIDER_ENV",
    "CudaProvider",
    "CudaPythonProvider",
    "DevicePtr",
    "HostPtr",
    "NullCudaProvider",
    "RawDeviceProperties",
    "StreamHandle",
    "TorchCudaProvider",
    "get_cuda_provider",
    "reset_cuda_provider",
]

PROVIDER_ENV = "SHIPINFER_CUDA_PROVIDER"

_provider: CudaProvider | None = None
_lock = threading.Lock()


def get_cuda_provider() -> CudaProvider:
    """The process-wide low-level provider, probed once and cached."""
    global _provider
    if _provider is not None:
        return _provider
    with _lock:
        if _provider is None:
            _provider = _select()
            LOG.debug("low-level CUDA provider: %s", _provider.describe())
    return _provider


def _select() -> CudaProvider:
    forced = os.environ.get(PROVIDER_ENV, "").strip().lower()
    if forced:
        provider = PROVIDERS.get(forced).probe()
        if provider is None:
            raise DeviceError(f"{PROVIDER_ENV}={forced!r} but that provider is unusable here")
        return provider
    for entry in sorted(PROVIDERS, key=lambda e: e.resolve().priority):
        provider = entry.resolve().probe()
        if provider is not None and provider.device_count() > 0:
            return provider
    return NullCudaProvider()


def reset_cuda_provider() -> None:
    """Forget the cached provider. For tests that flip ``SHIPINFER_CUDA_PROVIDER``."""
    global _provider
    with _lock:
        _provider = None
