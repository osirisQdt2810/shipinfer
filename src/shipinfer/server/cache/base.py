"""The response-cache contract.

Triton calls this the *response cache*; vLLM's prefix cache is the same idea applied to
prompts. Both exist because identical work is common in real traffic and re-doing it on a
GPU is the most expensive way to get an answer you already had.

In this pipeline the hit comes from a specific place: a static camera looking at an
unchanged scene produces byte-identical crops frame after frame. Re-embedding a parked ship
sixty times a minute is pure waste, and it competes for the same GPU as the moving one.

Caching is **off by default** and opt-in per model, because it is only sound for a
*deterministic, stateless* model. Enabling it on a stateful or stochastic one returns a
stale answer with full confidence, which is far worse than being slow.
"""

from __future__ import annotations

import abc
import hashlib
from collections.abc import Mapping
from typing import ClassVar

from shipinfer.core.types import Tensor

__all__ = ["ResponseCache", "cache_key"]


def cache_key(model: str, version: int, inputs: Mapping[str, Tensor]) -> str:
    """A content hash over the model identity and every input byte.

    BLAKE2b at 16 bytes: cryptographically strong, faster than SHA-256, and short enough to
    keep the key table small. Hashing the *bytes* rather than an id means two separately
    produced but identical crops share a hit, which is exactly the case worth catching.

    Device-resident tensors are unhashable here by design — reading them back would mean a
    D2H copy on the way *into* a cache meant to save work.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(model.encode())
    digest.update(version.to_bytes(4, "little"))
    for name in sorted(inputs):
        tensor = inputs[name]
        digest.update(name.encode())
        digest.update(tensor.dtype.value.encode())
        digest.update(repr(tensor.shape).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ResponseCache(abc.ABC):
    """Maps an input hash to a previously computed output map."""

    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def get(self, key: str) -> dict[str, Tensor] | None: ...

    @abc.abstractmethod
    def put(self, key: str, outputs: Mapping[str, Tensor]) -> None: ...

    @abc.abstractmethod
    def stats(self) -> dict[str, int]: ...

    def clear(self) -> None:
        """Drop everything. Called when a model is reloaded — a cached response from the
        previous version of a model is not a hit, it is a wrong answer."""

    @property
    def hit_rate(self) -> float:
        stats = self.stats()
        total = stats.get("hits", 0) + stats.get("misses", 0)
        return stats.get("hits", 0) / total if total else 0.0
