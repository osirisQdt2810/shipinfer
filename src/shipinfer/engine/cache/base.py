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

__all__ = ["ResponseCache", "cache_key", "freeze_outputs"]


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

    def key_for(self, model: str, version: int, inputs: Mapping[str, Tensor]) -> str | None:
        """The key these inputs would be stored under, or ``None`` if they cannot be.

        Hashing lives behind the cache object rather than at the call site so the disabled
        case costs one virtual call instead of a BLAKE2b pass over every input byte. That
        pass is by far the expensive half of a lookup, and a model with caching off must
        not pay it — see :class:`~shipinfer.engine.cache.null.NullResponseCache`.
        """
        try:
            return cache_key(model, version, inputs)
        except RuntimeError:
            # Device-resident inputs: the D2H copy needed to hash them would cost more
            # than the hit could ever save.
            return None

    @abc.abstractmethod
    def get(self, key: str) -> dict[str, Tensor] | None:
        """The stored outputs, or ``None`` on a miss.

        The returned mapping is fresh but the tensors inside it are shared with every
        other hit on this key and sealed against writes by :func:`freeze_outputs`. Callers
        treat a cached response as read-only; one that needs to mutate makes its own copy.
        """

    @abc.abstractmethod
    def put(self, key: str, outputs: Mapping[str, Tensor]) -> None:
        """Store the outputs of a completed request.

        Implementations must run the outputs through :func:`freeze_outputs` — that is what
        makes the read-only contract on :meth:`get` true rather than merely documented.

        Only ever called for a request that succeeded, with host-resident outputs. A
        cached exception would be served forever, and device memory is not cacheable.
        """

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


def freeze_outputs(outputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Detach outputs from the batch they were scattered out of, then seal them.

    Two failure modes, one fix. A scattered output is a *view* into the whole batch's
    array (:meth:`~shipinfer.core.types.Tensor.slice_batch`), so storing it as-is would
    pin an entire batch in memory behind one small entry and make the cache's byte
    accounting a fiction. And a hit hands the same object to every later caller, so one
    caller writing into it silently corrupts every subsequent hit.

    So the bytes are copied once on the way in and the copy is marked non-writeable: a
    caller that mutates a cached response gets a ``ValueError`` from numpy at the line
    with the bug, instead of a wrong answer somewhere else an hour later.

    Args:
        outputs: host-resident tensors. Filtering device-resident ones out is the caller's
            job — reading one here would mean a D2H copy inside the cache.

    Raises:
        RuntimeError: if a tensor is device-resident, from ``Tensor.numpy()``.
    """
    sealed: dict[str, Tensor] = {}
    for name, tensor in outputs.items():
        array = tensor.numpy().copy()
        array.flags.writeable = False
        sealed[name] = Tensor.from_numpy(array)
    return sealed
