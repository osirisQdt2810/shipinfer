"""The default: no caching."""

from __future__ import annotations

from collections.abc import Mapping

from shipinfer.core.types import Tensor
from shipinfer.engine.cache.base import ResponseCache
from shipinfer.engine.cache.registry import RESPONSE_CACHES

__all__ = ["NullResponseCache"]


@RESPONSE_CACHES.register("null", "none", "off")
class NullResponseCache(ResponseCache):
    """Never hits, never stores.

    A real object rather than an ``if cache is not None`` at three call sites: the branch
    would be in the hot path, and this way the disabled case costs one virtual call that
    returns ``None``.
    """

    name = "null"

    def key_for(self, model: str, version: int, inputs: Mapping[str, Tensor]) -> None:
        """Never hash.

        This override is the point of the null object. The base implementation runs
        BLAKE2b over every input byte, and a model with caching off — which is every model
        by default — must not pay that on each of the ~1000 requests a second the server
        accepts. Returning ``None`` here tells :class:`~shipinfer.engine.model.Model` the
        request is not cacheable, so neither :meth:`get` nor :meth:`put` is reached.
        """
        return

    def get(self, key: str) -> dict[str, Tensor] | None:
        return None

    def put(self, key: str, outputs: Mapping[str, Tensor]) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": 0, "entries": 0}
