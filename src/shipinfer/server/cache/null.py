"""The default: no caching."""

from __future__ import annotations

from collections.abc import Mapping

from shipinfer.core.types import Tensor
from shipinfer.server.cache.base import ResponseCache
from shipinfer.server.cache.registry import RESPONSE_CACHES

__all__ = ["NullResponseCache"]


@RESPONSE_CACHES.register("null", "none", "off")
class NullResponseCache(ResponseCache):
    """Never hits, never stores.

    A real object rather than an ``if cache is not None`` at three call sites: the branch
    would be in the hot path, and this way the disabled case costs one virtual call that
    returns ``None``.
    """

    name = "null"

    def get(self, key: str) -> dict[str, Tensor] | None:
        return None

    def put(self, key: str, outputs: Mapping[str, Tensor]) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": 0, "entries": 0}
