"""The graph-cache contract."""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from shipinfer.core.types import Device
from shipinfer.runtime.stream import Stream

__all__ = ["CapturedGraph", "GraphCache"]


class CapturedGraph(abc.ABC):
    """One captured graph, replayable against fixed buffers.

    The buffers are the contract. Replay takes no arguments: it re-executes the exact
    kernels that were recorded, against the exact addresses that were recorded. A caller
    copies inputs *into* :attr:`static_inputs`, calls :meth:`replay`, reads
    :attr:`static_outputs`.
    """

    batch_size: int
    static_inputs: dict[str, Any]
    static_outputs: dict[str, Any]

    @abc.abstractmethod
    def replay(self, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def replays(self) -> int: ...


class GraphCache(abc.ABC):
    """Per-instance cache of captured graphs, keyed on batch size.

    Owns the *failure* policy as well as the cache. "We tried to capture and it did not
    work" has to be remembered, or an uncapturable model pays a failed capture attempt on
    every batch — overhead added to the path it was meant to accelerate.
    """

    name: ClassVar[str] = "abstract"

    def __init__(
        self,
        device: Device,
        *,
        enabled: bool = True,
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        max_failures: int = 3,
    ) -> None:
        self._device = device
        self._batch_sizes = frozenset(batch_sizes)
        self._max_failures = max_failures
        self._failures = 0

    @property
    @abc.abstractmethod
    def enabled(self) -> bool: ...

    @abc.abstractmethod
    def get(self, batch_size: int) -> CapturedGraph | None: ...

    @abc.abstractmethod
    def capture(
        self,
        batch_size: int,
        stream: Stream,
        static_inputs: Mapping[str, Any],
        run: Callable[[], Mapping[str, Any]],
    ) -> CapturedGraph | None:
        """Capture ``run`` for ``batch_size``.

        Returns ``None`` when capture is disabled or failed. That is not an error: it means
        "take the ordinary launch path", which is slower and equally correct.
        """

    def should_capture(self, batch_size: int) -> bool:
        return self.enabled and batch_size in self._batch_sizes and self.get(batch_size) is None

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def stats(self) -> dict[str, int]: ...
