"""Placement targets: ``cpu`` and ``cuda:N``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["Device", "DeviceKind"]

DeviceKind = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True, order=True)
class Device:
    """Where a tensor lives or an instance runs.

    Frozen and hashable so it can key a dict of per-device pools, and ordered so device
    lists sort predictably in logs.
    """

    kind: DeviceKind = "cpu"
    index: int = 0

    @classmethod
    def cpu(cls) -> Device:
        return cls("cpu", 0)

    @classmethod
    def cuda(cls, index: int) -> Device:
        if index < 0:
            raise ValueError(f"cuda device index must be >= 0, got {index}")
        return cls("cuda", index)

    @classmethod
    def parse(cls, text: str) -> Device:
        """Parse ``"cpu"``, ``"cuda"`` (index 0) or ``"cuda:N"``."""
        kind, _, idx = text.strip().partition(":")
        kind = kind.lower()
        if kind == "cpu":
            return cls.cpu()
        if kind == "cuda":
            return cls.cuda(int(idx) if idx else 0)
        raise ValueError(f"unknown device string: {text!r}")

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def __str__(self) -> str:
        return self.kind if self.kind == "cpu" else f"cuda:{self.index}"
