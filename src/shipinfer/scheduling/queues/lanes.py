"""The per-key FIFO lane used by the fair queue."""

from __future__ import annotations

from collections import deque

from shipinfer.scheduling.work import WorkItem

__all__ = ["Lane"]


class Lane:
    """One priority level: per-key FIFOs plus a round-robin cursor over active keys.

    Both :meth:`push` and :meth:`pop` are O(1). The obvious alternative — scan every camera
    on each tick and take one — is O(cameras) per request, which at 50 cameras and 15 000
    requests/s is exactly the kind of quiet waste this project exists to remove.
    """

    __slots__ = ("by_key", "order", "size")

    def __init__(self) -> None:
        self.by_key: dict[str, deque[WorkItem]] = {}
        self.order: deque[str] = deque()
        self.size = 0

    def push(self, item: WorkItem) -> None:
        key = item.fairness_key
        bucket = self.by_key.get(key)
        if bucket is None:
            bucket = self.by_key[key] = deque()
            self.order.append(key)
        bucket.append(item)
        self.size += 1

    def pop(self) -> WorkItem:
        key = self.order.popleft()
        bucket = self.by_key[key]
        item = bucket.popleft()
        if bucket:
            self.order.append(key)  # this key goes to the back: round-robin
        else:
            del self.by_key[key]
        self.size -= 1
        return item

    def peek(self) -> WorkItem | None:
        """The item :meth:`pop` would return, without removing it.

        Needed because a batch is bounded in *rows* and an item carries however many rows
        its request does, so the drain has to see an item's size before committing to it —
        popping first and pushing back would send it to the back of its own key's FIFO and
        reorder a camera's frames.
        """
        if not self.order:
            return None
        return self.by_key[self.order[0]][0]

    def evict_from_longest(self) -> WorkItem | None:
        """Drop the oldest request of whichever key is hogging this lane.

        Deliberately *not* "the globally oldest". The request that has waited longest is
        usually the victim of a flood, not its cause; penalising the loudest camera is what
        keeps a quiet camera's frames alive. Evicting the global oldest is precisely the
        behaviour that made the previous system silently lose the quiet cameras.
        """
        if not self.by_key:
            return None
        key = max(self.by_key, key=lambda k: len(self.by_key[k]))
        bucket = self.by_key[key]
        item = bucket.popleft()
        self.size -= 1
        if not bucket:
            del self.by_key[key]
            self.order.remove(key)
        return item

    def drain(self) -> list[WorkItem]:
        items = [item for bucket in self.by_key.values() for item in bucket]
        self.by_key.clear()
        self.order.clear()
        self.size = 0
        return items

    def __len__(self) -> int:
        return self.size
