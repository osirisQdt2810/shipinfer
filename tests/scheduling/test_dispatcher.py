"""Dispatch and spillover — turning a policy's guess into a delivery guarantee."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from shipinfer.core.errors import QueueFullError, ServerStateError
from shipinfer.core.request import InferenceRequest, ResponseFuture
from shipinfer.core.types import Device, Tensor
from shipinfer.scheduling.dispatcher import Dispatcher
from shipinfer.scheduling.policies import JoinShortestQueuePolicy, RoundRobinPolicy
from shipinfer.scheduling.work import WorkItem


@dataclass
class FakeInstance:
    device: Device
    capacity: int = 2
    is_ready: bool = True
    ewma_latency_us: float = 0.0
    accepted: list = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.accepted)

    def enqueue(self, item: WorkItem) -> None:
        if len(self.accepted) >= self.capacity:
            raise QueueFullError(str(self.device), len(self.accepted), self.capacity)
        self.accepted.append(item)


def _item() -> WorkItem:
    request = InferenceRequest(
        model_name="m", inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))}
    )
    return WorkItem(request, ResponseFuture(request))


def _enqueue(instance, item):
    instance.enqueue(item)


def test_dispatch_places_on_the_policy_choice() -> None:
    a = FakeInstance(Device.cuda(0), capacity=4)
    b = FakeInstance(Device.cuda(1), capacity=4)
    a.accepted.append(_item())
    dispatcher = Dispatcher("m", [a, b], JoinShortestQueuePolicy())

    result = dispatcher.dispatch(_item(), _enqueue)

    assert result.instance is b
    assert result.spilled is False
    assert result.attempts == 1


def test_dispatch_spills_when_the_first_choice_is_full() -> None:
    """Without this, a transient full queue becomes a dropped frame."""
    a = FakeInstance(Device.cuda(0), capacity=0)
    b = FakeInstance(Device.cuda(1), capacity=4)
    spills: list[tuple] = []
    dispatcher = Dispatcher(
        "m", [a, b], RoundRobinPolicy(), on_spill=lambda w, g: spills.append((w, g))
    )

    result = dispatcher.dispatch(_item(), _enqueue)

    assert result.instance is b
    assert result.spilled is True
    assert spills == [(a, b)]


def test_dispatch_raises_only_when_the_whole_pool_is_saturated() -> None:
    """The honest signal: not "one GPU is unlucky" but "the pool cannot keep up"."""
    instances = [FakeInstance(Device.cuda(i), capacity=0) for i in range(4)]
    dispatcher = Dispatcher("m", instances, RoundRobinPolicy())

    with pytest.raises(QueueFullError):
        dispatcher.dispatch(_item(), _enqueue)


def test_dispatch_skips_instances_that_are_not_ready() -> None:
    dead = FakeInstance(Device.cuda(0), capacity=4, is_ready=False)
    alive = FakeInstance(Device.cuda(1), capacity=4)
    dispatcher = Dispatcher("m", [dead, alive], RoundRobinPolicy())

    for _ in range(4):
        assert dispatcher.dispatch(_item(), _enqueue).instance is alive
    assert dead.accepted == []


def test_dispatch_raises_when_nothing_is_ready() -> None:
    instances = [FakeInstance(Device.cuda(i), is_ready=False) for i in range(2)]
    dispatcher = Dispatcher("m", instances, RoundRobinPolicy())
    with pytest.raises(ServerStateError):
        dispatcher.dispatch(_item(), _enqueue)


def test_dispatcher_refuses_to_exist_without_instances() -> None:
    with pytest.raises(ServerStateError):
        Dispatcher("m", [], RoundRobinPolicy())
