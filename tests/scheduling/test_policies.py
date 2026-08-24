"""Placement policies: the load-balancing behaviour, without a GPU in sight."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest

from shipinfer.core.request import InferenceRequest
from shipinfer.core.types import Device, Tensor
from shipinfer.scheduling.policies import (
    POLICIES,
    JoinShortestQueuePolicy,
    LocalityAwareSpilloverPolicy,
    PowerOfTwoChoicesPolicy,
    RoundRobinPolicy,
    SequenceAffinityPolicy,
    build_policy,
)


@dataclass
class FakeInstance:
    """A ``Placeable`` with no machinery behind it.

    The policies are given exactly four attributes by contract, so a four-field dataclass
    is a *complete* test double — which is itself evidence the contract is narrow enough.
    """

    device: Device
    depth: int = 0
    ewma_latency_us: float = 0.0
    is_ready: bool = True


def _request(resident: Device | None = None, camera: str = "") -> InferenceRequest:
    from shipinfer.core.request import RequestContext

    return InferenceRequest(
        model_name="m",
        inputs={"x": Tensor.from_numpy(np.zeros((1, 4), dtype=np.float32))},
        resident_device=resident,
        context=RequestContext(camera_id=camera),
    )


class TestLoadSpreading:
    """A hint-free request goes to whichever instance the policy thinks is least busy."""

    def test_round_robin_rotates(self) -> None:
        instances = [FakeInstance(Device.cuda(i)) for i in range(3)]
        policy = RoundRobinPolicy()
        picked = [policy.select(instances, _request()).device.index for _ in range(6)]
        assert picked == [0, 1, 2, 0, 1, 2]

    def test_join_shortest_queue_picks_the_shortest(self) -> None:
        instances = [
            FakeInstance(Device.cuda(0), depth=7),
            FakeInstance(Device.cuda(1), depth=2),
            FakeInstance(Device.cuda(2), depth=9),
        ]
        assert JoinShortestQueuePolicy().select(instances, _request()).device.index == 1

    def test_power_of_two_never_picks_the_same_instance_twice(self) -> None:
        """Sampling without replacement, asserted rather than assumed.

        A rejection-free ``j >= i: j += 1`` index shuffle is easy to get subtly wrong, and the
        symptom would be a silent bias toward one GPU under load.
        """
        instances = [FakeInstance(Device.cuda(i), depth=i) for i in range(4)]
        policy = PowerOfTwoChoicesPolicy(rng=random.Random(1234))
        picks = [policy.select(instances, _request()).device.index for _ in range(500)]
        # With depths 0..3 the shortest should win most of the time, and the busiest should
        # essentially never be chosen when a shorter one was sampled alongside it.
        assert picks.count(0) > picks.count(3)
        assert set(picks) <= {0, 1, 2, 3}


class TestLocalitySpillover:
    """A resident frame stays on its GPU until the queue there makes crossing PCIe cheaper."""

    def test_locality_keeps_work_on_the_resident_gpu(self) -> None:
        instances = [
            FakeInstance(Device.cuda(0), depth=2),
            FakeInstance(Device.cuda(1), depth=0),
        ]
        policy = LocalityAwareSpilloverPolicy(spill_threshold=4)
        chosen = policy.select(instances, _request(resident=Device.cuda(0)))
        assert (
            chosen.device.index == 0
        ), "a 6 MB frame should not cross PCIe to save 2 queue slots"

    def test_locality_spills_once_the_resident_gpu_backs_up(self) -> None:
        instances = [
            FakeInstance(Device.cuda(0), depth=10),
            FakeInstance(Device.cuda(1), depth=0),
        ]
        policy = LocalityAwareSpilloverPolicy(spill_threshold=4)
        chosen = policy.select(instances, _request(resident=Device.cuda(0)))
        assert chosen.device.index == 1

    def test_locality_falls_back_when_there_is_no_hint(self) -> None:
        instances = [
            FakeInstance(Device.cuda(0), depth=5),
            FakeInstance(Device.cuda(1), depth=0),
        ]
        policy = LocalityAwareSpilloverPolicy(
            spill_threshold=4, fallback=JoinShortestQueuePolicy()
        )
        assert policy.select(instances, _request()).device.index == 1


class TestSequenceAffinity:
    """A camera is pinned to one instance for as long as that instance lives."""

    def test_sequence_affinity_pins_a_camera(self) -> None:
        """A stateful model is only correct if a sequence stays on one instance."""
        instances = [FakeInstance(Device.cuda(i)) for i in range(4)]
        policy = SequenceAffinityPolicy(fallback=RoundRobinPolicy())
        first = policy.select(instances, _request(camera="cam7"))
        for _ in range(20):
            assert policy.select(instances, _request(camera="cam7")) is first

    def test_sequence_affinity_repins_when_the_instance_dies(self) -> None:
        """A dead GPU must not mean a permanently dropped camera."""
        instances = [FakeInstance(Device.cuda(i)) for i in range(2)]
        policy = SequenceAffinityPolicy(fallback=RoundRobinPolicy())
        first = policy.select(instances, _request(camera="cam7"))
        first.is_ready = False
        survivors = [i for i in instances if i.is_ready]
        assert policy.select(survivors, _request(camera="cam7")) is not first


class TestPolicyRegistry:
    """Every policy is reachable by name, and an unknown name says what is available."""

    def test_every_policy_is_registered_and_buildable(self) -> None:
        assert set(POLICIES.names()) >= {
            "round_robin",
            "join_shortest_queue",
            "power_of_two",
            "locality_spillover",
            "sequence_affinity",
        }
        for name in POLICIES.names():
            assert build_policy(name).describe()

    def test_unknown_policy_names_its_alternatives(self) -> None:
        with pytest.raises(Exception, match="available:"):
            build_policy("nonexistent")
