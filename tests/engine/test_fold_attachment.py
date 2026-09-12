"""Who may attach a device fold to a model, and what happens when two slots disagree.

The chain states the fold and the composition root applies it — the route the other plane
takes, where the resolved plan hands a `MaskAreaSpec` to `TrtEngineAdapter`. What can go wrong
here is not arithmetic: it is two chain slots pointing at one model and asking it to fold
differently, which is a chain that cannot be served rather than a last-writer-wins.

Offline: the backends are doubles, because what is under test is the plumbing.
"""

from __future__ import annotations

from typing import Any

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.engine.instance import ModelInstance
from shipinfer.topology.elements.masks import InstanceMaskArea


class FoldingBackend:
    """A backend that has somewhere to put a fold, and advertises what it then returns."""

    def __init__(self) -> None:
        self.fold: Any = None

    def set_fold(self, fold: Any) -> None:
        self.fold = fold

    @property
    def output_specs(self) -> tuple[str, ...]:
        return ("output0", self.fold.name) if self.fold else ("output0", "output1")


class PlainBackend:
    """A backend that does not — onnx, TorchScript, a mock. Not a failure."""

    output_specs: tuple[str, ...] = ("output0", "output1")


class RecordingBatcher:
    """Stands in for `StackingBatcher`, which every instance holds a reference to."""

    def __init__(self) -> None:
        self.specs: Any = None

    def set_output_specs(self, specs: Any) -> None:
        self.specs = tuple(specs)


def instance_with(backend: Any) -> ModelInstance:
    made = object.__new__(ModelInstance)
    made._backend = backend
    return made


def spec(name: str = "mask_area_px") -> InstanceMaskArea:
    return InstanceMaskArea(crop_hw=(640, 640), name=name)


class TestOneInstance:
    def test_a_backend_that_folds_takes_it_and_says_so(self) -> None:
        backend = FoldingBackend()

        took = instance_with(backend).attach_fold(spec())

        assert took is True
        assert backend.fold is not None and backend.fold.name == "mask_area_px"

    def test_a_backend_that_does_not_fold_is_not_an_error(self) -> None:
        """The host fold is what every backend did until now, so declining is an answer.

        Returned rather than raised because a chain is allowed to run on onnx, and refusing
        would make the device fold a requirement the chain never stated.
        """
        assert instance_with(PlainBackend()).attach_fold(spec()) is False


class FakeModel:
    """`Model.attach_fold`'s body, over instances this test controls.

    Bound off the real class so the refusal under test is the shipped one; only the
    collaborators are doubles.
    """

    def __init__(self, backends: list[Any]) -> None:
        from shipinfer.engine.model import Model

        self.name = "ship_segmenter"
        self._fold = None
        #: The batcher every instance was handed at construction. Recorded so the test can
        #: see that attaching a fold updates it in place rather than replacing it.
        self._batcher = RecordingBatcher()
        self._instances = [instance_with(b) for b in backends]
        self.attach_fold = Model.attach_fold.__get__(self)  # type: ignore[attr-defined]


class TestOneModel:
    def test_every_instance_is_offered_the_fold(self) -> None:
        backends = [FoldingBackend(), FoldingBackend()]
        model = FakeModel(backends)

        took = model.attach_fold(spec())

        assert took is True
        assert all(b.fold is not None for b in backends), "one model, all of its instances"
        assert model._batcher.specs == ("output0", "mask_area_px"), (
            "the batcher follows the backend, or the scatter refuses a response for an "
            "output the engine no longer returns"
        )

    def test_the_same_fold_twice_is_not_a_conflict(self) -> None:
        """A chain reopened, or two slots that agree. Only disagreement is the fault."""
        model = FakeModel([FoldingBackend()])

        model.attach_fold(spec())
        model.attach_fold(spec())

    def test_two_slots_that_fold_it_differently_are_refused(self) -> None:
        """One model has one set of instances, so the second slot cannot be served at all —
        and silently overwriting the first would publish the wrong area for that slot."""
        model = FakeModel([FoldingBackend()])
        model.attach_fold(spec())

        with pytest.raises(ConfigurationError, match="already folds"):
            model.attach_fold(spec(name="area_v2"))

    def test_detaching_is_always_allowed(self) -> None:
        """`None` is "no fold" and cannot conflict with anything, so a chain reopened with a
        different one must not be refused by the fold the last one left behind."""
        model = FakeModel([FoldingBackend()])
        model.attach_fold(spec())

        model.attach_fold(None)
        model.attach_fold(spec(name="area_v2"))  # a different slot, after a clean detach

    def test_a_model_whose_backends_all_decline_says_false(self) -> None:
        """So the caller can report the host path once for the model rather than guess."""
        assert FakeModel([PlainBackend(), PlainBackend()]).attach_fold(spec()) is False
