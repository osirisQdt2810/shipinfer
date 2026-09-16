"""A backend whose `_do_initialize` throws part-way must still release what it took.

The C++ twin is `TrtInstance::teardown`, called by the constructor's `catch(...)`. On this
plane the trap is `initialize()` setting `_initialized` only AFTER `_do_initialize` returns:
`finalize()` guards on that flag, so the one method that spells out the release order --
bindings, then context, then engine -- was unreachable on exactly the failing path.

TensorRT allocates its execution context before its bindings, and a device a neighbour has
filled throws between the two. That is the case this covers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from shipinfer.backends.base import ModelBackend
from shipinfer.core.errors import BackendLoadError
from shipinfer.repository import ModelConfig


class PartBuiltBackend(ModelBackend):
    """Takes a resource, then fails -- the shape of context-then-bindings."""

    platform = "part-built"
    requires_gpu = False

    def __init__(self, context, *, finalize_raises: bool = False) -> None:
        super().__init__(context)
        self.held: list[str] = []
        self.released: list[str] = []
        self._finalize_raises = finalize_raises

    def _do_initialize(self) -> None:
        self.held.append("context")
        raise BackendLoadError("device 5 is full")

    def _do_finalize(self) -> None:
        if self._finalize_raises:
            raise RuntimeError("cleanup itself failed")
        self.released.extend(self.held)

    def execute(self, inputs, batch_size):  # pragma: no cover - never reached
        raise AssertionError("initialize failed; execute must not run")


def _backend(**kwargs) -> PartBuiltBackend:
    """A duck-typed context, as `test_warmup_wiring` does: this needs no device and no pool."""
    config = ModelConfig(
        name="m",
        platform="part-built",
        max_batch_size=1,
        inputs=[{"name": "x", "data_type": "FP32", "dims": [2]}],
        outputs=[{"name": "y", "data_type": "FP32", "dims": [2]}],
    )
    context = SimpleNamespace(
        config=config,
        artifact=SimpleNamespace(path=None, config=config, name="m", version=1),
        device=SimpleNamespace(is_cuda=False),
        instance_name="m:0",
    )
    return PartBuiltBackend(context, **kwargs)


class TestAPartBuiltInitializeReleasesWhatItTook:
    def test_do_finalize_runs_and_the_original_error_propagates(self) -> None:
        backend = _backend()

        with pytest.raises(BackendLoadError, match="device 5 is full"):
            backend.initialize()

        # THE POINT: without the guard this list is empty, because `finalize()` returns at its
        # `_initialized` check and `_do_finalize` is never reached.
        assert backend.released == ["context"]
        assert backend.is_initialized is False

    def test_a_cleanup_failure_does_not_replace_the_real_one(self) -> None:
        """The caller has to see why the load failed, not why the tidy-up did."""
        backend = _backend(finalize_raises=True)

        with pytest.raises(BackendLoadError, match="device 5 is full"):
            backend.initialize()

    def test_finalize_afterwards_is_still_safe_and_does_not_release_twice(self) -> None:
        backend = _backend()
        with pytest.raises(BackendLoadError):
            backend.initialize()

        backend.finalize()

        assert backend.released == ["context"]
