"""The TensorRT backend's contract with the rest of the runtime, tested without TensorRT.

Two defects shipped in this file's subject because nothing exercised the call *shape*
between the backend and the pieces it drives:

  * ``runtime/graph.py`` was refactored into the ``runtime/graphs/`` package with a new
    four-argument ``capture`` and a ``replay`` method, and the only caller kept calling the
    old three-argument ``capture`` and a ``launch`` that no longer existed. Every CUDA-graph
    batch would have raised ``TypeError`` on the first request.
  * ``execute`` issued its torch copies on torch's current stream while TensorRT enqueued on
    a different one, with nothing ordering them. Silently wrong numbers.

Neither needs hardware to catch, so neither is a GPU test. The backend is built with
``object.__new__`` and its collaborators injected: that runs the *real* methods against
fakes, which is what makes it a contract test rather than a mirror of the implementation.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from shipinfer.backends.tensorrt.backend import TensorRTBackend
from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.runtime.graphs.base import CapturedGraph, GraphCache

INPUTS = (TensorSpec("images", DataType.FP32, (3, 8, 8)),)
OUTPUTS = (TensorSpec("boxes", DataType.FP32, (4,)),)


# -- fakes ---------------------------------------------------------------------------------


class RecordingStream:
    """A Stream that records whether work was issued inside its activation."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.active = False
        self.handle = 0xBEEF

    @contextmanager
    def activate(self):
        self.active = True
        self._log.append("activate:enter")
        try:
            yield self
        finally:
            self.active = False
            self._log.append("activate:exit")

    def synchronize(self) -> None:
        self._log.append("synchronize")


@dataclass
class RecordingBindings:
    """Stands in for BindingSet, recording every transfer and whether it was ordered."""

    log: list[str]
    stream: RecordingStream
    unordered: list[str] = field(default_factory=list)
    tensors: dict[str, object] = field(default_factory=dict)

    def _note(self, what: str) -> None:
        self.log.append(what)
        if not self.stream.active:
            self.unordered.append(what)

    def stage_input(self, name, array, stream, *, async_copy):  # noqa: ANN001
        self._note(f"stage:{name}")

    def fetch_output(self, name, batch_size, stream, *, async_copy):  # noqa: ANN001
        self._note(f"fetch:{name}")
        return np.zeros((batch_size, 4), dtype=np.float32)

    def device_tensor(self, name: str) -> object:
        return self.tensors.setdefault(name, object())

    def total_bytes(self) -> int:
        return 0


class RecordingGraphCache(GraphCache):
    """A real GraphCache subclass, so an ABC signature change breaks this test too."""

    name = "recording"

    def __init__(self, log: list[str], *, capture_succeeds: bool = True) -> None:
        super().__init__(_Device(), enabled=True, batch_sizes=(1, 2, 4, 8))
        self._log = log
        self._graphs: dict[int, CapturedGraph] = {}
        self._capture_succeeds = capture_succeeds
        self.capture_calls: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return True

    def get(self, batch_size: int) -> CapturedGraph | None:
        return self._graphs.get(batch_size)

    def capture(self, batch_size, stream, static_inputs, run):  # noqa: ANN001
        self.capture_calls.append(
            {"batch_size": batch_size, "stream": stream, "static_inputs": dict(static_inputs)}
        )
        self._log.append(f"capture:{batch_size}")
        outputs = run()
        if not self._capture_succeeds:
            return None
        graph = RecordedGraph(batch_size, dict(static_inputs), dict(outputs), self._log)
        self._graphs[batch_size] = graph
        return graph

    def close(self) -> None:
        self._graphs.clear()

    def stats(self) -> dict[str, int]:
        return {"captured": len(self._graphs)}


class RecordedGraph(CapturedGraph):
    def __init__(self, batch_size, static_inputs, static_outputs, log):  # noqa: ANN001
        self.batch_size = batch_size
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self._log = log
        self._replays = 0

    def replay(self, inputs=None):  # noqa: ANN001
        assert inputs is None, (
            "the TensorRT backend must replay with no inputs: staging already wrote into "
            "the binding tensors the graph recorded, so copying them would be a self-copy"
        )
        self._log.append("replay")
        self._replays += 1
        return self.static_outputs

    def close(self) -> None:
        return None

    @property
    def replays(self) -> int:
        return self._replays


@dataclass
class _Device:
    kind: str = "cuda"
    index: int = 0
    is_cuda: bool = True

    def __str__(self) -> str:
        return "cuda:0"


@dataclass
class _Config:
    input_specs: tuple = INPUTS
    output_specs: tuple = OUTPUTS
    max_batch_size: int = 8
    parameters: dict = field(default_factory=dict)


@dataclass
class _Execution:
    async_transfers: bool = True


@dataclass
class _Context:
    graphs: Any
    config: _Config = field(default_factory=_Config)
    execution: _Execution = field(default_factory=_Execution)
    instance_name: str = "fake:1@cuda:0"


def _backend(
    log: list[str], graphs: Any
) -> tuple[TensorRTBackend, RecordingStream, RecordingBindings]:
    """A TensorRTBackend with its collaborators replaced but its own methods intact."""
    backend = object.__new__(TensorRTBackend)
    stream = RecordingStream(log)
    bindings = RecordingBindings(log, stream)
    backend._context = _Context(graphs=graphs)
    backend._loaded = None  # so input_specs/output_specs fall back to the config
    backend._bindings = bindings
    backend._ctx = object()
    backend._stream = stream
    backend._enqueues = 0
    backend._graph_replays = 0
    backend._set_input_shapes = lambda batch_size: log.append(f"shapes:{batch_size}")
    backend._enqueue = lambda s: (
        log.append("enqueue"),
        bindings._note("enqueue"),
    )
    return backend, stream, bindings


# -- the contract with the graph cache ------------------------------------------------------


def test_backend_calls_capture_with_the_signature_the_abc_declares() -> None:
    """The regression test for the refactor that broke the graph path entirely.

    Binding the recorded call against the ABC's own signature means renaming or reordering
    a parameter on either side fails here, not on a production GPU.
    """
    log: list[str] = []
    graphs = RecordingGraphCache(log)
    backend, stream, _ = _backend(log, graphs)

    backend._maybe_replay(4, stream)

    assert len(graphs.capture_calls) == 1
    call = graphs.capture_calls[0]
    # Would raise TypeError if the backend passed the wrong number or names of arguments.
    inspect.signature(GraphCache.capture).bind(
        graphs, call["batch_size"], call["stream"], call["static_inputs"], lambda: {}
    )


def test_capture_receives_the_persistent_binding_tensors() -> None:
    """ADR-008: a graph records addresses, so the buffers handed to capture must be the
    ones staging writes into — not a fresh allocation."""
    log: list[str] = []
    graphs = RecordingGraphCache(log)
    backend, stream, bindings = _backend(log, graphs)

    backend._maybe_replay(4, stream)

    static_inputs = graphs.capture_calls[0]["static_inputs"]
    assert set(static_inputs) == {"images"}
    assert static_inputs["images"] is bindings.device_tensor("images")


def test_second_batch_of_a_captured_size_replays_instead_of_capturing() -> None:
    log: list[str] = []
    graphs = RecordingGraphCache(log)
    backend, stream, _ = _backend(log, graphs)

    backend._maybe_replay(4, stream)
    backend._maybe_replay(4, stream)

    assert log.count("capture:4") == 1
    assert log.count("replay") == 2
    assert backend._graph_replays == 2


def test_a_failed_capture_falls_back_to_the_ordinary_launch() -> None:
    """Capture returning None means "no graph", not "error"."""
    log: list[str] = []
    graphs = RecordingGraphCache(log, capture_succeeds=False)
    backend, stream, _ = _backend(log, graphs)

    assert backend._maybe_replay(4, stream) is None
    assert backend._graph_replays == 0


def test_no_graph_cache_is_not_an_error() -> None:
    log: list[str] = []
    backend, stream, _ = _backend(log, None)
    assert backend._maybe_replay(4, stream) is None


# -- the contract with the stream -----------------------------------------------------------


def test_every_transfer_and_the_enqueue_share_one_stream() -> None:
    """The regression test for the unordered-copies defect.

    Staging, the enqueue and the readback must all be issued inside the instance stream's
    activation. Anything issued outside it lands on torch's current stream, which nothing
    orders against the stream TensorRT was handed.
    """
    log: list[str] = []
    backend, stream, bindings = _backend(log, None)

    backend.execute({"images": Tensor.from_numpy(np.zeros((2, 3, 8, 8), np.float32))}, 2)

    assert (
        bindings.unordered == []
    ), f"issued outside the stream activation: {bindings.unordered}"
    assert log[0] == "activate:enter"
    assert log[-1] == "activate:exit"
    assert log.index("stage:images") < log.index("enqueue") < log.index("fetch:boxes")


def test_execute_returns_one_row_per_request_row() -> None:
    log: list[str] = []
    backend, _, _ = _backend(log, None)

    outputs = backend.execute(
        {"images": Tensor.from_numpy(np.zeros((3, 3, 8, 8), np.float32))}, 3
    )

    assert set(outputs) == {"boxes"}
    assert outputs["boxes"].shape == (3, 4)


def test_execute_before_initialise_is_refused() -> None:
    from shipinfer.core.errors import InferenceError

    backend = object.__new__(TensorRTBackend)
    backend._context = _Context(graphs=None)
    backend._bindings = None
    backend._ctx = None
    backend._stream = None
    with pytest.raises(InferenceError, match="not initialised"):
        backend.execute({}, 1)
