"""The TensorRT backend — the production execution path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shipinfer.backends.base import BackendContext, ModelBackend
from shipinfer.backends.tensorrt.bindings import BindingSet
from shipinfer.backends.tensorrt.engine import LoadedEngine, load_engine
from shipinfer.backends.tensorrt.logger import build_trt_logger
from shipinfer.core.errors import BackendUnavailableError, ConfigurationError, InferenceError
from shipinfer.core.logging import get_logger
from shipinfer.core.types import Tensor, TensorSpec
from shipinfer.runtime.platform import require_torch
from shipinfer.runtime.stream import Stream

__all__ = ["TensorRTBackend"]

_LOG = get_logger("backends.tensorrt")


class TensorRTBackend(ModelBackend):
    """Executes a serialised TensorRT engine, with CUDA-graph replay where possible.

    The execution path, in the order it matters:

    1. **Persistent bindings.** Device and pinned-host buffers are allocated once at load
       time, sized for ``max_batch_size``. Nothing is allocated per batch.
    2. **Async staging.** Inputs go host -> pinned -> device with ``cudaMemcpyAsync`` on
       this instance's stream, so the copy overlaps whatever the other streams are doing.
    3. **Graph replay when the shape is captured.** For a batch size with a captured graph,
       the whole enqueue collapses into one ``cudaGraphLaunch``. For the small models here
       that is the difference between being launch-bound and being compute-bound.
    4. **Ordinary enqueue otherwise.** Correct, just not as fast — a model with dynamic
       control flow lives here permanently and that is fine.

    Engine I/O is read from the engine itself, not from ``config.yaml``. A config that has
    drifted from its engine then fails at load with both shapes printed, instead of
    producing quietly wrong output.
    """

    platform = "tensorrt"
    requires_gpu = True

    def __init__(self, context: BackendContext) -> None:
        super().__init__(context)
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise BackendUnavailableError(
                'TensorRT is not installed. Install it with: pip install "shipinfer[tensorrt]"'
            ) from exc
        self._trt: Any = trt
        self._torch: Any = require_torch()
        self._logger = build_trt_logger(
            trt, verbose=bool(context.config.parameters.get("verbose"))
        )
        self._loaded: LoadedEngine | None = None
        self._ctx: Any = None
        self._bindings: BindingSet | None = None
        self._stream: Stream | None = None
        self._uses_named_tensors = False
        self._graph_replays = 0
        self._enqueues = 0

    # -- lifecycle -----------------------------------------------------------------------

    def _do_initialize(self) -> None:
        context = self.context
        engine_file = context.config.parameters.get("engine_file", "model.plan")
        path = context.artifact.file(str(engine_file))

        self._torch.cuda.set_device(self.device.index)
        self._loaded = load_engine(self._trt, self._logger, path)
        self._ctx = self._loaded.engine.create_execution_context()
        if self._ctx is None:
            raise InferenceError(f"could not create an execution context for {path}")
        self._uses_named_tensors = hasattr(self._loaded.engine, "num_io_tensors")

        self._validate_against_config()
        self._allocate_bindings()
        self._stream = (
            context.streams.next() if context.streams is not None else Stream(self.device)
        )

    def _validate_against_config(self) -> None:
        """Refuse a config that disagrees with the engine."""
        assert self._loaded is not None
        strip = self.context.config.max_batch_size > 0
        engine_specs = {t.name: t.to_spec(strip) for t in self._loaded.io}

        for declared in (*self.context.config.input_specs, *self.context.config.output_specs):
            actual = engine_specs.get(declared.name)
            if actual is None:
                raise ConfigurationError(
                    f"{self.context.instance_name}: config declares tensor "
                    f"{declared.name!r} but the engine has {sorted(engine_specs)}"
                )
            if actual.dtype is not declared.dtype:
                raise ConfigurationError(
                    f"{self.context.instance_name}: {declared.name!r} is "
                    f"{actual.dtype.value} in the engine but {declared.dtype.value} in config"
                )

    def _allocate_bindings(self) -> None:
        assert self._loaded is not None
        context = self.context
        batch = context.config.effective_max_batch_size
        self._bindings = BindingSet(device=self.device, staging=context.memory.staging)
        for tensor in self._loaded.io:
            self._bindings.add(
                tensor.name,
                is_input=tensor.is_input,
                dtype=tensor.dtype,
                shape=self._concrete_shape(tensor.shape, batch),
            )
        _LOG.info(
            "%s: allocated %.1f MiB of persistent bindings",
            context.instance_name,
            self._bindings.total_bytes() / (1 << 20),
        )

    @staticmethod
    def _concrete_shape(shape: tuple[int, ...], batch: int) -> tuple[int, ...]:
        """Resolve dynamic extents to their maximum so the buffer fits any batch.

        Dim 0 becomes ``max_batch_size``. Any other dynamic dim has no principled maximum,
        so it must be pinned in the config — guessing one is how a buffer overflow becomes
        a silent wrong answer.
        """
        if not shape:
            return (batch,)
        resolved = [batch if i == 0 else dim for i, dim in enumerate(shape)]
        unresolved = [i for i, dim in enumerate(resolved) if dim < 0]
        if unresolved:
            raise ConfigurationError(
                f"engine tensor has dynamic dimension(s) at {unresolved} beyond the batch "
                "axis; build the engine with a fixed optimisation profile for those axes"
            )
        return tuple(resolved)

    def _do_finalize(self) -> None:
        if self._bindings is not None:
            self._bindings.close()
            self._bindings = None
        self._ctx = None
        self._loaded = None

    # -- introspection -------------------------------------------------------------------

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        """The engine's truth, once loaded; the config's claim before that."""
        if self._loaded is None:
            return super().input_specs
        strip = self.context.config.max_batch_size > 0
        return tuple(t.to_spec(strip) for t in self._loaded.inputs)

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        if self._loaded is None:
            return super().output_specs
        strip = self.context.config.max_batch_size > 0
        return tuple(t.to_spec(strip) for t in self._loaded.outputs)

    # -- execution -----------------------------------------------------------------------

    def execute(self, inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        if self._bindings is None or self._ctx is None or self._stream is None:
            raise InferenceError(f"{self.context.instance_name} is not initialised")

        bindings = self._bindings
        stream = self._stream
        async_copy = self.context.execution.async_transfers

        # EVERYTHING in one execution happens on one stream, and that is not a style
        # choice. `stage_input`'s copy_ and `fetch_output`'s .to("cpu") are torch ops, so
        # they go to torch's *current* stream; TensorRT enqueues on the handle it is given.
        # Without this block those are two different streams with nothing ordering them, so
        # the enqueue can start before a ~39 MB H2D has landed and the readback can run
        # before the kernels retire — wrong numbers, no error, no crash.
        with stream.activate():
            for name, tensor in inputs.items():
                bindings.stage_input(name, tensor.numpy(), stream, async_copy=async_copy)

            self._set_input_shapes(batch_size)

            if self._maybe_replay(batch_size, stream) is None:
                self._enqueue(stream)
            self._enqueues += 1

            outputs: dict[str, Tensor] = {}
            for spec in self.output_specs:
                array = bindings.fetch_output(
                    spec.name, batch_size, stream, async_copy=async_copy
                )
                outputs[spec.name] = Tensor.from_numpy(array)
        return outputs

    def _set_input_shapes(self, batch_size: int) -> None:
        """Tell the context this batch's real shape, when the engine is dynamic."""
        assert self._loaded is not None and self._bindings is not None
        if not self._loaded.has_dynamic_shapes:
            return
        for tensor in self._loaded.inputs:
            shape = (batch_size, *self._bindings[tensor.name].shape[1:])
            if self._uses_named_tensors:
                self._ctx.set_input_shape(tensor.name, shape)
            else:
                self._ctx.set_binding_shape(
                    self._loaded.engine.get_binding_index(tensor.name), shape
                )

    def _enqueue(self, stream: Stream) -> None:
        """Issue the inference on ``stream``.

        Two API generations, because a deployment should not be pinned to one TensorRT
        minor version: ``execute_async_v3`` with named tensor addresses (>= 8.5), or
        ``execute_async_v2`` with an ordered pointer list.
        """
        assert self._bindings is not None and self._loaded is not None
        if self._uses_named_tensors:
            for name in self._bindings.names():
                self._ctx.set_tensor_address(name, self._bindings.device_ptr(name))
            ok = self._ctx.execute_async_v3(stream_handle=stream.handle)
        else:
            pointers = [
                self._bindings.device_ptr(self._loaded.engine.get_binding_name(i))
                for i in range(self._loaded.engine.num_bindings)
            ]
            ok = self._ctx.execute_async_v2(bindings=pointers, stream_handle=stream.handle)
        if not ok:
            raise InferenceError(f"{self.context.instance_name}: TensorRT enqueue failed")

    def _capture_io(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The binding tensors a captured graph reads and writes.

        These *are* the static buffers ADR-008 requires: allocated once at load, sized for
        ``max_batch_size``, never reallocated. Because staging writes straight into them,
        a replay needs no input copy at all — the data is already at the addresses the
        graph recorded, which is the whole reason the bindings are persistent.
        """
        assert self._bindings is not None
        inputs = {
            spec.name: self._bindings.device_tensor(spec.name) for spec in self.input_specs
        }
        outputs = {
            spec.name: self._bindings.device_tensor(spec.name) for spec in self.output_specs
        }
        return inputs, outputs

    def _maybe_replay(self, batch_size: int, stream: Stream) -> Any:
        """Replay a captured graph for this batch size, capturing one if it is new.

        Returns ``None`` to mean "no graph — take the ordinary launch path", which the
        caller treats as a normal outcome rather than a failure.
        """
        graphs = self.context.graphs
        if graphs is None or not graphs.enabled:
            return None

        captured = graphs.get(batch_size)
        if captured is None and graphs.should_capture(batch_size):
            static_inputs, static_outputs = self._capture_io()

            def run() -> Mapping[str, Any]:
                # Issues the work and names the buffers it produced, which is the contract
                # GraphCache.capture expects. The cache warms this on a side stream before
                # recording, so TensorRT's first-call scratch allocation and tactic
                # selection do not get baked into a graph that replays forever.
                self._enqueue(stream)
                return static_outputs

            captured = graphs.capture(batch_size, stream, static_inputs, run)

        if captured is None:
            return None
        # No arguments: staging already wrote into the binding tensors, which are the exact
        # objects the graph recorded. Passing inputs here would copy them onto themselves.
        captured.replay()
        self._graph_replays += 1
        return captured

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "enqueues": self._enqueues,
            "graph_replays": self._graph_replays,
        }
        if self._bindings is not None:
            stats["binding_bytes"] = self._bindings.total_bytes()
        if self.context.graphs is not None:
            stats.update({f"graph_{k}": v for k, v in self.context.graphs.stats().items()})
        return stats
