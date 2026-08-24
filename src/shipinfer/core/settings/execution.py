"""Execution-path optimisations: streams, CUDA graphs, and the native fast path."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shipinfer.core.settings.enums import ExecutionProvider

__all__ = ["ExecutionSettings"]


class ExecutionSettings(BaseModel):
    """Knobs that change *how fast* an execution runs, never *what it computes*.

    Each one is here because it is a measurable win on this workload, and each one is a
    knob rather than a constant because the right value depends on the model:

    * **CUDA graphs** collapse the per-launch CPU cost of a whole inference into a single
      ``cudaGraphLaunch``. For the small, launch-bound models in this pipeline (a 512-d
      embedder at batch 8) launch overhead is a large share of wall time, and this is the
      same technique vLLM uses for its decode step. Graphs require *static* shapes and
      addresses, which is why capture is per (model, batch size) and why the I/O buffers
      must come from the pool rather than being freshly allocated.
    * **Multiple streams per instance** overlap H2D of batch *n+1* with compute of batch
      *n* and D2H of batch *n-1*. Without it a GPU sits idle for the whole transfer.
    * **The native provider** moves the queue, the batcher and the pre/post-processing
      into ``shipinfer._C``. Python is fine for the control plane; it is not fine for a
      15 000 req/s data plane.
    """

    model_config = ConfigDict(extra="forbid")

    provider: ExecutionProvider = ExecutionProvider.AUTO

    # -- CUDA graphs -------------------------------------------------------------------
    #: Off by default, and that default is a measurement rather than a preference.
    #:
    #: The TensorRT execute path cannot be captured as written: `fetch_output` synchronises
    #: the stream and may allocate a pinned staging buffer, and CUDA forbids both inside a
    #: capture region. Attempting it costs three failed captures per instance at every
    #: start-up — measured at 24 instances, that alone took the GPU tier's end-to-end test
    #: past its 90 s timeout — and the failures do not clean up perfectly: the caching
    #: allocator still reports `captures_underway` afterwards.
    #:
    #: The staging pool refuses before the driver call and a partial graph is never stored,
    #: so a failed capture cannot corrupt a *replay*. It is still not free, and paying for
    #: it on every start-up to reach a path that cannot succeed is the wrong default.
    #:
    #: Turn it on per model once execute keeps device work and host synchronisation apart,
    #: or globally with `SHIPINFER_CUDA_GRAPHS=on` to measure the difference.
    cuda_graphs: bool = False
    #: Which implementation captures them. ``torch`` (the default) uses
    #: ``torch.cuda.CUDAGraph``, which warms up on a side stream and shares one memory pool
    #: across batch sizes. ``custom`` is this project's raw-driver reference implementation
    #: — readable, but without those safeguards (ADR-003).
    graph_cache: Literal["torch", "custom"] = "torch"
    #: Batch sizes to capture a graph for. A request whose padded batch is not in this set
    #: runs the ordinary launch path — correct, just not as fast.
    cuda_graph_batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    #: Give up on graph capture after this many failures and log once. A model with
    #: dynamic control flow simply cannot be captured, and retrying every batch is waste.
    cuda_graph_max_capture_failures: int = Field(default=3, ge=0)

    # -- streams -----------------------------------------------------------------------
    #: Streams per instance, i.e. how many batches may be in flight. 0 means "use the
    #: model config's ``instance_group.streams``".
    streams_per_instance: int = Field(default=0, ge=0)
    #: Use ``cudaMemcpyAsync`` on the instance's stream rather than a blocking copy. Only
    #: valid from pinned memory, which the pool guarantees.
    async_transfers: bool = True

    # -- warm-up -----------------------------------------------------------------------
    #: Batches pushed through each instance at load time, so the first real request does
    #: not pay for lazy CUDA module loading, cuBLAS autotuning and TensorRT's first-call
    #: allocations. Skipping this makes the first p99 of every deploy meaningless.
    warmup_iterations: int = Field(default=3, ge=0)
    #: Capture graphs during warm-up rather than on the first live request.
    warmup_captures_graphs: bool = True

    # -- pre/post-processing -----------------------------------------------------------
    #: Run letterbox-resize + colour conversion + normalise + NHWC->NCHW as a single fused
    #: CUDA kernel instead of four passes over the image. Four passes over a 1080p frame is
    #: four times the memory traffic for identical output.
    fused_preprocess: bool = True
    #: Run NMS on the device. Copying 25 000 candidate boxes to the host to filter them
    #: down to 20 is the single most common self-inflicted bottleneck in this kind of
    #: pipeline.
    gpu_postprocess: bool = True
