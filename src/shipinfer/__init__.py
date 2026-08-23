"""ShipInfer — a Triton-shaped, hackable multi-GPU inference server.

Built for one workload and honest about it: 50 cameras x 20 fps of ship and person
perception, fanned across 16 GPUs, where the hard part is not raw throughput but **load
balance and tail latency**.

Layering, one-way, enforced by ``tests/test_architecture.py``::

    core/         pure types, errors, settings, logging, metrics, the registry primitive
      ^
    repository/   the on-disk model repository (Triton's layout)
    scheduling/   queues, batching, placement policies, the dispatcher   [pure]
      ^
    runtime/      devices, streams, memory pools, CUDA graphs, image ops [the CUDA seam]
      ^
    backends/     TensorRT / ONNX Runtime / TorchScript / mock
      ^
    server/       instances, models, the engine, health, HTTP
      ^
    pipeline/     the ship+person DAG built on top of the server

Quick start::

    from shipinfer import InferenceServer, InferenceRequest, ServerSettings, Tensor

    with InferenceServer(ServerSettings(model_repository="model_repository")) as server:
        response = server.infer_sync(
            InferenceRequest("ship_detector", {"images": Tensor.from_numpy(batch)})
        )
"""

from shipinfer.core import (
    DataType,
    Device,
    InferenceRequest,
    InferenceResponse,
    Priority,
    RequestContext,
    ServerSettings,
    ShipInferError,
    Tensor,
    TensorSpec,
    configure_logging,
    get_logger,
)

__version__ = "0.1.0"

__all__ = [
    "DataType",
    "Device",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceServer",
    "Priority",
    "RequestContext",
    "ServerSettings",
    "ShipInferError",
    "Tensor",
    "TensorSpec",
    "__version__",
    "configure_logging",
    "get_logger",
]


def __getattr__(name: str) -> object:
    """Import the server lazily.

    ``import shipinfer`` should stay cheap enough for a CLI that only lists a repository.
    Pulling in the server would drag the whole runtime and backend registry along with it,
    so it is resolved on first attribute access instead.
    """
    if name == "InferenceServer":
        from shipinfer.server import InferenceServer

        return InferenceServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
