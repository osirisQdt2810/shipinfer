"""The HTTP surface, speaking KServe v2 (the Triton HTTP protocol).

This is the engine's **side-door** (arch.md §6): a caller that brings its own tensors posts
them to ``/v2/models/{model}/infer`` and gets a response back, without a chain, a camera or
a frame anywhere in the picture. It is optional — ``shipinfer serve`` without ``--http``
runs a warm engine with no ingress at all — and it is the one package allowed to name
FastAPI and uvicorn, which is why it sits beside the engine rather than inside it: an
in-process caller must not pay for a web framework to reach the model pool.

The imports of FastAPI and uvicorn are *inside* the functions that need them (see
:func:`~shipinfer.api.app.create_app`), so importing this package costs nothing on a host
where the ``server`` extra was never installed and the failure is a typed
``ConfigurationError`` naming the extra rather than an ``ImportError`` at start-up.
"""

from shipinfer.api.app import create_app, serve_http
from shipinfer.api.routes import build_router

__all__ = ["build_router", "create_app", "serve_http"]
