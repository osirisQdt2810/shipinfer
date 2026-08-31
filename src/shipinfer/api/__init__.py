"""The HTTP surface: the camera door, and the KServe side-door.

``/streams`` is how cameras enter a running deployment (served by ``shipinfer run --http``);
the routes reach the runner only through the :class:`~shipinfer.api.streams.CameraController`
protocol, so this package can *use* a runner but cannot *build* one. ``/v2/...`` is the
engine's side-door for callers that bring their own tensors; it is optional, and this is the
one package allowed to name FastAPI and uvicorn (arch.md §2 and §6).

FastAPI/uvicorn imports live inside the functions that need them, so importing this package
costs nothing without the ``server`` extra and fails as a typed ``ConfigurationError``.
"""

from shipinfer.api.app import (
    BackgroundHttpServer,
    create_app,
    require_server_extra,
    serve_http,
)
from shipinfer.api.errors import http_error
from shipinfer.api.routes import build_router
from shipinfer.api.streams import CameraController, build_streams_router

__all__ = [
    "BackgroundHttpServer",
    "CameraController",
    "build_router",
    "build_streams_router",
    "create_app",
    "http_error",
    "require_server_extra",
    "serve_http",
]
