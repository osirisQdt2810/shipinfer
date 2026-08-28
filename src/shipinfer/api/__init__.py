"""The HTTP surface: the camera door, and the KServe side-door.

Two doors, and they answer different callers (arch.md §2 and §6).

``/streams`` is where **cameras and videos enter a running deployment** — ``POST /streams
{"url": "rtsp://..."}`` and a runner starts reading it. It is served by ``shipinfer run
--http``, because the runner that owns the cameras is what that command composes; the routes
reach it through the five-member :class:`~shipinfer.api.streams.CameraController` protocol, so
this package can *use* a runner and cannot *build* one.

``/v2/...`` is the engine's **side-door**: a caller that brings its own tensors posts them to
``/v2/models/{model}/infer`` and gets a response back, without a chain, a camera or a frame
anywhere in the picture. It is optional — ``shipinfer serve`` without ``--http`` runs a warm
engine with no ingress at all — and this is the one package allowed to name FastAPI and
uvicorn, which is why it sits beside the engine rather than inside it: an in-process caller
must not pay for a web framework to reach the model pool.

The imports of FastAPI and uvicorn are *inside* the functions that need them (see
:func:`~shipinfer.api.app.create_app`), so importing this package costs nothing on a host
where the ``server`` extra was never installed and the failure is a typed
``ConfigurationError`` naming the extra rather than an ``ImportError`` at start-up.
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
