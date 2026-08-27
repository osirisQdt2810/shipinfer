"""Building and running the ASGI application."""

from __future__ import annotations

from typing import Any

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.engine.pool import InferenceServer
from shipinfer.server.api.routes import build_router

__all__ = ["create_app", "serve_http"]

_LOG = get_logger("server.api")


def create_app(server: InferenceServer) -> Any:
    """Wrap a running server in a FastAPI application.

    The server's lifecycle is deliberately *not* tied to the app's: whoever created the
    server started it and will stop it. Tying them would make the HTTP layer load-bearing
    for a library whose main use is in-process.
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise ConfigurationError(
            'the HTTP API needs FastAPI: pip install "shipinfer[server]"'
        ) from exc

    from shipinfer import __version__

    app = FastAPI(
        title="ShipInfer",
        version=__version__,
        description="KServe v2 inference API",
        docs_url="/docs",
    )
    app.include_router(build_router(server))
    return app


def serve_http(server: InferenceServer, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the API with uvicorn until interrupted."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigurationError(
            'the HTTP API needs uvicorn: pip install "shipinfer[server]"'
        ) from exc

    _LOG.info("serving KServe v2 on http://%s:%d (docs at /docs)", host, port)
    uvicorn.run(
        create_app(server),
        host=host,
        port=port,
        log_level=server.settings.observability.log_level.lower(),
        # The server manages its own thread pool per instance; letting uvicorn add
        # another layer of workers would multiply the queues and defeat the balancing.
        workers=1,
    )
