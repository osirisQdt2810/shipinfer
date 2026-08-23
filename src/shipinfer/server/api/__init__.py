"""The optional HTTP facade, speaking KServe v2 (the Triton HTTP protocol)."""

from shipinfer.server.api.app import create_app, serve_http
from shipinfer.server.api.routes import build_router

__all__ = ["build_router", "create_app", "serve_http"]
