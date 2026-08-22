"""The optional KServe-v2 HTTP facade."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["HttpSettings"]


class HttpSettings(BaseModel):
    """Bind address and paths for the HTTP API."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    metrics_path: str = "/metrics"
    #: Reject a request body larger than this. A perception payload is a few MB; anything
    #: much larger is a mistake or an attack, and either way should not reach the pool.
    max_body_mb: int = Field(default=64, ge=1)
