"""Where a camera's effective settings come from: camera, then fleet, then environment.

Three layers, one direction, stated once. A camera field wins because it is the most
specific thing anyone wrote down; :class:`IngestSettings` is next because fifty cameras that
all want TCP should say so once; the process environment is last because it is how a
container is retargeted without editing a config file at all.

These live here rather than as methods on the settings objects for a layering reason: the
last fallback is :mod:`shipinfer.envs`, and ``core`` sits below it and may not import upward
(ADR-001). Keeping them pure functions also means the precedence is asserted directly in a
test instead of inferred from a source's behaviour.
"""

from __future__ import annotations

from shipinfer import envs
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings

__all__ = [
    "resolve_hwaccel",
    "resolve_latency_ms",
    "resolve_open_timeout_s",
    "resolve_read_timeout_s",
    "resolve_source_name",
    "resolve_transport",
]


def resolve_source_name(config: CameraConfig, settings: IngestSettings | None = None) -> str:
    """Which registered source implementation this camera should use."""
    if config.source:
        return config.source
    if settings is not None and settings.backend:
        return settings.backend
    return envs.SHIPINFER_INGEST_BACKEND


def resolve_hwaccel(config: CameraConfig, settings: IngestSettings | None = None) -> bool:
    """Whether to prefer hardware decode for this camera."""
    if config.hwaccel is not None:
        return config.hwaccel
    if settings is not None and settings.hwaccel is not None:
        return settings.hwaccel
    return envs.SHIPINFER_INGEST_HWACCEL


def resolve_transport(config: CameraConfig, settings: IngestSettings | None = None) -> str:
    """RTSP lower transport for this camera."""
    if config.transport is not None:
        return config.transport
    if settings is not None and settings.transport is not None:
        return settings.transport
    return envs.SHIPINFER_INGEST_RTSP_TRANSPORT


def resolve_latency_ms(config: CameraConfig, settings: IngestSettings | None = None) -> int:
    """Jitter-buffer depth for this camera, in milliseconds."""
    if config.latency_ms is not None:
        return config.latency_ms
    if settings is not None and settings.latency_ms is not None:
        return settings.latency_ms
    return envs.SHIPINFER_INGEST_LATENCY_MS


def resolve_read_timeout_s(settings: IngestSettings | None = None) -> float:
    """How long a single read may block. Fleet-wide; not a per-camera knob."""
    if settings is not None and settings.read_timeout_ms is not None:
        return settings.read_timeout_ms / 1000.0
    return envs.SHIPINFER_INGEST_READ_TIMEOUT_S


def resolve_open_timeout_s(settings: IngestSettings | None = None) -> float:
    """How long a connection attempt may take before it counts as a failure."""
    if settings is not None and settings.open_timeout_ms is not None:
        return settings.open_timeout_ms / 1000.0
    return envs.SHIPINFER_INGEST_OPEN_TIMEOUT_S
