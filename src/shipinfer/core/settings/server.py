"""The root settings object."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shipinfer.core.settings.device import DeviceSettings
from shipinfer.core.settings.execution import ExecutionSettings
from shipinfer.core.settings.http import HttpSettings
from shipinfer.core.settings.ingest import IngestSettings
from shipinfer.core.settings.memory import MemorySettings
from shipinfer.core.settings.observability import ObservabilitySettings
from shipinfer.core.settings.scheduler import SchedulerSettings

__all__ = ["ServerSettings"]


class ServerSettings(BaseSettings):
    """Everything that varies per *deployment*.

    Anything that varies per *model* lives in that model's ``config.yaml`` instead. Keeping
    the split sharp is what lets one model repository run unchanged on a 2-GPU dev box and
    a 16-GPU node.

    Construct once in the entry point and inject it; nothing here is a global.
    """

    model_config = SettingsConfigDict(
        env_prefix="SHIPINFER_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    #: Directory holding ``<model>/config.yaml`` and ``<model>/<version>/`` artefacts.
    model_repository: Path = Path("model_repository")
    load_all_models: bool = True
    startup_models: list[str] = Field(default_factory=list)
    #: A start-up that keeps going after a model fails to load hides a broken deployment.
    strict_startup: bool = True
    #: Seconds to wait for in-flight batches when shutting down.
    shutdown_grace_s: float = Field(default=10.0, ge=0.0)

    devices: DeviceSettings = Field(default_factory=DeviceSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    http: HttpSettings = Field(default_factory=HttpSettings)

    @model_validator(mode="after")
    def _startup_models_need_selection(self) -> ServerSettings:
        if not self.load_all_models and not self.startup_models:
            raise ValueError("load_all_models=false requires a non-empty startup_models list")
        return self
