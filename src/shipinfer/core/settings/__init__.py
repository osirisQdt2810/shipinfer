"""Server settings: one tree, one file per section, layered defaults -> file -> env.

Environment overrides use the ``SHIPINFER_`` prefix and ``__`` for nesting::

    SHIPINFER_SCHEDULER__PLACEMENT_POLICY=power_of_two
    SHIPINFER_EXECUTION__CUDA_GRAPHS=false
    SHIPINFER_DEVICES__VISIBLE_GPUS='[0,1,2,3]'
"""

from shipinfer.core.settings.device import DeviceSettings
from shipinfer.core.settings.enums import ExecutionProvider, OverflowPolicy
from shipinfer.core.settings.execution import ExecutionSettings
from shipinfer.core.settings.http import HttpSettings
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.core.settings.memory import MemorySettings
from shipinfer.core.settings.observability import ObservabilitySettings
from shipinfer.core.settings.scheduler import SchedulerSettings
from shipinfer.core.settings.server import ServerSettings

__all__ = [
    "DeviceSettings",
    "ExecutionProvider",
    "ExecutionSettings",
    "CameraConfig",
    "HttpSettings",
    "IngestSettings",
    "MemorySettings",
    "ObservabilitySettings",
    "OverflowPolicy",
    "SchedulerSettings",
    "ServerSettings",
]
