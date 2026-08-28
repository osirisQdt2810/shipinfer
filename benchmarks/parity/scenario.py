"""The scenario format: what each camera's decode path is told to do, call by call.

Line-oriented and not JSON, so the C++ half parses it with no vendored library. One
directive per line, whitespace-separated, ``#`` to end of line is a comment::

    scenario reconnect            # exactly once, first
    records_min 24                # the vacuity floor this scenario promises
    reconnect_initial_ms 2        # a fleet setting, before the first camera
    camera cam0 [disabled]
    open ok|SourceOpenError|SourceUnavailableError [detail]
    read frame|empty|exhaust|FrameDecodeError [detail]
    sink accept|full|closed

Each list's last entry repeats for ever, so every enabled camera must end somewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "FLOAT_SETTINGS",
    "INT_SETTINGS",
    "OPEN_OUTCOMES",
    "READ_OUTCOMES",
    "SINK_OUTCOMES",
    "CameraScript",
    "Scenario",
    "load_scenario",
]

#: Fleet settings a scenario may set, by the name both planes already use for them.
INT_SETTINGS = frozenset(
    {
        "empty_reads_before_reconnect",
        "empty_read_sleep_ms",
        "failures_before_unhealthy",
        "reconnect_initial_ms",
        "reconnect_max_ms",
    }
)
FLOAT_SETTINGS = frozenset({"reconnect_factor", "reconnect_jitter"})

#: Stated by every scenario rather than inherited: these four are what each plane's actor
#: builds its `ExponentialBackoff` from -- and the `retry` record reports that backoff's own
#: peek, so a scenario that left them to the default would compare two different backoffs.
#: `empty_read_sleep_ms` paces a run the trace does not record.
REQUIRED_SETTINGS = (
    "empty_read_sleep_ms",
    "reconnect_initial_ms",
    "reconnect_max_ms",
    "reconnect_factor",
    "reconnect_jitter",
)

OPEN_OUTCOMES = ("ok", "SourceOpenError", "SourceUnavailableError")
READ_OUTCOMES = ("frame", "empty", "exhaust", "FrameDecodeError")
SINK_OUTCOMES = ("accept", "full", "closed")


@dataclass(frozen=True, slots=True)
class CameraScript:
    """One camera's outcomes, in the order its actor will consume them."""

    camera_id: str
    enabled: bool
    opens: tuple[tuple[str, str], ...]
    reads: tuple[tuple[str, str], ...]
    sinks: tuple[str, ...]

    def open_at(self, index: int) -> tuple[str, str]:
        return self.opens[min(index, len(self.opens) - 1)]

    def read_at(self, index: int) -> tuple[str, str]:
        return self.reads[min(index, len(self.reads) - 1)]

    def sink_at(self, index: int) -> str:
        return self.sinks[min(index, len(self.sinks) - 1)] if self.sinks else "accept"


@dataclass(frozen=True, slots=True)
class Scenario:
    """A whole run: the fleet settings, every camera's script, and the vacuity floor."""

    name: str
    records_min: int
    settings: Mapping[str, str]
    cameras: tuple[CameraScript, ...]

    def int_setting(self, key: str) -> int:
        return int(self.settings[key])

    def float_setting(self, key: str) -> float:
        return float(self.settings[key])

    def camera(self, camera_id: str) -> CameraScript:
        for script in self.cameras:
            if script.camera_id == camera_id:
                return script
        raise ConfigurationError(
            f"scenario {self.name!r} has no camera {camera_id!r}; it has "
            f"{[c.camera_id for c in self.cameras]}"
        )


def _refuse(path: Path, number: int, why: str) -> ConfigurationError:
    return ConfigurationError(f"{path}:{number}: {why}")


def load_scenario(path: Path) -> Scenario:
    """Parse one ``.scn`` file.

    Raises:
        ConfigurationError: any malformed line, naming ``path:line`` -- a scenario that
            half-parses would drive half a run and compare it against a whole golden.
    """
    if not path.is_file():
        raise ConfigurationError(f"no parity scenario at {path}")
    name = ""
    records_min = -1
    settings: dict[str, str] = {}
    cameras: list[dict[str, object]] = []
    for number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        directive, rest = tokens[0], tokens[1:]
        if directive == "scenario":
            if name or len(rest) != 1:
                raise _refuse(path, number, "expected exactly one `scenario <name>`, first")
            name = rest[0]
        elif directive == "records_min":
            if len(rest) != 1 or not rest[0].isdigit():
                raise _refuse(path, number, "expected `records_min <non-negative integer>`")
            records_min = int(rest[0])
        elif directive in INT_SETTINGS or directive in FLOAT_SETTINGS:
            if cameras:
                raise _refuse(path, number, f"setting {directive!r} must precede any camera")
            if len(rest) != 1:
                raise _refuse(path, number, f"expected `{directive} <value>`")
            settings[directive] = rest[0]
        elif directive == "camera":
            if not rest or len(rest) > 2 or (len(rest) == 2 and rest[1] != "disabled"):
                raise _refuse(path, number, "expected `camera <id> [disabled]`")
            if any(c["camera_id"] == rest[0] for c in cameras):
                raise _refuse(path, number, f"camera {rest[0]!r} is declared twice")
            cameras.append(
                {
                    "camera_id": rest[0],
                    "enabled": len(rest) == 1,
                    "open": [],
                    "read": [],
                    "sink": [],
                }
            )
        elif directive in ("open", "read", "sink"):
            if not cameras:
                raise _refuse(path, number, f"`{directive}` before any `camera <id>`")
            allowed = {"open": OPEN_OUTCOMES, "read": READ_OUTCOMES, "sink": SINK_OUTCOMES}[
                directive
            ]
            if not rest or rest[0] not in allowed:
                raise _refuse(
                    path, number, f"`{directive}` outcome must be one of {list(allowed)}"
                )
            if len(rest) > (2 if directive != "sink" else 1):
                raise _refuse(
                    path, number, f"`{directive}` takes an outcome and at most one detail"
                )
            entry = (
                rest[0] if directive == "sink" else (rest[0], rest[1] if len(rest) > 1 else "")
            )
            cameras[-1][directive].append(entry)  # type: ignore[union-attr]
        else:
            raise _refuse(path, number, f"unknown directive {directive!r}")
    return _finished(path, name, records_min, settings, cameras)


def _finished(
    path: Path,
    name: str,
    records_min: int,
    settings: dict[str, str],
    cameras: list[dict[str, object]],
) -> Scenario:
    """Turn the parsed lines into a scenario, refusing one that could never end."""
    if not name:
        raise ConfigurationError(f"{path}: no `scenario <name>` line")
    if records_min < 0:
        raise ConfigurationError(f"{path}: no `records_min <n>` line; a golden with no floor")
    missing = [key for key in REQUIRED_SETTINGS if key not in settings]
    if missing:
        raise ConfigurationError(f"{path}: missing required setting(s) {missing}")
    scripts = []
    for entry in cameras:
        script = CameraScript(
            camera_id=str(entry["camera_id"]),
            enabled=bool(entry["enabled"]),
            opens=tuple(entry["open"]),  # type: ignore[arg-type]
            reads=tuple(entry["read"]),  # type: ignore[arg-type]
            sinks=tuple(entry["sink"]),  # type: ignore[arg-type]
        )
        if script.enabled:
            _refuse_endless(path, script)
        scripts.append(script)
    return Scenario(
        name=name, records_min=records_min, settings=settings, cameras=tuple(scripts)
    )


def _refuse_endless(path: Path, script: CameraScript) -> None:
    """A camera whose script never terminates would hang the harness, not fail it."""
    if not script.opens or not script.reads:
        raise ConfigurationError(
            f"{path}: camera {script.camera_id!r} needs at least one `open` and one `read`"
        )
    ends = (
        script.reads[-1][0] == "exhaust"
        or script.opens[-1][0] == "SourceUnavailableError"
        or "closed" in script.sinks
    )
    if not ends:
        raise ConfigurationError(
            f"{path}: camera {script.camera_id!r} never finishes -- the last entry of each "
            f"list repeats for ever, so a script must end in `read exhaust`, "
            f"`open SourceUnavailableError` or a `sink closed`"
        )
