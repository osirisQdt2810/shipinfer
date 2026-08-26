"""Shared CLI plumbing: settings construction and console output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import configure
from shipinfer.core.settings import ServerSettings
from shipinfer.envs import SHARD_CAMERAS

__all__ = ["build_settings", "console", "print_table"]


def build_settings(
    repository: Path | None = None,
    *,
    gpus: str | None = None,
    policy: str | None = None,
    log_level: str = "INFO",
    **overrides: Any,
) -> ServerSettings:
    """Assemble settings from CLI flags on top of the environment.

    Flags win over ``SHIPINFER_*`` env vars, which win over defaults — the ordering people
    expect, and the reason a one-off ``--policy round_robin`` does not require unsetting
    anything.
    """
    configure(log_level, force=True)

    data: dict[str, Any] = dict(overrides)
    if repository is not None:
        data["model_repository"] = repository
    if gpus is not None:
        parsed = [int(g) for g in gpus.replace(" ", "").split(",") if g != ""]
        data["devices"] = {"visible_gpus": parsed}
    if policy is not None:
        data["scheduler"] = {"placement_policy": policy}
    settings = ServerSettings(**data)
    return _narrow_to_shard(settings)


def _narrow_to_shard(settings: ServerSettings) -> ServerSettings:
    """Keep only this shard's cameras, when the launcher says which ones are ours.

    Every shard is started from the *same* configuration — that is the point, since a fleet
    described in two places is a fleet that can disagree with itself — and each is told which
    slice of it to read through :data:`shipinfer.envs.SHARD_CAMERAS`. With the variable unset this is
    the identity, which is what a single-process run is.

    A named camera that the configuration does not have is refused rather than skipped. The
    plan and the config are two views of one fleet; if they disagree, some camera is going
    unread, and the shard that would have read it is the only thing that can notice.
    """
    raw = os.environ.get(SHARD_CAMERAS.name)
    if raw is None:
        return settings
    if not raw.strip():
        # Set-but-empty is a launcher bug, and reading it as "everything" would give every
        # shard every camera. `EnvVar.get` treats a blank as unset, so the check is explicit.
        raise ConfigurationError(
            f"{SHARD_CAMERAS.name} is set but empty. A shard with no cameras still loads "
            f"engines and holds a CUDA context; unset the variable to serve the whole fleet"
        )
    wanted = list(SHARD_CAMERAS.get())  # names only separators -> refused, variable named
    available = {camera.camera_id: camera for camera in settings.ingest.cameras}
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ConfigurationError(
            f"{SHARD_CAMERAS.name} names {unknown}, which this configuration does not define "
            f"(it has {sorted(available)}). The plan and the config are two views of one "
            f"fleet, so a disagreement means a camera is going unread"
        )
    return settings.model_copy(
        update={
            "ingest": settings.ingest.model_copy(
                update={"cameras": [available[name] for name in wanted]}
            )
        }
    )


def console() -> Any:
    """A rich console when rich is installed, else a shim with the same ``print``."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:  # pragma: no cover - only on a bare install

        class _Plain:
            @staticmethod
            def print(*args: Any, **_: Any) -> None:
                print(*args)

        return _Plain()


def print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Render a table with rich, or as aligned text without it."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title, header_style="bold")
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        Console().print(table)
        return
    except ImportError:  # pragma: no cover
        pass

    widths = [
        max(len(columns[i]), *(len(r[i]) for r in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    print(f"\n{title}")
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
