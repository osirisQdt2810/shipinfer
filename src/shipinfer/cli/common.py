"""Shared CLI plumbing: settings construction and console output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shipinfer.core.logging import configure
from shipinfer.core.settings import ServerSettings

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

    It used to narrow the camera list to ``SHIPINFER_SHARD_CAMERAS`` as well. It does not any
    more: a shard is told its cameras over ``AddCamera``, on a process that is already running
    (arch.md section 2), so every process built from these settings reads the whole configured
    fleet — which is what a single-process run always did.
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
    return ServerSettings(**data)


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
