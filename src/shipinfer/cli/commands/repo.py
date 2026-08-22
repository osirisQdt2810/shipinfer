"""``shipinfer repo`` — inspect a model repository without starting a server."""

from __future__ import annotations

from pathlib import Path

from shipinfer.cli.common import console, print_table
from shipinfer.repository import ModelRepository
from shipinfer.runtime.platform import device_count

__all__ = ["repo_list", "repo_show"]


def repo_list(repository: Path) -> int:
    """List models, versions, platforms and how many instances each would create.

    Runs without loading a single engine, which is the point: on a laptop with no CUDA you
    can still check that a repository is well-formed before shipping it to the node.
    """
    repo = ModelRepository.load(repository)
    visible = tuple(range(device_count()))

    rows = []
    for entry in repo:
        try:
            placements = entry.config.placements(visible)
            spread = ", ".join(sorted({str(p.device) for p in placements}))
            instances = f"{len(placements)} ({spread})"
        except Exception as exc:
            instances = f"[red]{exc}[/red]"
        rows.append(
            [
                entry.name,
                entry.config.platform,
                ",".join(str(v) for v in entry.versions),
                str(entry.config.max_batch_size),
                instances,
            ]
        )
    print_table(
        f"Models in {repo.root}",
        ["name", "platform", "versions", "max batch", "instances here"],
        rows,
    )
    return 0


def repo_show(repository: Path, name: str) -> int:
    """Print one model's resolved configuration."""
    out = console()
    repo = ModelRepository.load(repository)
    artifact = repo.resolve(name)
    config = artifact.config

    out.print(f"[bold]{config.name}[/bold] v{artifact.version}  platform={config.platform}")
    out.print(f"  path: {artifact.path}")
    out.print(f"  max_batch_size: {config.max_batch_size}")
    out.print(f"  inputs:  {[s.describe() for s in config.input_specs]}")
    out.print(f"  outputs: {[s.describe() for s in config.output_specs]}")
    batching = config.dynamic_batching
    out.print(
        f"  dynamic_batching: enabled={batching.enabled} "
        f"delay={batching.max_queue_delay_us}us preferred={batching.preferred_batch_sizes}"
    )
    for group in config.instance_groups:
        out.print(
            f"  instance_group: kind={group.kind.value} count={group.count} "
            f"gpus={group.gpus or 'all'} streams={group.streams}"
        )
    if config.parameters:
        out.print(f"  parameters: {config.parameters}")
    return 0
