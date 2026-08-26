"""``shipinfer fleet`` — run the deployment as several processes instead of one."""

from __future__ import annotations

from pathlib import Path

from shipinfer.cli.common import build_settings, console
from shipinfer.core.errors import ConfigurationError, ShardExitedError
from shipinfer.runtime.containment import require_container
from shipinfer.server.launcher import Fleet, forward_signals
from shipinfer.server.topology import build_topology

__all__ = ["fleet"]


def fleet(
    repository: Path,
    *,
    shards: int | None = None,
    gpus: str | None = None,
    policy: str | None = None,
    topology: str | None = None,
    log_level: str = "INFO",
    dry_run: bool = False,
    drain_s: float | None = None,
    http_port_base: int | None = None,
) -> int:
    """Split the configured fleet across ``shards`` processes and supervise them.

    Each shard is one ``shipinfer serve`` in its own process, reading the *same*
    configuration and told which slice of the cameras is its own. That is the point rather
    than a shortcut: a fleet described in two places is a fleet that can disagree with
    itself, and the disagreement is silent — a camera in the plan and not in the config is
    simply never read.

    ``--dry-run`` prints the plan and stops. Worth having because the plan is the decision:
    which camera lands on which GPU is stable across restarts, so it is worth looking at
    before fifty of them start reconnecting.
    """
    # A shard is a server, so the same gate applies — and it applies here rather than only in
    # the children, so a host run fails before sixteen processes have started.
    if not dry_run:
        require_container("`shipinfer fleet`")
    out = console()

    settings = build_settings(repository, gpus=gpus, policy=policy, log_level=log_level)
    # `fps` is 0 for "whatever the source delivers", which is most RTSP cameras. Weighting a
    # plan by zero would make every camera equal and turn the balance back into a count —
    # rebuilding the failure this exists to fix. So an unspecified rate counts as 1.0, and a
    # fleet that declares none is balanced by count, which is the right answer when the only
    # thing known about the cameras is how many there are.
    cameras = {c.camera_id: (c.fps or 1.0) for c in settings.ingest.cameras}
    if not cameras:
        raise ConfigurationError(
            "this configuration defines no cameras, so there is nothing to shard. `serve` is "
            "the command for a model-only deployment"
        )
    devices = list(settings.devices.visible_gpus or [])
    if not devices:
        # Empty means "every device the driver reports", as it does for `serve` and for
        # `DeviceManager` — a fleet that refused what a single process accepts would be the
        # odd one out. The driver is asked here, once, and only when nothing was configured.
        from shipinfer.runtime.platform import device_count

        devices = list(range(device_count()))
    if not devices:
        raise ConfigurationError(
            "no visible GPUs: none configured and the driver reports none. A shard's device set "
            "comes from the plan, so the plan needs to know what there is — pass --gpus, or set "
            "devices.visible_gpus"
        )

    # The topology decides the plan and what the children are told. `--topology`, `--shards`
    # and `--drain` override the settings section; unset, the section decides, and the
    # section's own default for `shards` is one process per visible GPU (ADR-006).
    chosen = build_topology(topology or settings.topology.kind)
    count = shards if shards is not None else (settings.topology.shards or len(devices))
    drain = drain_s if drain_s is not None else settings.topology.drain_s
    out.print(f"topology: {chosen.name} — {chosen.describe()}")
    plan = chosen.plan(settings, cameras=cameras, gpus=devices, shards=count)
    out.print(plan.describe())
    if plan.device_imbalance > 0.4:
        # Not an error: an imbalanced plan can be the best available split of a lopsided
        # fleet. It is printed loudly because the fleet's throughput is bounded by its
        # busiest *device*, so this is the number that decides whether sharding helped.
        out.print(
            f"[yellow]warning[/yellow]: GPUs differ by {plan.device_imbalance:.0%} in offered "
            f"load. The fleet is bounded by its busiest shard, so this plan adds less than "
            f"its shard count suggests"
        )
    if dry_run:
        return 0

    running = Fleet(
        plan=plan,
        command=lambda shard: chosen.command(
            shard, repository=str(repository), http_port_base=http_port_base
        ),
        env=chosen.environment(settings),
        shard_env=chosen.shard_environment,
        drain_s=drain,
    )
    # The handler goes in before the first child exists: a signal in the window between
    # `start()` and the install would escape as KeyboardInterrupt and orphan children that
    # `start_new_session=True` has already detached from the terminal.
    forward_signals(running)
    running.start()
    try:
        running.supervise()
    except ShardExitedError as exc:
        out.print(f"[red]{exc}[/red]")
        return 1
    finally:
        running.stop()
    return 0
