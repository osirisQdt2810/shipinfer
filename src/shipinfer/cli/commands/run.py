"""``shipinfer run`` — execute a topology, under the runner the operator picks.

The command arch.md §2 names for the offline mode: *"``shipinfer run --topology
ship_person.yaml --inputs a.mp4 b.mp4 …``"*. One chain definition, three executions — the
chain is a file, and ``--runner`` is which of them runs it:

* ``inprocess`` — the whole chain on a thread pool here. Dev, tests, few cameras.
* ``fleet`` — one shard process per GPU, driven over gRPC. The production default.
* ``deepstream`` — the chain compiled into a GStreamer graph (phase E).

It replaces ``shipinfer fleet``, and the difference is the whole point of the phase. That
command took a *model repository* and a *placement* named "topology", and rendered a command
line for each child. This one takes the **chain** and a **runner**, and the children are told
what to do over the control plane.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from shipinfer.cli.common import build_settings, console
from shipinfer.core.errors import ConfigurationError, ShardExitedError
from shipinfer.core.settings import DeviceSettings, ServerSettings

if TYPE_CHECKING:  # pragma: no cover - typing only; `runners` is imported inside `run()`
    from shipinfer.runners.base import Runner

__all__ = ["run"]


def run(
    topology: Path,
    *,
    runner: str | None = None,
    repository: Path | None = None,
    inputs: list[str] | None = None,
    shards: int | None = None,
    gpus: str | None = None,
    policy: str | None = None,
    drain_s: float | None = None,
    log_level: str = "INFO",
    dry_run: bool = False,
) -> int:
    """Load a chain and run it. Blocks until interrupted.

    Args:
        topology: the chain file. Read **once**, here: the text is what a shard is sent (the
            loader lives on the shard, ADR-017) and the parsed chain is what this process
            validates before anything is spawned. A mistyped chain therefore fails on this
            line rather than on sixteen children at once.
        runner: which runner executes it. ``None`` takes ``runner.runner`` from the settings.
        repository: the model repository, for the runners that load one.
        inputs: files or URLs to shard across the fleet at start (arch.md §2's offline mode).
            **Accepted and stored, not yet wired**: turning one into a running camera is the
            ingest half of phase B, and a flag that silently did nothing would be worse than
            one that says so — so a non-empty list is refused rather than ignored.
        shards: how many shard processes, for the runners that have any. ``None`` leaves
            ``runner.shards``, whose own default is one per visible GPU (ADR-006).
        gpus: which devices, as ``0,1,2``. ``None`` leaves ``devices.visible_gpus``, and
            only when *that* is empty too is the driver asked, once (:func:`_fill_in_gpus`).
            A flag is not the only way an operator answers this question.
        drain_s: seconds a shard gets to finish before it is killed. ``None`` leaves
            ``runner.drain_s``. It is ``shipinfer fleet --drain`` under its settings-tree
            name: the same budget, on the command that replaced it, so an operator who had
            tuned it for a slow decoder does not silently get the default back.
        dry_run: resolve everything and print it, spawn nothing. The plan is the decision
            worth reading before fifty cameras start reconnecting.

    Raises:
        ConfigurationError: an unknown runner, an invalid chain, or ``--inputs`` (see above).
    """
    from shipinfer.runners import build_runner
    from shipinfer.runtime.containment import require_container
    from shipinfer.topology import ChainSpec, Topology

    out = console()
    if inputs:
        raise ConfigurationError(
            f"--inputs is not wired yet ({len(inputs)} given): the chain runs, but nothing "
            "opens a file or an RTSP URL for it until the ingest half of phase B. Until then "
            "a camera reaches a running fleet through the control plane"
        )
    # The gate lives here as well as in the shell hook: a deny-list over command text cannot
    # be made sound, and this command loads engines and drives GPUs (`serve` says the same).
    # It comes BEFORE anything is resolved so that `_fill_in_gpus()` - the one thing here that
    # can touch the driver - runs below it and not above: a refusal should not have queried
    # the hardware it is refusing to use. `--dry-run` is exempt because it spawns nothing; it
    # is the mode for reading a plan on a laptop.
    if not dry_run:
        require_container("`shipinfer run`")
    # Every flag lands in the *settings tree* rather than in per-runner keyword arguments, so
    # this command names no runner: `--shards` is `runner.shards`, `--drain-s` is
    # `runner.drain_s` and `--gpus` is `devices.visible_gpus`, and a runner that cares reads
    # them. A CLI that special-cased `fleet` here would be the if/elif the registry exists to
    # prevent (CONVENTIONS 2.3).
    runner_keys: dict[str, object] = {}
    if shards is not None:
        runner_keys["shards"] = shards
    if drain_s is not None:
        runner_keys["drain_s"] = drain_s
    settings = build_settings(
        repository,
        gpus=gpus,
        policy=policy,
        log_level=log_level,
        **({} if not runner_keys else {"runner": runner_keys}),
    )
    # AFTER the settings, never as an argument to them - see `_fill_in_gpus`.
    settings = _fill_in_gpus(settings)
    chain_yaml = _read(topology)
    chain = Topology.from_spec(ChainSpec.from_yaml(chain_yaml, name=Path(topology).stem))
    chosen = runner or settings.runner.runner
    out.print(f"topology: {chain.name} ({len(list(chain))} element(s)) — runner {chosen}")

    built = build_runner(chosen, chain, settings, chain_yaml=chain_yaml)
    if dry_run:
        # On the contract, not probed for: `Runner.describe_plan` has an in-process default
        # ("no plan: one process") and the fleet overrides it with the plan it would run.
        out.print(built.describe_plan())
        return 0

    built.start()
    try:
        _wait(built)
    except ShardExitedError as exc:
        out.print(f"[red]{exc}[/red]")
        return 1
    finally:
        built.stop()
    return 0


def _read(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read topology {path}: {exc}") from exc


def _fill_in_gpus(settings: ServerSettings) -> ServerSettings:
    """Put every device the driver reports into ``devices.visible_gpus``, if nothing did.

    **Resolved after the settings, and that ordering is the whole correctness of this
    function.** ``build_settings`` hands a flag to ``ServerSettings(**data)`` as an init
    keyword argument, which is pydantic-settings' *highest*-priority source: a driver answer
    passed in there outranks ``SHIPINFER_DEVICES__VISIBLE_GPUS``, so an operator who
    restricted this deployment to the two devices they own would silently get eight shards on
    eight GPUs - each holding a CUDA context on a box everybody shares - with nothing saying
    the setting had been read and dropped. So the flag goes in alone, the resolved tree is
    what answers "did anybody say", and the driver is asked only when the answer is no. A
    ``--dry-run`` on a configured host therefore touches no driver at all.

    Resolved **here** rather than inside a runner, and that reason survives too: ``runners``
    imports no ``runtime`` (``check_layers.py``), because ``device_count()`` initialises CUDA
    in whichever process calls it and the launcher is the one process in the deployment that
    should hold no context. The CLI is already above that line, and this is the same call
    ``shipinfer fleet`` made.

    The section is rebuilt through :class:`~shipinfer.core.settings.DeviceSettings` rather
    than ``model_copy``d on the leaf so its field validators run - the same argument
    :func:`shipinfer.cli.shard.apply_sharing` makes for ``shared_by``. A driver that reports
    none leaves the tree alone, and the runner that needs devices refuses by name.
    """
    if settings.devices.visible_gpus:
        return settings
    from shipinfer.runtime.platform import device_count

    reported = list(range(device_count()))
    if not reported:
        return settings
    devices = DeviceSettings(**{**settings.devices.model_dump(), "visible_gpus": reported})
    return settings.model_copy(update={"devices": devices})


def _wait(built: Runner) -> None:
    """Block until Ctrl-C, SIGTERM, or a shard dies.

    Signals are installed here rather than by the runner: handlers are process-global state,
    and a library that installed them behind your back is a library you cannot embed
    (``launch/signals.py`` makes the same argument for the fleet).

    Every runner supervises — :meth:`Runner.supervise` waits to be told to go, and the fleet
    overrides it to also notice a shard that exits while the rest keep reporting healthy. So
    there is no probe here for which kind this is: the contract is the contract, and a
    ``getattr`` fallback would have silently downgraded a fleet whose method got renamed into
    one that never watched its shards.

    Raises:
        ShardExitedError: a shard exited; the fleet is already stopped.
    """
    from shipinfer.launch import forward_signals

    forward_signals(built)
    built.supervise()
