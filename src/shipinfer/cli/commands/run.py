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

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from shipinfer.cli.common import build_settings, console
from shipinfer.core.errors import ConfigurationError, ShardExitedError
from shipinfer.core.settings import DeviceSettings, ServerSettings
from shipinfer.launch.control import CameraSpec, mint_camera_id

if TYPE_CHECKING:  # pragma: no cover - typing only; `runners` is imported inside `run()`
    from shipinfer.runners.base import Runner

__all__ = [
    "cameras_from_inputs",
    "cameras_from_settings",
    "cameras_to_place",
    "place_cameras",
    "refuse_if_it_manages_no_cameras",
    "run",
]


def run(
    topology: Path,
    *,
    runner: str | None = None,
    repository: Path | None = None,
    inputs: list[str] | None = None,
    loop: bool = True,
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
        inputs: files or URLs to run as cameras (arch.md §2's offline mode). Each becomes one
            camera, named by its position, placed on the runner once it is up — *after* the
            cameras the settings tree already configures, which are placed the same way
            (:func:`cameras_to_place`). A dry run reports how many there are and places none,
            because placing one means starting a decoder thread — but it still refuses a
            runner that manages no cameras, because that is a fact about the ``--runner``
            chosen rather than about this run.
        loop: whether an ``--inputs`` file restarts at EOF. ``True`` is the historical
            behaviour and what a stress run wants; ``--no-loop`` is how a file is processed
            once. It applies to ``--inputs`` only: a camera the settings tree configures
            already has its own ``loop:`` and keeps it.
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
        ConfigurationError: an unknown runner, an invalid chain, a runner that manages no
            cameras given ``--inputs``, or an input the runner refuses.
    """
    from shipinfer.runners import build_runner
    from shipinfer.runtime.containment import require_container
    from shipinfer.topology import ChainSpec, Topology

    out = console()
    from_inputs = cameras_from_inputs(inputs, loop=loop)
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
    # The configured fleet and `--inputs`, in that order, as ONE list to place. Nobody starts
    # a camera but this: `InprocessRunner` used to start `ingest.cameras` inside its own
    # `_do_start`, which is right for one process and catastrophic for a shard, because a
    # shard is an `InprocessRunner` whose env-only settings inherit the operator's whole
    # fleet -- so all fifty cameras ran on all eight of them. Placing them here instead means
    # the fleet runner spreads them over its shards through `add_camera` and the in-process
    # runner starts them locally, from one line, with no runner knowing which deployment it is.
    cameras = cameras_to_place(settings, inputs, loop=loop)
    # The capability refusal is made HERE, before the dry-run branch, because it is a fact
    # about the runner the operator picked and not about this particular execution: reached
    # only after `start()`, `--dry-run --inputs deepstream` printed a plan and exited 0 for a
    # combination that can never work, and the operator learned it on the real run. The
    # *placement* still happens after `start()` -- a camera is placed on a running runner --
    # so the two halves of the check sit either side of it deliberately.
    refuse_if_it_manages_no_cameras(built, cameras)
    configured = len(cameras) - len(from_inputs)
    if configured:
        out.print(f"cameras: {configured} configured ({cameras[0].camera_id} ...)")
    if from_inputs:
        out.print(f"cameras: {len(from_inputs)} from --inputs ({from_inputs[0].camera_id} ...)")
    if dry_run:
        # On the contract, not probed for: `Runner.describe_plan` has an in-process default
        # ("no plan: one process") and the fleet overrides it with the plan it would run.
        out.print(built.describe_plan())
        return 0

    built.start()
    try:
        # After `start`, because a camera is placed on a *running* runner: the chain has to be
        # open and its workers up before a decoder thread starts publishing into them. A
        # refusal here therefore travels through the `finally`, which stops what did come up.
        place_cameras(built, cameras)
        _wait(built)
    except ShardExitedError as exc:
        out.print(f"[red]{exc}[/red]")
        return 1
    finally:
        built.stop()
    return 0


def cameras_from_inputs(inputs: Sequence[str] | None, *, loop: bool = True) -> list[CameraSpec]:
    """``--inputs a.mp4 rtsp://b`` as the camera specs a runner takes.

    Identity is **positional** — ``cam-000``, ``cam-001``, minted by
    :func:`~shipinfer.launch.control.mint_camera_id`, which is also what ``POST /streams``
    uses for a request that supplies no id — and that is the only sensible
    answer: a file path is not a camera id (two directories can hold the same ``clip.mp4``,
    and a path is not a legal metric label), and the offline mode has nobody to ask. Stable
    across restarts for a fixed argument list, which is what a downstream tracker keyed on
    ``camera_id`` needs, and deliberately zero-padded so ``cam-010`` sorts after ``cam-009``
    in every log and dashboard that sorts strings.

    ``loop`` is ``--loop/--no-loop``, and it is on the spec rather than on the settings tree
    because these cameras exist nowhere in it: they are minted here, so ``ingest.cameras[]``
    has no entry whose ``loop:`` could reach them, and ``shipinfer run --inputs clip.mp4``
    had no configuration anywhere that would let the file finish.

    An empty or absent list gives an empty list, not an error: ``shipinfer run`` with no
    inputs is the normal way to bring a chain up and add cameras over the control plane.
    """
    return [
        CameraSpec(camera_id=mint_camera_id(index), url=url, fps=0.0, loop=loop)
        for index, url in enumerate(inputs or ())
    ]


def cameras_from_settings(settings: ServerSettings) -> list[CameraSpec]:
    """``ingest.cameras`` plus ``ingest.camera_db`` as the camera specs a runner takes.

    The deployment's own fleet, in the launcher's vocabulary, so that it is placed through
    the same door as everything else. A runner used to read this list itself, which made the
    camera set a property of *whatever settings a process happened to load* — and a shard
    loads the operator's, so every shard read every camera (see :func:`run`).

    Only the four fields a launcher decides travel. The rest of a camera's configuration is
    deployment settings and the process that runs it resolves them from its own tree, which
    is the split :class:`~shipinfer.launch.control.CameraSpec` exists to state: the shard
    keeps the codec, the transport and the priority band from its settings, and is *told*
    which camera to open.

    The import is inside the function: ``shipinfer.ingest`` reaches a decode runtime through
    its source registry, and ``shipinfer repo ls`` must not pay for one.

    Raises:
        ConfigurationError: the camera database cannot be read, or declares a camera the
            inline list already names.
    """
    from shipinfer.ingest import configured_cameras

    return [
        CameraSpec(camera_id=camera.camera_id, url=camera.uri, fps=camera.fps, loop=camera.loop)
        for camera in configured_cameras(settings.ingest)
    ]


def cameras_to_place(
    settings: ServerSettings, inputs: Sequence[str] | None, *, loop: bool = True
) -> list[CameraSpec]:
    """Every camera this run should open, in the order they are offered to the runner.

    **Configured first, ``--inputs`` after**, and the order is a decision rather than a
    coincidence: the configured fleet is the deployment, the inputs are what this invocation
    adds to it, and a placement policy that fills the least-loaded shard first will spread
    the standing fleet evenly before the extras land on top of it. It also makes the failure
    legible when the two collide — a ``--inputs`` camera whose id is already configured is
    refused naming the id, rather than the configured camera being refused by the input.
    """
    return [*cameras_from_settings(settings), *cameras_from_inputs(inputs, loop=loop)]


def refuse_if_it_manages_no_cameras(runner: Runner, cameras: Sequence[CameraSpec]) -> None:
    """Refuse cameras on a runner that owns no ingest plane. Starts nothing.

    ``cameras`` is whatever this run would place — ``--inputs``, the configured fleet, or
    both — because the refusal is about the *runner*: a chain executor with no ingest plane
    opens neither kind, and a message that named only the flag would send an operator whose
    fleet is in ``ingest.camera_db`` looking for a flag they never typed.

    Separate from :func:`place_cameras` because the two answer at different moments.
    Placement needs a *running* runner — a camera is placed on an open chain — while this is a
    fact about the class the operator named, known as soon as it is built. Asking it late made
    ``--dry-run --inputs`` print a plan and exit ``0`` for a combination that cannot run at
    all, which is the one thing a dry run exists to catch.

    Raises:
        ConfigurationError: the runner manages no cameras (the ``--runner`` chosen executes a
            chain but owns no ingest plane, so nothing would open the files).
    """
    if not cameras or runner.manages_cameras:
        return
    raise ConfigurationError(
        f"{len(cameras)} camera(s) were given — `--inputs`, or `ingest.cameras` / "
        f"`ingest.camera_db` in the settings — but runner {runner.name!r} manages no "
        "cameras, so nothing would open them; choose a runner that does with `--runner` "
        "(`shipinfer runners` lists them), or feed the chain through the control plane"
    )


def place_cameras(runner: Runner, cameras: Sequence[CameraSpec]) -> None:
    """Start every camera on the runner, or refuse naming the first one that would not.

        Stops at the first failure rather than placing the rest, because the failures reachable
        here are configuration ones — a duplicate id, a source the chain names that nobody
        registered — and each of them means the same thing about every camera behind it. Carrying
        on would report the last one's message for a mistake made in the first.

        The capability refusal is :func:`refuse_if_it_manages_no_cameras`'s and is made before the
        runner is started, but it is repeated here rather than assumed: this function is public and
        a caller that placed cameras on a camera-less runner would otherwise get
        ``Runner.add_camera``'s default ``ServerStateError`` per camera instead of one message
        naming what to do about it.

        Raises:
            ConfigurationError: the runner manages no cameras, or a camera was refused. In the
                second case the original message is kept and prefixed with **both** the id and
                the url, because ``ingest/manager.py``'s message names only the camera id — which
                for an ``--inputs`` camera is minted from its position and appears nowhere in what
                the operator typed.
            NoShardAvailableError: a fleet whose shards all refused. Deliberately **not**
                re-labelled with the input path: it is not a fact about this input at all -- the
                camera is fine and there is nowhere to put it -- and wrapping it in a
                ``ConfigurationError`` would turn a 503 into a 400 for the same condition reached
                over ``POST /streams`` (``api/errors.py``). Its own message names the camera and
                what every shard said.
    """
    refuse_if_it_manages_no_cameras(runner, cameras)
    for camera in cameras:
        try:
            runner.add_camera(camera)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"camera {camera.camera_id} ({camera.url!r}) was refused: {exc}"
            ) from exc


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
