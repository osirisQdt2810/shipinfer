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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from shipinfer.cli.common import build_settings, console
from shipinfer.core.errors import ConfigurationError, ShardExitedError
from shipinfer.core.settings import DeviceSettings, ServerSettings
from shipinfer.launch.control import CameraSpec, mint_camera_id

if TYPE_CHECKING:  # pragma: no cover - typing only; every one is imported inside a function
    from rich.console import Console

    from shipinfer.engine import InferenceServer
    from shipinfer.runners.base import Runner
    from shipinfer.runtime.ops import ImageOps
    from shipinfer.topology import Topology

__all__ = [
    "cameras_from_inputs",
    "cameras_from_settings",
    "cameras_to_place",
    "image_ops_are_needed",
    "model_pool_is_needed",
    "place_cameras",
    "refuse_flags_that_would_be_ignored",
    "refuse_if_it_manages_no_cameras",
    "run",
]

#: What ``--http`` binds when the operator names neither. Loopback, because ``/streams``
#: starts and stops decoding on a shared GPU box and phase B puts no authentication in front
#: of it. Named constants rather than defaults in :func:`run`'s signature so ``None`` can mean
#: "the operator did not say", which is what :func:`refuse_flags_that_would_be_ignored` needs
#: to tell an ignored flag from an unmentioned one.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


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
    http: bool = False,
    host: str | None = None,
    port: int | None = None,
    log_level: str = "INFO",
    dry_run: bool = False,
) -> int:
    """Load a chain and run it. Blocks until interrupted.

    The whole command in one reading: refuse what can be refused for free, resolve the plan,
    say what it is, build it, serve it. Each of those is one function below, and the order is
    the design — every refusal that costs no CUDA context is above :func:`_bring_up`, because
    :class:`~shipinfer.engine.InferenceServer` takes a primary context per visible GPU
    (~200 MiB) that nothing in this process gives back.

    ``None`` for ``runner``, ``shards``, ``gpus``, ``drain_s`` and ``port`` means "the settings
    tree decides". ``--inputs`` files become cameras placed after the configured ones and
    ``--no-loop`` applies to them only. ``--http`` serves arch.md §2's camera door on a thread,
    bound to loopback unless the operator says otherwise — those routes start and stop decoding
    on a shared box and phase B has no authentication — and ``--host``/``--port`` without it are
    refused, not ignored. ``--dry-run`` resolves and prints the plan and spawns nothing.

    Raises:
        ConfigurationError: an unknown runner, an invalid chain, an unreadable
            ``ingest.camera_db``, a runner that manages no cameras, an input it refuses,
            ``--host``/``--port`` without ``--http``, a missing ``server`` extra, an HTTP port
            that would not bind, or an element that will not open.
        ServerStateError: the model pool this chain needs would not start.
        BackendUnavailableError: a model needs a runtime this host has not got.
        ModelNotFoundError: a ``pool`` element names a model the pool did not load.

    Note:
        The list is not closed — a backend may raise a bare ``ValueError``. It is about what an
        operator reads, not about what is held: every one of them leaves the GPU as it found
        it, which is :func:`_bring_up`'s job and stated there.
    """
    from shipinfer.runners import RUNNERS
    from shipinfer.runtime.containment import require_container

    out = console()
    # The gate lives here as well as in the shell hook: a deny-list over command text cannot
    # be made sound, and this command loads engines and drives GPUs (`serve` says the same).
    # It comes BEFORE anything is resolved so that `_fill_in_gpus()` — the one thing here
    # that can touch the driver — runs below it and not above. `--dry-run` is exempt because
    # it spawns nothing; it is the mode for reading a plan on a laptop.
    if not dry_run:
        require_container("`shipinfer run`")
    # A fact about the command line alone, and the cheapest refusal there is.
    refuse_flags_that_would_be_ignored(http=http, host=host, port=port)

    plan = _resolve(
        out,
        topology,
        runner=runner,
        repository=repository,
        inputs=inputs,
        loop=loop,
        shards=shards,
        gpus=gpus,
        policy=policy,
        drain_s=drain_s,
        log_level=log_level,
    )
    if http:
        # Whether this host can serve HTTP at all is a fact about the host, not about this
        # execution, so it is asked here. Asked inside `_wait` — after `start()` — it cost a
        # fleet sixteen spawned shards, a placed camera set and a full shutdown to learn that
        # `pip install "shipinfer[server]"` had never been run. Imported inside the branch
        # because `shipinfer.api` reaches the engine, and a run without `--http` must not pay
        # for it.
        from shipinfer.api import require_server_extra

        require_server_extra()
    if plan.configured:
        out.print(f"cameras: {plan.configured} configured ({plan.cameras[0].camera_id} ...)")
    if from_inputs := plan.cameras[plan.configured :]:
        out.print(f"cameras: {len(from_inputs)} from --inputs ({from_inputs[0].camera_id} ...)")
    # A fact about the runner the operator picked, not about this execution, and it needs no
    # runner *instance*: `manages_cameras` and `name` are `ClassVar`s read off the registered
    # class. Reached only after `start()`, `--dry-run --inputs deepstream` printed a plan and
    # exited 0 for a combination that can never work. The *placement* still happens after
    # `start()`, because a camera is placed on a running runner; the two halves of the check
    # sit either side of it deliberately.
    refuse_if_it_manages_no_cameras(RUNNERS.get(plan.runner), plan.cameras)

    engine, built = _bring_up(plan, dry_run=dry_run)
    if dry_run:
        # On the contract, not probed for: `Runner.describe_plan` has an in-process default
        # ("no plan: one process") and the fleet overrides it with the plan it would run.
        out.print(built.describe_plan())
        return 0
    return _serve(
        built,
        plan,
        engine=engine,
        http=http,
        host=_DEFAULT_HOST if host is None else host,
        port=_DEFAULT_PORT if port is None else port,
        log_level=log_level,
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    """What one ``shipinfer run`` resolved, before any device was touched.

    Everything here is decided from the command line, the settings tree and two files on
    disk. Nothing in it holds a CUDA context, a thread or a socket, which is what lets every
    refusal that reads it be made above :func:`_bring_up`.
    """

    settings: ServerSettings
    chain: Topology
    #: The chain file's **text**, because that is what a shard is sent — the loader lives on
    #: the shard (ADR-017) and the parsed :attr:`chain` is what this process validates.
    chain_yaml: str
    runner: str
    #: Configured fleet first, ``--inputs`` after, as one list. Nobody starts a camera but
    #: this command: `InprocessRunner` used to start `ingest.cameras` inside its own
    #: `_do_start`, which is right for one process and catastrophic for a shard — a shard is
    #: an `InprocessRunner` whose env-only settings inherit the operator's whole fleet, so
    #: all fifty cameras ran on all eight of them.
    cameras: tuple[CameraSpec, ...]
    #: How many of :attr:`cameras` came from the settings rather than ``--inputs``. Split by
    #: **position**, not by building the input specs a second time: two lists had to be
    #: passed the same ``loop`` to agree, and only one of them was ever placed.
    configured: int


def _resolve(
    out: Console,
    topology: Path,
    *,
    runner: str | None,
    repository: Path | None,
    inputs: list[str] | None,
    loop: bool,
    shards: int | None,
    gpus: str | None,
    policy: str | None,
    drain_s: float | None,
    log_level: str,
) -> _Plan:
    """Settings, chain and camera list — the whole plan, with no device touched.

    Reports the chain as soon as it parses and *before* the camera list is read off disk, so
    a mistyped chain and an unreadable ``ingest.camera_db`` are told apart by what the
    operator has already seen.

    Every flag lands in the **settings tree** rather than in per-runner keyword arguments, so
    this command names no runner: ``--shards`` is ``runner.shards``, ``--drain-s`` is
    ``runner.drain_s``, ``--gpus`` is ``devices.visible_gpus``, and a runner that cares reads
    them. A CLI that special-cased ``fleet`` here would be the ``if/elif`` the registry exists
    to prevent (CONVENTIONS 2.3).

    Raises:
        ConfigurationError: an invalid chain, an unreadable ``ingest.camera_db``, a duplicate
            camera id, or an input this build refuses.
    """
    from shipinfer.topology import ChainSpec, Topology

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
    # AFTER the settings, never as an argument to them — see `_fill_in_gpus`.
    settings = _fill_in_gpus(settings)
    chain_yaml = _read(topology)
    chain = Topology.from_spec(ChainSpec.from_yaml(chain_yaml, name=Path(topology).stem))
    chosen = runner or settings.runner.runner
    out.print(f"topology: {chain.name} ({len(list(chain))} element(s)) — runner {chosen}")
    cameras = cameras_to_place(settings, inputs, loop=loop)
    return _Plan(
        settings=settings,
        chain=chain,
        chain_yaml=chain_yaml,
        runner=chosen,
        cameras=tuple(cameras),
        configured=len(cameras) - len(inputs or ()),
    )


def _bring_up(plan: _Plan, *, dry_run: bool) -> tuple[InferenceServer | None, Runner]:
    """Build the dependencies the chain cannot build for itself, then start everything.

    Under **one** ``except BaseException``, because every line of it can fail with a started
    engine and no other reference to it. The case that made it necessary is ``built.start()``:
    :meth:`Runner.start` unwinds its own elements and re-raises, so a chain whose ``pool``
    element names a model the repository has not got left an engine at ``_started = True``,
    its worker threads alive and its CUDA contexts held, reachable from nothing.
    ``BaseException`` and not ``Exception``, because a ``KeyboardInterrupt`` during a chain
    that is opening decoders and sockets is the likeliest way this path is taken by hand, and
    ``forward_signals`` is not installed until :func:`_wait`.

    The engine is constructed as late as the ``models=`` keyword allows, and the lateness is
    the point: ``InferenceServer.__init__`` builds a ``DeviceManager``, which validates every
    visible device — one CUDA primary context per GPU, ~200 MiB each on this box, and nothing
    in the process gives them back. Every refusal that can be made without them is made in
    :func:`run`, above this call.

    Returns:
        The engine this run owns (``None`` when nothing would ask for one — a chain naming no
        model, a ``--dry-run``, or a ``fleet`` whose shards each build their own) and the built
        runner. Under ``--dry-run`` the runner is built but **not** started.
    """
    from shipinfer.runners import build_runner

    engine = None
    ops = None
    try:
        if not dry_run and model_pool_is_needed(plan.runner, plan.chain):
            from shipinfer.engine import InferenceServer

            engine = InferenceServer(plan.settings)
        if not dry_run and image_ops_are_needed(plan.runner, plan.chain):
            ops = _image_ops_for(plan, engine)
        built = build_runner(
            plan.runner,
            plan.chain,
            plan.settings,
            chain_yaml=plan.chain_yaml,
            models=engine,
            ops=ops,
        )
        if not dry_run:
            # The pool comes up BEFORE the chain does: `built.start()` is what calls `open()`
            # on every element, and that is where a `pool` element resolves its model against
            # a pool that has to be loaded by then. Whatever a backend raises comes out of
            # `engine.start()` unwrapped, and not all of it is typed — `ModelBackend` still
            # refuses a `requires_gpu` backend on a CPU device with a bare `ValueError`.
            if engine is not None:
                engine.start()
            built.start()
    except BaseException:
        # `stop()` is documented safe on a server whose `start` raised half-way, and on one
        # that never started it is the `already_stopped` no-op.
        if engine is not None:
            engine.stop()
        raise
    return engine, built


def _image_ops_for(plan: _Plan, engine: InferenceServer | None) -> ImageOps:
    """The other dependency an element cannot resolve for itself: one image-ops delegate.

    ``topology`` may not import ``runtime``, so the composition root is where an
    implementation is chosen and handed over.

    **Thread-local, and that is the whole point of the call.** ``pipeline.workers`` threads
    walk this chain at once, one ``PoolDetect`` is shared by all of them, and every
    implementation ``get_image_ops`` can return is per-thread by contract — the native one
    keeps a staging ring inside the extension, the torch one binds a device on the
    constructing thread. One instance across four workers is four threads in one ring:
    plausible pixels, no error, and invisible to the offline tier because ``NumpyImageOps``
    is stateless. It also put every camera's pre-processing on ``cuda:0`` however many GPUs
    the process can see, which is this project's founding bug one layer up.

    The devices come from the engine's ``DeviceManager`` when there is one and are *not*
    resolved by building one here (:func:`_bring_up` says why), so a chain that needs ops and
    no pool falls back to ``devices.visible_gpus``, then to ``(0,)``. That fallback binds no
    worker thread (ADR-002) and claims no pinned staging pool — each delegate is *constructed
    with* its own index, so it is unbound rather than misplaced. Pinned by
    ``tests/cli/test_run_engine.py::TestAChainThatNeedsOpsAndNoPool``.
    """
    from shipinfer.runtime.ops import get_thread_local_image_ops

    manager = getattr(engine, "devices", None)
    return get_thread_local_image_ops(
        plan.settings.execution.provider,
        devices=tuple(getattr(manager, "visible_gpus", ()) or ())
        or tuple(plan.settings.devices.visible_gpus),
        device_manager=manager,
        memory=getattr(engine, "memory", None),
    )


def _serve(
    built: Runner,
    plan: _Plan,
    *,
    engine: InferenceServer | None,
    http: bool,
    host: str,
    port: int,
    log_level: str,
) -> int:
    """Place the cameras, block until interrupted, and give everything back.

    Cameras are placed **after** ``start``, because a camera is placed on a *running* runner:
    the chain has to be open and its workers up before a decoder thread publishes into them. A
    refusal here therefore travels through the ``finally``, which stops what did come up.

    Teardown is ingress, then runner, then engine — the reverse of the order they came up in.
    :func:`_wait` has already stopped the web server in its own ``finally``, so nothing places
    a camera on a runner that is going down; the workers stop next; and only then does the pool
    they were submitting to go away, because a worker still walking a frame would otherwise
    lose the models mid-request. Nested rather than sequential so that a runner whose stop
    raises still gives the GPU back — a crash must not be what frees the device.

    Returns:
        ``0``, or ``1`` when a shard exited under the fleet: an operator-facing failure with
        its own message, not a traceback.
    """
    out = console()
    try:
        place_cameras(built, plan.cameras)
        _wait(built, http=http, host=host, port=port, log_level=log_level)
    except ShardExitedError as exc:
        out.print(f"[red]{exc}[/red]")
        return 1
    finally:
        try:
            built.stop()
        finally:
            if engine is not None:
                engine.stop()
    return 0


#: The dependencies a chain is *handed* rather than imports: the ``build_runner`` keyword,
#: and the :class:`~shipinfer.topology.base.Element` attribute by which an implementation
#: declares it needs one. A table rather than a function each, because the two were the same
#: four lines with one attribute name changed and phase D adds a third (a DataPool) — at which
#: point the copies would be the design rather than an accident.
#:
#: Both halves are **declarations**, never a check against ``"inprocess"`` or ``"pool"``, so a
#: new runner or a new element answers for itself instead of being added to a condition here
#: (CONVENTIONS 2.3).
_HANDED_IN: Mapping[str, str] = {
    "models": "needs_model",
    "ops": "needs_image_ops",
}


def dependency_is_needed(keyword: str, runner: str, chain: Topology) -> bool:
    """Whether this run has to build ``keyword=`` and hand it to the runner.

    Two questions, and neither is answered by a name:

    * **does this runner open the chain here?** ``Runner.needs_model_pool``, read off the
      registered class before anything is built. The attribute is named for the pool because
      that was the first dependency to need it; what it declares is "this runner calls
      ``open()`` on these elements in this process", the condition for every handed-in
      dependency. ``fleet`` answers ``False`` — its shards each build their own
      (``cli/shard.py``), and a launcher that built one too would hold a CUDA context on every
      device it can see while running no inference.
    * **does anything in the chain ask for it?** The element attribute from
      :data:`_HANDED_IN`, declared by the implementation. Asking ``node.kind in MODEL_KINDS``
      instead looks right and is not: every ``detect`` element is a model kind and must name a
      ``model:``, so a chain whose ``detect`` slot resolves an implementation that needs no
      pool would still load the whole repository.

    The two are asked separately because they come apart in both directions: a chain of
    ``pool`` embedders needs a pool and no ops, and the first element that crops without
    running a repository model will need ops and no pool.

    Raises:
        ConfigurationError: no runner is registered under that name; the message lists the ones
            that are. The same refusal :func:`~shipinfer.runners.build_runner` makes.
    """
    from shipinfer.runners import RUNNERS

    if not RUNNERS.get(runner).needs_model_pool:
        return False
    attribute = _HANDED_IN[keyword]
    return any(getattr(node.element, attribute) for node in chain)


def model_pool_is_needed(runner: str, chain: Topology) -> bool:
    """Whether this run has to build a model pool and hand it in as ``models=``.

    A name for :func:`dependency_is_needed`'s ``models`` row, kept because it is what the call
    site reads like and what the tests ask for.
    """
    return dependency_is_needed("models", runner, chain)


def image_ops_are_needed(runner: str, chain: Topology) -> bool:
    """Whether this run has to resolve image ops and hand them in as ``ops=``.

    A name for :func:`dependency_is_needed`'s ``ops`` row. Only ``PoolDetect`` answers
    ``True`` today, and the gate is worth having because ``get_image_ops`` is not free: under
    a non-``AUTO`` provider it constructs a torch implementation bound to a device.
    """
    return dependency_is_needed("ops", runner, chain)


def cameras_from_inputs(inputs: Sequence[str] | None, *, loop: bool = True) -> list[CameraSpec]:
    """``--inputs a.mp4 rtsp://b`` as the camera specs a runner takes.

    Identity is **positional** — ``cam-000``, ``cam-001``, minted by
    :func:`~shipinfer.launch.control.mint_camera_id`, which is also what ``POST /streams`` uses
    for a request supplying no id. A file path is not a camera id (two directories can hold the
    same ``clip.mp4``, and a path is not a legal metric label) and the offline mode has nobody
    to ask. Stable across restarts for a fixed argument list, which is what a downstream tracker
    keyed on ``camera_id`` needs, and zero-padded so ``cam-010`` sorts after ``cam-009``.

    ``loop`` is ``--loop/--no-loop``, and it is on the spec rather than the settings tree
    because these cameras exist nowhere in it: they are minted here, so no ``ingest.cameras[]``
    entry's ``loop:`` could reach them.

    An empty or absent list gives an empty list, not an error: ``shipinfer run`` with no inputs
    is the normal way to bring a chain up and add cameras over the control plane.
    """
    return [
        CameraSpec(camera_id=mint_camera_id(index), url=url, fps=0.0, loop=loop)
        for index, url in enumerate(inputs or ())
    ]


def cameras_from_settings(settings: ServerSettings) -> list[CameraSpec]:
    """``ingest.cameras`` plus ``ingest.camera_db`` as the camera specs a runner takes.

    The deployment's own fleet, in the launcher's vocabulary, so that it is placed through the
    same door as everything else. A runner used to read this list itself, which made the camera
    set a property of *whatever settings a process happened to load* — and a shard loads the
    operator's, so every shard read every camera.

    Only the five fields a launcher decides travel; the rest is deployment settings the running
    process resolves from its own tree, which is the split
    :class:`~shipinfer.launch.control.CameraSpec` exists to state.

    ``priority`` is the exception, and the reason this function is read. A band used to be
    settings like the rest — until a fleet shard's ``ingest.cameras`` was stripped to stop eight
    shards each opening all fifty cameras, which left the shard with no table to resolve a band
    from and put every RPC-placed camera in ``normal``. **This process** still has the
    operator's fleet config, so the band is read where it is true and carried to whichever shard
    ends up holding the camera.

    The import is inside the function: ``shipinfer.ingest`` reaches a decode runtime through its
    source registry, and ``shipinfer repo ls`` must not pay for one.

    Raises:
        ConfigurationError: the camera database cannot be read, or declares a camera the inline
            list already names.
    """
    from shipinfer.ingest import configured_cameras

    return [
        CameraSpec(
            camera_id=camera.camera_id,
            url=camera.uri,
            fps=camera.fps,
            loop=camera.loop,
            priority=camera.priority,
        )
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


def refuse_flags_that_would_be_ignored(
    *, http: bool, host: str | None, port: int | None
) -> None:
    """Refuse ``--host``/``--port`` without the ``--http`` that gives them meaning.

    Both flags configure exactly one thing — where :func:`_wait` puts the web server — and
    without ``--http`` there is no web server to configure. Accepting them silently is a small
    instance of the failure this command keeps meeting: an operator who typed
    ``shipinfer run --port 9000`` and forgot ``--http`` gets a healthy deployment, no ingress,
    and exit ``0``, with the flag they typed nowhere in the logs to contradict them.

    ``None`` rather than typer's own defaults is what makes this honest: with the default as the
    sentinel, ``--host 127.0.0.1`` typed out in full is indistinguishable from not typing it at
    all, and refusing a value the operator did not give is as rude as ignoring one they did.

    Raises:
        ConfigurationError: either flag was given and ``--http`` was not. Typed like every other
            refusal in this module rather than a ``typer.BadParameter``, because this module
            names no typer — it is called by the CLI and by tests alike.
    """
    if http:
        return
    given = [name for name, value in (("--host", host), ("--port", port)) if value is not None]
    if not given:
        return
    raise ConfigurationError(
        f"{' and '.join(given)} {'configure' if len(given) > 1 else 'configures'} the HTTP "
        "server and this run starts none; add `--http` to serve /streams on it, or drop "
        f"{'them' if len(given) > 1 else 'it'}"
    )


def refuse_if_it_manages_no_cameras(
    runner: Runner | type[Runner], cameras: Sequence[CameraSpec]
) -> None:
    """Refuse cameras on a runner that owns no ingest plane. Starts nothing.

    ``cameras`` is whatever this run would place — ``--inputs``, the configured fleet, or both
    — because the refusal is about the *runner*: a chain executor with no ingest plane opens
    neither kind, and a message naming only the flag would send an operator whose fleet is in
    ``ingest.camera_db`` looking for a flag they never typed.

    **A class or an instance**, because the two attributes read here are ``ClassVar``s and the
    answer is known before anything is built. :func:`run` passes ``RUNNERS.get(name)`` so the
    refusal is made above ``InferenceServer(...)``, whose construction takes a CUDA primary
    context per visible device; :func:`place_cameras` passes the running runner it holds.

    Separate from :func:`place_cameras` because the two answer at different moments. Placement
    needs a *running* runner; this is a fact about the class the operator named. Asking it late
    made ``--dry-run --inputs`` print a plan and exit ``0`` for a combination that cannot run at
    all, which is the one thing a dry run exists to catch.

    Raises:
        ConfigurationError: the runner manages no cameras, so nothing would open the files.
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
    here are configuration ones — a duplicate id, a source nobody registered — and each means
    the same thing about every camera behind it. Carrying on would report the last one's
    message for a mistake made in the first.

    The capability refusal is :func:`refuse_if_it_manages_no_cameras`'s and is made before the
    runner starts, but it is repeated here rather than assumed: this function is public, and a
    caller that placed cameras on a camera-less runner would otherwise get
    ``Runner.add_camera``'s default ``ServerStateError`` per camera.

    Raises:
        ConfigurationError: the runner manages no cameras, or a camera was refused. In the
            second case the original message is kept and prefixed with **both** the id and the
            url, because ``ingest/manager.py``'s message names only the id — which for an
            ``--inputs`` camera is minted from its position and appears nowhere in what the
            operator typed.
        NoShardAvailableError: a fleet whose shards all refused. Deliberately **not** re-labelled
            with the input path: the camera is fine and there is nowhere to put it, and wrapping
            it in a ``ConfigurationError`` would turn a 503 into a 400 for the same condition
            reached over ``POST /streams`` (``api/errors.py``).
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
    passed in there outranks ``SHIPINFER_DEVICES__VISIBLE_GPUS``, so an operator who restricted
    this deployment to the two devices they own would silently get eight shards on eight GPUs —
    each holding a CUDA context on a shared box — with nothing saying the setting had been read
    and dropped. So the flag goes in alone, the resolved tree answers "did anybody say", and the
    driver is asked only when the answer is no. A ``--dry-run`` on a configured host therefore
    touches no driver at all.

    Resolved **here** rather than inside a runner because ``runners`` imports no ``runtime``
    (``check_layers.py``): ``device_count()`` initialises CUDA in whichever process calls it,
    and the launcher is the one process that should hold no context.

    The section is rebuilt through :class:`~shipinfer.core.settings.DeviceSettings` rather than
    ``model_copy``d on the leaf so its field validators run. A driver that reports none leaves
    the tree alone, and the runner that needs devices refuses by name.
    """
    if settings.devices.visible_gpus:
        return settings
    from shipinfer.runtime.platform import device_count

    reported = list(range(device_count()))
    if not reported:
        return settings
    devices = DeviceSettings(**{**settings.devices.model_dump(), "visible_gpus": reported})
    return settings.model_copy(update={"devices": devices})


def _wait(
    built: Runner,
    *,
    http: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "INFO",
) -> None:
    """Block until Ctrl-C, SIGTERM, or a shard dies, answering ``/streams`` meanwhile.

    Signals are installed here rather than by the runner: handlers are process-global state,
    and a library that installed them behind your back is one you cannot embed
    (``launch/signals.py`` makes the same argument for the fleet).

    Every runner supervises — :meth:`Runner.supervise` waits to be told to go, and the fleet
    overrides it to also notice a shard that exits while the rest report healthy. So there is
    no probe here for which kind this is: a ``getattr`` fallback would silently downgrade a
    fleet whose method got renamed into one that never watched its shards.

    **The main thread stays the supervising thread**, which is what decides where the web
    server goes. ``forward_signals`` is installed *first* and is the only handler this process
    installs: uvicorn installs its own inside ``Server.serve``, and a Ctrl-C that stopped the
    web server while fifty decoder threads kept running is the shutdown ADR-005 and
    ``launch/signals.py`` exist to avoid. Running it on a thread prevents that, and also lets
    ``supervise()`` return the moment the runner is told to go rather than one HTTP tick later.

    Raises:
        ShardExitedError: a shard exited; the fleet is already stopped.
        ConfigurationError: ``--http`` was asked for and FastAPI or uvicorn is not installed,
            or the server could not bind ``host:port``. The first is the last line of defence
            and not where an operator meets it — :func:`run` probes the import before starting
            anything — and stays for a caller that reached here directly. The second cannot be
            probed early (a port is free until it is not), so ``start()`` raises it *before*
            ``supervise()`` is entered and therefore before the deployment looks healthy. Both
            leave nothing to stop: no server was assigned, and :func:`_serve`'s ``finally``
            stops the runner.
    """
    from shipinfer.launch import forward_signals

    forward_signals(built)
    if not http:
        built.supervise()
        return

    # Imported here, not at module scope: `--http` is the only thing in this command that
    # needs the `server` extra, and `shipinfer run` without it must work on a host that never
    # installed FastAPI.
    from shipinfer.api import BackgroundHttpServer, create_app

    server = BackgroundHttpServer(
        create_app(cameras=built), host=host, port=port, log_level=log_level
    ).start()
    try:
        built.supervise()
    finally:
        server.stop()
