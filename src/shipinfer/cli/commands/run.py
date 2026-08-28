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
from pathlib import Path
from typing import TYPE_CHECKING

from shipinfer.cli.common import build_settings, console
from shipinfer.core.errors import ConfigurationError, ShardExitedError
from shipinfer.core.settings import DeviceSettings, ServerSettings
from shipinfer.launch.control import CameraSpec, mint_camera_id

if TYPE_CHECKING:  # pragma: no cover - typing only; `runners` is imported inside `run()`
    from shipinfer.runners.base import Runner
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
            already has its own ``loop:`` and keeps it, and one posted to ``/streams`` carries
            its own (``StreamRequest.loop``, which defaults the same way this flag does).
        shards: how many shard processes, for the runners that have any. ``None`` leaves
            ``runner.shards``, whose own default is one per visible GPU (ADR-006).
        gpus: which devices, as ``0,1,2``. ``None`` leaves ``devices.visible_gpus``, and
            only when *that* is empty too is the driver asked, once (:func:`_fill_in_gpus`).
            A flag is not the only way an operator answers this question.
        http: serve ``/streams`` (arch.md §2's camera door) beside the running chain, on a
            thread. Off by default: a chain driven by ``--inputs`` needs no ingress, and a
            port that nobody asked for is a port somebody has to firewall.
        host: what the HTTP server binds; ``None`` means the operator did not say and takes
            ``127.0.0.1``. **Loopback by default**, unlike ``shipinfer serve``: these routes
            start and stop video decoding on a shared GPU box and phase B has no
            authentication on them, so reaching them from another machine is a decision an
            operator makes explicitly — by putting an authenticating proxy in front, or, at
            their own risk, by passing ``--host 0.0.0.0``. Given *without* ``--http`` it is
            refused rather than ignored (:func:`refuse_flags_that_would_be_ignored`).
        port: what it binds, or ``None`` for 8000. ``0`` is not special-cased; ask the OS for
            one only if you have a way to find out which it chose. Refused without ``--http``
            for ``host``'s reason.
        drain_s: seconds a shard gets to finish before it is killed. ``None`` leaves
            ``runner.drain_s``. It is ``shipinfer fleet --drain`` under its settings-tree
            name: the same budget, on the command that replaced it, so an operator who had
            tuned it for a slow decoder does not silently get the default back.
        dry_run: resolve everything and print it, spawn nothing. The plan is the decision
            worth reading before fifty cameras start reconnecting.

    Raises:
        ConfigurationError: an unknown runner, an invalid chain, an unreadable
            ``ingest.camera_db``, a runner that manages no cameras given ``--inputs``, an
            input the runner refuses, ``--host``/``--port`` without the ``--http`` that gives
            them meaning, ``--http`` on a host where the ``server`` extra was never installed,
            an HTTP port that could not be bound
            (:meth:`shipinfer.api.BackgroundHttpServer.start`), or a chain element that will
            not open -- a ``pool`` element with no ``model:``, for instance. The HTTP bind is
            raised after the runner is up and therefore travels through the ``finally`` that
            stops it, which is what an operator who typed ``--http`` is asking for: no
            ingress, no deployment.
        ServerStateError: the model pool this chain needs would not start, or a model it needs
            has no live instance. Raised by
            :meth:`~shipinfer.engine.InferenceServer.start`, unwrapped, because it already
            names the model that would not load.
        BackendUnavailableError: a model in the repository needs a runtime this host has not
            got (no TensorRT, no onnxruntime). Also from the engine's start.
        ModelNotFoundError: a ``pool`` element names a model the pool did not load. Raised
            from ``built.start()``, one line further on than the engine's own start, which is
            why every one of these travels through the guard below rather than past it.

    Note:
        **Every one of them leaves the GPU as it found it.** The bring-up -- the engine's
        construction, the runner's construction, both ``start()`` calls -- sits under one
        ``except BaseException`` that stops the engine if one was built, and the cheap
        refusals are made above it so that no context is taken to answer them.

        The list above is not closed. The engine's start surfaces whatever a backend raises,
        and not every backend raises a typed error: a ``requires_gpu`` backend placed on a
        CPU device is refused by :class:`~shipinfer.backends.base.ModelBackend` with a bare
        ``ValueError``. It leaves the GPU as the typed ones do -- the guard catches
        ``BaseException`` -- so this is about what an operator reads, not about what is held.
    """
    from shipinfer.runners import RUNNERS, build_runner
    from shipinfer.runtime.containment import require_container
    from shipinfer.topology import ChainSpec, Topology

    out = console()
    # The gate lives here as well as in the shell hook: a deny-list over command text cannot
    # be made sound, and this command loads engines and drives GPUs (`serve` says the same).
    # It comes BEFORE anything is resolved so that `_fill_in_gpus()` - the one thing here that
    # can touch the driver - runs below it and not above: a refusal should not have queried
    # the hardware it is refusing to use. `--dry-run` is exempt because it spawns nothing; it
    # is the mode for reading a plan on a laptop.
    if not dry_run:
        require_container("`shipinfer run`")
    # Before the settings are built and long before anything is spawned: this is a fact about
    # the command line alone, and it is the cheapest refusal in the function.
    refuse_flags_that_would_be_ignored(http=http, host=host, port=port)
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

    # The configured fleet and `--inputs`, in that order, as ONE list to place. Nobody starts
    # a camera but this: `InprocessRunner` used to start `ingest.cameras` inside its own
    # `_do_start`, which is right for one process and catastrophic for a shard, because a
    # shard is an `InprocessRunner` whose env-only settings inherit the operator's whole
    # fleet -- so all fifty cameras ran on all eight of them. Placing them here instead means
    # the fleet runner spreads them over its shards through `add_camera` and the in-process
    # runner starts them locally, from one line, with no runner knowing which deployment it is.
    #
    # Resolved BEFORE the engine is constructed, because it reads `ingest.camera_db` off the
    # disk and raises `ConfigurationError` on an unreadable file or a duplicate id -- see the
    # engine comment below for why the order of the refusals is load-bearing rather than
    # stylistic.
    cameras = cameras_to_place(settings, inputs, loop=loop)
    if http:
        # The same argument one flag further out, and the same place to make it: whether this
        # host can serve HTTP at all is a fact about the host, not about this execution. Asked
        # inside `_wait` -- after `built.start()` -- it cost a fleet sixteen spawned shards, a
        # placed camera set and a full shutdown to learn that `pip install "shipinfer[server]"`
        # had never been run. Imported here rather than at module scope for `_wait`'s reason:
        # `shipinfer.api` reaches the engine, and `shipinfer run` without `--http` must not pay
        # for it.
        from shipinfer.api import require_server_extra

        require_server_extra()
    # Split for the report by POSITION, not by building the input specs a second time.
    # `cameras_from_inputs` mints exactly one camera per input, in order, and
    # `cameras_to_place` appends them after the configured fleet — so the tail of that list
    # *is* the inputs. A second build was two lists that had to be passed the same `loop` to
    # agree, and only one of them was ever placed.
    configured = len(cameras) - len(inputs or ())
    from_inputs = cameras[configured:]
    if configured:
        out.print(f"cameras: {configured} configured ({cameras[0].camera_id} ...)")
    if from_inputs:
        out.print(f"cameras: {len(from_inputs)} from --inputs ({from_inputs[0].camera_id} ...)")

    # The capability refusal is made HERE, before anything is built, because it is a fact
    # about the runner the operator picked and not about this particular execution: reached
    # only after `start()`, `--dry-run --inputs deepstream` printed a plan and exited 0 for a
    # combination that can never work, and the operator learned it on the real run. It needs
    # no runner *instance* either -- `manages_cameras` and `name` are `ClassVar`s, read off
    # the registered class exactly as `model_pool_is_needed` reads `needs_model_pool` one line
    # on -- so it belongs above the constructor, with the other refusals that cost no CUDA
    # context. The *placement* still happens after `start()`, because a camera is placed on a
    # running runner; the two halves of the check sit either side of it deliberately.
    refuse_if_it_manages_no_cameras(RUNNERS.get(chosen), cameras)

    # The model pool this run owns, or `None` when nothing here would ask for one: a chain of
    # mocks, a `--dry-run` that spawns nothing, or a `fleet` whose shards each build their own
    # (`cli/shard.py`). Without it a real chain could not run in this process at all -- a
    # `pool` element is opened with `ElementContext.models` and refuses when it is `None`, so
    # `--runner inprocess` over any topology with a model in it failed at `start()`.
    #
    # Constructed HERE, as late as the `models=` constructor argument allows, and the lateness
    # is the fix rather than a preference. `InferenceServer.__init__` builds a `DeviceManager`,
    # which validates every visible device -- one `torch.cuda.mem_get_info` per GPU, i.e. one
    # CUDA primary context per GPU, ~200 MiB each on this box. Nothing gives those back inside
    # the process: `stop()` on a server that was never started takes the `already_stopped`
    # branch and returns without reaching `_release`, and there is no other teardown to call.
    # So every refusal that can be made without them has to be made above this line, and the
    # two that used to sit below it -- an unreadable `ingest.camera_db` and a `--http` on a
    # host with no `server` extra -- now do, and so does the camera-capability refusal above,
    # which only ever read two `ClassVar`s. What is left below is `build_runner`, and it is
    # below because `models=` is the keyword this engine is built to be.
    engine = None
    ops = None
    try:
        if not dry_run and model_pool_is_needed(chosen, chain):
            from shipinfer.engine import InferenceServer

            engine = InferenceServer(settings)
        if not dry_run and image_ops_are_needed(chosen, chain):
            # The other dependency an element cannot resolve for itself, gated the same way and
            # for the same reason: `topology` may not import `runtime`, so the composition root
            # is where an ops implementation is chosen and handed over.
            #
            # `get_thread_local_image_ops`, not `get_image_ops`, and the difference is the
            # whole point of the line. `pipeline.workers` threads walk this chain at once
            # (`runners/inprocess.py`), one `PoolDetect` instance is shared by all of them, and
            # every implementation `get_image_ops` can return is per-thread by contract: the
            # native one keeps a staging ring inside the extension, the torch one binds a
            # device on the constructing thread and caches an event on the instance. One
            # instance across four workers is four threads in one ring -- plausible pixels, no
            # error, and invisible to the offline tier because `NumpyImageOps` is stateless.
            # It also put every camera's pre-processing on `cuda:0` regardless of how many GPUs
            # this process can see, which is this project's founding bug one layer up.
            #
            # The devices come from the engine's `DeviceManager` when there is one, and are
            # *not* resolved by building one here: `DeviceManager.__init__` takes a CUDA
            # primary context per visible GPU (~200 MiB each) that nothing in this process
            # gives back, which is the same reason the engine itself is constructed as late as
            # it is. A chain that needs ops and no pool (the first `crop` element will be one)
            # therefore falls back to what the operator pinned in `devices.visible_gpus`, and
            # to `(0,)` when that is empty -- on a host with no accelerator the delegate
            # degrades to `NumpyImageOps` and the index is never used, which is what keeps the
            # offline tier running the real element rather than a stubbed one.
            from shipinfer.runtime.ops import get_thread_local_image_ops

            manager = getattr(engine, "devices", None)
            ops = get_thread_local_image_ops(
                settings.execution.provider,
                devices=tuple(getattr(manager, "visible_gpus", ()) or ())
                or tuple(settings.devices.visible_gpus),
                device_manager=manager,
                memory=getattr(engine, "memory", None),
            )

        built = build_runner(
            chosen, chain, settings, chain_yaml=chain_yaml, models=engine, ops=ops
        )
        if dry_run:
            # On the contract, not probed for: `Runner.describe_plan` has an in-process default
            # ("no plan: one process") and the fleet overrides it with the plan it would run.
            out.print(built.describe_plan())
            return 0

        # The pool comes up BEFORE the chain does: `built.start()` is what calls `open()` on
        # every element, and that is where a `pool` element resolves its model
        # (`topology/elements/pool.py`) -- against a pool that has to be loaded by then.
        if engine is not None:
            # Whatever a backend raises comes out of here unwrapped, and not all of it is
            # typed: `ModelBackend` (`backends/base.py`) still refuses a `requires_gpu`
            # backend placed on a CPU device with a bare `ValueError` where a
            # `ConfigurationError` belongs. Left as a follow-up rather than fixed from this
            # command, because `backends/` cannot be touched without running the GPU tier.
            engine.start()
        built.start()
    except BaseException:
        # One guard over the whole bring-up, because every line of it can fail with a started
        # engine and no other reference to it. The case that made this necessary is
        # `built.start()`: `Runner.start` unwinds its own elements and re-raises, so a chain
        # whose `pool` element names a model the repository does not have left an engine at
        # `_started = True`, with its instance worker threads alive and its CUDA contexts held,
        # reachable from nothing -- the leak `InferenceServer.start` (`engine/pool.py`) and
        # `_ShardProcess.build` (`cli/shard.py`) each refuse in their own scope. `engine.start()`
        # failing is covered by the same line and costs nothing: `stop()` is documented safe on
        # a server whose `start` raised half-way, and on one that never started it is the
        # `already_stopped` no-op.
        #
        # `BaseException`, not `Exception`, for `InferenceServer.start`'s reason: a
        # `KeyboardInterrupt` during a chain that is opening decoders and sockets is the
        # likeliest way this path is taken by hand, and `forward_signals` is not installed
        # until `_wait`, which is two lines further on.
        if engine is not None:
            engine.stop()
        raise
    try:
        # After `start`, because a camera is placed on a *running* runner: the chain has to be
        # open and its workers up before a decoder thread starts publishing into them. A
        # refusal here therefore travels through the `finally`, which stops what did come up.
        place_cameras(built, cameras)
        _wait(
            built,
            http=http,
            host=_DEFAULT_HOST if host is None else host,
            port=_DEFAULT_PORT if port is None else port,
            log_level=log_level,
        )
    except ShardExitedError as exc:
        out.print(f"[red]{exc}[/red]")
        return 1
    finally:
        # Ingress, then runner, then engine -- the reverse of the order they came up in.
        # `_wait` has already stopped the web server in its own `finally`, so nothing places a
        # camera on a runner that is going down; the workers stop next; and only then does the
        # pool they were submitting to go away, because a worker still walking a frame would
        # otherwise lose the models mid-request. Nested rather than two statements so that a
        # runner whose stop raises still gives the GPU back: a crash must not be what frees
        # the device (CLAUDE.md's hygiene rule), and `InferenceServer.stop` never raises.
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
      registered class before anything is built, because these are constructor arguments. The
      attribute is named for the pool because that was the first dependency to need it; what
      it declares is "this runner calls ``open()`` on these elements in this process", which
      is the condition for every dependency an element is handed. ``fleet`` answers ``False``
      — its shards each build their own (``cli/shard.py``), and a launcher that built one too
      would hold a CUDA context on every device it can see while running no inference.
    * **does anything in the chain ask for it?** The element attribute from
      :data:`_HANDED_IN`, declared by the implementation. Asking ``node.kind in MODEL_KINDS``
      instead is the version of this that looks right and is not: every ``detect`` element is
      a model kind and must name a ``model:``, so a chain of mocks would load the whole
      repository to run elements that invent a box.

    The two dependencies are asked separately and not folded into one answer, because they
    come apart in both directions: a chain of ``pool`` embedders needs a pool and no ops
    (somebody upstream shaped the tensor), and the first element that crops without running a
    repository model will need ops and no pool.

    Raises:
        ConfigurationError: no runner is registered under that name; the message lists the
            ones that are. The same refusal :func:`~shipinfer.runners.build_runner` makes, one
            line earlier, because this asks the same registry.
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

    Only the five fields a launcher decides travel. The rest of a camera's configuration is
    deployment settings and the process that runs it resolves them from its own tree, which
    is the split :class:`~shipinfer.launch.control.CameraSpec` exists to state: the shard
    keeps the codec, the transport and the decode size from its settings, and is *told* which
    camera to open.

    ``priority`` is the exception, and it is the one this function is read for. A band used to
    be settings like the rest — until a fleet shard's ``ingest.cameras`` was stripped to stop
    eight shards each opening all fifty cameras (``runners/inprocess.py::_ingest``), which
    left the shard with no table to resolve a band from and put every RPC-placed camera in
    ``normal``. **This process** still has the operator's fleet config, so the band is read
    here, where it is true, and carried to whichever shard ends up holding the camera.

    The import is inside the function: ``shipinfer.ingest`` reaches a decode runtime through
    its source registry, and ``shipinfer repo ls`` must not pay for one.

    Raises:
        ConfigurationError: the camera database cannot be read, or declares a camera the
            inline list already names.
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
    without ``--http`` there is no web server to configure. Accepting them silently is a
    small instance of the failure this command keeps meeting: an operator who typed
    ``shipinfer run --port 9000`` and forgot ``--http`` gets a healthy deployment, no
    ingress, and exit ``0``, with the flag they typed nowhere in the logs to contradict them.
    A refusal naming the missing flag costs one line and is read before anything is spawned.

    ``None`` rather than typer's own defaults is what makes this honest: with the default as
    the sentinel, ``--host 127.0.0.1`` typed out in full is indistinguishable from not typing
    it at all, and refusing a value the operator did not give is as rude as ignoring one they
    did. So ``cli/__init__.py`` declares both options with ``None`` and this sees only what
    was actually said.

    Raises:
        ConfigurationError: either flag was given and ``--http`` was not. Typed like every
            other refusal in this module rather than a ``typer.BadParameter``, because this
            module names no typer — it is called by the CLI and by tests, and a library
            caller deserves the same message.
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

    ``cameras`` is whatever this run would place — ``--inputs``, the configured fleet, or
    both — because the refusal is about the *runner*: a chain executor with no ingest plane
    opens neither kind, and a message that named only the flag would send an operator whose
    fleet is in ``ingest.camera_db`` looking for a flag they never typed.

    **A class or an instance**, because the two attributes read here are ``ClassVar``s and
    the answer is therefore known before anything is built. :func:`run` passes
    ``RUNNERS.get(name)`` so that the refusal is made above ``InferenceServer(...)``, whose
    construction takes a CUDA primary context per visible device that nothing gives back
    inside the process; :func:`place_cameras` passes the running runner it already holds.

    Separate from :func:`place_cameras` because the two answer at different moments.
    Placement needs a *running* runner — a camera is placed on an open chain — while this is a
    fact about the class the operator named, known as soon as that name is resolved. Asking it
    late made ``--dry-run --inputs`` print a plan and exit ``0`` for a combination that cannot
    run at all, which is the one thing a dry run exists to catch.

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
    and a library that installed them behind your back is a library you cannot embed
    (``launch/signals.py`` makes the same argument for the fleet).

    Every runner supervises — :meth:`Runner.supervise` waits to be told to go, and the fleet
    overrides it to also notice a shard that exits while the rest keep reporting healthy. So
    there is no probe here for which kind this is: the contract is the contract, and a
    ``getattr`` fallback would have silently downgraded a fleet whose method got renamed into
    one that never watched its shards.

    **The main thread stays the supervising thread**, which is what decides where the web
    server goes. ``forward_signals`` is installed *first* and is the only handler this process
    installs: uvicorn installs its own inside ``Server.serve``, and a Ctrl-C that stopped the
    web server while fifty decoder threads kept running is precisely the shutdown this
    codebase spent ADR-005 and ``launch/signals.py`` avoiding. Running it on a thread is what
    prevents that (``api/app.py::BackgroundHttpServer`` says how), and it is also what lets
    ``supervise()`` return the moment the runner is told to go rather than one HTTP tick later.

    The server is stopped in a ``finally`` and the runner is stopped by the caller's, in that
    order: the ingress closes first, so nothing places a camera on a runner that is going down.

    Raises:
        ShardExitedError: a shard exited; the fleet is already stopped.
        ConfigurationError: ``--http`` was asked for and FastAPI or uvicorn is not installed,
            or the server could not bind ``host:port``. The first is typed and names the
            extra; it is the last line of defence and not where an operator meets it, because
            :func:`run` probes both imports before it starts anything, so a real ``shipinfer
            run --http`` refuses with no shard spawned. It stays here for a caller that
            reached ``_wait`` directly. The second cannot be probed early -- a port is free
            until it is not -- so it is raised by ``start()`` below, *before* ``supervise()``
            is entered and therefore before the deployment settles into looking healthy. Both
            leave nothing to stop: ``start()`` raising means no server was assigned, so the
            ``finally`` that would stop one is never reached, and :func:`run`'s own
            ``finally`` stops the runner.
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
