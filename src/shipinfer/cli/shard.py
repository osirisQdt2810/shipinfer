"""``python -m shipinfer.cli.shard --shard-id N --control-port P`` — one shard process.

The child half of arch.md section 2. It takes **two flags and nothing else**: who it is and
where to be reached. Everything that used to ride in argv and in the environment — which
cameras to read, which GPUs to use, which topology to run, how the device is shared — arrives
as RPCs after this process reports ready (V140: *"xóa luôn cách dùng gọi command giữa 2 tiến
trình"*). A third flag is refused rather than ignored, because the argv is the contract:
:meth:`shipinfer.runners.fleet.FleetRunner.shard_command` renders it, this parses it, and
nothing else holds the two ends together.

**The one thing that cannot arrive by RPC** is ``CUDA_VISIBLE_DEVICES``: it has to be in the
environment before this interpreter imports torch, which happens several frames below the
first ``InferenceServer``. The launcher puts it there at ``exec`` (``launch/supervisor.py``),
and that is the whole boundary of what the environment still carries.

WHY THIS LIVES IN ``cli/`` AND NOT IN ``launch/``
-------------------------------------------------
It is a composition root: it builds an engine, a topology and a runner, and hands them to a
servicer. ``launch`` may import none of those — a launcher that imported the thing it launches
would pay for the model pool in the parent process, and ``check_layers.py`` plus
``tests/test_architecture.py`` both enforce that direction. ``runners`` may not import the
engine either, for the same reason a runner is *handed* a
:class:`~shipinfer.topology.base.ModelResolver` rather than importing one. ``cli`` is the
layer whose job is exactly this wiring, and it is where ``serve`` already builds an engine.

WHAT THE SHARD DOES, IN ORDER
-----------------------------
1. bind the control port and start answering — **before** anything heavy, so a launcher
   learns "this process is alive" in a second rather than after an engine load;
2. answer ``Ready`` with ``starting``: alive, nothing installed, takes no cameras;
3. on ``UpdateTopology``, apply the sharing to the settings tree, load the models, build the
   in-process runner over the chain, and start it. That call is the slow one — it is where a
   shard's engines are deserialised;
4. on ``Stop``, the runner stops and this process notices and exits, releasing its CUDA
   context. A shard that stayed alive after ``Stop`` would hold ~250 MiB of VRAM until the
   launcher's SIGTERM caught up with it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from shipinfer.cli.common import build_settings
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.core.settings import DeviceSettings, ServerSettings

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime imports are inside main()
    from shipinfer.runners.base import Runner
    from shipinfer.topology import ChainSpec

__all__ = ["apply_sharing", "build_parser", "main"]

_LOG = get_logger("cli.shard")

#: How long the shard's runner gets to be built and started before the launcher's
#: ``UpdateTopology`` deadline is the one that matters. Not enforced here — it is the
#: *client's* deadline that decides — and named here only so the two numbers can be compared:
#: ``FleetRunner.topology_timeout_s`` must be larger than an engine load on a cold cache.
_TYPICAL_START_S = 120.0


def build_parser() -> argparse.ArgumentParser:
    """The two flags, and the refusal of a third.

    ``argparse`` rather than typer: a shard must start on a host that installed the runtime
    and not the ``cli`` extra, and it has no options to be pretty about.
    """
    parser = argparse.ArgumentParser(
        prog="python -m shipinfer.cli.shard",
        description=(
            "One shard process. It is told who it is and where to be reached; its cameras, "
            "its topology and its device sharing arrive over gRPC (arch.md section 2)."
        ),
    )
    parser.add_argument(
        "--shard-id", type=int, required=True, help="Which shard this is, from the plan."
    )
    parser.add_argument(
        "--control-port",
        type=int,
        required=True,
        help="Port for this shard's gRPC control plane. 0 picks an ephemeral one.",
    )
    return parser


def apply_sharing(
    settings: ServerSettings, shared_by: Sequence[int], share_rank: Sequence[int]
) -> ServerSettings:
    """Put the launcher's device sharing into the settings tree, where the engine reads it.

    This is the whole of what ``SHIPINFER_DEVICES__SHARED_BY`` and
    ``SHIPINFER_DEVICES__SHARE_RANK`` used to do from the child's environment, moved onto the
    wire. It is not decoration: ``ModelConfig.placements`` divides every model's configured
    ``instance_group.count`` by ``shared_by`` and gives this process the ``share_rank`` slice,
    so **two shards on one GPU each load half the instances**. A shard that is never told
    about its co-tenant loads the whole count, and the device silently holds twice the engines
    and twice the VRAM for the same total throughput — with nothing anywhere saying so.

    Empty lists mean "one process per device", which is what a single-process run is, so the
    settings are returned unchanged rather than rebuilt.

    Raises:
        ValidationError: a share of zero or a negative rank. Rebuilt through
            :class:`~shipinfer.core.settings.DeviceSettings` rather than ``model_copy``d
            precisely so the field validators run: a launcher bug reaches the launcher as a
            refused ``UpdateTopology``, not as a division by zero inside a model load.
    """
    if not shared_by and not share_rank:
        return settings
    devices = DeviceSettings(
        **{
            **settings.devices.model_dump(),
            "shared_by": list(shared_by),
            "share_rank": list(share_rank),
        }
    )
    return settings.model_copy(update={"devices": devices})


class _ShardProcess:
    """What this process owns: an engine, a runner, and the servicer over them.

    A class rather than three closures because the two failure paths need the *same* release:
    a factory that raised half-way and a ``Stop`` that succeeded both have to leave no CUDA
    context behind (CLAUDE.md's GPU hygiene rule — this box is shared).
    """

    def __init__(self, settings: ServerSettings, shard_id: int) -> None:
        self._settings = settings
        self._shard_id = shard_id
        self._engine: Any = None

    def build(
        self, spec: ChainSpec, shared_by: Sequence[int], share_rank: Sequence[int]
    ) -> Runner:
        """The :data:`~shipinfer.runners.service.RunnerFactory` this shard hands its servicer.

        Called from inside ``UpdateTopology``, with the chain the launcher sent. The order is
        load-bearing: the sharing goes into the settings **before** the engine is built,
        because the engine decides how many instances of each model to load while it starts
        and cannot be told afterwards.

        Called *again* after a start the servicer refused, which is why it releases first: a
        second engine assigned over the first would leave the first one's CUDA context held by
        nothing that can ever stop it (CLAUDE.md's GPU hygiene rule - this box is shared).
        :meth:`release` is idempotent and safe when nothing was ever built.

        The engine is **assigned before it is started**, not after. ``start()`` loads models
        one at a time and can raise on the fifth with four already running; the object that
        owns those four is the one this method is holding, and if the assignment waits for
        ``start()`` to return there is no assignment at all on that path — the engine, its
        threads and its contexts are unreachable, and :meth:`release` has nothing to give
        back. Assigning first costs an unstarted engine on the failure path, which
        :meth:`InferenceServer.stop` handles.
        """
        from shipinfer.engine import InferenceServer
        from shipinfer.runners import build_runner
        from shipinfer.topology import Topology

        self.release()
        settings = apply_sharing(self._settings, shared_by, share_rank)
        topology = Topology.from_spec(spec)
        engine = self._engine = InferenceServer(settings)
        try:
            engine.start()
            # Always `inprocess`: a shard *is* the process the fleet placed, so a shard that
            # built a `fleet` runner would spawn shards of its own.
            return build_runner(
                "inprocess",
                topology,
                settings,
                shard_id=self._shard_id,
                models=engine,
            )
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        """Give the GPU back.

        Idempotent, safe when nothing was ever built, and safe for an engine whose
        ``start()`` raised: :meth:`InferenceServer.stop` unwinds a partial start and never
        raises, which is what lets this be called unconditionally from a ``finally``.
        """
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.stop()


def main(argv: Sequence[str] | None = None) -> int:
    """Serve one shard's control plane until it is told to stop.

    Returns:
        ``0`` on an orderly stop. Argument errors exit ``2`` through argparse, which is what
        makes a launcher that renders the wrong argv fail loudly on the first shard rather
        than quietly on all of them.
    """
    from shipinfer.runners.service import serve_shard
    from shipinfer.runtime.containment import require_container

    args = build_parser().parse_args(argv)
    # After parsing, so `--help` works anywhere, and before anything is loaded: a shard is a
    # server, and a deny-list over command text cannot be made sound (`serve` says the same).
    require_container("a shipinfer shard")

    settings = build_settings()
    process = _ShardProcess(settings, args.shard_id)
    shard = serve_shard(
        None,
        shard_id=args.shard_id,
        control_port=args.control_port,
        build=process.build,
    )
    _LOG.info(
        "shard %d waiting for a topology on port %d (a build takes up to ~%.0fs)",
        args.shard_id,
        shard.identity.control_port,
        _TYPICAL_START_S,
    )
    try:
        # Polled rather than a single blocking wait: `Stop` stops the *runner*, and this
        # process is what has to notice and exit. Leaving it alive would hold a CUDA context
        # until the launcher's SIGTERM arrived, which is exactly the leak the fleet's
        # shared-deadline stop exists to avoid.
        while not shard.wait_for_termination(1.0):
            # `service.stopped`, not `service.state()`: the flag is the whole question here,
            # and `state()` with no snapshot in hand takes one - a full `runner.health()`
            # across every element, once a second, to answer a boolean.
            if shard.service.stopped:
                break
    except KeyboardInterrupt:  # pragma: no cover - the operator's Ctrl-C on a hand-run shard
        _LOG.info("shard %d interrupted", args.shard_id)
    finally:
        shard.stop()
        runner = shard.service.runner
        if runner is not None:
            runner.stop()
        process.release()
    _LOG.info("shard %d exited", args.shard_id)
    return 0


if __name__ == "__main__":  # pragma: no cover - the child entry point
    try:
        raise SystemExit(main())
    except ConfigurationError as exc:
        # Caught here rather than inside `main` so a library caller still gets the exception:
        # only the *process* turns a typed refusal into an exit code and a line an operator
        # can read, instead of a traceback the launcher only sees in a log.
        print(f"shard: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
