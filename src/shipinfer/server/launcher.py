"""The two argv lines a shard is started with — the mechanism arch.md §2 deletes.

The supervision this file used to hold has moved to :mod:`shipinfer.launch`: spawning,
watching and stopping a fleet is the launcher's permanent job and it now lives in the package
named after it. What is left here is the part that is *not* permanent — rendering a child's
command line.

arch.md §2 is explicit about why. A process must still be spawned, but "the child receives
nothing in argv beyond its identity (``--shard-id N --control-port P``); everything else —
camera set, GPU binding, topology, config — arrives as RPCs after the child reports ready",
and the mechanism where a topology object renders a command string is deleted outright. These
two functions and the ``server/topology/`` classes that call them go together in A2 PR-6, so
they are deliberately left where they are rather than carried into the new package and
deleted from it one PR later.

Nothing above imports them except ``server/topology/``. A caller that wants a fleet wants
:class:`shipinfer.launch.Fleet`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from shipinfer.scheduling.sharding import Shard

__all__ = ["deepstream_command", "serve_command"]


#: How a shard learns which cameras are its own.
#:
#: An environment variable rather than a flag, because ``shipinfer serve`` has no camera flag
#: and inventing one for this would put fleet-splitting into the vocabulary of a command that
#: serves models. It also matches how the rest of the configuration already travels — the
#: settings tree reads ``SHIPINFER_*`` with ``__`` for nesting — so a shard is configured the
#: same way an operator would configure one by hand.
#:
#: A comma-separated list of ids, not a JSON fleet: the child already has the whole fleet from
#: its own configuration, and sending the full camera objects would mean two descriptions of
#: one camera that can disagree.
def serve_command(
    shard: Shard,
    *,
    repository: str,
    extra: Sequence[str] = (),
    http_port_base: int | None = None,
) -> list[str]:
    """The default argv: this interpreter, running ``shipinfer serve`` on one repository.

    ``sys.executable`` rather than ``"shipinfer"`` so a child lands in the same virtualenv as
    the parent — the console script may not be on ``PATH`` inside a container, and a shard that
    starts under a different interpreter is a debugging session nobody wants.

    Two things are deliberately **not** on this line.

    The cameras. ``serve`` takes no such flag, and the first version of this function invented
    one — an argv that reads plausibly and that the CLI rejects. They travel in
    :data:`shipinfer.envs.SHARD_CAMERAS` instead, which
    :meth:`shipinfer.launch.Fleet._spawn` sets.

    The GPUs. The child sees only its own devices, because ``CUDA_VISIBLE_DEVICES`` was in its
    environment before it started, so to the child they are ``0..n-1`` — passing the physical
    ordinals as well would ask it to select devices it cannot see.

    Which leaves one per-shard thing on the line, and only when asked for: the HTTP port. A
    warm shard needs no ingress, so the default is `serve`'s default — no HTTP at all. With
    ``http_port_base`` every shard serves HTTP on ``base + shard.index``, so a fleet is
    addressable shard by shard (the `service` topology's tests drive one shard and read the
    other's statistics). One port for all of them is not an option: the second bind fails.
    Everything else per-shard travels in the environment, so the rest of the argv is identical
    for every shard.
    """
    argv = [sys.executable, "-m", "shipinfer", "serve", "-r", repository, *extra]
    if http_port_base is not None:
        argv += ["--http", "--port", str(http_port_base + shard.index)]
    return argv


def deepstream_command(
    shard: Shard,  # noqa: ARG001 - the topology contract hands one over; nothing is on the argv
    *,
    repository: str,
    extra: Sequence[str] = (),
) -> list[str]:
    """The argv for a `deepstream` shard: this interpreter, running one DeepStream graph.

    Shaped exactly like :func:`serve_command`, and short for the same reasons: the cameras
    travel in :data:`shipinfer.envs.SHARD_CAMERAS` and the devices in ``CUDA_VISIBLE_DEVICES``,
    both already in the child's environment before its interpreter starts.

    What is missing compared to `serve` is the HTTP port, and that is not an oversight. A
    DeepStream shard runs no :class:`~shipinfer.engine.InferenceServer`: there is no model
    table, no KServe v2 endpoint and nothing for ``base + shard.index`` to bind, so a port on
    this line would be a flag the child parses and ignores. The topology refuses
    ``http_port_base`` outright rather than dropping it silently — an operator who asks a
    DeepStream fleet for an HTTP API has a wrong expectation, and a fleet that starts anyway
    confirms it.

    ``server`` may not import ``pipeline`` (the layering rule), so this names its child by argv
    only. That is enough: the CLI command is the seam, and it is the same one an operator uses
    by hand.
    """
    return [sys.executable, "-m", "shipinfer", "deepstream", "-r", repository, *extra]
