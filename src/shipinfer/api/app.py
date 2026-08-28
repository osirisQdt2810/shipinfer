"""Building and running the ASGI application."""

from __future__ import annotations

import threading
import time
from typing import Any

from shipinfer.api.routes import build_router
from shipinfer.api.streams import CameraController, build_streams_router
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import LOG
from shipinfer.engine.pool import InferenceServer

__all__ = ["BackgroundHttpServer", "create_app", "require_server_extra", "serve_http"]

# The logger name stays "server.api" on purpose: an operator's log filter is behaviour,
# and this move promises none changed. It is retargeted when server/ is deleted (A2 PR-6).

#: One message for the one missing thing, so a host that installed neither is told about
#: whichever import failed first and always about the same extra.
_MISSING = 'the HTTP API needs {name}: pip install "shipinfer[server]"'


def require_server_extra() -> None:
    """Refuse now if FastAPI or uvicorn is missing. Builds nothing, binds nothing.

    For the composition root, which learns whether it can serve HTTP *before* it starts
    anything: ``shipinfer run --http`` on a host without the extra used to get as far as
    sixteen shard processes and a placed camera set, and only then failed on the import
    inside :class:`BackgroundHttpServer` -- so the operator paid a full start-up and a full
    shutdown to be told about a ``pip install``. The refusal is a *fact about the host*, known
    as soon as the flag is read, which is the same argument
    ``cli/commands/run.py::refuse_if_it_manages_no_cameras`` makes one line above the call.

    It does not replace the checks in :func:`create_app` and :class:`BackgroundHttpServer`.
    Those are where the import actually happens and they are reachable from a library caller
    who never went through the CLI; this is the early word, and a probe that let them be
    deleted would be a probe every other entry point has to remember to call.

    Raises:
        ConfigurationError: either import failed. Typed and naming the extra, never an
            ``ImportError``: the caller asked for something this host is not set up to do.
    """
    import importlib

    for name, shown in (("fastapi", "FastAPI"), ("uvicorn", "uvicorn")):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ConfigurationError(_MISSING.format(name=shown)) from exc


def create_app(
    server: InferenceServer | None = None, *, cameras: CameraController | None = None
) -> Any:
    """Wrap a running server, a camera controller, or both, in a FastAPI application.

    Two doors, two arguments, and each router is mounted only when the thing behind it exists
    (arch.md section 2): ``server`` brings the KServe side-door for a caller who already has
    tensors, ``cameras`` brings ``/streams`` for a caller who has a camera. They are separate
    because the two commands that serve them are: ``shipinfer serve`` owns an engine and no
    runner, ``shipinfer run --http`` owns a runner and — with ``--runner fleet`` — no engine
    in this process at all. Mounting a router with nothing behind it would answer 500 for
    every request to it, which is a worse answer than 404.

    Neither lifecycle is tied to the app's: whoever created the server or the runner started
    it and will stop it. Tying them would make the HTTP layer load-bearing for a library whose
    main use is in-process.

    Raises:
        ConfigurationError: FastAPI is not installed (the ``server`` extra), or neither a
            server nor a camera controller was given — an app with no routers answers 404 for
            everything, and finding that out from a probe is more expensive than being told
            here.
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise ConfigurationError(_MISSING.format(name="FastAPI")) from exc
    if server is None and cameras is None:
        raise ConfigurationError(
            "create_app was given neither an engine nor a camera controller, so the "
            "application would have no routes; pass server= for the KServe side-door, "
            "cameras= for /streams, or both"
        )

    from shipinfer import __version__

    app = FastAPI(
        title="ShipInfer",
        version=__version__,
        description="KServe v2 inference API",
        docs_url="/docs",
    )
    if server is not None:
        app.include_router(build_router(server))
    if cameras is not None:
        app.include_router(build_streams_router(cameras))
    return app


def serve_http(server: InferenceServer, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the API with uvicorn until interrupted."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigurationError(_MISSING.format(name="uvicorn")) from exc

    LOG.info("serving KServe v2 on http://%s:%d (docs at /docs)", host, port)
    uvicorn.run(
        create_app(server),
        host=host,
        port=port,
        log_level=server.settings.observability.log_level.lower(),
        # The server manages its own thread pool per instance; letting uvicorn add
        # another layer of workers would multiply the queues and defeat the balancing.
        workers=1,
    )


class BackgroundHttpServer:
    """uvicorn on a thread, so the process's main thread keeps supervising.

    ``shipinfer run --http`` has two jobs at once: supervise a runner until it is told to stop
    (``launch/signals.py``'s invariant — the handler *records*, the supervising thread does the
    blocking work) and answer ``/streams``. Only one of them can own the main thread, and it
    has to be the supervisor: a signal handler installed by anything else is a Ctrl-C that
    stops the web server and leaves fifty decoder threads running.

    So uvicorn runs here instead, and **the thread is what keeps the signal handlers ours**.
    ``uvicorn.Server.capture_signals`` installs its own ``SIGINT``/``SIGTERM`` handlers, and
    the only thing that stops it is that it early-returns off the main thread (uvicorn 0.52
    removed the ``install_signal_handlers`` flag that used to say this explicitly; the thread
    rule is what is left, and ``tests/cli/test_run_http.py`` asserts the outcome rather than
    the mechanism — after the server is up, ``SIGINT`` is still routed to the runner).

    Shutdown is ``should_exit``, which is uvicorn's own cooperative flag: the serving loop
    notices it within one tick, finishes what it is answering and returns. There is no kill —
    a request in flight is somebody's ``POST /streams``, and cutting it off would leave a
    camera placed with nobody told.

    This class lives in ``api/`` and not in ``cli/`` because ``uvicorn`` may be named in
    exactly one layer (``scripts/hooks/check_layers.py``), and because the missing-extra
    refusal belongs next to the import that fails.
    """

    def __init__(
        self,
        app: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_level: str = "info",
        bind_timeout_s: float = 5.0,
    ) -> None:
        """Build the server. Binds nothing until :meth:`start`.

        Args:
            bind_timeout_s: how long :meth:`start` waits for uvicorn to report it is
                serving before it calls the start a failure. Generous rather than tight:
                what is being waited for is a ``listen()`` on a socket this process already
                chose, so a second is already long -- but this thread starts while a fleet is
                spawning shards, and a deadline short enough to lose that race would refuse a
                deployment that was about to work. Nothing waits on it in the normal case,
                because the loop below returns the moment ``started`` flips.

        Raises:
            ConfigurationError: uvicorn is not installed. Typed and naming the extra, for the
                reason the whole package imports lazily: a host that never installed
                ``shipinfer[server]`` must still be able to import this module.
        """
        try:
            import uvicorn
        except ImportError as exc:
            raise ConfigurationError(_MISSING.format(name="uvicorn")) from exc

        self._host = host
        self._port = port
        self._bind_timeout_s = bind_timeout_s
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                # One worker, for `serve_http`'s reason: the runner already owns a thread pool
                # per element and letting uvicorn add another layer would multiply the queues.
                workers=1,
            )
        )
        self._thread: threading.Thread | None = None

    def start(self) -> BackgroundHttpServer:
        """Serve on a daemon thread, and do not return until it is serving. Idempotent.

        Daemon on purpose: the runner's shutdown is what the process waits for, and an HTTP
        thread that could keep a stopped deployment alive would turn a failed
        :meth:`stop` into a hang instead of a warning.

        **The bind is confirmed, and that is the whole reason this method can fail.** A
        daemon thread that dies takes its exception with it: ``uvicorn.Server.startup``
        catches the ``OSError`` from a port already in use, logs it and calls
        ``sys.exit(3)``, and off the main thread ``threading.excepthook`` discards
        ``SystemExit`` without a word. So ``shipinfer run --http --port 8000`` against a
        taken port used to spawn every shard, place every camera, log *"serving /streams on
        ..."*, run the deployment with no ingress at all, and exit ``0`` -- there was nothing
        for a supervisor or a readiness probe to notice. Confirming ``started`` turns that
        into a refusal the operator reads before ``supervise()`` is ever entered, and the
        INFO line moved below the wait so it can no longer assert something untrue.

        Failure leaves nothing running: the thread reference is dropped, so :meth:`stop`
        stays a safe no-op, and ``should_exit`` is set first for the case the deadline lost a
        race rather than the bind failing -- a server that binds one tick after it was
        declared failed must not go on holding the port. It is not joined, because a startup
        wedged behind a lifespan hook would then hang the refusal it is meant to deliver, and
        the thread is a daemon.

        Raises:
            ConfigurationError: uvicorn did not report itself serving within
                ``bind_timeout_s`` -- in practice the address is in use, or the host does not
                own it. Named with the ``host:port`` because that is the one thing the
                operator can change, and typed like every other refusal in this module so it
                travels out of ``cli/commands/run.py::_wait`` through the ``finally`` that
                stops the runner.
        """
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._server.run, name="shipinfer-http", daemon=True
        )
        self._thread.start()
        if not self._serving_within(self._bind_timeout_s):
            self._server.should_exit = True
            self._thread = None
            raise ConfigurationError(
                f"the HTTP server did not come up on {self._host}:{self._port} within "
                f"{self._bind_timeout_s:.1f}s (the address is most likely already in use; "
                "uvicorn logged the reason) -- choose another --port, or stop what holds it"
            )
        LOG.info("serving /streams on http://%s:%d (docs at /docs)", self._host, self._port)
        return self

    def _serving_within(self, timeout_s: float) -> bool:
        """Whether uvicorn reported itself serving before the deadline.

        ``Server.started`` is uvicorn's own flag, set at the end of ``startup()`` once every
        listener is up; polling it is what there is, because ``Server`` offers no event and
        the loop it would be set on belongs to the thread being waited for. Polled at 5 ms so
        the normal case costs one tick rather than the deadline.

        The thread is watched as well as the clock, and that is what makes the common failure
        fast: a bind that fails ends the thread within milliseconds, so this returns then
        rather than after ``bind_timeout_s`` of a port that was never going to open.
        """
        deadline = time.monotonic() + timeout_s
        thread = self._thread
        while not self._server.started:
            if thread is None or not thread.is_alive() or time.monotonic() >= deadline:
                # Re-read rather than return False: the flag may have flipped in the window
                # between the loop test and the thread ending.
                return bool(self._server.started)
            time.sleep(0.005)
        return True

    def stop(self, timeout_s: float = 10.0) -> None:
        """Ask uvicorn to finish and wait for the thread. Idempotent, and never raises.

        A server that will not stop is logged rather than raised: this runs in the ``finally``
        that also stops the runner, and an exception here would mask why the deployment was
        going down in the first place.

        Safe after a :meth:`start` that refused, and safe before one: a failed start drops
        the thread reference, so this returns on the first line rather than setting
        ``should_exit`` on a server that never served.
        """
        if self._thread is None:
            return
        self._server.should_exit = True
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            LOG.warning(
                "the HTTP thread did not finish within %.1fs; it is a daemon and the "
                "process will not wait for it",
                timeout_s,
            )
        self._thread = None
