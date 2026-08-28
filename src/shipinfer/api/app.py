"""Building and running the ASGI application."""

from __future__ import annotations

import threading
import time
from typing import Any

from shipinfer.api.routes import build_router
from shipinfer.api.streams import CameraController, build_streams_router
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.engine.pool import InferenceServer

__all__ = ["BackgroundHttpServer", "create_app", "require_server_extra", "serve_http"]

# The logger name stays "server.api" on purpose: an operator's log filter is behaviour,
# and this move promises none changed. It is retargeted when server/ is deleted (A2 PR-6).
_LOG = get_logger("api")

#: One message for the one missing thing, so a host that installed neither is told about
#: whichever import failed first and always about the same extra.
_MISSING = 'the HTTP API needs {name}: pip install "shipinfer[server]"'


def require_server_extra() -> None:
    """Refuse now if FastAPI or uvicorn is missing. Builds nothing, binds nothing.

    Lets the composition root learn the host cannot serve HTTP *before* starting anything;
    it does not replace the checks in :func:`create_app` and :class:`BackgroundHttpServer`,
    which sit where the import actually happens and are reachable from library callers.

    Raises:
        ConfigurationError: either import failed; typed and naming the extra.
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

    Each router is mounted only when the thing behind it exists (arch.md section 2):
    ``server`` brings the KServe side-door, ``cameras`` brings ``/streams``. Neither
    lifecycle is tied to the app's: whoever created the server or runner stops it.

    Raises:
        ConfigurationError: FastAPI is not installed (the ``server`` extra), or neither
            argument was given — an app with no routers would answer 404 for everything.
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

    _LOG.info("serving KServe v2 on http://%s:%d (docs at /docs)", host, port)
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
    """uvicorn on a thread, so the main thread keeps supervising the runner.

    The thread is what keeps the signal handlers ours: uvicorn installs its own
    ``SIGINT``/``SIGTERM`` handlers only when run on the main thread (uvicorn 0.52 removed
    the flag that said this explicitly; ``tests/cli/test_run_http.py`` asserts the outcome —
    after the server is up, ``SIGINT`` still routes to the runner). Shutdown is uvicorn's
    cooperative ``should_exit`` flag — no kill, because a request in flight may be somebody's
    ``POST /streams``. Lives in ``api/`` because ``uvicorn`` may be named in exactly one
    layer (``scripts/hooks/check_layers.py``).
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
                serving. Generous on purpose: this thread starts while a fleet is spawning
                shards, and a deadline that lost that race would refuse a deployment about
                to work; the wait returns the moment ``started`` flips.

        Raises:
            ConfigurationError: uvicorn is not installed (the ``server`` extra).
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

        The bind must be confirmed: a daemon thread that dies takes its exception with it
        silently, so a taken port once ran a whole deployment with no ingress and exited 0.
        A failed start leaves nothing running -- ``should_exit`` set (a bind that wins the
        race one tick late must not keep the port), thread reference dropped so :meth:`stop`
        stays a no-op, and not joined (a wedged startup must not hang the refusal).

        Raises:
            ConfigurationError: not serving within ``bind_timeout_s``; names the host:port.
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
        _LOG.info("serving /streams on http://%s:%d (docs at /docs)", self._host, self._port)
        return self

    def _serving_within(self, timeout_s: float) -> bool:
        """Whether uvicorn reported itself serving before the deadline.

        ``Server.started`` is uvicorn's own flag; polling (5 ms) is what there is, because
        ``Server`` offers no event and its loop belongs to the thread being waited on. The
        thread is watched as well as the clock, so a failed bind returns within
        milliseconds rather than after ``bind_timeout_s``.
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
            _LOG.warning(
                "the HTTP thread did not finish within %.1fs; it is a daemon and the "
                "process will not wait for it",
                timeout_s,
            )
        self._thread = None
