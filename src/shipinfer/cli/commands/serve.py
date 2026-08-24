"""``shipinfer serve`` — run the server."""

from __future__ import annotations

import signal
import threading
from pathlib import Path

from shipinfer.cli.common import build_settings, console
from shipinfer.runtime.containment import require_container
from shipinfer.server import InferenceServer, check_health

__all__ = ["serve"]


def serve(
    repository: Path,
    *,
    gpus: str | None = None,
    policy: str | None = None,
    http: bool = False,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "INFO",
) -> int:
    """Load the repository and serve until interrupted.

    Without ``--http`` this is a *warm* server with no ingress: models loaded, engines
    deserialised, graphs captured. That is the shape a pipeline embedding the library in
    the same process wants, and it is also the fastest way to find out whether a
    repository will actually come up on this node.
    """
    # The gate lives here, not only in the shell hook: a deny-list over command
    # text cannot be made sound, and this command loads engines and drives GPUs.
    require_container("`shipinfer serve`")
    out = console()
    settings = build_settings(
        repository,
        gpus=gpus,
        policy=policy,
        log_level=log_level,
        http={"enabled": http, "host": host, "port": port},
    )

    server = InferenceServer(settings)
    stop = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        out.print(f"\nreceived signal {signum}; draining")
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        server.start()
        report = check_health(server)
        out.print(f"[bold]{report.status.value}[/bold]: {report.detail}")

        if http:
            from shipinfer.server.api import serve_http

            serve_http(server, host=host, port=port)
        else:
            stop.wait()
    finally:
        server.stop()
    return 0
