"""``shipinfer run --http``: the web server goes on a thread, and the signals stay ours.

Offline, over a fake ``uvicorn``. What is under test is the *wiring*, and every line of it is
a decision that fails silently if it is wrong:

* the server runs on a **thread**, because the main thread is the supervising thread — that
  is ``launch/signals.py``'s invariant (the handler records, the supervising thread blocks),
  and it is what lets ``supervise()`` return the instant the runner is told to go;
* ``forward_signals`` stays the **only** handler this process installs. ``uvicorn.Server``
  installs its own inside ``serve()`` and skips it only off the main thread, so a
  main-thread server would make Ctrl-C stop the web server and leave fifty decoder threads
  reading;
* the supervise loop returning sets ``should_exit`` and joins, rather than leaving a daemon
  thread to be shot at interpreter exit with a request half-answered.

The shape of the file follows from the first bullet: ``_wait`` must run on the **main**
thread, because ``signal.signal`` refuses anywhere else — so the assertions that need the
server up run inside :meth:`SupervisingRunner.supervise`, which is exactly the moment
``shipinfer run`` spends its whole life in.

A real ``shipinfer run --http`` is container-tier evidence (``shipinfer run`` loads engines
and drives GPUs, so ``runtime/containment.py`` gates it); this file is the part of it that can
be pinned on a laptop, and ``tests/api/test_streams_over_a_runner.py`` is the other part —
the routes against a real runner.
"""

from __future__ import annotations

import signal
import sys
import threading
import types
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from shipinfer.cli.commands.run import _wait
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import ResponseFuture
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainItem, ChainSpec, Topology

CHAIN = """
name: mock_chain
elements:
  decode: {impl: mock}
  detect: {impl: mock, model: ship_detector}
  output: {impl: mock}
"""


class SupervisingRunner(Runner):
    """A runner whose ``supervise`` runs the test's assertions and then returns.

    A real :class:`Runner` subclass rather than a mock, because ``_wait`` installs
    ``forward_signals`` over it and the whole point of that call is that the ABC's
    ``request_stop`` is what the handler reaches.

    ``probe`` is called *inside* ``supervise``: the web server is up by then and the process
    is in the state it spends its life in, and it runs on the main thread because that is
    where ``_wait`` has to be called from.
    """

    name: ClassVar[str] = "supervising"
    manages_cameras: ClassVar[bool] = True

    def __init__(self, topology: Topology, **kwargs: Any) -> None:
        super().__init__(topology, **kwargs)
        self.probe: Callable[[], None] | None = None
        self.supervising = threading.Event()

    def supervise(self, **kwargs: Any) -> None:
        self.supervising.set()
        if self.probe is not None:
            self.probe()

    def _do_start(self) -> None:
        return None

    def _do_stop(self, timeout_s: float) -> None:
        return None

    def _do_submit(self, item: ChainItem) -> ResponseFuture:  # pragma: no cover - unused
        raise NotImplementedError

    def add_camera(self, camera: Any) -> None:  # pragma: no cover - never called here
        return None


class FakeUvicornServer:
    """Records what it was configured with, and blocks in ``run`` until ``should_exit``."""

    instances: ClassVar[list[FakeUvicornServer]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.should_exit = False
        self.serving = threading.Event()
        self.finished = threading.Event()
        self.ran_on = ""
        FakeUvicornServer.instances.append(self)

    def run(self) -> None:
        self.ran_on = threading.current_thread().name
        self.serving.set()
        while not self.should_exit:
            self.finished.wait(0.005)
        self.finished.set()


@pytest.fixture()
def uvicorn(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A ``uvicorn`` module that binds no socket. Real FastAPI, fake server.

    Patched into ``sys.modules`` rather than onto an attribute, because ``api/app.py`` imports
    uvicorn *inside* the constructor — the property ``tests/api/test_optional_dependency.py``
    pins, and this fixture must not work around it.
    """
    pytest.importorskip("fastapi")
    module = types.ModuleType("uvicorn")
    module.Config = lambda app, **options: {"app": app, **options}  # type: ignore[attr-defined]
    module.Server = FakeUvicornServer  # type: ignore[attr-defined]
    FakeUvicornServer.instances.clear()
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return module


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Put the interpreter's handlers back: ``_wait`` installs process-global state."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture()
def runner() -> SupervisingRunner:
    return SupervisingRunner(Topology.from_spec(ChainSpec.from_yaml(CHAIN))).start()


def serving() -> FakeUvicornServer:
    """The server ``_wait`` built, once its thread has got as far as answering."""
    assert FakeUvicornServer.instances, "no uvicorn server was built"
    server = FakeUvicornServer.instances[-1]
    assert server.serving.wait(5.0), "uvicorn never started"
    return server


class TestTheServerRunsBesideTheSupervisor:
    def test_uvicorn_is_started_on_a_thread_of_its_own(self, runner, uvicorn) -> None:
        seen: dict[str, str] = {}
        runner.probe = lambda: seen.update(ran_on=serving().ran_on)

        _wait(runner, http=True, host="127.0.0.1", port=8123)

        assert seen["ran_on"] != threading.main_thread().name
        assert seen["ran_on"].startswith("shipinfer-http")

    def test_the_host_port_and_log_level_reach_uvicorn(self, runner, uvicorn) -> None:
        config: dict[str, Any] = {}
        runner.probe = lambda: config.update(serving().config)

        _wait(runner, http=True, host="127.0.0.1", port=8123, log_level="WARNING")

        assert config["host"] == "127.0.0.1"
        assert config["port"] == 8123
        assert config["log_level"] == "warning", "uvicorn wants it lower-cased"
        assert config["workers"] == 1, "the runner owns its own threads"

    def test_the_supervise_loop_returning_stops_the_server_and_joins_it(
        self, runner, uvicorn
    ) -> None:
        """``should_exit`` is uvicorn's own cooperative flag; the thread is then joined.

        Not left to interpreter exit: the thread is a daemon, and a daemon killed mid-request
        is a ``POST /streams`` whose caller never learns whether the camera was placed.
        """
        runner.probe = serving

        _wait(runner, http=True)

        server = FakeUvicornServer.instances[-1]
        assert server.should_exit is True
        assert server.finished.is_set(), "the HTTP thread was not joined"

    def test_no_http_means_no_server_at_all(self, runner, uvicorn) -> None:
        """A chain driven by ``--inputs`` needs no ingress, and pays for none."""
        _wait(runner, http=False)

        assert runner.supervising.is_set()
        assert FakeUvicornServer.instances == []


class TestTheSignalHandlersStayTheRunners:
    def test_ctrl_c_still_reaches_the_runner_while_the_server_is_up(
        self, runner, uvicorn
    ) -> None:
        """The one that matters.

        ``uvicorn.Server`` installs SIGINT/SIGTERM handlers inside ``serve()`` and skips it
        only off the main thread. If one were ever installed here, Ctrl-C would stop the web
        server and leave every decoder thread reading — and the deployment would look up.
        """
        handlers: dict[str, Any] = {}

        def probe() -> None:
            serving()
            handler = signal.getsignal(signal.SIGINT)
            handlers["sigint"] = handler
            assert callable(handler)
            handler(signal.SIGINT, None)

        runner.probe = probe
        _wait(runner, http=True)

        assert handlers["sigint"].__module__ == "shipinfer.launch.signals", handlers
        assert runner.stop_requested, "the handler did not reach the runner"

    def test_sigterm_is_the_same_handler(self, runner, uvicorn) -> None:
        handlers: dict[str, Any] = {}

        def probe() -> None:
            serving()
            handlers["sigint"] = signal.getsignal(signal.SIGINT)
            handlers["sigterm"] = signal.getsignal(signal.SIGTERM)

        runner.probe = probe
        _wait(runner, http=True)

        assert handlers["sigterm"] is handlers["sigint"]


class TestWhenTheExtraIsMissing:
    def test_http_without_the_server_extra_is_a_typed_refusal_naming_it(
        self, runner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raised before the supervise loop, so the operator is told at start-up.

        ``None`` in ``sys.modules`` is the interpreter's own "this import is blocked" marker.
        """
        for name in ("fastapi", "uvicorn"):
            monkeypatch.setitem(sys.modules, name, None)

        with pytest.raises(ConfigurationError, match=r"shipinfer\[server\]"):
            _wait(runner, http=True)

        assert not runner.supervising.is_set(), "it supervised with no ingress up"

    def test_a_run_without_http_needs_no_extra_at_all(
        self, runner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("fastapi", "uvicorn"):
            monkeypatch.setitem(sys.modules, name, None)

        _wait(runner, http=False)

        assert runner.supervising.is_set()
