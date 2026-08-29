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
  thread to be shot at interpreter exit with a request half-answered;
* **the bind is confirmed before the deployment is allowed to look healthy.** A daemon thread
  that dies takes its exception with it, and uvicorn answers a taken port by calling
  ``sys.exit`` inside ``startup()``, which ``threading.excepthook`` discards without a word.
  So ``--http`` on a taken port used to spawn every shard, place every camera, log *"serving
  /streams"*, serve nothing at all and exit ``0``. :class:`TestABindThatFails` is that case
  against a **real** socket and a real uvicorn, because the whole failure lives in what the
  library does off the main thread.

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
import socket
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from shipinfer.cli.commands.run import _wait, run
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import ResponseFuture
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainItem, ChainSpec, Topology

#: A file source and a sink that counts, and nothing between them: what is under test is the
#: web server in front of a runner, and a model slot would make every case here build an
#: engine out of the demo repository.
CHAIN = """
name: http_chain
elements:
  decode: {impl: replay}
  output: {impl: none}
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
    """Records what it was configured with, and blocks in ``run`` until ``should_exit``.

    ``started`` is uvicorn's own flag and is faithfully part of the fake, because
    ``BackgroundHttpServer.start`` now waits for it before it claims to be serving. Set
    *after* the socket would be open, which is where the real ``Server.startup`` sets it.

    ``fail_to_bind`` is the other half of the real thing: uvicorn catches the ``OSError`` from
    a taken port inside ``startup()`` and calls ``sys.exit``, so the thread ends with
    ``started`` still False and the ``SystemExit`` is swallowed by ``threading.excepthook``.
    """

    instances: ClassVar[list[FakeUvicornServer]] = []
    #: Set by a test before the server is built; every instance reads it at ``run``.
    fail_to_bind: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.started = False
        self.should_exit = False
        self.serving = threading.Event()
        self.finished = threading.Event()
        self.ran_on = ""
        FakeUvicornServer.instances.append(self)

    def run(self) -> None:
        self.ran_on = threading.current_thread().name
        if FakeUvicornServer.fail_to_bind:
            self.finished.set()
            raise SystemExit(3)
        self.started = True
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
    FakeUvicornServer.fail_to_bind = False
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
def chain_file(tmp_path: Path) -> Path:
    """A chain on disk, for the tests that go through ``run()`` rather than ``_wait``."""
    path = tmp_path / "mock_chain.yaml"
    path.write_text(CHAIN)
    return path


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


class TestABindThatFailsIsRefusedRatherThanLoggedAsSuccess:
    """Against a real socket and the installed uvicorn, because that is where the bug lived.

    A fake server cannot pin this: what made the old code wrong was a *library* behaviour off
    the main thread -- ``Server.startup`` catching the ``OSError`` and calling ``sys.exit``,
    and ``threading.excepthook`` silently dropping the resulting ``SystemExit``. Asserting it
    over a stand-in that raises on demand would prove only that the stand-in raises.

    No GPU, no container, no marker: a loopback socket is all this needs, which is why it
    belongs in the offline tier despite touching the network stack.
    """

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_a_port_that_is_already_taken_is_a_typed_refusal_and_not_a_dead_thread(
        self,
    ) -> None:
        """The ignored warning is the bug: pytest sees the ``SystemExit`` a deployment does
        not, because ``threading.excepthook`` drops it and pytest's hook does not."""
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        from shipinfer.api import BackgroundHttpServer, create_app

        runner = SupervisingRunner(Topology.from_spec(ChainSpec.from_yaml(CHAIN)))
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]
            server = BackgroundHttpServer(
                create_app(cameras=runner), host="127.0.0.1", port=port, bind_timeout_s=5.0
            )

            with pytest.raises(ConfigurationError, match=rf"127\.0\.0\.1:{port}"):
                server.start()

            assert server._thread is None, "a failed start left a thread behind"
            assert not [
                thread for thread in threading.enumerate() if thread.name == "shipinfer-http"
            ], "the HTTP thread outlived the refusal"
            # The `finally` in `_wait` and in `run()` reaches this either way; it must not
            # raise or touch a server that never served.
            server.stop()
        finally:
            occupied.close()

    def test_a_bind_that_works_reports_started_and_stops_cleanly(self) -> None:
        """The other side of the same gate, so the confirmation cannot be vacuously true.

        Port ``0`` rather than a port picked and released, which would be a race: the OS
        chooses a free one and uvicorn binds it, and what is under test is that ``start()``
        returns only once ``started`` is set -- no sleep, no polling in the assertion.
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        from shipinfer.api import BackgroundHttpServer, create_app

        runner = SupervisingRunner(Topology.from_spec(ChainSpec.from_yaml(CHAIN)))
        server = BackgroundHttpServer(
            create_app(cameras=runner), host="127.0.0.1", port=0, bind_timeout_s=10.0
        )
        try:
            assert server.start() is server
            assert server._server.started is True, "start() returned before the bind"
            thread = server._thread
            assert thread is not None and thread.is_alive()
        finally:
            server.stop(timeout_s=10.0)

        assert thread is not None and not thread.is_alive(), "stop() did not join the thread"
        assert server._thread is None, "a stopped server still holds a thread"


class TestTheCommandDoesNotSuperviseWithNoIngress:
    """The wiring, over the fake: a refused bind must stop the run rather than continue it."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_a_failed_bind_travels_out_of_wait_before_anything_supervises(
        self, runner, uvicorn
    ) -> None:
        """Raised *before* ``supervise()``, which is what makes it reach ``run()``'s finally.

        The alternative -- logging it and supervising anyway -- is the bug: the deployment
        runs, the camera door is shut, and nothing in the process has a reason to say so.
        """
        FakeUvicornServer.fail_to_bind = True

        with pytest.raises(ConfigurationError, match=r"127\.0\.0\.1:8123"):
            _wait(runner, http=True, host="127.0.0.1", port=8123)

        assert not runner.supervising.is_set(), "it supervised with no ingress up"

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_the_command_exits_non_zero_and_stops_the_runner(
        self, chain_file: Path, uvicorn, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through ``run()``: the refusal escapes rather than being logged.

        ``run()`` returns ``0`` on a clean shutdown, so "the refusal escapes" *is* the
        non-zero exit: ``cli/__init__.py`` wraps the return value in ``typer.Exit`` and never
        constructs one when the call raises, and click's standalone mode does not swallow a
        ``ShipInferError``. What must not happen is the old behaviour -- a normal return, exit
        ``0``, and a deployment with no ingress.

        ``supervise`` is stubbed out for a reason worth stating: without it, a regression in
        the bind check does not *fail* this test, it **hangs** it -- a real ``InprocessRunner``
        supervises until it is signalled, which is exactly what ``run()`` is supposed to be
        prevented from reaching here. A test whose failure mode is a hang is a test that gets
        deleted; this one fails in a second and names what it was waiting for.
        """
        from shipinfer.runners.inprocess import InprocessRunner

        supervised = threading.Event()
        FakeUvicornServer.fail_to_bind = True
        monkeypatch.setattr(
            "shipinfer.runtime.containment.require_container", lambda *a, **k: None
        )
        monkeypatch.setattr(
            InprocessRunner, "supervise", lambda self, **kwargs: supervised.set()
        )

        with pytest.raises(ConfigurationError, match=r"did not come up"):
            run(chain_file, runner="inprocess", http=True, port=8123)

        assert FakeUvicornServer.instances, "no server was ever built"
        assert not supervised.is_set(), "it supervised a deployment with no ingress"


class TestFlagsThatWouldBeIgnored:
    """``--host``/``--port`` say where the web server goes; without ``--http`` there is none.

    Accepted silently, they are a deployment that looks configured and is not -- the same
    shape as the bind failure above, one flag out and one layer earlier, and the refusal costs
    a line. ``None`` is the sentinel precisely so ``--host 127.0.0.1`` typed out in full is
    still refused: the operator asked for something this run will not do.
    """

    def test_a_port_without_http_is_refused_naming_the_flag_that_is_missing(
        self, chain_file: Path
    ) -> None:
        with pytest.raises(ConfigurationError, match=r"--port.*`--http`"):
            run(chain_file, runner="inprocess", port=9000, dry_run=True)

    def test_a_host_without_http_is_refused_too(self, chain_file: Path) -> None:
        with pytest.raises(ConfigurationError, match=r"--host"):
            run(chain_file, runner="inprocess", host="0.0.0.0", dry_run=True)

    def test_both_are_named_in_one_message(self, chain_file: Path) -> None:
        with pytest.raises(ConfigurationError, match=r"--host and --port"):
            run(chain_file, runner="inprocess", host="0.0.0.0", port=9000, dry_run=True)

    def test_saying_nothing_is_not_saying_the_default(self, chain_file: Path) -> None:
        """The reason both options are declared ``None`` rather than with their real values."""
        assert run(chain_file, runner="inprocess", dry_run=True) == 0

    def test_with_http_they_are_exactly_what_they_configure(self, runner, uvicorn) -> None:
        config: dict[str, Any] = {}
        runner.probe = lambda: config.update(serving().config)

        _wait(runner, http=True, host="0.0.0.0", port=9000)

        assert (config["host"], config["port"]) == ("0.0.0.0", 9000)


class TestTheUvicornBehaviourAllOfThisRestsOn:
    def test_uvicorn_installs_no_handler_from_a_non_main_thread(self) -> None:
        """The mechanism every test above depends on, pinned against the installed uvicorn.

        Those tests assert the *outcome* over a fake server, which is the right level -- but
        the outcome is only true because ``Server.capture_signals`` early-returns off the main
        thread. uvicorn 0.52 removed ``install_signal_handlers``, the flag that used to say so
        out loud; if a release ever drops the thread check as well, ``shipinfer run --http``
        goes quietly back to a Ctrl-C that stops the web server and leaves fifty decoder
        threads reading. This is the guard that would notice, and it costs one import.
        """
        uvicorn = pytest.importorskip("uvicorn")
        server = uvicorn.Server(uvicorn.Config(app=None))
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        during: dict[Any, Any] = {}

        def enter() -> None:
            with server.capture_signals():
                during.update({sig: signal.getsignal(sig) for sig in before})

        thread = threading.Thread(target=enter, name="not-the-main-thread")
        thread.start()
        thread.join(5.0)

        assert not thread.is_alive(), "capture_signals never returned"
        assert during == before, "uvicorn installed a handler from a non-main thread"


class TestWhenTheExtraIsMissing:
    def test_the_command_refuses_before_it_builds_or_starts_anything(
        self, chain_file: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where an operator actually meets this: ``shipinfer run --http`` on a bare host.

        Probed beside the ``manages_cameras`` refusal and for its argument -- whether this
        host can serve HTTP is a fact about the host, not about this execution. Asked inside
        ``_wait``, which runs after ``built.start()``, a fleet spawned sixteen shard processes
        and placed every camera before anything looked for FastAPI, and the operator paid a
        full start-up and shutdown to be told about a ``pip install``.

        ``--dry-run`` is what makes the *ordering* visible: the refusal comes out ahead of the
        plan, which is itself ahead of ``start()``.
        """

        def never(self: Runner) -> Runner:  # pragma: no cover - the assertion is that it is
            raise AssertionError("the runner was started")

        monkeypatch.setitem(sys.modules, "fastapi", None)
        monkeypatch.setattr(Runner, "start", never)

        with pytest.raises(ConfigurationError, match=r"shipinfer\[server\]"):
            run(chain_file, runner="inprocess", http=True, dry_run=True)

        assert "no plan" not in capsys.readouterr().out, "it got as far as printing the plan"

    def test_a_dry_run_without_http_still_needs_no_extra(
        self, chain_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe is behind the flag: a chain driven by ``--inputs`` serves nothing."""
        for name in ("fastapi", "uvicorn"):
            monkeypatch.setitem(sys.modules, name, None)

        assert run(chain_file, runner="inprocess", dry_run=True) == 0

    def test_http_without_the_server_extra_is_a_typed_refusal_naming_it(
        self, runner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_wait``'s own refusal: the last line of defence, not where an operator meets it.

        It stays because ``_wait`` is reachable without going through :func:`run`, and because
        the import that fails belongs next to the refusal. Raised before the supervise loop,
        so nothing supervises with no ingress up.

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
