"""One camera, one thread, for the thread's whole life.

This is the ingest half of ADR-002. A camera actor is **stateful** — it owns the RTSP
connection, the decoder, the ``frame_id`` counter and the reconnect schedule — and it does
**no** inference. Everything it produces goes into a sink and is picked up by the
stateless, GPU-pooled half of the system, which is what lets fifty uneven cameras share
sixteen GPUs instead of being pinned three-to-a-device.

What it deliberately does **not** own is a queue. The reference system gave every camera a
share of one 1000-slot buffer that evicted the *oldest* entry when full, so a crowded camera
starved a quiet one and nothing logged it. This actor publishes into a
:class:`~shipinfer.ingest.sink.FrameSink` — in production one backed by the fair, bounded
queue in :mod:`shipinfer.scheduling.queues`, which has per-camera lanes and sheds the
loudest camera rather than its victim (ADR-005). Writing another queue *here* would
reintroduce exactly the bug the project exists to fix; depending on the scheduler directly
would put dispatch policy in the decode path, which is the other half of the same mistake.

One policy decision is worth calling out because it is not the obvious one: **a successful
connection does not reset the failure count — a successful frame does.** An RTSP source that
accepts a connection and then delivers nothing is the most common real failure mode of a
camera fleet, and treating "opened" as "healthy" is precisely how it stays invisible.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from shipinfer.core.errors import (
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
    SourceUnavailableError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.redact import redact, redact_in
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.camera.health import CameraHealth, CameraState
from shipinfer.ingest.frame.frame import Frame
from shipinfer.ingest.frame.tag import FrameCounter
from shipinfer.ingest.metrics import IngestMetrics
from shipinfer.ingest.registry import create_source
from shipinfer.ingest.sink import FrameSink
from shipinfer.ingest.timing.backoff import ExponentialBackoff

__all__ = ["CameraActor", "SourceFactory"]

_LOG = get_logger("ingest.camera")

#: How a caller substitutes a source. The counter is passed in, never created by the
#: factory, because frame ids must survive the reconnect that replaces the source.
SourceFactory = Callable[[CameraConfig, FrameCounter], FrameSource]

#: Window over which :attr:`CameraActor.health` measures fps. Long enough to be stable at
#: 20 fps, short enough that a camera that stopped a few seconds ago reads as 0.
_FPS_WINDOW_S = 2.0

#: Shortest partial window that gives a usable fps estimate before the first full one.
_FPS_MIN_WINDOW_S = 0.25


class CameraActor:
    """Pulls one camera and publishes tagged frames into a sink.

    Args:
        config: the camera to run.
        sink: where frames go. Any :class:`~shipinfer.ingest.sink.FrameSink`; in production
            one backed by the fair, bounded queue (ADR-005), in a benchmark a
            :class:`~shipinfer.ingest.sink.CountingSink`.
        settings: fleet-wide ingest defaults.
        metrics: shared handles. One :class:`IngestMetrics` for the whole fleet, labelled by
            camera, so an operator gets per-camera numbers without per-camera objects.
        source_factory: builds the source. Injected for tests and for a deployment with a
            source this package does not ship; defaults to
            :func:`~shipinfer.ingest.registry.create_source`.
        sleep: how the actor waits out a reconnect delay. Defaults to
            :meth:`_wait_or_stop`, which waits on the stop event rather than the clock;
            injected so the offline tier can assert the *sequence* of reconnect delays
            rather than merely that a retry happened.
    """

    def __init__(
        self,
        config: CameraConfig,
        sink: FrameSink,
        *,
        settings: IngestSettings | None = None,
        metrics: IngestMetrics | None = None,
        source_factory: SourceFactory | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or IngestSettings()
        self.metrics = metrics or IngestMetrics()
        self._sink = sink
        self._stop = threading.Event()
        self._sleep: Callable[[float], None] = self._wait_or_stop if sleep is None else sleep
        self._factory: SourceFactory = source_factory or self._default_factory
        self._counter = FrameCounter(config.camera_id, config.first_frame_id)
        self._backoff = ExponentialBackoff(
            self.settings.reconnect_initial_ms / 1000.0,
            self.settings.reconnect_max_ms / 1000.0,
            factor=self.settings.reconnect_factor,
            jitter=self.settings.reconnect_jitter,
        )
        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._fatal = False

        # Everything below is written by the actor thread and read by anyone; the lock is
        # taken once per frame at most, which at 20 fps per camera is free.
        self._lock = threading.Lock()
        self._state = CameraState.IDLE
        self._frames_read = 0
        self._frames_published = 0
        self._frames_dropped = 0
        self._empty_reads = 0
        self._connects = 0
        self._connect_failures = 0
        self._consecutive_empty = 0
        self._last_error = ""
        self._last_frame_unix_ns = 0
        self._last_frame_ns = 0
        self._fps = 0.0
        self._fps_window_start = 0.0
        self._fps_window_frames = 0

    # -- identity ----------------------------------------------------------------------

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_counter(self) -> FrameCounter:
        """The counter, so a caller can see where the tag sequence has reached."""
        return self._counter

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Start the actor thread. Not restartable once stopped.

        Raises:
            ServerStateError: if already started. A restarted actor would need a second
                opinion on where its frame counter stands; the manager builds a fresh actor
                instead, which has one.
        """
        if self._thread is not None:
            raise ServerStateError(
                f"camera {self.camera_id!r} has already been started; "
                "build a new actor rather than restarting this one"
            )
        self._stop.clear()
        self._set_state(CameraState.CONNECTING)
        self._thread = threading.Thread(
            target=self._run, name=f"ingest-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the actor to finish, without waiting for it.

        Separate from :meth:`stop` because shutting down fifty cameras should cost one read
        timeout, not fifty: signal them all, then join them all.
        """
        self._stop.set()

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Ask the actor to finish, and wait for it. Idempotent.

        A no-op on an actor that was never started and on one that has already stopped,
        because shutdown paths call this from more than one place and neither may hang.

        The thread notices within one read timeout, which is why every source bounds its
        read. A thread still alive after ``timeout_s`` is logged and abandoned: it is a
        daemon blocked inside a decoder, and holding up the whole process's shutdown behind
        it would be the worse failure.

        Returns ``False`` when the thread had to be abandoned, ``True`` on a clean stop —
        the same contract as the C++ plane's ``CameraActor::stop``, so a caller can count
        its abandonments instead of grepping the log for them.
        """
        self.request_stop()
        abandoned = False
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout_s)
            if thread.is_alive():
                abandoned = True
                _LOG.warning(
                    "camera %s did not stop within %.1fs; abandoning the thread",
                    self.camera_id,
                    timeout_s,
                    extra=log_context(camera_id=self.camera_id),
                )
        if not self._state_is_final():
            self._set_state(CameraState.STOPPED)
        return not abandoned

    def __enter__(self) -> CameraActor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- the actor loop ----------------------------------------------------------------

    def _wait_or_stop(self, delay: float) -> None:
        """Wait out a reconnect delay, or wake the moment the actor is asked to stop.

        The default because ``time.sleep`` is not interruptible and this delay grows to the
        ``reconnect_max_ms`` cap — 30 s in the shipped settings. A camera that had just
        failed to connect therefore ignored :meth:`request_stop` for up to half a minute, so
        :meth:`stop` timed out and abandoned a thread that was still holding a decoder, and
        ``IngestManager.remove_camera`` returned while the removed camera was still alive.
        Waiting on the event costs the same and answers immediately.
        """
        self._stop.wait(delay)

    def _run(self) -> None:
        _LOG.info(
            "camera %s: ingest actor started (%s)",
            self.camera_id,
            redact(self.config.uri),
            extra=log_context(camera_id=self.camera_id),
        )
        try:
            while not self._stop.is_set():
                try:
                    if self._source is None and not self._connect():
                        continue
                    if not self._pump():
                        break
                except Exception as exc:  # a camera must outlive one bad frame
                    # The safety net. A bug in a decoder or a sink must degrade one camera,
                    # not kill its thread and leave the fleet quietly 49 cameras wide. The
                    # backoff means a *persistent* bug does not become a hot loop either.
                    self._record_failure(exc)
                    _LOG.exception(
                        "camera %s: unexpected ingest failure; backing off",
                        self.camera_id,
                        extra=log_context(camera_id=self.camera_id),
                    )
                    self._teardown()
                    self._sleep(self._backoff.next_delay())
        finally:
            self._teardown()
            if not self._state_is_final():
                self._set_state(CameraState.STOPPED)
            _LOG.info(
                "camera %s: ingest actor stopped after %d frame(s), %d dropped",
                self.camera_id,
                self._frames_read,
                self._frames_dropped,
                extra=log_context(camera_id=self.camera_id),
            )

    def _connect(self) -> bool:
        """One connection attempt. Returns False after sleeping out the backoff.

        A missing decode runtime is fatal rather than retried: no amount of waiting installs
        PyGObject, and hammering a reconnect loop against an ImportError buries the one log
        line that says what to do about it.
        """
        self._mark_connecting()
        try:
            source = self._factory(self.config, self._counter)
            source.open()
        except SourceUnavailableError as exc:
            self._record_failure(exc)
            self._fatal = True
            self._set_state(CameraState.UNHEALTHY)
            _LOG.error(
                "camera %s: %s — giving up; retrying cannot fix this",
                self.camera_id,
                redact_in(str(exc)),
                extra=log_context(camera_id=self.camera_id),
            )
            self._stop.set()
            return False
        except Exception as exc:  # a decoder can fail in any number of ways
            self._record_failure(exc)
            delay = self._backoff.next_delay()
            _LOG.warning(
                "camera %s: connect attempt %d failed (%s); retrying in %.2fs",
                self.camera_id,
                self._backoff.attempts,
                redact_in(str(exc)),
                delay,
                extra=log_context(camera_id=self.camera_id),
            )
            self._sleep(delay)
            return False

        self._source = source
        with self._lock:
            self._connects += 1
            self._consecutive_empty = 0
        self.metrics.reconnects_total.inc(camera=self.camera_id)
        _LOG.info(
            "camera %s: connected (%dx%d @ %g fps)",
            self.camera_id,
            source.width,
            source.height,
            source.fps,
            extra=log_context(camera_id=self.camera_id),
        )
        return True

    def _pump(self) -> bool:
        """Read one frame and publish it. Returns False when the actor should exit."""
        source = self._source
        assert source is not None  # guarded by the caller
        try:
            frame = source.read()
        except Exception as exc:  # decode failures are the expected case here
            self._record_failure(exc)
            delay = self._backoff.next_delay()
            _LOG.warning(
                "camera %s: read failed (%s); reconnecting in %.2fs",
                self.camera_id,
                redact_in(str(exc)),
                delay,
                extra=log_context(camera_id=self.camera_id),
            )
            self._teardown()
            self._sleep(delay)
            return True

        if frame is None:
            return self._on_empty_read(source)

        # A *frame* is what proves the camera works, so this is where the failure state is
        # cleared — not on a successful connect.
        self._backoff.reset()
        with self._lock:
            self._consecutive_empty = 0
            self._last_error = ""
        self._publish(frame)
        return True

    def _on_empty_read(self, source: FrameSource) -> bool:
        """No frame this time. Decide between waiting, reconnecting and finishing."""
        self.metrics.empty_reads_total.inc(camera=self.camera_id)
        with self._lock:
            self._empty_reads += 1
            self._consecutive_empty += 1
            consecutive = self._consecutive_empty

        if source.is_exhausted:
            # A finite replay source finishing is not a fault, and reconnecting to it would
            # loop forever. This is what lets a bench or a test terminate on its own.
            _LOG.info(
                "camera %s: source exhausted after %d frame(s); actor finishing",
                self.camera_id,
                self._frames_read,
                extra=log_context(camera_id=self.camera_id),
            )
            self._set_state(CameraState.EXHAUSTED)
            return False

        if consecutive >= self.settings.empty_reads_before_reconnect:
            self._record_failure(TimeoutError(f"{consecutive} consecutive empty reads"))
            delay = self._backoff.next_delay()
            _LOG.warning(
                "camera %s: %d consecutive empty read(s); reconnecting in %.2fs",
                self.camera_id,
                consecutive,
                delay,
                extra=log_context(camera_id=self.camera_id),
            )
            self._teardown()
            self._sleep(delay)
            return True

        self._set_state(CameraState.DEGRADED)
        # A source that returns None immediately (a fake, a stream between frames) would
        # otherwise spin a core at 100%.
        if self.settings.empty_read_sleep_ms:
            self._sleep(self.settings.empty_read_sleep_ms / 1000.0)
        return True

    # -- publishing --------------------------------------------------------------------

    def _publish(self, frame: Frame) -> None:
        """Hand one frame to the sink, or count the drop.

        Backpressure is honest here on purpose, and this is the only place in the system that
        can be. An RTSP camera cannot be told to slow down, so *something* must drop a frame
        when the consumer falls behind — and this is the one component that knows which
        camera a frame came from, so it is the only one that can count the drop against the
        camera that caused it. The alternative is what the previous system did: accept
        everything and silently evict somebody else's work three stages later (ADR-005).
        """
        self._record_frame(frame)
        try:
            self._sink.put(frame)
        except QueueFullError as exc:
            self._record_drop("sink_full")
            _LOG.debug(
                "camera %s: frame %d dropped, %s",
                self.camera_id,
                frame.frame_id,
                redact_in(str(exc)),
                extra=log_context(camera_id=self.camera_id, frame_id=frame.frame_id),
            )
            return
        except RequestCancelledError:
            # The consumer is gone: the server is shutting down, so finish cleanly rather
            # than logging one line per frame for as long as the process lives.
            self._record_drop("sink_closed")
            _LOG.info(
                "camera %s: sink closed; actor finishing",
                self.camera_id,
                extra=log_context(camera_id=self.camera_id),
            )
            self._stop.set()
            return
        with self._lock:
            self._frames_published += 1
        self.metrics.frames_published.inc(camera=self.camera_id)

    def _default_factory(self, config: CameraConfig, counter: FrameCounter) -> FrameSource:
        return create_source(config, counter, settings=self.settings)

    # -- bookkeeping -------------------------------------------------------------------

    def _record_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        with self._lock:
            self._frames_read += 1
            if self._last_frame_ns:
                self.metrics.frame_interval_us.observe(
                    (frame.captured_ns - self._last_frame_ns) / 1_000.0, camera=self.camera_id
                )
            self._last_frame_ns = frame.captured_ns
            self._last_frame_unix_ns = frame.captured_unix_ns
            self._state = CameraState.STREAMING
            if self._fps_window_start == 0.0:
                self._fps_window_start = now
                self._fps_window_frames = 1
            else:
                self._fps_window_frames += 1
                elapsed = now - self._fps_window_start
                if elapsed >= _FPS_WINDOW_S:
                    self._fps = self._fps_window_frames / elapsed
                    self._fps_window_start = now
                    self._fps_window_frames = 0
        self.metrics.frames_total.inc(camera=self.camera_id)

    def _record_drop(self, reason: str) -> None:
        with self._lock:
            self._frames_dropped += 1
        self.metrics.frames_dropped.inc(camera=self.camera_id, reason=reason)

    def _record_failure(self, error: BaseException) -> None:
        """Count a failure and set the state the operator will see.

        Called *before* the backoff advances, so ``attempts + 1`` is the number of
        consecutive failures including this one.
        """
        with self._lock:
            self._connect_failures += 1
            self._last_error = f"{type(error).__name__}: {error}"
            self._fps = 0.0
            failures = self._backoff.attempts + 1
            self._state = (
                CameraState.UNHEALTHY
                if failures >= self.settings.failures_before_unhealthy
                else CameraState.DEGRADED
            )
        self.metrics.connect_failures_total.inc(camera=self.camera_id)

    def _mark_connecting(self) -> None:
        """Enter CONNECTING unless the camera is already known to be unhealthy.

        Without the guard, a camera retrying every 30 s flaps between UNHEALTHY and
        CONNECTING, and a dashboard that samples it sees whichever it happened to catch.
        Health is sticky until a frame clears it.
        """
        with self._lock:
            if self._state is not CameraState.UNHEALTHY:
                self._state = CameraState.CONNECTING

    def _set_state(self, state: CameraState) -> None:
        with self._lock:
            self._state = state
            if state in (CameraState.STOPPED, CameraState.EXHAUSTED):
                self._fps = 0.0

    def _state_is_final(self) -> bool:
        """Whether the current state must survive shutdown.

        ``STOPPED``/``EXHAUSTED`` are already terminal. ``_fatal`` matters too: a camera that
        gave up because its decode runtime is missing should report UNHEALTHY afterwards, not
        STOPPED — otherwise it is indistinguishable from one an operator removed on purpose,
        which is the difference between "install python3-gi" and "no action needed".
        """
        with self._lock:
            return self._fatal or self._state in (CameraState.STOPPED, CameraState.EXHAUSTED)

    def _teardown(self) -> None:
        source, self._source = self._source, None
        if source is None:
            return
        try:
            source.close()
        except Exception as exc:  # closing a broken stream can raise
            _LOG.debug(
                "camera %s: error closing source: %s",
                self.camera_id,
                redact_in(str(exc)),
                extra=log_context(camera_id=self.camera_id),
            )

    # -- observability -----------------------------------------------------------------

    @property
    def health(self) -> CameraHealth:
        """An immutable snapshot, safe to read from any thread."""
        with self._lock:
            fps = self._fps
            if (
                fps == 0.0
                and self._state is CameraState.STREAMING
                and self._fps_window_start > 0.0
            ):
                elapsed = time.monotonic() - self._fps_window_start
                if elapsed >= _FPS_MIN_WINDOW_S:
                    fps = self._fps_window_frames / elapsed
            health = CameraHealth(
                camera_id=self.camera_id,
                state=self._state,
                frames_read=self._frames_read,
                frames_published=self._frames_published,
                frames_dropped=self._frames_dropped,
                empty_reads=self._empty_reads,
                connects=self._connects,
                connect_failures=self._connect_failures,
                consecutive_failures=self._backoff.attempts,
                fps=fps,
                last_frame_unix_ns=self._last_frame_unix_ns,
                last_error=self._last_error,
            )
        self.metrics.camera_fps.set(health.fps, camera=self.camera_id)
        return health

    def __repr__(self) -> str:
        return (
            f"<CameraActor {self.camera_id} {self._state.value} "
            f"read={self._frames_read} dropped={self._frames_dropped}>"
        )
