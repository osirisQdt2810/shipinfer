#include "shipinfer/ingest/camera/actor.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <utility>

#include "shipinfer/core/redact.h"
#include "shipinfer/core/types.h"

namespace shipinfer {
    namespace {

        double monotonic_s() {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

        // Rate-limited stderr. There is no logging framework in this plane, and a camera
        // retrying a dead switch every 30 s for a night would otherwise be the only thing in
        // the log — the failure the reference system's "Can not read frame" line already was.
        void shout(const std::string& line) {
            static std::atomic<int> said{0};
            if (said.fetch_add(1) < 200) std::cerr << line << "\n";
        }

    }  // namespace

    namespace {

        // Validation runs *before* the backoff is built from these numbers: the backoff's own
        // constructor also checks them, but it cannot name the camera, and "backoff factor
        // must be > 1" from a fifty-camera fleet is a search where "camera 'cam7':
        // reconnect_factor must be > 1" is an answer. The manager validates earlier still —
        // this covers direct construction, which is public API.
        const IngestConfig& validated(const IngestConfig& config) {
            config.validate();
            return config;
        }

    }  // namespace

    CameraActor::CameraActor(IngestConfig config, FrameSink& sink, SourceFactory factory,
                             WaitFn wait)
        : config_(std::move(config)),
          sink_(sink),
          factory_(std::move(factory)),
          wait_(std::move(wait)),
          counter_(validated(config_).camera_id, config_.first_frame_id),
          backoff_(config_.reconnect_initial_ms / 1000.0, config_.reconnect_max_ms / 1000.0,
                   config_.reconnect_factor, config_.reconnect_jitter) {
        if (!factory_) {
            factory_ = [](const IngestConfig& config, FrameCounter& counter, StopSignal& stop) {
                return create_source(config, counter, stop);
            };
        }
        if (!wait_) {
            wait_ = [this](double seconds) { return wait_or_stop(seconds); };
        }
    }

    CameraActor::~CameraActor() {
        stop();
        // `stop()` either joined the thread or detached it. A joinable `std::thread` destroyed
        // here would call `std::terminate` and take the process with it, so this is the
        // invariant rather than a nicety.
        assert(!thread_.joinable());
    }

    // -- identity ---------------------------------------------------------------------------

    bool CameraActor::is_running() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return started_ && !is_finished_;
    }

    CameraState CameraActor::state() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    // -- lifecycle --------------------------------------------------------------------------

    void CameraActor::start() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (started_) {
                throw ServerStateError("camera '" + config_.camera_id +
                                       "' has already been started; build a new actor rather "
                                       "than restarting this one");
            }
            started_ = true;
            state_ = CameraState::Connecting;
        }
        stop_.clear();
        {
            // Under the lifecycle lock: a concurrent `stop()` reads `thread_.joinable()`,
            // and an unsynchronised read against this write is the race, not just the
            // join/detach below it.
            std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
            thread_ = std::thread([this] { run(); });
        }
    }

    void CameraActor::request_stop() {
        stop_.set();
    }

    bool CameraActor::stop(std::chrono::milliseconds timeout) {
        request_stop();
        bool abandoned = false;
        // A stop from the actor's own thread can only signal: joining would deadlock on the
        // thread doing the joining. The Python original guards the same way on
        // `threading.current_thread()` — the id is read from its atomic copy because this
        // guard cannot take the lifecycle lock (a stopper holds it across its grace wait FOR
        // this very thread), yet a bare `thread_.get_id()` would race `start()`'s write.
        //
        // The whole joinable/join/detach section sits under the lifecycle lock: the manager
        // itself can enter here from two threads at once (its `stop()` and `add_camera`'s
        // re-check, #35), and without the lock one caller joins while the other detaches —
        // or both detach — which is `std::terminate`, in CI's own hammer test (flip-proven:
        // removing this lock aborts this binary 3/3). The second caller blocks for at most
        // the first one's grace, then finds the thread already joined or detached and
        // returns; only the caller that performed the detach reports the abandonment, so an
        // actor is never counted (or parked) twice.
        if (thread_id_.load() != std::this_thread::get_id()) {
            std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
            if (thread_.joinable()) {
                bool finished = false;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    finished =
                        finished_.wait_for(lock, timeout, [this] { return is_finished_; });
                }
                if (finished) {
                    thread_.join();
                } else {
                    thread_.detach();
                    thread_abandoned_ = true;
                    shout("camera " + config_.camera_id + " did not stop within " +
                          std::to_string(timeout.count()) + "ms; abandoning the thread");
                }
            }
            // Read under the same lock: the thread's fate, whichever stopper sealed it. A
            // loser that answered "clean" for a thread its rival detached would zero the
            // fleet count that keeps the sink alive (#39 round 1).
            abandoned = thread_abandoned_;
        }
        if (!state_is_final()) set_state(CameraState::Stopped);
        return !abandoned;
    }

    // -- the actor loop ---------------------------------------------------------------------

    bool CameraActor::wait_or_stop(double seconds) {
        return stop_.wait_for(seconds);
    }

    void CameraActor::run() {
        // The child publishes its own id: the parent's store after the spawn left a window
        // where a self-stop from the first frames (a sink calling stop() from publish) read
        // the default id, missed the self-stop guard, and waited its grace for itself
        // (#39 round 1). An external stopper reading the default id simply proceeds, which
        // is the correct outcome.
        thread_id_.store(std::this_thread::get_id());
        try {
            while (!stop_.is_set()) {
                try {
                    if (!source_ && !connect()) continue;
                    if (!pump()) break;
                } catch (const std::exception& error) {
                    // The safety net. A bug in a decoder or a sink must degrade one camera, not
                    // kill its thread and leave the fleet quietly 49 cameras wide. The backoff
                    // means a *persistent* bug does not become a hot loop either.
                    record_failure(error.what());
                    shout(
                        "camera " + config_.camera_id +
                        ": unexpected ingest failure; backing off: " + redact_in(error.what()));
                    teardown();
                    wait_(backoff_.next_delay());
                }
            }
        } catch (...) {  // NOLINT — nothing may escape a thread; that is std::terminate
            shout("camera " + config_.camera_id + ": actor loop ended on an unknown exception");
        }
        teardown();
        if (!state_is_final()) set_state(CameraState::Stopped);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_finished_ = true;
        }
        finished_.notify_all();
    }

    bool CameraActor::connect() {
        mark_connecting();
        std::unique_ptr<FrameSource> source;
        try {
            source = factory_(config_, counter_, stop_);
            if (!source) {
                throw SourceError("camera '" + config_.camera_id +
                                  "': the source factory returned nothing");
            }
            source->open();
        } catch (const SourceUnavailableError& error) {
            // A missing decode runtime is fatal rather than retried: no amount of waiting
            // installs it, and hammering a reconnect loop against it buries the one log line
            // that says what to do about it.
            record_failure(error.what());
            {
                std::lock_guard<std::mutex> lock(mutex_);
                fatal_ = true;
            }
            set_state(CameraState::Unhealthy);
            shout("camera " + config_.camera_id + ": " + redact_in(error.what()) +
                  " — giving up; retrying cannot fix this");
            stop_.set();
            return false;
        } catch (const std::exception& error) {
            record_failure(error.what());
            const double delay = backoff_.next_delay();
            shout("camera " + config_.camera_id + ": connect attempt " +
                  std::to_string(backoff_.attempts()) + " failed (" + redact_in(error.what()) +
                  "); retrying in " + std::to_string(delay) + "s");
            wait_(delay);
            return false;
        }

        source_ = std::move(source);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++connects_;
            consecutive_empty_ = 0;
        }
        // The URI is redacted here and nowhere else is it printed: this line runs on every
        // reconnect, so an unredacted one publishes the fleet's password once a minute forever.
        shout("camera " + config_.camera_id + ": connected to " + redact_uri(config_.uri) +
              " (" + std::to_string(source_->width()) + "x" +
              std::to_string(source_->height()) + " @ " + std::to_string(source_->fps()) +
              " fps)");
        return true;
    }

    bool CameraActor::pump() {
        std::optional<Frame> frame;
        try {
            frame = source_->read();
        } catch (const std::exception& error) {
            record_failure(error.what());
            const double delay = backoff_.next_delay();
            shout("camera " + config_.camera_id + ": read failed (" + redact_in(error.what()) +
                  "); reconnecting in " + std::to_string(delay) + "s");
            teardown();
            wait_(delay);
            return true;
        }

        if (!frame) return on_empty_read(*source_);

        // A *frame* is what proves the camera works, so this is where the failure state is
        // cleared — not on a successful connect.
        backoff_.reset();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            consecutive_empty_ = 0;
            consecutive_failures_ = 0;
            last_error_.clear();
        }
        publish(std::move(*frame));
        return true;
    }

    bool CameraActor::on_empty_read(const FrameSource& source) {
        uint64_t consecutive = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++empty_reads_;
            ++consecutive_empty_;
            consecutive = consecutive_empty_;
        }

        if (source.is_exhausted()) {
            // A finite replay source finishing is not a fault, and reconnecting to it would
            // loop forever. This is what lets a bench or a test terminate on its own.
            shout("camera " + config_.camera_id + ": source exhausted; actor finishing");
            set_state(CameraState::Exhausted);
            return false;
        }

        if (consecutive >= static_cast<uint64_t>(config_.empty_reads_before_reconnect)) {
            record_failure(std::to_string(consecutive) + " consecutive empty reads");
            const double delay = backoff_.next_delay();
            shout("camera " + config_.camera_id + ": " + std::to_string(consecutive) +
                  " consecutive empty read(s); reconnecting in " + std::to_string(delay) + "s");
            teardown();
            wait_(delay);
            return true;
        }

        set_state(CameraState::Degraded);
        // A source that returns nothing immediately (a fake, a stream between frames) would
        // otherwise spin a core at 100%.
        if (config_.empty_read_sleep_ms > 0) wait_(config_.empty_read_sleep_ms / 1000.0);
        return true;
    }

    // -- publishing -------------------------------------------------------------------------

    void CameraActor::publish(Frame&& frame) {
        record_frame(frame);
        // Backpressure is honest here on purpose, and this is the only place in the system that
        // can be. A camera cannot be told to slow down, so *something* must drop a frame when
        // the consumer falls behind — and this is the one component that knows which camera a
        // frame came from, so it is the only one that can charge the drop to the camera that
        // caused it. The alternative is what the previous system did: accept everything and
        // silently evict somebody else's work three stages later (ADR-005).
        try {
            sink_.put(std::move(frame));
        } catch (const QueueFullError&) {
            record_drop();
            return;
        } catch (const RequestCancelledError&) {
            // The consumer is gone: the server is shutting down, so finish cleanly rather than
            // logging one line per frame for as long as the process lives.
            record_drop();
            shout("camera " + config_.camera_id + ": sink closed; actor finishing");
            stop_.set();
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        ++frames_published_;
    }

    // -- bookkeeping ------------------------------------------------------------------------

    void CameraActor::record_frame(const Frame& frame) {
        const double now = monotonic_s();
        std::lock_guard<std::mutex> lock(mutex_);
        ++frames_read_;
        last_frame_unix_ns_ = frame.tag.captured_unix_ns;
        state_ = CameraState::Streaming;
        if (fps_window_start_ == 0.0) {
            fps_window_start_ = now;
            fps_window_frames_ = 1;
            return;
        }
        ++fps_window_frames_;
        const double elapsed = now - fps_window_start_;
        if (elapsed >= kFpsWindowS) {
            fps_ = static_cast<double>(fps_window_frames_) / elapsed;
            fps_window_start_ = now;
            fps_window_frames_ = 0;
        }
    }

    void CameraActor::record_drop() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++frames_dropped_;
    }

    void CameraActor::record_failure(const std::string& reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++connect_failures_;
        ++consecutive_failures_;
        last_error_ = redact_in(reason);
        // Zeroed rather than left stale: a camera that has just failed is not still delivering
        // at whatever rate it last managed, and a dashboard reading the old number would say
        // so.
        fps_ = 0.0;
        state_ =
            consecutive_failures_ >= static_cast<uint64_t>(config_.failures_before_unhealthy)
                ? CameraState::Unhealthy
                : CameraState::Degraded;
    }

    void CameraActor::mark_connecting() {
        // Without the guard, a camera retrying every 30 s flaps between UNHEALTHY and
        // CONNECTING, and a dashboard that samples it sees whichever it happened to catch.
        // Health is sticky until a frame clears it.
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != CameraState::Unhealthy) state_ = CameraState::Connecting;
    }

    void CameraActor::set_state(CameraState state) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = state;
        if (state == CameraState::Stopped || state == CameraState::Exhausted) fps_ = 0.0;
    }

    bool CameraActor::state_is_final() const {
        // STOPPED/EXHAUSTED are already terminal. `fatal_` matters too: a camera that gave up
        // because its decode runtime is missing should report UNHEALTHY afterwards, not STOPPED
        // — otherwise it is indistinguishable from one an operator removed on purpose, which is
        // the difference between "install the runtime" and "no action needed".
        std::lock_guard<std::mutex> lock(mutex_);
        return fatal_ || state_ == CameraState::Stopped || state_ == CameraState::Exhausted;
    }

    void CameraActor::teardown() {
        std::unique_ptr<FrameSource> source = std::move(source_);
        source_.reset();
        if (!source) return;
        try {
            source->close();
        } catch (const std::exception& error) {  // closing a broken stream can raise
            shout("camera " + config_.camera_id +
                  ": error closing source: " + redact_in(error.what()));
        }
    }

    // -- observability ----------------------------------------------------------------------

    CameraHealth CameraActor::health() const {
        std::lock_guard<std::mutex> lock(mutex_);
        double fps = fps_;
        if (fps == 0.0 && state_ == CameraState::Streaming && fps_window_start_ > 0.0) {
            // Before the first full window closes there is still an honest answer, and
            // reporting 0 for the first two seconds of every camera makes a start-up look like
            // an outage.
            const double elapsed = monotonic_s() - fps_window_start_;
            if (elapsed >= kFpsMinWindowS) {
                fps = static_cast<double>(fps_window_frames_) / elapsed;
            }
        }
        CameraHealth health;
        health.camera_id = config_.camera_id;
        health.state = state_;
        health.frames_read = frames_read_;
        health.frames_published = frames_published_;
        health.frames_dropped = frames_dropped_;
        health.empty_reads = empty_reads_;
        health.connects = connects_;
        health.connect_failures = connect_failures_;
        health.consecutive_failures = consecutive_failures_;
        health.fps = fps;
        health.last_frame_unix_ns = last_frame_unix_ns_;
        health.last_error = last_error_;
        return health;
    }

}  // namespace shipinfer
