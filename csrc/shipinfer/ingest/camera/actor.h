// One camera, one thread, for the thread's whole life — `ingest/camera/actor.py`.
//
// This is the ingest half of ADR-002. A camera actor is **stateful** — it owns the connection,
// the decoder, the `frame_id` counter and the reconnect schedule — and it does **no**
// inference. Everything it produces goes into a sink and is picked up by the stateless,
// GPU-pooled half of the system, which is what lets fifty uneven cameras share sixteen GPUs
// instead of being pinned three-to-a-device.
//
// What it deliberately does **not** own is a queue. The reference system gave every camera a
// share of one 1000-slot buffer that evicted the *oldest* entry when full, so a crowded camera
// starved a quiet one and nothing logged it. This actor publishes into a `FrameSink` — in
// production one backed by the fair, bounded queue in `scheduling/queues/`, which has
// per-camera lanes and sheds the loudest camera rather than its victim (ADR-005). Writing
// another queue *here* would reintroduce exactly the bug the project exists to fix; depending
// on the scheduler directly would put dispatch policy in the decode path, which is the other
// half of the same mistake.
//
// One policy decision is worth calling out because it is not the obvious one: **a successful
// connection does not reset the failure count — a successful frame does.** A source that
// accepts a connection and then delivers nothing is the most common real failure mode of a
// camera fleet, and treating "opened" as "healthy" is precisely how it stays invisible.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "shipinfer/core/stop_signal.h"
#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/camera/health.h"
#include "shipinfer/ingest/config.h"
#include "shipinfer/ingest/frame.h"
#include "shipinfer/ingest/registry.h"
#include "shipinfer/ingest/sink.h"
#include "shipinfer/ingest/timing/backoff.h"

namespace shipinfer {

    class CameraActor {
      public:
        // How the actor waits out a delay. Returns **true when the wait was cut short by a stop
        // request**. Injected so the offline tier can assert the *sequence* of reconnect delays
        // rather than merely that a retry happened; the default waits on the stop signal rather
        // than on the clock, for the reason `StopSignal` exists.
        using WaitFn = std::function<bool(double)>;

        // `factory` empty means `create_source`; `wait` empty means wait on the stop signal.
        // `sink` and anything the factory captures must outlive the actor.
        CameraActor(IngestConfig config, FrameSink& sink, SourceFactory factory = {},
                    WaitFn wait = {});
        ~CameraActor();

        CameraActor(const CameraActor&) = delete;
        CameraActor& operator=(const CameraActor&) = delete;

        const std::string& camera_id() const { return config_.camera_id; }
        const IngestConfig& config() const { return config_; }
        bool is_running() const;
        CameraState state() const;
        // An immutable snapshot, safe to read from any thread.
        CameraHealth health() const;

        // Start the actor thread. Not restartable once stopped.
        //
        // Throws ServerStateError if already started: a restarted actor would need a second
        // opinion on where its frame counter stands, and the manager builds a fresh actor
        // instead, which has one.
        void start();

        // Ask the actor to finish, without waiting for it. Separate from `stop` because
        // shutting down fifty cameras should cost one read timeout, not fifty: signal them all,
        // then join them all.
        void request_stop();

        // Ask the actor to finish, and wait for it. Idempotent, and a no-op on an actor that
        // was never started, because shutdown paths call this from more than one place and
        // neither may hang. Safe to call from the actor's own thread (it signals and returns
        // rather than joining itself), but **not** safe against a concurrent `stop` from
        // another thread — the owner serialises it, which for a fleet is `IngestManager`.
        //
        // Returns **false when the thread had to be abandoned**. A thread still alive after
        // `timeout` is one blocked inside a decoder, and holding up the whole process's
        // shutdown behind it would be the worse failure — so it is detached (never left
        // joinable: `~thread` on a joinable thread calls `std::terminate`) and reported. The
        // caller then owes the detached thread a `this` that stays valid; `IngestManager` is
        // what pays that debt.
        bool stop(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

      private:
        bool wait_or_stop(double seconds);
        void run();
        bool connect();
        bool pump();
        bool on_empty_read(const FrameSource& source);
        void publish(Frame&& frame);
        void record_frame(const Frame& frame);
        void record_drop();
        void record_failure(const std::string& reason);
        void mark_connecting();
        void set_state(CameraState state);
        bool state_is_final() const;
        void teardown();

        IngestConfig config_;
        FrameSink& sink_;
        SourceFactory factory_;
        WaitFn wait_;
        StopSignal stop_;
        // Owned by the actor and outliving every source it builds: a counter dying with the
        // source would hand a downstream tracker a second frame 0 after every reconnect.
        FrameCounter counter_;
        ExponentialBackoff backoff_;
        std::unique_ptr<FrameSource> source_;
        std::thread thread_;
        // Serialises the thread's lifecycle — `start()`'s assignment, `stop()`'s
        // joinable/join/detach — against a concurrent `stop()`. The fleet manager itself
        // enters `stop()` from two threads (its own `stop()` and `add_camera`'s re-check,
        // #35 rounds 2–3): without this, both pass the unsynchronised `joinable()` read and
        // one joins while the other detaches, or both detach — `std::terminate` on the
        // shutdown path. Distinct from `mutex_` (the state lock): a stopper holds this
        // across its whole grace wait, and health reads must not queue behind that.
        std::mutex lifecycle_mutex_;
        // The self-stop guard's own copy of the id, atomic because the guard cannot take
        // `lifecycle_mutex_` (a stopper holds it across its grace wait FOR this thread —
        // taking it here would deadlock the shutdown), yet reading `thread_.get_id()` bare
        // would race `start()`'s assignment.
        std::atomic<std::thread::id> thread_id_{};

        // Everything below is written by the actor thread and read by anyone; the lock is taken
        // once per frame at most, which at 20 fps per camera is free.
        mutable std::mutex mutex_;
        std::condition_variable finished_;
        bool started_ = false;
        bool is_finished_ = false;
        // Set when retrying cannot possibly help. It is what makes the difference between
        // "install the decode runtime" and "no action needed" visible after shutdown.
        bool fatal_ = false;
        CameraState state_ = CameraState::Idle;
        uint64_t frames_read_ = 0;
        uint64_t frames_published_ = 0;
        uint64_t frames_dropped_ = 0;
        uint64_t empty_reads_ = 0;
        uint64_t connects_ = 0;
        uint64_t connect_failures_ = 0;
        uint64_t consecutive_empty_ = 0;
        // A guarded mirror of the backoff's attempt count. The backoff itself is touched only
        // by the actor thread, and reading it from `health()` under this lock — as the Python
        // original does — would be a data race in this language rather than merely a stale
        // read.
        uint64_t consecutive_failures_ = 0;
        std::string last_error_;
        int64_t last_frame_unix_ns_ = 0;
        double fps_ = 0.0;
        double fps_window_start_ = 0.0;
        uint64_t fps_window_frames_ = 0;
    };

}  // namespace shipinfer
