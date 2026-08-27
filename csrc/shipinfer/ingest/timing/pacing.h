// Frame pacing that absorbs lateness rather than repaying it — `ingest/timing/pacing.py`.
//
// `sleep(1/fps)` between frames is the obvious implementation and it is wrong: the sleep is
// only part of the loop, so the real period is `1/fps + decode + publish` and a "20 fps" replay
// source delivers 17. The error is systematic, it compounds over a long run, and it silently
// makes every throughput measurement taken with it optimistic about the server and pessimistic
// about the load. Accumulating an absolute deadline fixes it.
//
// THE OTHER HALF: LATENESS IS ABSORBED, NOT CAUGHT UP
// ---------------------------------------------------
// If a camera falls behind it does *not* burst to make up the deficit. A burst turns a
// transient stall into a spike that the fleet's queue then has to absorb, and at 50 cameras
// those spikes align. So a late frame is simply late, the deadline restarts from now, and the
// shortfall is *counted* — which is what makes `frames_read` an honest measure of offered load
// rather than a restatement of the configured target.
#pragma once

#include <cstdint>
#include <functional>

namespace shipinfer {

    // How a pacer waits. Returns **true when the wait was interrupted** — the actor was asked
    // to stop — so the caller produces no frame. Injected rather than hard-coded to
    // `sleep_for` for the reason `StopSignal` exists at all: an uninterruptible sleep inside a
    // decode loop is a shutdown that times out.
    using WaitFn = std::function<bool(double)>;

    class DeadlinePacer {
      public:
        // `fps <= 0` disables pacing entirely and `wait` becomes a no-op, which is what a live
        // RTSP source wants — the camera sets the rate.
        explicit DeadlinePacer(double fps = 0.0);

        bool enabled() const { return interval_s_ > 0.0; }
        double interval_s() const { return interval_s_; }

        // How many times the loop was already late when `wait` was called. Non-zero on a replay
        // run means the *consumer* could not keep up, which is exactly what a stress test wants
        // to know.
        uint64_t behind() const { return behind_; }

        // Start the schedule from now. Called when a source opens or reopens.
        void reset();

        // Block until the next frame is due. Returns true when `wait_fn` reported an
        // interruption, in which case the deadline is **left where it was** so the next call
        // asks for the same instant rather than skipping a frame's worth of schedule.
        bool wait(const WaitFn& wait_fn);

      private:
        double interval_s_ = 0.0;
        double deadline_s_ = 0.0;
        uint64_t behind_ = 0;
    };

}  // namespace shipinfer
