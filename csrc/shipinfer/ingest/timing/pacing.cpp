#include "shipinfer/ingest/timing/pacing.h"

#include <chrono>

namespace shipinfer {
    namespace {

        double monotonic_s() {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

    }  // namespace

    DeadlinePacer::DeadlinePacer(double fps) : interval_s_(fps > 0.0 ? 1.0 / fps : 0.0) {}

    void DeadlinePacer::reset() {
        deadline_s_ = monotonic_s();
    }

    bool DeadlinePacer::wait(const WaitFn& wait_fn) {
        if (!enabled()) return false;
        if (deadline_s_ == 0.0) reset();
        const double due = deadline_s_ + interval_s_;
        const double now = monotonic_s();
        if (due <= now) {
            // Already late. Absorb: the schedule restarts from *now* rather than being left in
            // the past, which is what stops the deficit from being repaid as a burst of
            // back-to-back frames no downstream queue asked for.
            ++behind_;
            deadline_s_ = now;
            return false;
        }
        if (wait_fn && wait_fn(due - now)) return true;
        deadline_s_ = due;
        return false;
    }

}  // namespace shipinfer
