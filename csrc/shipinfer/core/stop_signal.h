// One-shot, waitable "please finish" — Python's `threading.Event`, reduced to what a worker
// thread needs from it.
//
// WHY THIS EXISTS RATHER THAN A SLEEP
// -----------------------------------
// A camera's reconnect delay grows to a 30 s cap, and the obvious implementation of that delay
// is `std::this_thread::sleep_for`. It is not interruptible. A camera that had just failed to
// connect therefore ignored a stop request for up to half a minute, so `stop()` timed out and
// abandoned a thread that was still holding a decoder, and removing a camera returned while the
// removed camera was still alive. Waiting on this costs the same and answers immediately.
//
// Header-only and dependency-free so it can be used from `core`, `ingest` and anything above
// them without adding a translation unit to a link line.
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace shipinfer {

    class StopSignal {
      public:
        // Ask everyone waiting to finish. Sticky: once set it stays set until `clear()`, so a
        // thread that was not yet waiting still sees it. Idempotent.
        void set() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                set_ = true;
            }
            signalled_.notify_all();
        }

        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            set_ = false;
        }

        bool is_set() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return set_;
        }

        // Wait up to `seconds`. Returns **true when the signal is set** — that is, "stop now"
        // — and false when the wait ran out with nothing to report. The polarity is the
        // caller's question, not the clock's: every call site reads `if (wait_for(delay))
        // return;`.
        //
        // A non-positive `seconds` is a poll: it reports the current state and does not block.
        bool wait_for(double seconds) const {
            std::unique_lock<std::mutex> lock(mutex_);
            if (set_) return true;
            if (seconds <= 0.0) return false;
            const auto budget = std::chrono::duration<double>(seconds);
            return signalled_.wait_for(
                lock, std::chrono::duration_cast<std::chrono::steady_clock::duration>(budget),
                [this] { return set_; });
        }

        // Wait until set, however long that takes.
        void wait() const {
            std::unique_lock<std::mutex> lock(mutex_);
            signalled_.wait(lock, [this] { return set_; });
        }

      private:
        // `mutable` because `is_set`/`wait_for`/`wait` are logically const reads that still
        // have to take the lock — a caller holding a `const StopSignal&` (every source does)
        // must be able to ask.
        mutable std::mutex mutex_;
        mutable std::condition_variable signalled_;
        bool set_ = false;
    };

}  // namespace shipinfer
