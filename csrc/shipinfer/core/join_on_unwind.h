// Join watched threads on the way out, so an exception does not turn into an abort.
//
// WHY THIS EXISTS
// ---------------
// `cli/bench.cpp` spawns its workers and its sweeper BEFORE it builds the cameras, and
// building them throws: `ReplayLibrary::acquire` on a folder it cannot read, `create_source`
// on a name this binary does not link, the ingest manager on a camera it will not accept. That
// throw unwinds past a `std::vector<std::thread>` holding joinable threads, and a joinable
// `std::thread` destroyed is `terminate called without an active exception` -- exit 134, no
// message, and indistinguishable from a crash mid-run (`CSRC-BENCH-STARTUP-ABORT`).
//
// A FAILURE PATH ONLY. It WATCHES threads it does not own, so a normal shutdown still stops
// and joins in its own order -- which is load-bearing where models must stop between two sets
// of threads. After a clean stop every watched thread is already joined and this does nothing.
#pragma once

#include <atomic>
#include <exception>
#include <functional>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace shipinfer {

    class JoinOnUnwind {
      public:
        explicit JoinOnUnwind(std::atomic<bool>& stopping) : stopping_(stopping) {}
        JoinOnUnwind(const JoinOnUnwind&) = delete;
        JoinOnUnwind& operator=(const JoinOnUnwind&) = delete;

        // Every step swallows its own failure, because this runs while an exception is in
        // flight: one escaping here is `std::terminate` again with the original error lost --
        // the same outcome by a longer road. A thread that will not join is left joinable, and
        // `~thread` aborting is the honest end for a thread that ignored the stop.
        ~JoinOnUnwind() {
            stopping_.store(true);
            for (const std::function<void()>& wake : wakers_) {
                try {
                    wake();
                } catch (const std::exception& error) {
                    std::cerr << "join_on_unwind: waking failed: " << error.what() << "\n";
                }
            }
            // Reverse registration order, matching the destruction order of the scope the
            // watched threads live in.
            for (auto it = threads_.rbegin(); it != threads_.rend(); ++it) {
                try {
                    if ((*it)->joinable()) (*it)->join();
                } catch (const std::exception& error) {
                    std::cerr << "join_on_unwind: joining failed: " << error.what() << "\n";
                }
            }
        }

        // Watched, not owned: the thread outlives this guard and the normal path joins it.
        void watch(std::thread& thread) { threads_.push_back(&thread); }
        // Setting `stopping` is not enough for a thread parked in a blocking wait; this is how
        // that thread is woken -- closing a queue, signalling a condition variable.
        void wake_with(std::function<void()> waker) { wakers_.push_back(std::move(waker)); }

      private:
        std::atomic<bool>& stopping_;
        std::vector<std::thread*> threads_;
        std::vector<std::function<void()>> wakers_;
    };

}  // namespace shipinfer
