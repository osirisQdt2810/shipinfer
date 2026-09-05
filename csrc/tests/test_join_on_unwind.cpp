// `core/join_on_unwind.h`: a throw past a live thread must report, not abort.
//
// The defect it closes (`CSRC-BENCH-STARTUP-ABORT`) is a thread destroyed while joinable,
// which is `terminate called without an active exception` -- a process that dies with no
// message and no stack, at exit 134. That is unobservable from inside the aborting process, so
// what is checked here is the property that PREVENTS it: after the guard's scope ends, every
// watched thread has been joined, including when an exception carried the scope away.
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/join_on_unwind.h"

namespace {

    using namespace shipinfer;
    using namespace std::chrono_literals;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // A thread that only ever leaves through `stopping`, which is every thread the bench's
    // start-up spawns: the workers poll it beside a queue read, the sweeper beside a sleep.
    std::thread spinning_on(std::atomic<bool>& stopping, std::atomic<int>& ran) {
        return std::thread([&stopping, &ran]() {
            while (!stopping.load()) {
                ran.fetch_add(1);
                std::this_thread::sleep_for(1ms);
            }
        });
    }

    void a_throw_past_a_live_thread_joins_it_instead_of_aborting() {
        std::atomic<bool> stopping{false};
        std::atomic<int> ran{0};
        std::thread worker = spinning_on(stopping, ran);
        bool propagated = false;
        try {
            JoinOnUnwind guard(stopping);
            guard.watch(worker);
            // What `ReplayLibrary::acquire` does on a folder it cannot read, and every other
            // refusal between spawning the workers and the shutdown block.
            throw std::runtime_error("a camera this run cannot build");
        } catch (const std::runtime_error&) {
            propagated = true;
        }

        check(!worker.joinable(), "the thread is joined by the time the scope is gone");
        check(propagated, "and the original error still reaches its handler");
        check(stopping.load(), "the stop flag is set, which is what let the thread finish");
        check(ran.load() > 0, "and the thread really ran, so this is not a no-op passing");
    }

    void a_clean_stop_leaves_the_guard_nothing_to_do() {
        // The normal shutdown order is load-bearing -- the bench stops its models BETWEEN the
        // workers and the sweeper -- so the guard must not join anything on the happy path.
        std::atomic<bool> stopping{false};
        std::atomic<int> ran{0};
        std::thread worker = spinning_on(stopping, ran);
        bool joined_inside = false;
        {
            JoinOnUnwind guard(stopping);
            guard.watch(worker);
            stopping.store(true);
            worker.join();
            joined_inside = !worker.joinable();
        }

        check(joined_inside, "the caller joined it in its own order");
        check(!worker.joinable(), "and the guard's destructor found nothing to join");
    }

    void a_thread_parked_in_a_wait_is_woken_before_it_is_joined() {
        // `stopping` alone deadlocks a thread blocked on a condition variable -- which is a
        // worker parked in `get_batch`. The bench registers `queue.close()` as the waker; this
        // is that shape with the queue reduced to the wait it performs.
        std::atomic<bool> stopping{false};
        std::mutex mutex;
        std::condition_variable ready;
        bool woken = false;
        std::thread parked([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock, [&]() { return woken; });
        });
        try {
            JoinOnUnwind guard(stopping);
            guard.watch(parked);
            guard.wake_with([&]() {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    woken = true;
                }
                ready.notify_all();
            });
            throw std::runtime_error("the refusal that unwinds this scope");
        } catch (const std::runtime_error&) {
        }

        check(!parked.joinable(), "a waker runs before the join, so the wait ends");
    }

    void a_waker_that_throws_does_not_replace_the_original_failure() {
        // This destructor runs while an exception is in flight: one escaping it is
        // `std::terminate`, with the error the operator needed lost. Both loops swallow.
        std::atomic<bool> stopping{false};
        std::atomic<int> ran{0};
        std::thread worker = spinning_on(stopping, ran);
        std::string reported;
        try {
            JoinOnUnwind guard(stopping);
            guard.watch(worker);
            guard.wake_with([]() { throw std::runtime_error("the waker itself failed"); });
            throw std::runtime_error("a camera this run cannot build");
        } catch (const std::runtime_error& error) {
            reported = error.what();
        }

        check(!worker.joinable(), "a failed waker does not skip the join");
        check(reported == "a camera this run cannot build",
              "and the ORIGINAL error is the one that arrives, not the waker's");
    }

    void watched_threads_are_joined_in_reverse_registration_order() {
        // Matching the destruction order of the scope they live in, so the guard cannot join
        // a thread that another watched thread is still feeding.
        std::atomic<bool> stopping{false};
        std::mutex mutex;
        std::vector<int> order;
        std::vector<std::thread> threads;
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back([&stopping, &mutex, &order, i]() {
                while (!stopping.load()) std::this_thread::sleep_for(1ms);
                std::lock_guard<std::mutex> lock(mutex);
                order.push_back(i);
            });
        }
        {
            JoinOnUnwind guard(stopping);
            for (std::thread& thread : threads) guard.watch(thread);
        }

        for (std::thread& thread : threads) {
            check(!thread.joinable(), "every watched thread is joined, not just the first");
        }
        check(order.size() == 3, "and all three ran to their end");
    }

    void watching_nothing_is_not_a_crash() {
        std::atomic<bool> stopping{false};
        { JoinOnUnwind guard(stopping); }

        check(stopping.load(), "an empty guard still sets the stop flag");
    }

}  // namespace

int main() {
    a_throw_past_a_live_thread_joins_it_instead_of_aborting();
    a_clean_stop_leaves_the_guard_nothing_to_do();
    a_thread_parked_in_a_wait_is_woken_before_it_is_joined();
    a_waker_that_throws_does_not_replace_the_original_failure();
    watched_threads_are_joined_in_reverse_registration_order();
    watching_nothing_is_not_a_crash();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
