// The default queue: priority lanes, round-robin fair within a lane.
// `scheduling/queues/fair.py` (`FairPriorityQueue`), ported seam for seam.
//
// This class is the direct answer to the failure documented in the reference system's
// `docs/flow.md`: every camera fed one shared 1000-slot buffer that evicted the *oldest* entry
// when full, so a crowded camera silently starved a quiet one. Two choices fix it, and both
// live here: fair queueing (requests are bucketed by camera and drained round-robin, so a
// camera producing 30 crops per frame cannot occupy 30 consecutive batch slots) and honest
// overflow (a full queue refuses by default; backpressure that reaches the producer is a
// signal, a silent eviction three stages downstream is a bug that takes a week to find).
//
// WHAT THE FIRST C++ QUEUE GOT DIFFERENT, AND WHY THIS ONE DOES NOT. It evicted the *newest*
// frame of the greediest camera where the Python queue evicts its *oldest* — a different
// latency profile under sustained overload, recorded in ADR-014 as the one place the planes had
// already diverged. It had no priority lanes, no batch window (a fixed wait for the first item
// and then whatever was there), no expiry, and an O(cameras) rotation. The parity harness
// (ledger P6) drives both planes with one trace and expects the same batches and the same
// evictions; that is only possible if the arithmetic is the same, so it is.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/scheduling/queues/base.h"
#include "shipinfer/scheduling/queues/lanes.h"

namespace shipinfer {

    template <typename T>
    class FairPriorityQueue {
      public:
        FairPriorityQueue(std::string name, size_t capacity,
                          Overflow overflow = Overflow::Reject, int block_timeout_ms = 50,
                          bool drop_expired = true, DropHandler<T> on_drop = {})
            : name_(std::move(name)),
              capacity_(capacity),
              overflow_(overflow),
              block_timeout_ms_(block_timeout_ms),
              drop_expired_(drop_expired),
              on_drop_(std::move(on_drop)) {
            if (capacity_ < 1) throw std::invalid_argument("queue capacity must be >= 1");
        }

        const std::string& name() const { return name_; }
        size_t capacity() const { return capacity_; }

        // -- introspection ----------------------------------------------------------------
        // Read without the lock by the placement policies: a slightly stale depth changes
        // which of two near-equal GPUs wins, nothing more, and a lock here would be taken
        // thousands of times a second.
        size_t depth() const { return size_.load(std::memory_order_relaxed); }
        bool is_closed() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return closed_;
        }
        // A snapshot, taken under the lock and returned by value — the maps are copies, as
        // the Python plane's are, so a caller that trims what it was given is not editing a
        // live queue's attribution. `depth_by_camera` is computed here rather than
        // maintained: O(cameras x priorities), once per stats call, instead of bookkeeping
        // on the path that runs 15 000 times a second.
        QueueStats stats() const {
            std::lock_guard<std::mutex> lock(mutex_);
            QueueStats copy = stats_;
            copy.depth = size_.load(std::memory_order_relaxed);
            copy.capacity = capacity_;
            for (const Lane<T>& lane : lanes_) lane.add_depths(copy.depth_by_camera);
            return copy;
        }

        // -- producer ---------------------------------------------------------------------
        // Takes the item only on acceptance: a refused or closed put leaves it with the caller,
        // the way the Python queue raises before taking ownership — the dispatcher's spill
        // depends on still holding the item after a refusal.
        PutStatus put(T&& item) {
            std::optional<T> evicted;
            std::unique_lock<std::mutex> lock(mutex_);
            if (closed_) return PutStatus::Closed;
            if (size_.load(std::memory_order_relaxed) >= capacity_ &&
                !make_room_locked(lock, evicted)) {
                ++stats_.rejected;
                ++stats_.rejected_by_camera[item.camera()];  // the one branch that needs it
                return PutStatus::Rejected;
            }
            if (closed_) return PutStatus::Closed;  // BLOCK may have waited through a close
            const int level = clamp_priority(item.priority());
            lanes_[level].push(std::move(item));
            const size_t now = size_.fetch_add(1, std::memory_order_relaxed) + 1;
            ++stats_.accepted;
            if (now > stats_.peak) stats_.peak = now;
            work_.notify_one();
            lock.unlock();
            // The drop handler runs outside the lock, as `close()` already does: a handler
            // that settles a promise is safe either way, but the next one someone writes may
            // touch the queue, and a callback under the queue's own mutex is a deadlock waiting
            // for its second author.
            if (evicted.has_value() && on_drop_)
                on_drop_(std::move(*evicted), DropReason::Evicted);
            return PutStatus::Accepted;
        }

        // -- consumer ---------------------------------------------------------------------
        // Two-phase wait — the classic dynamic-batching shape. 1. Wait (indefinitely, but
        // wake-able) for the *first* request: an idle model must not burn a core spinning.
        // 2. Once one has arrived, wait at most `max_delay_us` more for the batch to fill,
        // returning early the moment it reaches `max_batch_size` or a preferred size.
        // Returns an empty batch only when the queue has been closed, which is how a worker
        // thread learns to exit without a separate sentinel.
        std::vector<T> get_batch(const BatchWindow& window, int poll_ms = 50) {
            std::unique_lock<std::mutex> lock(mutex_);
            while (size_.load(std::memory_order_relaxed) == 0) {
                if (closed_) return {};
                work_.wait_for(lock, std::chrono::milliseconds(poll_ms));
            }
            if (window.max_delay_us > 0 &&
                size_.load(std::memory_order_relaxed) < window.max_batch_size) {
                wait_to_fill_locked(lock, window);
            }
            return drain_locked(window.max_batch_size);
        }

        // -- lifecycle --------------------------------------------------------------------
        // Close. Everything still queued is handed to the drop handler as `Closed` when one
        // is set, and returned to the caller otherwise — either way the caller can fail
        // exactly that much work and report it. A shutdown that silently discards 400
        // requests is not orderly.
        std::vector<T> close() {
            std::vector<T> drained;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                closed_ = true;
                for (Lane<T>& lane : lanes_) {
                    for (T& item : lane.drain()) drained.push_back(std::move(item));
                }
                size_.store(0, std::memory_order_relaxed);
            }
            work_.notify_all();
            space_.notify_all();
            if (on_drop_) {
                for (T& item : drained) on_drop_(std::move(item), DropReason::Closed);
                drained.clear();
            }
            return drained;
        }

      private:
        static int clamp_priority(int level) {
            return level < 0 ? 0 : (level >= kPriorityLevels ? kPriorityLevels - 1 : level);
        }

        // Try to free one slot. True if the caller may now enqueue. An evicted item is handed
        // back in `evicted` for the caller to fail once it has released the lock.
        bool make_room_locked(std::unique_lock<std::mutex>& lock, std::optional<T>& evicted) {
            if (overflow_ == Overflow::Reject) return false;
            if (overflow_ == Overflow::Block) {
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(block_timeout_ms_);
                while (size_.load(std::memory_order_relaxed) >= capacity_ && !closed_) {
                    if (space_.wait_until(lock, deadline) == std::cv_status::timeout &&
                        size_.load(std::memory_order_relaxed) >= capacity_) {
                        return false;
                    }
                }
                return !closed_;
            }
            // DROP_OLDEST: sacrifice from the *lowest*-priority non-empty lane, so a
            // BACKGROUND request can never displace a TRACKING_CRITICAL one.
            for (int level = kPriorityLevels - 1; level >= 0; --level) {
                std::optional<T> victim = lanes_[level].evict_from_longest();
                if (victim.has_value()) {
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    ++stats_.evicted;
                    // The victim's camera is the greediest by construction —
                    // `evict_from_longest` picks the key hogging the lane — so this names
                    // the flood, not the camera whose frame merely happened to be oldest.
                    ++stats_.evicted_by_camera[victim->camera()];
                    evicted = std::move(victim);
                    return true;
                }
            }
            return false;
        }

        void wait_to_fill_locked(std::unique_lock<std::mutex>& lock,
                                 const BatchWindow& window) {
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::microseconds(window.max_delay_us);
            // `size_` counts items and `max_batch_size` counts rows, so this is a lower bound
            // on fullness: with multi-row requests the batch reaches its row budget before the
            // item count does, and waiting past that only adds latency. Deliberately not made
            // exact — summing every queued request's rows on each wake would walk the whole
            // queue thousands of times a second to refine a wait heuristic.
            while (size_.load(std::memory_order_relaxed) < window.max_batch_size && !closed_) {
                if (window.preferred(size_.load(std::memory_order_relaxed))) return;
                if (std::chrono::steady_clock::now() >= deadline) return;
                work_.wait_until(lock, deadline);
            }
        }

        // Pop up to `max_rows` **rows** highest-priority-first, round-robin in a lane. Rows,
        // not items: a per-object request carries one row per crop, and counting items
        // against a row budget overfilled the batch on the first real 50-camera run. An item
        // whose own row count already exceeds the budget is still returned, alone — refusing
        // to dequeue it would park it at the head of its lane forever and stall the model.
        std::vector<T> drain_locked(size_t max_rows) {
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            std::vector<T> batch;
            size_t rows = 0;
            bool budget_hit = false;
            for (Lane<T>& lane : lanes_) {
                while (!lane.empty() && !budget_hit) {
                    const T* head = lane.peek();
                    const size_t head_rows = head->rows() == 0 ? 1 : head->rows();
                    if (!batch.empty() && rows + head_rows > max_rows) {
                        budget_hit = true;
                        break;
                    }
                    T item = lane.pop();
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    if (drop_expired_ && item.expired(now)) {
                        ++stats_.expired;
                        // Read before the move: `on_drop_` takes ownership below.
                        ++stats_.expired_by_camera[item.camera()];
                        if (on_drop_) on_drop_(std::move(item), DropReason::Expired);
                        continue;
                    }
                    batch.push_back(std::move(item));
                    rows += head_rows;
                    if (rows >= max_rows) budget_hit = true;
                }
                if (budget_hit) break;
            }
            // On *every* exit, not only the loop's natural end: the row-budget exit is the
            // common one under load, and a blocked producer must wake the instant a slot frees
            // rather than sleep out its whole timeout — measured at 500 against 50 ms in the
            // Python plane before its `finally` fixed it.
            if (overflow_ == Overflow::Block) space_.notify_all();
            return batch;
        }

        std::string name_;
        size_t capacity_;
        Overflow overflow_;
        int block_timeout_ms_;
        bool drop_expired_;
        DropHandler<T> on_drop_;

        mutable std::mutex mutex_;
        std::condition_variable work_;
        std::condition_variable space_;
        Lane<T> lanes_[kPriorityLevels];
        std::atomic<size_t> size_{0};
        bool closed_ = false;
        QueueStats stats_;
    };

}  // namespace shipinfer
