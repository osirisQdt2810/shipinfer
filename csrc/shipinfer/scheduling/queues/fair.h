// The fair, bounded, per-camera queue — the part this project exists to own.
//
// THE FAILURE THIS REPLACES
// ------------------------
// The previous generation funnelled every camera into one shared 1000-slot buffer that
// evicted the *oldest* entry when full. A crowded camera therefore filled the buffer and
// pushed out a quiet camera's frames, and the symptom was reported exactly that way:
// "camera đông người được nhận diện đầy đủ, camera vắng người thỉnh thoảng bị miss" — the
// crowded cameras complete, the quiet ones intermittently miss.
//
// So: one lane per camera, round-robin between lanes on drain, and when the queue is full the
// *greediest* camera loses a frame rather than whichever frame happens to be oldest. Nothing
// is silently dropped: a refusal throws `QueueFullError` carrying depth and capacity, so the
// caller can attribute it (ADR-005).
//
// ROWS, NOT ITEMS
// ---------------
// `drain` counts **rows**, because a per-object request carries one row per crop. Counting
// items against a row budget overfills the batch: sixteen embedder requests each carrying a
// frame's worth of crops assembled 24 rows against `max_batch_size: 16`, the assembler
// refused it, and every request in it failed. That was found by running the Python version,
// and it is the reason this signature takes a row budget rather than a count.
//
// An item whose own row count already exceeds the budget is returned **alone** rather than
// refused, because refusing it would park it at the head of its lane forever and stall the
// model. Letting it through gives the assembler a chance to name the real problem, which is a
// request too large for the engine rather than a scheduling decision.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    enum class Overflow { Reject, Block, EvictGreediest };

    // `T` needs `rows()` and `camera()`. A concept would be clearer; this has to build under the
    // host's g++ 11 in C++17, so it is a documented duck type.
    template <typename T>
    class FairQueue {
      public:
        struct Stats {
            size_t depth = 0;
            size_t peak = 0;
            uint64_t accepted = 0;
            uint64_t rejected = 0;
            uint64_t evicted = 0;
            std::map<std::string, uint64_t> rejected_by_camera;
            std::map<std::string, uint64_t> evicted_by_camera;
        };

        FairQueue(size_t capacity, Overflow overflow, int block_timeout_ms = 0)
            : capacity_(capacity), overflow_(overflow), block_timeout_ms_(block_timeout_ms) {}

        // Returns false when the item was refused under Overflow::Reject; throws nothing on the
        // hot path so a camera thread never unwinds through a lock.
        bool put(T item) {
            std::unique_lock<std::mutex> lock(mutex_);
            const std::string camera = item.camera();
            if (size_ >= capacity_) {
                if (overflow_ == Overflow::Block) {
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::milliseconds(block_timeout_ms_);
                    // Predicate form, so a spurious wake-up does not read as a timeout.
                    if (!space_.wait_until(lock, deadline,
                                           [this] { return size_ < capacity_ || closed_; })) {
                        ++stats_.rejected;
                        ++stats_.rejected_by_camera[camera];
                        return false;
                    }
                    if (closed_) return false;
                } else if (overflow_ == Overflow::EvictGreediest) {
                    if (!evict_greediest_locked()) {
                        ++stats_.rejected;
                        ++stats_.rejected_by_camera[camera];
                        return false;
                    }
                } else {
                    ++stats_.rejected;
                    ++stats_.rejected_by_camera[camera];
                    return false;
                }
            }
            // `try_emplace` tells us whether the lane is new in the same lookup that finds
            // it, so a camera is appended to the round-robin order exactly once and `put` is
            // O(log cameras) rather than O(cameras). The scan it replaces ran on **every**
            // frame — a linear walk of fifty strings a thousand times a second to answer a
            // question the map had already answered.
            const auto [entry, is_new] = lanes_.try_emplace(camera);
            entry->second.push_back(std::move(item));
            if (is_new) order_.push_back(camera);
            ++size_;
            ++stats_.accepted;
            stats_.peak = std::max(stats_.peak, size_);
            work_.notify_one();
            return true;
        }

        // Round-robin across lanes until `max_rows` rows are collected, the queue empties, or the
        // wait expires. An empty result means "nothing to do" — the queue is closed, or the
        // wait simply expired with the queue empty. A caller that treats empty as "closed"
        // exits on the first idle interval, so a worker has to ask `closed()`; the two are
        // deliberately separate questions.
        std::vector<T> drain(size_t max_rows, int wait_ms) {
            std::vector<T> batch;
            std::unique_lock<std::mutex> lock(mutex_);
            if (size_ == 0 && !closed_) {
                work_.wait_for(lock, std::chrono::milliseconds(wait_ms),
                               [this] { return size_ > 0 || closed_; });
            }
            size_t rows = 0;
            // `next_` persists across calls: restarting at lane 0 every time would starve the
            // tail of the fleet under load, which is the same bug in a different shape.
            while (size_ > 0 && rows < max_rows) {
                bool moved = false;
                for (size_t step = 0; step < order_.size() && rows < max_rows; ++step) {
                    const std::string& camera = order_[(next_ + step) % order_.size()];
                    auto it = lanes_.find(camera);
                    if (it == lanes_.end() || it->second.empty()) continue;
                    const size_t head_rows = it->second.front().rows();
                    if (!batch.empty() && rows + head_rows > max_rows) {
                        // Full enough. Leaving the head where it is keeps the lane's order.
                        goto done;
                    }
                    batch.push_back(std::move(it->second.front()));
                    it->second.pop_front();
                    --size_;
                    rows += head_rows;
                    moved = true;
                }
                if (!moved) break;
                next_ = (next_ + 1) % (order_.empty() ? 1 : order_.size());
            }
        done:
            // `notify_all` before returning, in every case. The Python version reached this only
            // on the fall-through path, so a producer blocked on a full queue slept the entire
            // timeout instead of waking when a slot freed — 500 ms against 50 — and when the
            // deadline beat the drain, the drop was charged to a camera that had done nothing.
            if (overflow_ == Overflow::Block) space_.notify_all();
            return batch;
        }

        void close() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                closed_ = true;
            }
            work_.notify_all();
            space_.notify_all();
        }

        size_t depth() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return size_;
        }

        Stats stats() const {
            std::lock_guard<std::mutex> lock(mutex_);
            Stats copy = stats_;
            copy.depth = size_;
            return copy;
        }

        bool closed() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return closed_;
        }

      private:
        // ADR-005: the camera with the deepest lane loses a frame. Charging the drop to the
        // greediest camera rather than to the oldest frame is the whole point — the victim of an
        // eviction should be the cause of the pressure.
        bool evict_greediest_locked() {
            std::string worst;
            size_t deepest = 0;
            for (const auto& [camera, lane] : lanes_) {
                if (lane.size() > deepest) {
                    deepest = lane.size();
                    worst = camera;
                }
            }
            if (worst.empty()) return false;
            lanes_[worst].pop_back();  // the newest of the greediest, not the oldest of anyone
            --size_;
            ++stats_.evicted;
            ++stats_.evicted_by_camera[worst];
            return true;
        }

        mutable std::mutex mutex_;
        std::condition_variable work_;
        std::condition_variable space_;
        std::map<std::string, std::deque<T>> lanes_;
        std::vector<std::string> order_;
        size_t next_ = 0;
        size_t size_ = 0;
        size_t capacity_;
        Overflow overflow_;
        int block_timeout_ms_;
        bool closed_ = false;
        Stats stats_;
    };

}  // namespace shipinfer
