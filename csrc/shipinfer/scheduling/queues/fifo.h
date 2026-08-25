// Plain FIFO — `scheduling/queues/fifo.py`. Kept so fairness can be A/B'd against it: the
// fair queue is a *choice*, and a choice needs a control.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/scheduling/queues/base.h"

namespace shipinfer {

    template <typename T>
    class FifoQueue {
      public:
        FifoQueue(std::string name, size_t capacity, Overflow overflow = Overflow::Reject,
                  int block_timeout_ms = 50, bool drop_expired = true, DropHandler<T> on_drop = {})
            : name_(std::move(name)),
              capacity_(capacity),
              overflow_(overflow),
              block_timeout_ms_(block_timeout_ms),
              drop_expired_(drop_expired),
              on_drop_(std::move(on_drop)) {
            if (capacity_ < 1) throw std::invalid_argument("queue capacity must be >= 1");
        }

        const std::string& name() const { return name_; }
        size_t depth() const { return size_.load(std::memory_order_relaxed); }
        bool is_closed() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return closed_;
        }
        QueueStats stats() const {
            std::lock_guard<std::mutex> lock(mutex_);
            QueueStats copy = stats_;
            copy.depth = size_.load(std::memory_order_relaxed);
            copy.capacity = capacity_;
            return copy;
        }

        PutStatus put(T item) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (closed_) return PutStatus::Closed;
            const std::string camera = item.camera();
            if (items_.size() >= capacity_ && !make_room_locked(lock)) {
                ++stats_.rejected;
                ++stats_.rejected_by_camera[camera];
                return PutStatus::Rejected;
            }
            if (closed_) return PutStatus::Closed;
            items_.push_back(std::move(item));
            size_.store(items_.size(), std::memory_order_relaxed);
            ++stats_.accepted;
            if (items_.size() > stats_.peak) stats_.peak = items_.size();
            work_.notify_one();
            return PutStatus::Accepted;
        }

        std::vector<T> get_batch(const BatchWindow& window, int poll_ms = 50) {
            std::unique_lock<std::mutex> lock(mutex_);
            while (items_.empty()) {
                if (closed_) return {};
                work_.wait_for(lock, std::chrono::milliseconds(poll_ms));
            }
            if (window.max_delay_us > 0 && items_.size() < window.max_batch_size) {
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::microseconds(window.max_delay_us);
                while (items_.size() < window.max_batch_size && !closed_) {
                    if (window.preferred(items_.size())) break;
                    if (std::chrono::steady_clock::now() >= deadline) break;
                    work_.wait_until(lock, deadline);
                }
            }
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            std::vector<T> batch;
            size_t rows = 0;
            // Rows, not items — a per-object request carries one row per crop.
            while (!items_.empty()) {
                const size_t head_rows = items_.front().rows() == 0 ? 1 : items_.front().rows();
                if (!batch.empty() && rows + head_rows > window.max_batch_size) break;
                T item = std::move(items_.front());
                items_.pop_front();
                if (drop_expired_ && item.expired(now)) {
                    ++stats_.expired;
                    if (on_drop_) on_drop_(std::move(item), DropReason::Expired);
                    continue;
                }
                batch.push_back(std::move(item));
                rows += head_rows;
                if (rows >= window.max_batch_size) break;
            }
            size_.store(items_.size(), std::memory_order_relaxed);
            if (overflow_ == Overflow::Block) space_.notify_all();
            return batch;
        }

        std::vector<T> close() {
            std::vector<T> drained;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                closed_ = true;
                for (T& item : items_) drained.push_back(std::move(item));
                items_.clear();
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
        bool make_room_locked(std::unique_lock<std::mutex>& lock) {
            if (overflow_ == Overflow::Reject) return false;
            if (overflow_ == Overflow::Block) {
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(block_timeout_ms_);
                while (items_.size() >= capacity_ && !closed_) {
                    if (space_.wait_until(lock, deadline) == std::cv_status::timeout &&
                        items_.size() >= capacity_) {
                        return false;
                    }
                }
                return !closed_;
            }
            // DROP_OLDEST, and here it *is* the globally oldest: this is the control the fair
            // queue is compared against, and its whole point is that it does not know cameras.
            T victim = std::move(items_.front());
            items_.pop_front();
            size_.store(items_.size(), std::memory_order_relaxed);
            ++stats_.evicted;
            ++stats_.evicted_by_camera[victim.camera()];
            if (on_drop_) on_drop_(std::move(victim), DropReason::Evicted);
            return true;
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
        std::deque<T> items_;
        std::atomic<size_t> size_{0};
        bool closed_ = false;
        QueueStats stats_;
    };

}  // namespace shipinfer
