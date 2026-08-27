// The request-queue contract — `src/shipinfer/scheduling/queues/base.py`, seam for seam.
//
// A queue owns three decisions: what order requests come out in, what happens when it is full,
// and how long a consumer waits for a batch to fill. Making it a contract means those three can
// be varied independently of the instance that drains it — and it means the fair-queueing
// behaviour that fixes this system's inherited starvation bug is a *choice* that can be A/B'd
// against plain FIFO rather than a hardcoded assumption.
//
// THE ITEM CONTRACT. A queue is templated on its item, which must provide:
//     std::string camera() const;       // the fairness key (Python: WorkItem.fairness_key)
//     size_t rows() const;              // rows this item adds to a batch (request.batch_size)
//     int priority() const;             // 0..3, lower served first (core.request.Priority)
//     bool expired(int64_t now_ns) const;  // request.is_expired(now)
// The Python queue *fails* an item it drops (evicted, expired, closed) through its future; here
// the queue hands the item to the `on_drop` callback given at construction, with the reason,
// and the caller does whatever "fail" means for that item. Same event, same moment.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace shipinfer {

    // `core.settings.OverflowPolicy`: what a full queue does. REJECT is the default: dropping a
    // frame at the edge, loudly and countably, beats half-processing it and evicting somebody
    // else's work three stages later — the silent eviction this system was rebuilt to remove.
    enum class Overflow { Reject, Block, DropOldest };

    // `core.request.Priority`: lower value == served first.
    enum Priority : int { TrackingCritical = 0, High = 1, Normal = 2, Background = 3 };
    constexpr int kPriorityLevels = 4;

    // Why an item left the queue without being executed. The Python queue raises the matching
    // error into the item's future; the C++ queue names it here and hands the item back.
    enum class DropReason { Evicted, Expired, Closed };

    inline const char* to_string(DropReason reason) {
        switch (reason) {
            case DropReason::Evicted:
                return "evicted";
            case DropReason::Expired:
                return "expired";
            case DropReason::Closed:
                return "closed";
        }
        return "?";
    }

    // What `put` says. Python raises `QueueFullError` (Rejected) or `RequestCancelledError`
    // (Closed); a C++ producer branches on the value.
    enum class PutStatus { Accepted, Rejected, Closed };

    // The batching contract a consumer asks the queue to honour (`BatchWindow`). The trade this
    // encodes is the whole point of dynamic batching: waiting `max_delay_us` costs every
    // request that latency, but lets the GPU run one large kernel launch instead of many small
    // ones.
    struct BatchWindow {
        size_t max_batch_size = 1;
        int64_t max_delay_us = 0;
        // Sizes worth stopping early at. A TensorRT engine profiled for {8, 16, 32} runs an
        // unprofiled batch of 31 on a fallback path that can be much slower than padding to 32.
        std::vector<size_t> preferred_sizes;

        BatchWindow() = default;
        BatchWindow(size_t max_batch, int64_t delay_us = 0, std::vector<size_t> preferred = {})
            : max_batch_size(max_batch),
              max_delay_us(delay_us),
              preferred_sizes(std::move(preferred)) {
            if (max_batch_size < 1) throw std::invalid_argument("max_batch_size must be >= 1");
            for (size_t s : preferred_sizes) {
                if (s < 1 || s > max_batch_size) {
                    throw std::invalid_argument(
                        "preferred_sizes must be within [1, max_batch_size]");
                }
            }
        }
        bool preferred(size_t size) const {
            for (size_t s : preferred_sizes) {
                if (s == size) return true;
            }
            return false;
        }
    };

    // A snapshot an operator can act on (`QueueStats`), plus the per-camera attribution ADR-005
    // exists to make possible — the Python side reports it from the ingest actors, the C++ side
    // from the queue, because here the queue is where a refusal is first known.
    struct QueueStats {
        size_t depth = 0;
        size_t capacity = 0;
        uint64_t accepted = 0;
        uint64_t rejected = 0;
        uint64_t evicted = 0;
        uint64_t expired = 0;
        size_t peak = 0;
        // Who paid for each outcome. `depth_by_camera` is built by `stats()` from the lanes
        // it already holds the lock over; the other three are counted at the drop sites. A
        // camera missing from a map lost nothing to that outcome — the maps are sparse on
        // purpose, so a 50-camera server does not report 200 zeroes.
        //
        // `close()` deliberately feeds none of them: shutdown loss is not a per-camera fault,
        // and charging it here would make an orderly stop read like a flood in the one view
        // an operator uses to find floods.
        std::map<std::string, uint64_t> depth_by_camera;
        std::map<std::string, uint64_t> rejected_by_camera;
        std::map<std::string, uint64_t> evicted_by_camera;
        std::map<std::string, uint64_t> expired_by_camera;

        double utilisation() const {
            return capacity == 0 ? 0.0
                                 : static_cast<double>(depth) / static_cast<double>(capacity);
        }
    };

    template <typename T>
    using DropHandler = std::function<void(T&&, DropReason)>;

}  // namespace shipinfer
