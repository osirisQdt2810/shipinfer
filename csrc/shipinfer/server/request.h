// The request a model is asked, the response it gives, and the item that carries both through a
// queue — `core/request/*` and `scheduling/work.py`, reduced to what the C++ plane moves today:
// one float tensor in, one float tensor out, N rows each.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/device.h"
#include "shipinfer/core/types.h"
#include "shipinfer/scheduling/queues/base.h"

namespace shipinfer {

    inline int64_t monotonic_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    struct InferenceRequest {
        std::string model_name;
        // The context: the frame this belongs to. Travels **unchanged** to the response
        // (ADR-002), which is what makes every reorder between submit and answer safe.
        FrameTag tag;
        int priority = Priority::Normal;
        // Absolute monotonic deadline in ns; 0 disables it.
        int64_t deadline_ns = 0;
        // The locality hint for the placement policy.
        std::optional<Device> resident_device;
        // The payload: `rows` rows of `row_elems` floats at `data`, on `payload_device`. The
        // caller owns the memory until the response arrives.
        const float* data = nullptr;
        size_t rows = 0;
        size_t row_elems = 0;
        Device payload_device = Device::cpu();
        // Whatever `data` points into, kept alive for as long as this request exists. The
        // Python plane gets this for free — a request holds a reference to its tensor —
        // and without it a stage that times out leaves a queued request pointing at a
        // buffer the next frame overwrites while the instance DMAs out of it.
        std::shared_ptr<const void> keepalive;
        int64_t received_ns = monotonic_ns();

        bool is_expired(int64_t now_ns) const {
            return deadline_ns != 0 && now_ns > deadline_ns;
        }
    };

    struct Timings {
        int64_t received_ns = 0;
        int64_t queued_ns = 0;
        int64_t batched_ns = 0;
        int64_t compute_start_ns = 0;
        int64_t compute_end_ns = 0;
        int64_t completed_ns = 0;
        double queue_us() const { return (batched_ns - queued_ns) / 1000.0; }
        double total_us() const { return (completed_ns - received_ns) / 1000.0; }
    };

    struct InferenceResponse {
        std::string model_name;
        FrameTag tag;
        std::vector<float> data;  // `rows` x `row_elems`, row-major, on the host
        size_t rows = 0;
        size_t row_elems = 0;
        Device executed_on;
        Timings timings;
        const float* row(size_t index) const { return data.data() + index * row_elems; }
    };

    // The queue item: the request plus the promise its caller is waiting on. Satisfies the
    // queue contract (camera, rows, priority, expired). Move-only, because a promise is.
    class WorkItem {
      public:
        explicit WorkItem(InferenceRequest request)
            : request_(std::move(request)), enqueued_ns_(monotonic_ns()) {}
        WorkItem(WorkItem&&) = default;
        WorkItem& operator=(WorkItem&&) = default;

        InferenceRequest& request() { return request_; }
        const InferenceRequest& request() const { return request_; }
        std::future<InferenceResponse> future() { return promise_.get_future(); }
        int64_t enqueued_ns() const { return enqueued_ns_; }

        // -- the queue contract --------------------------------------------------------------
        // By reference: this is read on every put and every lane push (CONVENTIONS 2.5, no
        // per-request allocation on the dispatch path). Non-video callers share one lane.
        const std::string& camera() const {
            static const std::string shared_lane = "-";
            return request_.tag.camera_id.empty() ? shared_lane : request_.tag.camera_id;
        }
        size_t rows() const { return request_.rows == 0 ? 1 : request_.rows; }
        int priority() const { return request_.priority; }
        bool expired(int64_t now_ns) const { return request_.is_expired(now_ns); }

        // Resolve, once. A second call is the "caller gave up while we computed" case in the
        // Python instance and is ignored the same way.
        void complete(InferenceResponse response) {
            if (settled_) return;
            settled_ = true;
            promise_.set_value(std::move(response));
        }
        void fail(std::exception_ptr error) {
            if (settled_) return;
            settled_ = true;
            promise_.set_exception(std::move(error));
        }
        template <typename E>
        void fail_with(E error) {
            fail(std::make_exception_ptr(std::move(error)));
        }
        bool settled() const { return settled_; }

      private:
        InferenceRequest request_;
        std::promise<InferenceResponse> promise_;
        int64_t enqueued_ns_;
        bool settled_ = false;
    };

    // `core.errors.RequestCancelledError`: the request left the system without an answer.
    struct RequestCancelledError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    // `core.errors.RequestTimeoutError`: a stage waited its whole budget for a model.
    struct RequestTimeoutError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

}  // namespace shipinfer
