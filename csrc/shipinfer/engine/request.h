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

    // `monotonic_ns()` is `core/types.h`'s: the clock that stamps a frame and the clock a
    // deadline is measured against have to be the same one, and there is exactly one
    // definition of it so they cannot drift apart.

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

    // One of the engine's outputs: `rows` x `row_elems` floats on the host, named as the
    // artefact names it. `InferenceResponse.outputs` on the Python plane is a `{name: Tensor}`
    // map and this is the same seam -- no output is privileged, because a YOLO-seg engine's
    // two are read together and neither is "the answer".
    struct OutputTensor {
        std::string name;
        std::vector<float> data;  // `rows` x `row_elems`, row-major
        size_t row_elems = 0;
        // ONE ROW's shape, without the batch dimension -- the artefact's own, as `TensorSpec`
        // declares it. `row_elems` is its product, and both are carried because a flattened
        // width cannot be un-flattened: a segmentation engine's `(300, 38)` rows and
        // `(32, 160, 160)` prototypes are two shapes the fold needs and one product hides.
        std::vector<int64_t> dims;
        const float* row(size_t index) const { return data.data() + index * row_elems; }
    };

    struct InferenceResponse {
        std::string model_name;
        FrameTag tag;
        // In the engine's declaration order. Never empty for a completed response: an engine
        // with no output is refused at load.
        std::vector<OutputTensor> outputs;
        size_t rows = 0;
        Device executed_on;
        Timings timings;

        // The first output. Every model this plane began with has exactly one, and a
        // single-output consumer means this by "the answer".
        const OutputTensor& first() const { return outputs.at(0); }
        size_t row_elems() const { return first().row_elems; }
        const float* row(size_t index) const { return first().row(index); }
        // By name, for a consumer that needs a SPECIFIC output -- the segmentation fold needs
        // the prototype bank, not "the second one", because which position it occupies is the
        // export's choice and not the chain's. `nullptr` where the engine has no such output,
        // which the caller turns into a refusal naming it.
        const OutputTensor* named(const std::string& name) const {
            for (const OutputTensor& output : outputs) {
                if (output.name == name) return &output;
            }
            return nullptr;
        }
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

    // `RequestCancelledError` is in `core/types.h`: the camera actor treats a closed sink as a
    // reason to finish, and `ingest/` may not include `server/`.
    //
    // `core.errors.RequestTimeoutError`: a stage waited its whole budget for a model.
    struct RequestTimeoutError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

}  // namespace shipinfer
