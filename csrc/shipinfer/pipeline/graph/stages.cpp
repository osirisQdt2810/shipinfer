#include "shipinfer/pipeline/graph/stages.h"

#include <algorithm>
#include <cstring>
#include <future>

#include "shipinfer/core/buffers.h"
#include "shipinfer/runtime/ops.h"

namespace shipinfer {

    // -- WorkerScratch
    // -------------------------------------------------------------------------

    WorkerScratch::WorkerScratch(Device device) : device_(device) {
        GPU_CHECK(gpuStreamCreate(&stream_));
    }

    WorkerScratch::~WorkerScratch() {
        if (stream_ != nullptr) gpuStreamDestroy(stream_);
    }

    std::shared_ptr<DeviceBuffer> WorkerScratch::acquire(const std::string& name,
                                                         size_t bytes) {
        auto& pool = pools_[name];
        for (auto& buffer : pool) {
            if (buffer.use_count() != 1) continue;  // a request still points into it
            if (buffer->bytes() < bytes) buffer = std::make_shared<DeviceBuffer>(bytes);
            return buffer;
        }
        if (pool.size() >= kMaxHeldPerName) {
            throw ServerStateError("worker scratch '" + name +
                                   "': " + std::to_string(pool.size()) +
                                   " payloads are still held by requests that have not "
                                   "completed (timed out or still queued); refusing to "
                                   "allocate more rather than grow without bound");
        }
        pool.push_back(std::make_shared<DeviceBuffer>(bytes));
        return pool.back();
    }

    size_t WorkerScratch::held(const std::string& name) const {
        const auto it = pools_.find(name);
        if (it == pools_.end()) return 0;
        size_t held = 0;
        for (const auto& buffer : it->second) held += buffer.use_count() != 1 ? 1 : 0;
        return held;
    }

    const float* WorkerScratch::upload_boxes(const std::vector<float>& boxes) {
        const size_t bytes = boxes.size() * sizeof(float);
        if (boxes_host_.bytes() < bytes) boxes_host_ = PinnedBuffer(bytes);
        if (boxes_device_.bytes() < bytes) boxes_device_ = DeviceBuffer(bytes);
        std::memcpy(boxes_host_.get(), boxes.data(), bytes);
        GPU_CHECK(gpuMemcpyAsync(boxes_device_.get(), boxes_host_.get(), bytes,
                                 gpuMemcpyHostToDevice, stream_));
        return boxes_device_.as<float>();
    }

    void WorkerScratch::synchronise() {
        GPU_CHECK(gpuStreamSynchronize(stream_));
    }

    // -- ModelStage
    // ----------------------------------------------------------------------------

    ModelStage::ModelStage(std::string name, Model& model, std::chrono::milliseconds timeout,
                           std::vector<std::string> consumes, std::vector<std::string> needs,
                           std::vector<std::string> produces)
        : Stage(std::move(name), std::move(consumes), std::move(needs), std::move(produces)),
          model_(model),
          timeout_(timeout) {
        if (timeout_.count() <= 0)
            throw ConfigError("stage " + this->name() + ": timeout must be > 0");
    }

    InferenceResponse ModelStage::infer(const FrameState& state, const float* data, size_t rows,
                                        size_t row_elems, Device device,
                                        std::shared_ptr<const void> keepalive) {
        // The request carries the frame's tag **unchanged** (ADR-002): batching, spillover to
        // another GPU and out-of-order completion are all fine because reassembly keys on the
        // tag rather than on arrival order.
        InferenceRequest request;
        request.model_name = model_.name();
        request.tag = state.tag();
        request.priority = state.priority();
        request.deadline_ns = state.deadline_ns();
        request.resident_device = device;
        request.data = data;
        request.rows = rows;
        request.row_elems = row_elems;
        request.payload_device = device;
        request.keepalive = std::move(keepalive);
        std::future<InferenceResponse> future = model_.infer(std::move(request));
        if (future.wait_for(timeout_) != std::future_status::ready) {
            // Bounded so a wedged instance costs one frame and one worker for this long rather
            // than forever; the stage fails and the event names it as missing.
            throw RequestTimeoutError(
                "stage " + name() + ": model " + model_.name() + " did not answer within " +
                std::to_string(timeout_.count()) + " ms for " + state.tag().key());
        }
        return future.get();
    }

    // -- DetectStage
    // ---------------------------------------------------------------------------

    DetectStage::DetectStage(std::string name, Model& detector, DetectConfig config,
                             WorkerScratch& scratch, std::chrono::milliseconds timeout)
        : ModelStage(std::move(name), detector, timeout, {FRAME_INPUT}, {FRAME_INPUT},
                     {DETECTIONS}),
          config_(config),
          scratch_(scratch) {}

    size_t DetectStage::do_run(FrameState& state) {
        const size_t row_elems = static_cast<size_t>(3) * config_.size * config_.size;
        std::shared_ptr<DeviceBuffer> owner =
            scratch_.acquire("letterbox", row_elems * sizeof(float));
        float* input = owner->as<float>();
        const LetterboxMap map = letterbox_into(
            state.image()->as<uint8_t>(), state.height(), state.width(), input, config_.size,
            config_.size, /*swap_rb=*/true, config_.pad_value, scratch_.stream());
        scratch_.synchronise();
        // Stored on the state, not recomputed downstream: the decode must undo exactly the
        // transform that was applied.
        state.set_letterbox(map.scale, map.pad_x, map.pad_y);

        const InferenceResponse response =
            infer(state, input, 1, row_elems, scratch_.device(), owner);
        if (response.row_elems % kDetectionStride != 0) {
            throw BackendError("stage " + name() + ": the detector's row of " +
                               std::to_string(response.row_elems) + " floats is not " +
                               std::to_string(kDetectionStride) + " per candidate");
        }
        // Model space back to original pixels. Cropping in letterboxed coordinates is where the
        // off-by-a-pad-bar bugs live.
        std::vector<Detection> detections;
        const size_t candidates = response.row_elems / kDetectionStride;
        for (size_t d = 0;
             d < candidates && detections.size() < static_cast<size_t>(config_.max_objects);
             ++d) {
            const float* row = response.row(0) + d * kDetectionStride;
            if (row[4] < config_.score_threshold) continue;
            Detection det;
            det.x1 = (row[0] - static_cast<float>(map.pad_x)) / map.scale;
            det.y1 = (row[1] - static_cast<float>(map.pad_y)) / map.scale;
            det.x2 = (row[2] - static_cast<float>(map.pad_x)) / map.scale;
            det.y2 = (row[3] - static_cast<float>(map.pad_y)) / map.scale;
            det.score = row[4];
            det.class_id = static_cast<int>(row[5]);
            detections.push_back(det);
        }
        const size_t count = detections.size();
        state.set_detections(std::move(detections));
        state.set_detected(true);
        return count;
    }

    // -- CropStage
    // -----------------------------------------------------------------------------

    CropStage::CropStage(std::string name, std::vector<CropSpec> crops, int max_objects,
                         WorkerScratch& scratch)
        : Stage(std::move(name), {DETECTIONS, FRAME_INPUT}, {DETECTIONS},
                [&crops] {
                    std::vector<std::string> names;
                    for (const CropSpec& spec : crops) names.push_back(spec.name);
                    return names;
                }()),
          crops_(std::move(crops)),
          max_objects_(max_objects),
          scratch_(scratch) {
        if (crops_.empty())
            throw ConfigError("stage " + this->name() + " must declare at least one crop set");
    }

    size_t CropStage::do_run(FrameState& state) {
        size_t total = 0;
        for (const CropSpec& spec : crops_) {
            std::vector<float> boxes;
            std::vector<int> indices;
            for (const Detection& det : state.detections()) {
                if (det.class_id != spec.class_id) continue;
                boxes.insert(boxes.end(), {det.x1, det.y1, det.x2, det.y2});
                indices.push_back(det.index);
            }
            DevicePayload payload;
            payload.name = spec.name;
            payload.class_name = spec.class_name;
            payload.device = scratch_.device();
            payload.row_elems = static_cast<size_t>(3) * spec.height * spec.width;
            if (indices.empty()) {
                // Zero rows rather than a missing entry: the name exists, so the graph is
                // valid, and it is empty, so no stage requiring it is planned.
                state.attach_payload(std::move(payload));
                continue;
            }
            const int count = static_cast<int>(indices.size());
            // Sized for what was detected, rounded up in steps of eight so the pool reuses a
            // buffer across frames of similar crowds — not for `max_objects_` every time:
            // at 64 objects a 640x640 crop set is 64 * 3 * 640^2 * 4 B = 315 MB per buffer
            // per worker, and the pool may hold a second one under a timeout.
            const int rows = std::min(max_objects_, ((count + 7) / 8) * 8);
            std::shared_ptr<DeviceBuffer> owner =
                scratch_.acquire(spec.name, static_cast<size_t>(std::max(count, rows)) *
                                                payload.row_elems * sizeof(float));
            float* dst = owner->as<float>();
            const float* boxes_device = scratch_.upload_boxes(boxes);
            crop_resize_into(state.image()->as<uint8_t>(), state.height(), state.width(),
                             boxes_device, count, dst, spec.height, spec.width,
                             /*swap_rb=*/true, scratch_.stream());
            scratch_.synchronise();
            payload.data = dst;
            payload.owner = std::move(owner);
            payload.rows = static_cast<size_t>(count);
            payload.object_indices = std::move(indices);
            payload.boxes = std::move(boxes);
            state.attach_payload(std::move(payload));
            total += static_cast<size_t>(count);
        }
        return total;
    }

    // -- ObjectStage
    // ---------------------------------------------------------------------------

    ObjectStage::ObjectStage(std::string name, Model& model, std::string source,
                             std::string output, std::chrono::milliseconds timeout)
        : ModelStage(std::move(name), model, timeout, {source}, {source}, {output}),
          source_(std::move(source)),
          output_(std::move(output)) {}

    size_t ObjectStage::do_run(FrameState& state) {
        const DevicePayload* payload = state.payload(source_);
        if (payload == nullptr)
            throw ConfigError("stage " + name() + ": no payload named " + source_);
        // Chunked to the engine's own batch — the plans are static, and submitting a whole
        // frame's crops as one request is what lost every crop in a 25-person frame against a
        // plan built at 16. One ObjectBatch, grown per chunk, attached once.
        const size_t limit = static_cast<size_t>(std::max(1, model().max_batch()));
        ObjectBatch out;
        out.name = output_;
        for (size_t start = 0; start < payload->rows; start += limit) {
            const size_t count = std::min(limit, payload->rows - start);
            const InferenceResponse response =
                infer(state, payload->data + start * payload->row_elems, count,
                      payload->row_elems, payload->device, payload->owner);
            if (response.rows != count) {
                throw BackendError("stage " + name() + ": model " + model().name() +
                                   " returned " + std::to_string(response.rows) +
                                   " row(s) for " + std::to_string(count) + " object(s)");
            }
            out.append(response.data.data(), static_cast<int>(count),
                       static_cast<int>(response.row_elems), payload->object_indices, start);
        }
        state.attach(std::move(out));
        return payload->rows;
    }

}  // namespace shipinfer
