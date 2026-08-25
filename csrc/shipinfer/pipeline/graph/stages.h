// The three stages of the perception DAG over `Model::infer` —
// `graph/{detect,crop,objects}.py`.
//
// Each stage submits one request per frame (or per chunk) to a Model and blocks on the future.
// Waiting is not the throughput problem it looks like: each stage's *model* batches across
// every frame in flight, so while frame A is embedding, frame B is detecting; the concurrency
// comes from the worker pool, not from this call. That is the Python plane's shape, and it
// replaces the pool-lease graph the first C++ binary had.
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/stage.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/server/model.h"

namespace shipinfer {

    // The detector's output row: x1, y1, x2, y2, score, class — the yolo26 end-to-end layout.
    constexpr size_t kDetectionStride = 6;

    // Per-worker device scratch, grown once and reused: a worker runs one frame at a time and
    // waits on every stage's future, so a slot is free again by the time the next frame needs
    // it. Kernels are launched on the worker's own stream and the stream is synchronised before
    // a payload is handed to a model — the model copies on *its* stream, and the two are only
    // ordered by that synchronise.
    class WorkerScratch {
      public:
        explicit WorkerScratch(Device device);
        ~WorkerScratch();
        WorkerScratch(const WorkerScratch&) = delete;
        WorkerScratch& operator=(const WorkerScratch&) = delete;
        Device device() const { return device_; }
        gpuStream_t stream() const { return stream_; }
        // A device slot of at least `bytes`, by name. Grown, never shrunk.
        float* slot(const std::string& name, size_t bytes);
        // The boxes upload path: page-locked host staging and a device buffer, both grow-only.
        const float* upload_boxes(const std::vector<float>& boxes);
        void synchronise();

      private:
        Device device_;
        gpuStream_t stream_ = nullptr;
        std::map<std::string, DeviceBuffer> slots_;
        PinnedBuffer boxes_host_;
        DeviceBuffer boxes_device_;
    };

    // Shared by the two model-driven stages: build the request, submit, wait with a budget.
    class ModelStage : public Stage {
      public:
        ModelStage(std::string name, Model& model, std::chrono::milliseconds timeout,
                   std::vector<std::string> consumes, std::vector<std::string> requires,
                   std::vector<std::string> produces);
        Model& model() const { return model_; }

      protected:
        InferenceResponse infer(const FrameState& state, const float* data, size_t rows,
                                size_t row_elems, Device device);

      private:
        Model& model_;
        std::chrono::milliseconds timeout_;
    };

    struct DetectConfig {
        int size = 640;
        float score_threshold = 0.25f;
        int max_objects = 64;
        float pad_value = 114.f / 255.f;  // TorchImageOps: fill 114, normalise mean 0 / std 255
    };

    // Letterbox one frame, run the detector, decode the boxes into frame pixels.
    class DetectStage : public ModelStage {
      public:
        DetectStage(std::string name, Model& detector, DetectConfig config,
                    WorkerScratch& scratch,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

      protected:
        size_t do_run(FrameState& state) override;

      private:
        DetectConfig config_;
        WorkerScratch& scratch_;
    };

    struct CropSpec {
        std::string name;        // the payload's name, e.g. "person_crops"
        std::string class_name;  // "person" / "ship", for the event builder
        int class_id = 0;
        int height = 256;
        int width = 128;
    };

    // Cut every detection out of the frame, once per configured crop set. A crop set with no
    // members is produced with zero rows rather than omitted — that is what makes conditional
    // execution work without a second mechanism.
    class CropStage : public Stage {
      public:
        CropStage(std::string name, std::vector<CropSpec> crops, int max_objects,
                  WorkerScratch& scratch);

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::vector<CropSpec> crops_;
        int max_objects_;
        WorkerScratch& scratch_;
    };

    // A model applied to every object of one frame at once, in chunks of the engine's batch.
    // Runs only when its source payload is non-empty (`requires`), so a frame with only people
    // never reaches the ship segmenter.
    class ObjectStage : public ModelStage {
      public:
        ObjectStage(std::string name, Model& model, std::string source, std::string output,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::string source_;
        std::string output_;
    };

}  // namespace shipinfer
