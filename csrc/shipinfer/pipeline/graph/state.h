// One frame in flight, and the records it becomes.
//
// ADR-002: a `FrameState` is owned by exactly one worker thread for the frame's whole life.
// The Python version broke that once — the reassembly sweeper read a state whose worker was
// still inside the graph — and the fix there is the design here: whatever the emitter needs is
// **captured** when the frame is finished, and the emitter never touches the state again.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // A per-object result: N rows, each belonging to one detection of one frame.
    struct ObjectBatch {
        std::string name;
        std::vector<int> object_indices;  // object_indices[row] is the detection that produced it
        std::vector<float> data;          // row-major, `width` floats per row
        int width = 0;

        bool empty() const { return object_indices.empty(); }
        const float* row(size_t index) const {
            return data.data() + index * static_cast<size_t>(width);
        }
    };

    // What the event builder needs, and nothing else. Captured under the collector's lock; built
    // outside it.
    struct EmissionInputs {
        FrameTag tag;
        int width = 0;
        int height = 0;
        float fps = 0.f;
        std::vector<Detection> detections;
        std::map<std::string, ObjectBatch> batches;
    };

    class FrameState {
      public:
        FrameState(FrameTag tag, int height, int width, float fps)
            : tag_(std::move(tag)), height_(height), width_(width), fps_(fps) {}

        const FrameTag& tag() const { return tag_; }
        int height() const { return height_; }
        int width() const { return width_; }
        float fps() const { return fps_; }

        void set_detections(std::vector<Detection> detections) {
            detections_ = std::move(detections);
            for (size_t i = 0; i < detections_.size(); ++i) {
                detections_[i].index = static_cast<int>(i);
            }
        }
        const std::vector<Detection>& detections() const { return detections_; }

        void attach(ObjectBatch batch) { batches_[batch.name] = std::move(batch); }
        void drop(const std::string& name) { batches_.erase(name); }

        // Four copies, cheap on purpose — see the header. The batches are moved out because the
        // state is finished by the time this is called and nothing will read them again.
        EmissionInputs capture() const {
            EmissionInputs inputs;
            inputs.tag = tag_;
            inputs.width = width_;
            inputs.height = height_;
            inputs.fps = fps_;
            inputs.detections = detections_;
            inputs.batches = batches_;
            return inputs;
        }

        // The frame's pixels, on the device it was decoded on (ADR-004: a frame stays where it was
        // decoded, and only its crops travel).
        void set_image(std::shared_ptr<DeviceBuffer> image, int device) {
            image_ = std::move(image);
            device_ = device;
        }
        const DeviceBuffer* image() const { return image_.get(); }
        int device() const { return device_; }
        // Released as soon as the last stage that needs pixels is done, because a 1080p frame is
        // 6 MB and a thousand of them in flight is the whole budget.
        void release_image() { image_.reset(); }

      private:
        FrameTag tag_;
        int height_ = 0;
        int width_ = 0;
        float fps_ = 0.f;
        int device_ = 0;
        std::vector<Detection> detections_;
        std::map<std::string, ObjectBatch> batches_;
        std::shared_ptr<DeviceBuffer> image_;
    };

}  // namespace shipinfer
