// One frame in flight, and the records it becomes.
//
// ADR-002: a `FrameState` is owned by exactly one worker thread for the frame's whole life.
// The Python version broke that once — the reassembly sweeper read a state whose worker was
// still inside the graph — and the fix there is the design here: whatever the emitter needs is
// **captured** when the frame is finished, and the emitter never touches the state again.
//
// One exception, and it is the designed path rather than a rare interleave: a frame *times out*
// precisely because its worker is still inside `run_objects`, so the sweeper's `capture()` runs
// while that worker is writing `detections_` or `batches_`. Two threads on one `std::map` is
// undefined behaviour, however brief. So the two containers the sweeper copies are behind a
// mutex — taken by the writer for the duration of a move and by `capture()` for the duration of
// a copy, both microseconds, once per stage per frame. The image, the tag and the sizes are set
// before the frame is opened and never change, and stay lock-free.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
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
        size_t rows() const { return object_indices.size(); }
        const float* row(size_t index) const {
            return data.data() + index * static_cast<size_t>(width);
        }

        // Append one engine chunk's real rows. A frame with more objects than the engine's
        // batch runs in chunks, and the first version built a *new* ObjectBatch per chunk and
        // attached it under the same name — `attach` assigns, so chunk k+1 replaced chunk k
        // and a 25-person frame reached the sink with the 9 embeddings of its last chunk,
        // sealed Complete. One batch per stage now, grown here; `attach` is called once.
        void append(const float* chunk, int count, int chunk_width, const std::vector<int>& indices,
                    size_t start) {
            if (count <= 0) return;
            if (width == 0) width = chunk_width;
            if (chunk_width != width) {
                throw std::logic_error("ObjectBatch " + name + ": chunk width " +
                                       std::to_string(chunk_width) + " differs from " +
                                       std::to_string(width));
            }
            if (start + static_cast<size_t>(count) > indices.size()) {
                throw std::logic_error("ObjectBatch " + name +
                                       ": chunk past the end of the indices");
            }
            data.insert(data.end(), chunk, chunk + static_cast<size_t>(count) * width);
            object_indices.insert(object_indices.end(), indices.begin() + static_cast<long>(start),
                                  indices.begin() + static_cast<long>(start) + count);
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
            for (size_t i = 0; i < detections.size(); ++i) {
                detections[i].index = static_cast<int>(i);
            }
            std::lock_guard<std::mutex> lock(mutex_);
            detections_ = std::move(detections);
        }
        // The owning worker's view, between `set_detections` and the frame's end. Not for the
        // sweeper: it copies through `capture()`, which takes the lock.
        const std::vector<Detection>& detections() const { return detections_; }

        void attach(ObjectBatch batch) {
            std::lock_guard<std::mutex> lock(mutex_);
            batches_[batch.name] = std::move(batch);
        }
        void drop(const std::string& name) {
            std::lock_guard<std::mutex> lock(mutex_);
            batches_.erase(name);
        }

        // Copies, cheap on purpose — see the header. Under the lock, because the sweeper calls
        // this on a frame whose worker may be mid-`attach`; the copy is what lets the emitter
        // never touch the state again.
        EmissionInputs capture() const {
            EmissionInputs inputs;
            inputs.tag = tag_;
            inputs.width = width_;
            inputs.height = height_;
            inputs.fps = fps_;
            std::lock_guard<std::mutex> lock(mutex_);
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
        // Not called yet, and kept deliberately. ADR-004: a frame stays on the GPU it was
        // decoded on and only its crops travel, so the pixels should be released the moment
        // the last stage that needs them is done — a 1080p frame is 6 MB and at the design
        // point a thousand of them in flight is the whole budget. The graph runs every stage
        // that needs pixels before it returns, so today the `shared_ptr` dies with the frame
        // and the effect is the same; this is the hook for when a stage runs after them.
        void release_image() { image_.reset(); }

      private:
        FrameTag tag_;
        int height_ = 0;
        int width_ = 0;
        float fps_ = 0.f;
        int device_ = 0;
        mutable std::mutex mutex_;  // guards the two containers below, and only them
        std::vector<Detection> detections_;
        std::map<std::string, ObjectBatch> batches_;
        std::shared_ptr<DeviceBuffer> image_;
    };

}  // namespace shipinfer
