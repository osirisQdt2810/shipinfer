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

#include "shipinfer/core/buffers.h"
#include "shipinfer/core/device.h"
#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/emission.h"

namespace shipinfer {

    // N rows of a per-object tensor living on a device — a crop set going *into* a model. Kept
    // apart from `ObjectBatch` (a model's *output*, host memory, copied into the emission)
    // because device memory is move-only and must never be copied into a capture. The buffer is
    // owned through `owner`, below, by everyone who still refers to it.
    struct DevicePayload {
        std::string name;
        std::string class_name;
        const float* data = nullptr;  // rows x row_elems floats, on `device`
        // What `data` points into. A request made from this payload keeps a copy, so the
        // buffer outlives a stage that gave up waiting on it — the worker's scratch cannot
        // hand it out again while an instance may still be reading it.
        std::shared_ptr<DeviceBuffer> owner;
        size_t rows = 0;
        size_t row_elems = 0;
        Device device;
        std::vector<int> object_indices;  // the detection each row belongs to
        std::vector<float> boxes;         // (rows, 4) source boxes, for the event builder
        bool empty() const { return rows == 0; }
    };

    // The frame's pixels AS THE DECODER LEFT THEM: one NV12 surface, luma plane first and the
    // interleaved chroma plane `uv_offset` bytes in, `stride` bytes per row.
    //
    // `uv_offset` is carried and not derived, which is the whole reason this is a struct rather
    // than a pointer. NVDEC decodes at a CODED height rounded up -- 1088 for 1080p -- so the
    // chroma begins at `stride * 1088` while the frame is 1080 tall, and `stride * height`
    // reads the last eight luma rows as chroma: right brightness, wrong colour, on every frame
    // (#153 round 4, #155, and the reason `runtime/ops.h` takes it as a parameter).
    //
    // `owner` is what keeps those bytes alive. For an NVDEC surface it unmaps a slot out of a
    // small pool, so dropping it early does not free the pixels -- it lets the next picture
    // overwrite them under a worker that is still reading.
    struct DeviceSurface {
        const uint8_t* nv12 = nullptr;
        int stride = 0;
        size_t uv_offset = 0;
        std::shared_ptr<const void> owner;
        bool empty() const { return nv12 == nullptr; }
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
        // A crop set (or any per-object device tensor) under its name, for the stages that read
        // it.
        void attach_payload(DevicePayload payload) {
            payloads_[payload.name] = std::move(payload);
        }
        const DevicePayload* payload(const std::string& name) const {
            auto it = payloads_.find(name);
            return it == payloads_.end() ? nullptr : &it->second;
        }
        const ObjectBatch* batch(const std::string& name) const {
            auto it = batches_.find(name);
            return it == batches_.end() ? nullptr : &it->second;
        }

        // What the letterbox applied, stored rather than recomputed: the decode must undo
        // exactly the transform that was applied, and these are the numbers that were applied.
        void set_letterbox(float scale, int pad_x, int pad_y) {
            scale_ = scale;
            pad_x_ = pad_x;
            pad_y_ = pad_y;
        }
        float scale() const { return scale_; }
        int pad_x() const { return pad_x_; }
        int pad_y() const { return pad_y_; }
        void set_detected(bool detected) { detected_ = detected; }

        int priority() const { return priority_; }
        void set_priority(int priority) { priority_ = priority; }
        int64_t deadline_ns() const { return deadline_ns_; }
        void set_deadline_ns(int64_t deadline_ns) { deadline_ns_ = deadline_ns; }

        // -- the planner's two questions -----------------------------------------------------
        // Names present right now: the image while it is held, the detections once the detector
        // answered (even if it found nothing), every payload and every batch.
        std::vector<std::string> available() const {
            std::vector<std::string> names;
            if (has_pixels()) names.push_back(FRAME_INPUT);
            if (detected_) names.push_back(DETECTIONS);
            for (const auto& [name, _] : payloads_) names.push_back(name);
            for (const auto& [name, _] : batches_) names.push_back(name);
            return names;
        }
        // Names present *and non-empty*: what a stage with `needs` waits for. The
        // conditional branch lives here — a ship crop set with zero rows is available (the
        // graph is valid) and not non-empty (the segmenter is never called for it).
        std::vector<std::string> non_empty() const {
            std::vector<std::string> names;
            if (has_pixels()) names.push_back(FRAME_INPUT);
            if (detected_ && !detections_.empty()) names.push_back(DETECTIONS);
            for (const auto& [name, payload] : payloads_) {
                if (!payload.empty()) names.push_back(name);
            }
            for (const auto& [name, batch] : batches_) {
                if (!batch.empty()) names.push_back(name);
            }
            return names;
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

        // The frame's pixels, on the device it was decoded on (ADR-004: a frame stays where it
        // was decoded, and only its crops travel).
        void set_image(std::shared_ptr<DeviceBuffer> image, int device) {
            image_ = std::move(image);
            device_ = device;
        }
        const DeviceBuffer* image() const { return image_.get(); }
        // The other representation, and the two are exclusive: a camera's pixels live in one
        // place for its whole life (`FrameCounter::latch_where` on the ingest side enforces
        // that a source cannot change its mind). `pipeline/graph/pixels.h` is the only thing
        // that asks which.
        void set_surface(DeviceSurface surface, int device) {
            surface_ = std::move(surface);
            device_ = device;
        }
        const DeviceSurface& surface() const { return surface_; }
        int device() const { return device_; }
        // Released as soon as the last stage that needs pixels is done, because a 1080p frame
        // is 6 MB and a thousand of them in flight is the whole budget. Not called yet, and
        // kept deliberately. ADR-004: a frame stays on the GPU it was decoded on and only its
        // crops travel, so the pixels should be released the moment the last stage that needs
        // them is done — a 1080p frame is 6 MB and at the design point a thousand of them in
        // flight is the whole budget. The graph runs every stage that needs pixels before it
        // returns, so today the `shared_ptr` dies with the frame and the effect is the same;
        // this is the hook for when a stage runs after them.
        void release_image() {
            image_.reset();
            surface_ = DeviceSurface{};
        }

      private:
        // Whether `FRAME_INPUT` is satisfied. EITHER representation counts, and asking this
        // rather than `image_` is what stops an NV12 frame looking like a frame with no pixels
        // -- which the planner would answer by skipping the detector on every frame, silently.
        bool has_pixels() const { return image_ != nullptr || !surface_.empty(); }

        FrameTag tag_;
        int height_ = 0;
        int width_ = 0;
        float fps_ = 0.f;
        int device_ = 0;
        float scale_ = 1.f;
        int pad_x_ = 0;
        int pad_y_ = 0;
        bool detected_ = false;
        int priority_ = 2;
        int64_t deadline_ns_ = 0;
        std::map<std::string, DevicePayload> payloads_;  // the owning worker's, never captured
        mutable std::mutex mutex_;  // guards the two containers below, and only them
        std::vector<Detection> detections_;
        std::map<std::string, ObjectBatch> batches_;
        std::shared_ptr<DeviceBuffer> image_;
        DeviceSurface surface_;
    };

}  // namespace shipinfer
