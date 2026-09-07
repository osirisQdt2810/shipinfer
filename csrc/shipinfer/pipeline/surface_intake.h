// A DECODER'S SURFACE IN, A SURFACE THE PIPELINE OWNS OUT — and the decoder's slot back at
// once.
//
// WHY A COPY IS THE RIGHT ANSWER HERE, AND NOT A COMPROMISE
// --------------------------------------------------------
// NVDEC hands out a mapped surface from a per-camera pool of two output surfaces. The fair
// queue's whole purpose is to HOLD frames while a batch assembles, so a surface travelling
// through it would exhaust that pool at the third frame -- `cuvidMapVideoFrame` fails, the
// camera reconnect-loops, and it happens under exactly the load the queue exists to serve.
// Raising the pool to the queue's depth is per-camera VRAM times fifty cameras; shortening the
// queue gives up the fairness this project exists to get right.
//
// So the hand-off copies, which is also what DeepStream does between the decoder's NVMM pool
// and `nvvideoconvert`'s (V86: the reference implementation's shape is the default). It stays
// V156's route -- ~3 MB DEVICE-TO-DEVICE for a 1080p NV12 frame, against ~6 MB down *and* 6 MB
// up for the host round trip the route exists to remove -- and the decoder's pool goes back to
// being bounded by decode depth rather than by queue depth.
//
// What comes out is TIGHT -- `uv_offset` is `stride * height`, exactly one luma plane. That is
// also what a `cuvidMapVideoFrame` surface reports (#159 measured it: the coded height sizes
// the decode surfaces, not the mapped output one), so the copy preserves the number rather
// than changing it, and no consumer can tell the two producers apart.
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "shipinfer/core/buffers.h"
#include "shipinfer/ingest/frame.h"
#include "shipinfer/pipeline/graph/state.h"

namespace shipinfer {

    // One camera's intake. Not thread-safe to CALL -- one actor thread owns a camera for its
    // whole life (ADR-002) -- but a buffer is RETURNED from whichever worker finishes the
    // frame, so the free list is behind a mutex.
    class SurfaceIntake {
      public:
        // `device` is the GPU the decoder is on, and the one every buffer is allocated on: a
        // frame stays where it was decoded (ADR-004).
        explicit SurfaceIntake(int device, size_t max_pooled = 8);

        // `image` copied into a pooled buffer. The returned surface owns that buffer and
        // returns it to the pool when the last reference goes; the CALLER still holds
        // `image.owner`, and dropping it is what gives the decode slot back.
        //
        // Throws `ConfigError` if `image` is not a usable NV12 surface, and the typed CUDA
        // error if the copy fails -- never a surface with fewer bytes than it claims.
        DeviceSurface take(const DeviceImage& image);

        // How many buffers are sitting in the free list. For the gate that proves reuse: an
        // allocation per frame at a thousand frames a second is the thing this class avoids.
        size_t pooled() const;

      private:
        // Returned by the deleter, or dropped if the pool is full or the size no longer
        // matches. A cap because a stalled consumer must not turn into unbounded VRAM.
        void give_back(std::unique_ptr<DeviceBuffer> buffer);

        int device_;
        size_t max_pooled_;
        mutable std::mutex mutex_;
        size_t bytes_ = 0;  // what the pooled buffers hold; a change empties the pool
        std::vector<std::unique_ptr<DeviceBuffer>> free_;
    };

}  // namespace shipinfer
