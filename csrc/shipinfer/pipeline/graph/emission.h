// The CUDA-free half of a frame in flight: what the emitter is handed.
//
// SPLIT OUT of `state.h` so it can be tested without a device. `ObjectBatch` and
// `EmissionInputs` are `std::vector<float>`, `std::map` and a tag -- host memory, every one --
// while `DevicePayload` and `FrameState` next door need `core/device.h` and therefore CUDA.
// Before the split, `pipeline/events/records.cpp` could only be reached from a binary that
// needs TensorRT, so the one translation unit that runs in production had no test at all and
// its first version dropped every embedding in silence.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // Why a frame's emission happened. Here rather than beside the collector because it
    // describes the EMISSION, and because a plain enum in a device-reaching header is what
    // stopped the record builder from having a test.
    enum class FinishReason { Complete, Incomplete, Timeout, Shutdown, Evicted };

    const char* to_string(FinishReason reason);

    // A per-object result: N rows, each belonging to one detection of one frame.
    struct ObjectBatch {
        std::string name;
        std::vector<int>
            object_indices;       // object_indices[row] is the detection that produced it
        std::vector<float> data;  // row-major, `width` floats per row
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
        void append(const float* chunk, int count, int chunk_width,
                    const std::vector<int>& indices, size_t start) {
            if (count <= 0) return;
            if (width == 0) width = chunk_width;
            if (chunk_width != width) {
                throw BackendError("ObjectBatch " + name + ": chunk width " +
                                   std::to_string(chunk_width) + " differs from " +
                                   std::to_string(width));
            }
            if (start + static_cast<size_t>(count) > indices.size()) {
                throw std::logic_error("ObjectBatch " + name +
                                       ": chunk past the end of the indices");
            }
            data.insert(data.end(), chunk, chunk + static_cast<size_t>(count) * width);
            object_indices.insert(object_indices.end(),
                                  indices.begin() + static_cast<long>(start),
                                  indices.begin() + static_cast<long>(start) + count);
        }
    };

    // What the event builder needs, and nothing else. Captured under the collector's lock;
    // built outside it.
    struct EmissionInputs {
        FrameTag tag;
        int width = 0;
        int height = 0;
        float fps = 0.f;
        std::vector<Detection> detections;
        std::map<std::string, ObjectBatch> batches;
    };

    //: The names the planner reads — `pipeline/graph/state.py`. `FRAME_INPUT` is the frame's
    //: own pixels; a stage that consumes it says so and the graph then knows when they may be
    //: released.
    inline const char* const FRAME_INPUT = "image";
    inline const char* const DETECTIONS = "detections";

}  // namespace shipinfer
