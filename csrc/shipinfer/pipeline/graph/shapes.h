// The engine's own shapes are the truth; the config is a claim. Checked once, at construction.
//
// The graph writes into a TensorRT input binding with a CUDA kernel, and the extent of that
// write comes from the *config* (`detect_size`, `crop_h`, `crop_w`) while the binding was sized
// from the *plan*. Nothing compared the two: a segmenter built at 512x512 fed 640x640 rows was
// an out-of-bounds device write per row that surfaced as `an illegal memory access` in another
// worker's synchronise several frames later, and a 1280 detector fed 640 rows inferred on
// uninitialised memory with no error at all. `engine.h` promised the check; this is it.
#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/backends/tensorrt/engine.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    inline std::string describe_dims(const std::vector<int64_t>& dims) {
        std::ostringstream out;
        out << "[";
        for (size_t i = 0; i < dims.size(); ++i) out << (i ? ", " : "") << dims[i];
        out << "]";
        return out.str();
    }

    // The rows the graph will feed `spec` must be exactly the plan's per-row shape. A negative
    // plan dimension is dynamic and matches anything; every other one has to agree.
    inline void expect_input_row(const TensorSpec& spec, const std::vector<int64_t>& fed,
                                 const std::string& who) {
        bool same = spec.dims.size() == fed.size();
        for (size_t i = 0; same && i < fed.size(); ++i) {
            same = spec.dims[i] < 0 || spec.dims[i] == fed[i];
        }
        if (!same) {
            throw ConfigError(who + ": the plan's input '" + spec.name + "' is " +
                              describe_dims(spec.dims) + " per row but the graph would feed " +
                              describe_dims(fed) +
                              " — the engine's own shapes are the truth; change the config to "
                              "match the plan, or rebuild the plan");
        }
    }

    // The graph writes float32 into inputs and reads float32 out of outputs. Any other element
    // size would be reinterpreted silently — an FP16 detector's boxes read as garbage floats.
    inline void expect_float32(const TensorSpec& spec, const std::string& who) {
        if (spec.element_size != 4) {
            throw ConfigError(
                who + ": tensor '" + spec.name + "' has " + std::to_string(spec.element_size) +
                "-byte elements but the graph moves float32; rebuild the plan with "
                "float32 I/O or add the conversion");
        }
    }

}  // namespace shipinfer
