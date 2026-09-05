// One row's shape, and the rule that a row has to have one — CUDA-free so a gate can reach it.
//
// `TensorSpec::elements_per_row` used to CLAMP a negative dimension to 1, and `row_bytes()`
// sizes both the device and the host buffers from it (`backends/tensorrt/engine.cpp`). So a
// dynamic NON-BATCH dimension did not merely report a wrong width: it allocated one element
// where the engine writes many, and the copy back reads past the end of a buffer sized from
// the lie. `dims` reported the `-1` verbatim, so the two an `OutputTensor` carries could also
// disagree (`ENGINE-DIMS-CAN-DISAGREE-WITH-WIDTH`).
//
// The batch is not in `dims` -- `TrtEngine` reads dimensions 1..n into it and handles the
// batch separately, taking the profile's max where it is dynamic. So a negative HERE is a
// dimension the plan never fixed, which is a shape this pipeline cannot feed: the crop
// elements resize to a fixed extent and the mask fold counts cells in a fixed bank.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // The number of elements in one row. Precondition: `require_static_row` accepted `dims`.
    inline size_t elements_per_row(const std::vector<int64_t>& dims) {
        size_t elements = 1;
        for (int64_t dim : dims) elements *= static_cast<size_t>(dim);
        return elements;
    }

    // Refuse a row shape this plane cannot size a buffer from, naming the tensor.
    //
    // At LOAD, where the artefact is still identifiable, rather than as an overflow later:
    // a `0` allocates nothing and a `-1` clamped to 1 allocated one element per row for an
    // output the engine fills with many.
    inline void require_static_row(const std::string& name, const std::vector<int64_t>& dims) {
        for (int64_t dim : dims) {
            if (dim > 0) continue;
            std::string shape;
            for (size_t i = 0; i < dims.size(); ++i) {
                shape += (i ? ", " : "") + std::to_string(dims[i]);
            }
            throw BackendError(
                "tensor '" + name + "' has the per-row shape (" + shape +
                "), which is not fixed. The batch dimension may be dynamic and is handled "
                "separately; every other one sizes a buffer, and a dimension the plan did not "
                "fix would size it for one element where the engine writes many");
        }
    }

}  // namespace shipinfer
