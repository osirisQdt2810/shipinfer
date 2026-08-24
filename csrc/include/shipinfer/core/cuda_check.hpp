// One place that turns a CUDA or TensorRT status into a typed exception with a location.
//
// The alternative — checking `cudaGetLastError()` somewhere later — reports the *next*
// operation's failure at the site of an unrelated one, which is how an afternoon disappears.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "shipinfer/core/types.hpp"

namespace shipinfer {

inline void cuda_check(cudaError_t status, const char* what, const char* file, int line) {
    if (status != cudaSuccess) {
        throw BackendError(std::string(what) + " failed: " + cudaGetErrorString(status) + " at " +
                           file + ":" + std::to_string(line));
    }
}

}  // namespace shipinfer

#define SHIPINFER_CUDA(expr) ::shipinfer::cuda_check((expr), #expr, __FILE__, __LINE__)
