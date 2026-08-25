// Vendor abstraction and the error check every call site uses.
//
// AMD's HIP is a near-exact rename of the CUDA runtime API, so a handful of aliases is
// genuinely all that separates the two for code of this kind. Doing it here — rather than with
// a second copy of every .cu file, or with hipify run over the tree at build time — keeps the
// kernels readable and makes a ROCm build a compiler flag rather than a port. That is ADR-003's
// argument, and the first version of this data plane dropped it by hard-coding the CUDA runtime
// throughout.
//
// This mirrors `3rdparty/shipvision/csrc/.../core/platform.hpp`, which is the house version and
// already existed. Two spellings of one convention is worse than either.
//
// `GPU_CHECK(expr)` is the convention every serious CUDA/HIP codebase uses: wrap the call,
// compare against `gpuSuccess`, throw with the expression, the file and the line. The
// alternative — checking the last error somewhere later — reports the *previous* operation's
// failure at the site of an unrelated one, which is how an afternoon disappears.
//
// This is the only header in `csrc/` that may name a vendor runtime. Everything else uses the
// aliases, and a `grep` for `cuda` outside this file is a bug report.

#pragma once

#include <string>

#include "shipinfer/core/types.h"

#if defined(SHIPINFER_WITH_HIP)
    #include <hip/hip_runtime.h>

using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
    #define gpuSuccess hipSuccess
    #define gpuGetErrorString hipGetErrorString
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemcpyAsync hipMemcpyAsync
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuMemsetAsync hipMemsetAsync
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuHostAlloc hipHostMalloc
    #define gpuHostAllocDefault hipHostMallocDefault
    #define gpuHostFree hipHostFree
    #define gpuHostRegister hipHostRegister
    #define gpuHostRegisterDefault hipHostRegisterDefault
    #define gpuHostUnregister hipHostUnregister
    #define gpuSetDevice hipSetDevice
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuGetLastError hipGetLastError
#else
    #include <cuda_runtime.h>

using gpuError_t = cudaError_t;
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
    #define gpuSuccess cudaSuccess
    #define gpuGetErrorString cudaGetErrorString
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyAsync cudaMemcpyAsync
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemsetAsync cudaMemsetAsync
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuHostAlloc cudaHostAlloc
    #define gpuHostAllocDefault cudaHostAllocDefault
    #define gpuHostFree cudaFreeHost
    #define gpuHostRegister cudaHostRegister
    #define gpuHostRegisterDefault cudaHostRegisterDefault
    #define gpuHostUnregister cudaHostUnregister
    #define gpuSetDevice cudaSetDevice
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuGetLastError cudaGetLastError
#endif

namespace shipinfer {

    // Never called directly — `GPU_CHECK` supplies the expression text and the location.
    inline void gpu_check(gpuError_t status, const char* expression, const char* file,
                          int line) {
        if (status != gpuSuccess) {
            throw BackendError(std::string(expression) +
                               " failed: " + gpuGetErrorString(status) + " at " + file + ":" +
                               std::to_string(line));
        }
    }

}  // namespace shipinfer

#define GPU_CHECK(expr) ::shipinfer::gpu_check((expr), #expr, __FILE__, __LINE__)
