// Vendor abstraction: one source tree, CUDA or HIP.
//
// AMD's HIP is a near-exact rename of the CUDA runtime API, so a handful of
// aliases is genuinely all that separates the two for kernels of this kind.
// Doing it here — rather than with a second copy of every .cu file, or with
// hipify run over the tree at build time — keeps the kernels readable and means
// a ROCm build is a CMake flag, not a port.
//
// The Python layer needs no equivalent: `torch.cuda` already *is* the HIP API
// on ROCm.

#pragma once

#include <stdexcept>
#include <string>

#if defined(SHIPINFER_WITH_HIP)
#include <hip/hip_runtime.h>

using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemsetAsync hipMemsetAsync
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetLastError hipGetLastError

#else
#include <cuda_runtime.h>

using gpuError_t = cudaError_t;
using gpuStream_t = cudaStream_t;
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemsetAsync cudaMemsetAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetLastError cudaGetLastError
#endif

namespace shipinfer {

  /// Thrown for any failed device call. pybind11 maps it to a Python
  /// RuntimeError, so a kernel launch failure surfaces as an ordinary exception
  /// instead of a silent wrong answer or an abort inside the interpreter.
  class GpuError : public std::runtime_error {
    public:
      explicit GpuError(const std::string& what) : std::runtime_error(what) {}
  };

  inline void check(gpuError_t status, const char* what) {
    if (status != gpuSuccess) {
      throw GpuError(std::string(what) + " failed: " + gpuGetErrorString(status));
    }
  }

  /// Check the *asynchronous* error slot after a launch.
  ///
  /// A kernel launch reports configuration errors immediately and execution
  /// errors only later; without this an out-of-bounds write shows up as an
  /// unrelated failure in whatever call happens next, which is one of the hardest
  /// CUDA bugs to trace.
  inline void check_launch(const char* what) {
    check(gpuGetLastError(), what);
  }

  /// Ceiling division, for grid sizing.
  constexpr int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
  }

  /// 256 threads: enough to hide memory latency on every architecture this
  /// targets, and small enough that the occupancy limit is registers rather than
  /// the block size.
  constexpr int kBlockSize = 256;

  /// A device-side view of one source image inside a ragged batch.
  ///
  /// The batch is ragged by nature — 50 cameras do not agree on resolution — so
  /// the kernel cannot index a single dense tensor. One descriptor per image,
  /// uploaded once per call, is what lets a single launch cover the whole batch
  /// instead of one launch per image.
  struct ImageView {
      const unsigned char* data; ///< HWC, 3 channels, uint8
      int height;
      int width;
      float scale; ///< resize factor applied to fit the destination
      int pad_x;   ///< letterbox offset in destination pixels
      int pad_y;
      int out_h; ///< resized extent inside the destination canvas
      int out_w;
  };

} // namespace shipinfer
