// Persistent, growable device and pinned-host scratch.
//
// The first version of this extension allocated and freed its scratch on every
// call, and it was no faster than the pure-torch path. The kernel was never the
// problem: `cudaMalloc` and `cudaFree` are *synchronising* calls, and a
// pageable `cudaMemcpy` runs at roughly half the bandwidth of a pinned one
// because the driver stages it through an internal bounce buffer. Between them
// they cost more than the work.
//
// So the scratch lives as long as the `ImageOps` object and grows
// monotonically. That is the same bet torch's caching allocator makes, made
// here because this extension does not link torch.

#pragma once

#include <algorithm>
#include <cstddef>

#include "shipinfer/platform.hpp"

namespace shipinfer {

/// Device scratch that grows to the high-water mark and is never handed back.
class DeviceScratch {
public:
  DeviceScratch() = default;
  ~DeviceScratch() { release(); }
  DeviceScratch(const DeviceScratch &) = delete;
  DeviceScratch &operator=(const DeviceScratch &) = delete;

  /// A pointer to at least `bytes`, reallocating only when the request grows.
  void *reserve(size_t bytes) {
    if (bytes <= capacity_)
      return ptr_;
    release();
    // Over-allocate by a quarter so a slowly growing batch does not reallocate
    // every call.
    const size_t target = bytes + bytes / 4;
    check(gpuMalloc(&ptr_, target), "gpuMalloc (device scratch)");
    capacity_ = target;
    return ptr_;
  }

  size_t capacity() const { return capacity_; }

  void release() {
    if (ptr_ != nullptr) {
      gpuFree(ptr_);
      ptr_ = nullptr;
      capacity_ = 0;
    }
  }

private:
  void *ptr_ = nullptr;
  size_t capacity_ = 0;
};

/// Page-locked host scratch, same growth policy.
///
/// Worth the trouble for one reason: `cudaMemcpyAsync` from pageable memory
/// silently degrades to a synchronous copy at about half the bandwidth. Staging
/// through pinned memory is what makes the upload both faster and genuinely
/// asynchronous.
class PinnedScratch {
public:
  PinnedScratch() = default;
  ~PinnedScratch() { release(); }
  PinnedScratch(const PinnedScratch &) = delete;
  PinnedScratch &operator=(const PinnedScratch &) = delete;

  unsigned char *reserve(size_t bytes) {
    if (bytes <= capacity_)
      return ptr_;
    release();
    const size_t target = bytes + bytes / 4;
#if defined(SHIPINFER_WITH_HIP)
    check(hipHostMalloc(reinterpret_cast<void **>(&ptr_), target),
          "hipHostMalloc");
#else
    check(cudaHostAlloc(reinterpret_cast<void **>(&ptr_), target,
                        cudaHostAllocDefault),
          "cudaHostAlloc");
#endif
    capacity_ = target;
    return ptr_;
  }

  size_t capacity() const { return capacity_; }

  void release() {
    if (ptr_ != nullptr) {
#if defined(SHIPINFER_WITH_HIP)
      hipHostFree(ptr_);
#else
      cudaFreeHost(ptr_);
#endif
      ptr_ = nullptr;
      capacity_ = 0;
    }
  }

private:
  unsigned char *ptr_ = nullptr;
  size_t capacity_ = 0;
};

} // namespace shipinfer
