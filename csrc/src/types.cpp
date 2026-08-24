#include "shipinfer/core/cuda_check.hpp"
#include "shipinfer/core/types.hpp"

namespace shipinfer {

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) return;
    SHIPINFER_CUDA(cudaMalloc(&ptr_, bytes));
}

DeviceBuffer::~DeviceBuffer() {
    // No throw from a destructor, and no check: a failing free during teardown is worth a
    // note in a log, not an abort in the middle of releasing a device someone else is waiting
    // for. Freeing the GPU promptly is a house rule on this box.
    if (ptr_ != nullptr) cudaFree(ptr_);
}

PinnedBuffer::PinnedBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) return;
    SHIPINFER_CUDA(cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault));
}

PinnedBuffer::~PinnedBuffer() {
    if (ptr_ != nullptr) cudaFreeHost(ptr_);
}

}  // namespace shipinfer
