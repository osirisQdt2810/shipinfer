#include "shipinfer/core/buffers.h"

#include "shipinfer/core/platform.h"

namespace shipinfer {

    DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes == 0) return;
        GPU_CHECK(gpuMalloc(&ptr_, bytes));
    }

    DeviceBuffer::~DeviceBuffer() {
        // No throw from a destructor, and no check: a failing free during teardown is worth a
        // note in a log, not an abort in the middle of releasing a device someone else is
        // waiting for. Freeing the GPU promptly is a house rule on this box.
        if (ptr_ != nullptr) gpuFree(ptr_);
    }

    PinnedBuffer::PinnedBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes == 0) return;
        GPU_CHECK(gpuHostAlloc(&ptr_, bytes, gpuHostAllocDefault));
    }

    PinnedBuffer::~PinnedBuffer() {
        if (ptr_ != nullptr) gpuHostFree(ptr_);
    }

}  // namespace shipinfer
