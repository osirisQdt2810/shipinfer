// Owning device and pinned-host buffers — the one place the core touches an allocator.
//
// Kept out of `types.h` on purpose: the value types cross every layer boundary and must stay
// header-only and CUDA-free, so the scheduler and the server can be built and tested on a
// machine with no driver. These two need `platform.h`, so they live beside it, and only the
// translation units that hold device memory include them.
#pragma once

#include <cstddef>
#include <utility>

namespace shipinfer {

    // A device buffer with a size. `cudaFree` on a raw pointer in a destructor is the one place
    // this codebase uses RAII hard: a leaked engine binding is 220-480 MiB of a shared box.
    class DeviceBuffer {
      public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(size_t bytes);
        ~DeviceBuffer();

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            swap(other);
            return *this;
        }

        void swap(DeviceBuffer& other) noexcept {
            std::swap(ptr_, other.ptr_);
            std::swap(bytes_, other.bytes_);
        }

        void* get() const { return ptr_; }
        template <typename T>
        T* as() const {
            return static_cast<T*>(ptr_);
        }
        size_t bytes() const { return bytes_; }
        bool empty() const { return ptr_ == nullptr; }

      private:
        void* ptr_ = nullptr;
        size_t bytes_ = 0;
    };

    // Pinned host memory. Separate type from DeviceBuffer because the allocator is different
    // and mixing them up is a silent 10x on every copy.
    class PinnedBuffer {
      public:
        PinnedBuffer() = default;
        explicit PinnedBuffer(size_t bytes);
        ~PinnedBuffer();

        PinnedBuffer(const PinnedBuffer&) = delete;
        PinnedBuffer& operator=(const PinnedBuffer&) = delete;
        PinnedBuffer(PinnedBuffer&& other) noexcept { swap(other); }
        PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
            swap(other);
            return *this;
        }

        void swap(PinnedBuffer& other) noexcept {
            std::swap(ptr_, other.ptr_);
            std::swap(bytes_, other.bytes_);
        }

        void* get() const { return ptr_; }
        template <typename T>
        T* as() const {
            return static_cast<T*>(ptr_);
        }
        size_t bytes() const { return bytes_; }

      private:
        void* ptr_ = nullptr;
        size_t bytes_ = 0;
    };

}  // namespace shipinfer
