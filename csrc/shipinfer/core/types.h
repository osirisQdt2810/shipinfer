// Small value types the whole data plane shares.
//
// Deliberately tiny and header-only: these cross every layer boundary, and a type that needs
// a translation unit becomes a link-order problem in a codebase with this many threads.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shipinfer {

    // The identity that must survive every path, including every error path. The Python side
    // calls this `RequestContext`; the invariant is the same and it is the one the previous
    // generation lost, which is how a crowded camera came to evict a quiet one's work.
    struct FrameTag {
        std::string camera_id;
        int64_t frame_id = 0;
        int64_t captured_ns = 0;

        std::string key() const { return camera_id + ":" + std::to_string(frame_id); }
    };

    // One detection. Boxes are in the *original* image's pixel coordinates, not the letterboxed
    // model input's — every downstream crop is taken from the full-resolution frame, which is
    // both cheaper and sharper than cropping a resized crop.
    struct Detection {
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        float score = 0;
        int class_id = 0;
        int index = 0;  // position within the frame, so a scattered row can find its owner
    };

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

    // The typed failure vocabulary, mirroring `core/errors/` on the Python side. One exception
    // type per domain, because "something went wrong" is not a diagnosis and a caller that
    // cannot tell a missing engine from a full queue cannot do anything useful about either.
    struct ConfigError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct BackendError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct QueueFullError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    // The server is not in a state to do this: no instance is ready, a model has no instances.
    struct ServerStateError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct SourceError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

}  // namespace shipinfer
