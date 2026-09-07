#include "shipinfer/pipeline/surface_intake.h"

#include <string>
#include <utility>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    SurfaceIntake::SurfaceIntake(int device, size_t max_pooled)
        : device_(device), max_pooled_(max_pooled) {}

    size_t SurfaceIntake::pooled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_.size();
    }

    void SurfaceIntake::give_back(std::unique_ptr<DeviceBuffer> buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_.size() >= max_pooled_ || buffer->bytes() != bytes_) return;  // freed instead
        free_.push_back(std::move(buffer));
    }

    DeviceSurface SurfaceIntake::take(const DeviceImage& image) {
        // The same predicate the ingest side already refuses on, asked again here because this
        // is where the numbers become a size: a surface whose `uv_offset` is below
        // `pitch * height` would size a buffer smaller than the plane it is about to read.
        if (image.empty()) {
            throw ConfigError(
                "a device surface reached the pipeline's intake incomplete; "
                "nothing here can size a buffer for it");
        }
        const size_t stride = static_cast<size_t>(image.pitch);
        const size_t luma = stride * static_cast<size_t>(image.height);
        const size_t bytes = luma + luma / 2;

        // THIS THREAD'S DEVICE, stated rather than inherited. An NVDEC actor thread has the
        // right primary context pushed already, so this is a no-op there; a source that hands
        // over a surface without having done so would otherwise copy onto whatever device the
        // thread last touched -- silently, and onto the wrong GPU.
        GPU_CHECK(gpuSetDevice(device_));

        std::unique_ptr<DeviceBuffer> buffer;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (bytes != bytes_) {
                // A resolution change. The source refuses one mid-stream, so this is a
                // reconnect having negotiated something else -- the pool's buffers are the
                // wrong size and holding them is just VRAM.
                free_.clear();
                bytes_ = bytes;
            }
            if (!free_.empty()) {
                buffer = std::move(free_.back());
                free_.pop_back();
            }
        }
        if (!buffer) buffer = std::make_unique<DeviceBuffer>(bytes);

        // Two copies and not one: the source's chroma starts at `uv_offset`, which for an NVDEC
        // surface is past `stride * height` by the coded padding. Copying `uv_offset + ...`
        // bytes in one go would carry that padding into the middle of this buffer and leave
        // every consumer needing to know about it -- what comes out of here is tight.
        GPU_CHECK(gpuMemcpy(buffer->get(), image.nv12, luma, gpuMemcpyDeviceToDevice));
        GPU_CHECK(gpuMemcpy(buffer->as<uint8_t>() + luma,
                            static_cast<const uint8_t*>(image.nv12) + image.uv_offset, luma / 2,
                            gpuMemcpyDeviceToDevice));

        DeviceSurface surface;
        surface.nv12 = buffer->as<uint8_t>();
        surface.stride = image.pitch;
        surface.uv_offset = luma;
        // The pool's own keepalive. `this` is safe to capture: an intake outlives its camera's
        // frames by construction -- it is owned by the sink, which is torn down after the
        // fleet has drained (`Fleet::stop`), and a frame cannot outlive the queue it was in.
        DeviceBuffer* raw = buffer.release();
        surface.owner = std::shared_ptr<const void>(raw->get(), [this, raw](const void*) {
            give_back(std::unique_ptr<DeviceBuffer>(raw));
        });
        return surface;
    }

}  // namespace shipinfer
