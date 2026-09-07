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
        size_t total = 0;
        for (const auto& [bytes, buffers] : free_) total += buffers.size();
        return total;
    }

    void SurfaceIntake::give_back(std::unique_ptr<DeviceBuffer> buffer) {
        // DECLARED BEFORE THE LOCK, so an over-cap buffer's `cudaFree` runs after it is
        // released. Freeing inside the lock is the same shape as the resolution case: a
        // device-synchronising call inside the mutex every camera on this GPU contends for.
        // It only fires above the cap -- not the steady state -- but that is precisely the
        // path a design-load run takes.
        std::unique_ptr<DeviceBuffer> discarded;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<std::unique_ptr<DeviceBuffer>>& bucket = free_[buffer->bytes()];
            if (bucket.size() >= max_pooled_) {
                discarded = std::move(buffer);
            } else {
                bucket.push_back(std::move(buffer));
            }
        }
    }

    DeviceSurface SurfaceIntake::take(const std::shared_ptr<SurfaceIntake>& self,
                                      const DeviceImage& image, const std::string& camera) {
        // The same predicate the ingest side already refuses on, asked again here because this
        // is where the numbers become a size: a surface whose `uv_offset` is below
        // `pitch * height` would size a buffer smaller than the plane it is about to read.
        if (image.empty()) {
            throw ConfigError("camera '" + camera +
                              "': a device surface reached the pipeline's intake incomplete; "
                              "nothing here can size a buffer for it");
        }
        const size_t stride = static_cast<size_t>(image.pitch);
        const size_t luma = stride * static_cast<size_t>(image.height);
        const size_t bytes = luma + luma / 2;

        // THIS THREAD'S DEVICE, stated rather than inherited. An NVDEC actor thread has the
        // right primary context pushed already, so this is a no-op there; a source that hands
        // over a surface without having done so would otherwise copy onto whatever device the
        // thread last touched -- silently, and onto the wrong GPU.
        GPU_CHECK(gpuSetDevice(self->device_));

        std::unique_ptr<DeviceBuffer> buffer;
        {
            std::lock_guard<std::mutex> lock(self->mutex_);
            // THIS SIZE'S bucket, and only this one. Fifty cameras on a GPU need not be one
            // resolution: a 1080p and a 720p camera alternating used to retire the whole free
            // list on every frame, which is a `cudaFree` per pooled buffer inside this lock and
            // then a `cudaMalloc` -- both device-synchronising, on the ingest thread.
            auto bucket = self->free_.find(bytes);
            if (bucket != self->free_.end() && !bucket->second.empty()) {
                buffer = std::move(bucket->second.back());
                bucket->second.pop_back();
            }
        }
        if (!buffer) buffer = std::make_unique<DeviceBuffer>(bytes);

        // Two copies and not one: the source's chroma starts at `uv_offset`, which need not be
        // `stride * height` -- a padded buffer's is above it. Copying `uv_offset + ...` bytes
        // in one go would carry that padding into the middle of this buffer and leave every
        // consumer needing to know about it. What comes out of here is tight.
        //
        // SYNCHRONOUS, on the legacy default stream, which is correct here and deliberate:
        // cuvid's post-processing lands on stream 0 (`ingest/sources/nvdec.cpp`) and every
        // stream in `csrc/` is blocking, so the ordering is free. It is two device-wide sync
        // points per frame on an ingest thread; invisible at 8x5, and worth watching at 50x20.
        GPU_CHECK(gpuMemcpy(buffer->get(), image.nv12, luma, gpuMemcpyDeviceToDevice));
        GPU_CHECK(gpuMemcpy(buffer->as<uint8_t>() + luma,
                            static_cast<const uint8_t*>(image.nv12) + image.uv_offset, luma / 2,
                            gpuMemcpyDeviceToDevice));

        DeviceSurface surface;
        surface.nv12 = buffer->as<uint8_t>();
        surface.stride = image.pitch;
        surface.uv_offset = luma;
        // THE POOL'S KEEPALIVE, and it holds the pool rather than pointing at it. Capturing
        // `this` was wrong for the reason `surface_intake.h` states: the sink that owns the
        // intakes is destroyed BEFORE the guard that stops the workers, so an unwind between
        // `manager.start()` and `queue.close()` had every queued surface's deleter locking a
        // destroyed mutex. Holding `self` makes the order irrelevant instead of asserted.
        DeviceBuffer* raw = buffer.release();
        surface.owner = std::shared_ptr<const void>(raw->get(), [self, raw](const void*) {
            self->give_back(std::unique_ptr<DeviceBuffer>(raw));
        });
        return surface;
    }

}  // namespace shipinfer
