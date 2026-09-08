#include "shipinfer/pipeline/surface_intake.h"

#include <string>
#include <utility>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    SurfaceIntake::SurfaceIntake(int device, size_t max_pooled)
        : device_(device), max_pooled_(max_pooled) {}

    namespace {

        // doc: long the guard has to cover the MEMBERS' destruction, not just the stream's
        // `device_` current for a scope, and the caller's device back after it. A surface holds
        // `self`, so the last reference can be dropped by a thread that is not this GPU's -- an
        // unwind, or a sweeper -- and a destructor that silently rebinds a thread's current
        // device is a landmine under ADR-002 even where today's callers make it a no-op.
        //
        // A GUARD rather than a line at the end, because the pooled `DeviceBuffer`s in `free_`
        // are freed AFTER the destructor body: restoring before that ran was the first version,
        // and `cudaFree` being address-based under unified addressing is luck, not design.
        struct OnDevice {
            int previous = 0;
            bool changed = false;

            explicit OnDevice(int device) {
                if (gpuGetDevice(&previous) != gpuSuccess) previous = device;
                changed = previous != device && gpuSetDevice(device) == gpuSuccess;
            }
            ~OnDevice() {
                if (changed) gpuSetDevice(previous);  // best effort: an exception may be flying
            }
        };

    }  // namespace

    SurfaceIntake::~SurfaceIntake() {
        // Declared FIRST, so it outlives the members destroyed after this body returns.
        const OnDevice on_device(device_);
        if (stream_ != nullptr) {
            gpuStreamDestroy(static_cast<gpuStream_t>(stream_));
            stream_ = nullptr;
        }
    }

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

    void* SurfaceIntake::stream() {
        // ONCE, whoever arrives first, with `device_` already current (`take` sets it): a
        // stream belongs to the device that was current when it was made, and this class is the
        // only thing that knows which that is. A throw leaves the flag unset, so the next
        // caller legitimately retries rather than using a null stream.
        std::call_once(stream_once_, [this] {
            gpuStream_t created = nullptr;
            GPU_CHECK(gpuStreamCreate(&created));
            stream_ = created;
        });
        return stream_;
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
        // doc: long the stream this runs on, and the drain the default stream cost
        // ON THIS INTAKE'S OWN BLOCKING STREAM, not the legacy default one. A synchronous
        // `gpuMemcpy` on the default stream makes every OTHER blocking stream wait too, so each
        // copy drained the whole device -- all 23 worker streams on that GPU -- and at the
        // design load that is ~1460 full-device synchronisations a second on the ingest
        // threads. Waiting on this stream alone drops the drain.
        //
        // THE ORDERING IS NOT INHERITED, and this comment used to claim it was: it said
        // cuvid's post-processing lands on stream 0 and a blocking stream waits for stream 0,
        // so the copies were ordered for free. Both clauses stopped being true when the
        // post-processing moved to its own non-blocking stream (#164), and a blocking stream is
        // ordered against the LEGACY DEFAULT stream only in any case. The edge is the
        // producer's event, below.
        gpuStream_t stream = static_cast<gpuStream_t>(self->stream());
        // WAITED FOR ON THE DEVICE, not the host: the producer recorded `ready` on whatever
        // stream wrote these bytes (`ingest/frame.h`), and this orders that write before the
        // two copies without the ingest thread blocking twice. Absent for a producer whose
        // bytes are already final, and then there is nothing to wait for.
        if (image.ready != nullptr) {
            GPU_CHECK(gpuStreamWaitEvent(stream, static_cast<gpuEvent_t>(image.ready), 0));
        }
        GPU_CHECK(
            gpuMemcpyAsync(buffer->get(), image.nv12, luma, gpuMemcpyDeviceToDevice, stream));
        GPU_CHECK(gpuMemcpyAsync(buffer->as<uint8_t>() + luma,
                                 static_cast<const uint8_t*>(image.nv12) + image.uv_offset,
                                 luma / 2, gpuMemcpyDeviceToDevice, stream));
        // WAITED FOR, and it has to be: the caller drops `image.owner` the moment this returns,
        // which lets NVDEC reuse the surface these copies are still reading. This orders
        // COPY -> UNMAP and nothing else -- the other direction is the event above.
        GPU_CHECK(gpuStreamSynchronize(stream));

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
