// One decoded frame, and the per-camera counter that stamps it.
//
// The counter is small enough to look pointless and important enough to be named here: **it
// belongs to the camera actor, not to the source**. A source is destroyed and rebuilt on every
// reconnect; if it owned the counter, a camera that dropped out would restart at zero and hand
// a downstream tracker a second frame 0 for the same camera — the same `(camera_id, frame_id)`
// key twice, which is the one thing ADR-002 relies on never happening.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // A decoded image on the host: uint8 HWC BGR, in the decoder's native layout. It is not
    // normalised, letterboxed or moved to a device here — that is the runtime's job, which is
    // what keeps this package free of any accelerator header.
    struct HostFrame {
        const uint8_t* pixels = nullptr;
        int height = 0;
        int width = 0;
        // Whatever `pixels` points into, kept alive for as long as this frame exists — the
        // keepalive idiom `InferenceRequest` already uses. A replay library outlives the source
        // a reconnect replaced *because of this field*: without it, a reconnect frees the pages
        // a worker is still DMAing out of, and the failure surfaces as an illegal access inside
        // an unrelated synchronise several frames later.
        std::shared_ptr<const void> owner;

        size_t bytes() const {
            return static_cast<size_t>(height) * static_cast<size_t>(width) * 3;
        }
        bool empty() const { return pixels == nullptr || height <= 0 || width <= 0; }
    };

    // doc: long a device image needs no CUDA header, and why the pitch is here
    // A decoded image in DEVICE memory: NV12, as NVDEC produces it.
    //
    // NO ACCELERATOR HEADER IS NEEDED TO HOLD ONE, which is what keeps `ingest/` free of them
    // (`HostFrame` above makes the same promise). A device pointer is an address; only reading
    // it needs CUDA, and that happens in `runtime/`, where `nv12_letterbox_into` already takes
    // exactly this shape.
    //
    // `pitch` is separate from `width` because NVDEC's surfaces are padded: the Y plane's rows
    // are `pitch` bytes apart. Assuming `pitch == width` reads the next row's left edge into
    // this row's right edge, a skew that looks like a decoder bug rather than an arithmetic
    // one.
    //
    // doc: long the field this carrier shipped without, and the wrong rule it then shipped WITH
    // `uv_offset` is separate for the SAME reason one plane up: the plane's offset is a
    // property of the buffer the frame was decoded into, and a consumer that derived it would
    // read luma rows as chroma -- right brightness, wrong colour, every frame, on every camera,
    // looking exactly like a model problem. `runtime/ops.h` takes it as a parameter for that.
    //
    // **What it IS, for every producer in this tree: `pitch * height`.** This header said the
    // opposite in capitals until #159 -- that NVDEC decodes at a coded height rounded up, so
    // the chroma begins at `pitch * 1088` for a 1080p camera -- and the source written from it
    // faulted `pitch * 8` bytes past the end of the mapping the first time anything read that
    // plane. The coded height sizes the DECODE surfaces, which no application sees;
    // `cuvidMapVideoFrame` hands back a post-processed OUTPUT surface at the TARGET extent. A
    // tight buffer's offset is the same value for its own reason, so the two producers shipped
    // today agree on the number.
    //
    // MEASURED, not re-read, because reading a header is how the first version happened:
    //   PROBE pitch=2048 display=1920x1080 coded_h=1088
    //         at_coded=cudaErrorInvalidValue  at_display=cudaSuccess
    //
    // Note what the guard in `runtime/ops.h` can and cannot do: it refuses an offset INSIDE the
    // luma plane, and `pitch * coded_height` is not inside it -- so a producer that gets this
    // wrong upward is caught by nothing until something reads the bytes.
    struct DeviceImage {
        const void* nv12 = nullptr;
        int height = 0;
        int width = 0;
        int pitch = 0;
        //: Bytes from `nv12` to the interleaved chroma plane: `pitch * height`, for a mapped
        //: decoder surface and a tight buffer alike. Carried, not derived -- see above.
        size_t uv_offset = 0;
        //: Which device the pointer belongs to. A frame decoded on GPU 3 is unreadable from a
        //: worker bound to GPU 1, and ADR-002 says one thread never touches another's memory,
        //: so the consumer checks rather than assumes.
        int device = -1;
        // Kept alive for as long as this frame exists, the way `HostFrame::owner` is -- and
        // here it is what UNMAPS the decoder's surface, because NVDEC hands out a slot from a
        // small pool and reuses it as soon as it is released. Dropping this early does not
        // free the pixels, it lets the next frame overwrite them.
        std::shared_ptr<const void> owner;

        bool empty() const {
            // `device < 0` for the same reason as `pitch < width`: a half-filled surface. The
            // field's own default is -1, and a decoder that forgot to set it would otherwise
            // reach a consumer that compares an ordinal against its own bound GPU, or calls
            // `cudaSetDevice(-1)`. ADR-002 says the consumer checks; this is what it checks.
            return nv12 == nullptr || height <= 0 || width <= 0 || pitch < width ||
                   device < 0 ||
                   uv_offset < static_cast<size_t>(pitch) * static_cast<size_t>(height);
        }
    };

    // A tagged frame: what a source produces and a sink consumes.
    //
    // EXACTLY ONE of `image` and `device` is populated. A source decodes to host memory or to
    // VRAM for its whole life, never both, and `FrameSource::read` latches which on the first
    // frame and refuses a change -- see `ingest/base.cpp`.
    struct Frame {
        FrameTag tag;
        HostFrame image;
        DeviceImage device;

        bool on_device() const { return !device.empty(); }
    };

    // Stamps decoded images with a monotonic, per-camera frame id and both clocks.
    //
    // Not thread-safe, and deliberately so: exactly one thread — the camera's own actor — ever
    // stamps a given camera's frames, so a lock here would be pure cost. One actor per camera
    // for the actor's whole life is what makes that true (ADR-002).
    class FrameCounter {
      public:
        explicit FrameCounter(std::string camera_id, int64_t start_at = 0)
            : camera_id_(std::move(camera_id)), next_(start_at) {
            if (start_at < 0) throw ConfigError("first_frame_id must be >= 0");
        }

        const std::string& camera_id() const { return camera_id_; }
        // The id the next `stamp` will use.
        int64_t next_frame_id() const { return next_; }
        // doc: long the latch lives here because it has to outlive one connection
        // Remember, then enforce, whether this CAMERA's pixels arrive on the device.
        //
        // ON THE COUNTER, not on the source, and the source's own header says why: "a source
        // is not expected to survive an error. The actor throws it away and builds a new one,
        // which is why the reconnect state -- the backoff, the frame counter, the stop
        // signal -- lives outside it". A latch inside a source resets on every reconnect, and
        // A RECONNECT IS THE WHOLE SCENARIO: the stream hiccups, the new source finds the
        // hardware decoder busy, falls back to software, and the graph moves onto the host
        // path silently. Which is exactly what this exists to refuse.
        //
        // Throws ConfigError naming both answers. `CameraActor::pump` treats that as fatal for
        // the camera rather than reconnecting, because retrying a contract violation is a hot
        // loop around a bug.
        void latch_where(bool on_device) {
            if (!where_latched_) {
                where_latched_ = true;
                reads_device_ = on_device;
                return;
            }
            if (reads_device_ == on_device) return;
            throw ConfigError(
                "camera '" + camera_id_ + "': a source answered from do_read" +
                (on_device ? "_device" : "") + " after this camera had answered from do_read" +
                (reads_device_ ? "_device" : "") +
                "; where a camera's pixels live is a property of the camera, and the chain is "
                "built once from it -- most plausibly a reconnect fell back to software "
                "decode");
        }
        // Whether this camera has answered yet, and from where. FOR A TEST -- not for a
        // report: this class is documented not thread-safe, so a reader off the manager thread
        // would be racing the actor that stamps.
        bool where_latched() const { return where_latched_; }
        bool reads_device() const { return reads_device_; }
        // How many frames this counter has stamped, across every reconnect.
        uint64_t stamped() const { return stamped_; }

        // Wrap `image` in a `Frame` and advance.
        //
        // **Both clocks are read here**, at the moment of decode, because this is the last
        // place that knows when the frame actually existed. A timestamp taken later measures
        // the queue, not the camera.
        Frame stamp(HostFrame image) {
            Frame frame = tagged();
            frame.image = std::move(image);
            return frame;
        }

        // The same stamp for a frame that never left the device. One counter for both, because
        // the `(camera_id, frame_id)` key ADR-002 relies on does not care where the pixels are.
        Frame stamp(DeviceImage image) {
            Frame frame = tagged();
            frame.device = std::move(image);
            return frame;
        }

      private:
        // The tag and the advance, shared by both `stamp` overloads so the two cannot drift on
        // which clocks they read or when the id moves.
        Frame tagged() {
            Frame frame;
            frame.tag.camera_id = camera_id_;
            frame.tag.frame_id = next_;
            frame.tag.captured_ns = monotonic_ns();
            frame.tag.captured_unix_ns = unix_ns();
            ++next_;
            ++stamped_;
            return frame;
        }

        std::string camera_id_;
        int64_t next_ = 0;
        uint64_t stamped_ = 0;
        bool where_latched_ = false;
        bool reads_device_ = false;
    };

}  // namespace shipinfer
