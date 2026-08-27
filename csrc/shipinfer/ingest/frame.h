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

    // A tagged frame: what a source produces and a sink consumes.
    struct Frame {
        FrameTag tag;
        HostFrame image;
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
        // How many frames this counter has stamped, across every reconnect.
        uint64_t stamped() const { return stamped_; }

        // Wrap `image` in a `Frame` and advance.
        //
        // **Both clocks are read here**, at the moment of decode, because this is the last
        // place that knows when the frame actually existed. A timestamp taken later measures
        // the queue, not the camera.
        Frame stamp(HostFrame image) {
            Frame frame;
            frame.tag.camera_id = camera_id_;
            frame.tag.frame_id = next_;
            frame.tag.captured_ns = monotonic_ns();
            frame.tag.captured_unix_ns = unix_ns();
            frame.image = std::move(image);
            ++next_;
            ++stamped_;
            return frame;
        }

      private:
        std::string camera_id_;
        int64_t next_ = 0;
        uint64_t stamped_ = 0;
    };

}  // namespace shipinfer
