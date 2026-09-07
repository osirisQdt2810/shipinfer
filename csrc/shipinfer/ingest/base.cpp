#include "shipinfer/ingest/base.h"

#include <utility>

namespace shipinfer {

    FrameSource::FrameSource(IngestConfig config, FrameCounter& counter, StopSignal& stop)
        : config_(std::move(config)), counter_(counter), stop_(stop) {
        if (counter_.camera_id() != config_.camera_id) {
            // A source stamping another camera's frames would produce a `(camera_id,
            // frame_id)` key that already belongs to somebody else — the one thing ADR-002
            // relies on never happening — so it is refused at construction rather than
            // discovered downstream.
            throw ConfigError("frame counter belongs to camera '" + counter_.camera_id() +
                              "', not '" + config_.camera_id + "'");
        }
    }

    void FrameSource::open() {
        if (is_open_) return;
        try {
            do_open();
        } catch (...) {
            // Unwind unconditionally and best-effort. The suppression is the point: the
            // original failure is the one worth propagating, and a `do_close` that throws while
            // cleaning up after a failed open would otherwise replace a useful diagnosis
            // ("cannot open rtsp://…: connection refused") with a useless one.
            try {
                do_close();
            } catch (...) {  // NOLINT — deliberate, see above
            }
            throw;
        }
        is_open_ = true;
    }

    std::optional<Frame> FrameSource::read() {
        if (!is_open_) {
            throw SourceOpenError(config_.camera_id, config_.uri, "read() before open()");
        }
        // Device first, because a source that has one answers only there; `do_read_device`
        // defaults to nothing, so this costs one branch for every host source.
        //
        // The latch is the COUNTER's, so it survives the reconnect that motivates it -- see
        // `FrameCounter::latch_where`.
        if (std::optional<DeviceImage> on_device = do_read_device()) {
            // CHECKED, not documented. `Frame::on_device()` is DEFINED as `!device.empty()`,
            // so an engaged-but-invalid surface was not rejected -- it was silently
            // reclassified as a HOST frame with a null pixel pointer, and `Frame`'s own
            // "exactly one of `image` and `device` is populated" became zero. Worse, round 2
            // adding `device < 0` to `empty()` WIDENED the set that got laundered that way.
            // Downstream reads as healthy: `pump()` resets the backoff and publishes, and the
            // detect stage letterboxes a 0x0 image while the fleet reports 50 streaming
            // (#153 round 3).
            if (on_device->empty()) {
                throw ConfigError(
                    "camera '" + config_.camera_id +
                    "': the decoder returned an incomplete device surface -- it needs a "
                    "pointer, a positive height and width, a pitch >= the width, a device "
                    "index >= 0, and a uv_offset at or past the end of the luma plane, which "
                    "is `pitch * height` for every producer here (see `ingest/frame.h`). A "
                    "decoder that forgot one will forget it on the rebuilt source too, so this "
                    "is fatal for the camera rather than a reconnect");
            }
            counter_.latch_where(true);
            return counter_.stamp(std::move(*on_device));
        }
        std::optional<HostFrame> image = do_read();
        if (!image) return std::nullopt;
        counter_.latch_where(false);
        return counter_.stamp(std::move(*image));
    }

    void FrameSource::close() {
        if (!is_open_) return;
        is_open_ = false;
        do_close();
    }

    void FrameSource::set_format(int height, int width, double fps) {
        height_ = height;
        width_ = width;
        fps_ = fps;
    }

}  // namespace shipinfer
