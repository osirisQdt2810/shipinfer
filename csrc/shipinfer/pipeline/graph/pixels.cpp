#include "shipinfer/pipeline/graph/pixels.h"

namespace shipinfer {

    namespace {

        // Neither representation present. A stage's `needs` is `FRAME_INPUT`, and the planner
        // only runs it once `available()` says so, so this is a wiring fault rather than data:
        // named as one, because a black frame is what the alternative looks like from outside.
        [[noreturn]] void no_pixels(const FrameState& state, const char* what) {
            throw ConfigError("frame " + state.tag().key() + ": cannot " + what +
                              " -- neither a host-uploaded image nor a device surface is "
                              "attached, so the graph was given a frame with no pixels");
        }

    }  // namespace

    LetterboxMap letterbox_frame(const FrameState& state, float* dst_device, int dst_h,
                                 int dst_w, bool swap_rb, float pad_value, gpuStream_t stream) {
        const DeviceSurface& surface = state.surface();
        if (!surface.empty()) {
            return nv12_letterbox_into(surface.nv12, state.height(), state.width(),
                                       surface.stride, surface.uv_offset, dst_device, dst_h,
                                       dst_w, swap_rb, pad_value, stream);
        }
        if (state.image() == nullptr) no_pixels(state, "letterbox a frame");
        return letterbox_into(state.image()->as<uint8_t>(), state.height(), state.width(),
                              dst_device, dst_h, dst_w, swap_rb, pad_value, stream);
    }

    void crop_frame(const FrameState& state, const float* boxes_device, int count,
                    float* dst_device, int dst_h, int dst_w, bool swap_rb, gpuStream_t stream) {
        const DeviceSurface& surface = state.surface();
        if (!surface.empty()) {
            nv12_crop_resize_into(surface.nv12, state.height(), state.width(), surface.stride,
                                  surface.uv_offset, boxes_device, count, dst_device, dst_h,
                                  dst_w, swap_rb, stream);
            return;
        }
        if (state.image() == nullptr) no_pixels(state, "crop a frame");
        crop_resize_into(state.image()->as<uint8_t>(), state.height(), state.width(),
                         boxes_device, count, dst_device, dst_h, dst_w, swap_rb, stream);
    }

}  // namespace shipinfer
