// The per-frame and per-object GPU work: letterbox, and crop-resize.
//
// These are the two kernels the whole measurement turns on. In the Python implementation the
// equivalent work went through torch or numpy per frame, and the pure-Python glue around it is
// what held the GIL and capped the process at five cores.
//
// Both write **directly into a TensorRT input binding**, so a frame's pixels go
// host -> device once and are never copied back. `_into` rather than a returning form, for
// exactly the reason `runtime/ops/base.py` gives on the Python side.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "shipinfer/core/types.hpp"

namespace shipinfer {

// Letterbox one uint8 HWC BGR image already on the device into a float NCHW row.
//
// Aspect ratio preserved, padded to `dst_h x dst_w` with `pad_value`, scaled by `1/255`, and
// optionally BGR->RGB. The scale and the two offsets are returned so the caller can map a
// detection in model space back to original pixels — a detour through the letterboxed
// coordinates is where off-by-a-pad-bar bugs live.
struct LetterboxMap {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

LetterboxMap letterbox_into(const uint8_t* src_device, int src_h, int src_w, float* dst_device,
                            int dst_h, int dst_w, bool swap_rb, float pad_value,
                            cudaStream_t stream);

// Crop `boxes` from one uint8 HWC BGR device image and resize each into a float NCHW row of
// `dst_device`. One kernel launch for the whole frame's objects, which is the difference
// between 15 launches and 1 at 15 objects a frame and 1000 frames a second.
//
// `boxes` is x1,y1,x2,y2 in *original* pixels, four floats per object. A degenerate box
// yields a black crop rather than a launch failure: a zero-area detection is data, not a bug,
// and the parity test on the Python side pins the same behaviour.
void crop_resize_into(const uint8_t* src_device, int src_h, int src_w, const float* boxes_device,
                      int count, float* dst_device, int dst_h, int dst_w, bool swap_rb,
                      cudaStream_t stream);

// NV12 (as NVDEC and the RTSP path produce) straight to a letterboxed float NCHW row, without
// an intermediate BGR image. The Python path could not do this without a host round trip, and
// at 1000 frames a second a 1080p BGR temporary is 6 MB of pure waste per frame.
LetterboxMap nv12_letterbox_into(const uint8_t* nv12_device, int src_h, int src_w, int stride,
                                 float* dst_device, int dst_h, int dst_w, bool swap_rb,
                                 float pad_value, cudaStream_t stream);

}  // namespace shipinfer
