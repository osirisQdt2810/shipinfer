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

#include <cstdint>
#include <vector>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    // Letterbox one uint8 HWC BGR image already on the device into a float NCHW row.
    //
    // Aspect ratio preserved, padded to `dst_h x dst_w` with `pad_value`, scaled by `1/255`,
    // and optionally BGR->RGB. The scale and the two offsets are returned so the caller can map
    // a detection in model space back to original pixels — a detour through the letterboxed
    // coordinates is where off-by-a-pad-bar bugs live.
    // How a source frame was placed on the canvas — `TorchImageOps._letterbox`'s arithmetic:
    // `scale = min(dst/src)`, the resized extent is `max(1, round(src * scale))` (round half to
    // even, as Python's `round`), and the bars are `(dst - new) // 2`. Detections map back with
    // `(x - pad) / scale`, on both planes.
    struct LetterboxMap {
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        int new_w = 0;  // the resized image's extent inside the canvas
        int new_h = 0;
    };

    // The placement alone, for callers and tests that need the geometry without a launch.
    LetterboxMap letterbox_fit(int src_h, int src_w, int dst_h, int dst_w);

    LetterboxMap letterbox_into(const uint8_t* src_device, int src_h, int src_w,
                                float* dst_device, int dst_h, int dst_w, bool swap_rb,
                                float pad_value, gpuStream_t stream);

    // Crop `boxes` from one uint8 HWC BGR device image and resize each into a float NCHW row of
    // `dst_device`. One kernel launch for the whole frame's objects, which is the difference
    // between 15 launches and 1 at 15 objects a frame and 1000 frames a second.
    //
    // `boxes` is x1,y1,x2,y2 in *original* pixels, four floats per object. A degenerate box
    // yields a black crop rather than a launch failure: a zero-area detection is data, not a
    // bug, and the parity test on the Python side pins the same behaviour.
    void crop_resize_into(const uint8_t* src_device, int src_h, int src_w,
                          const float* boxes_device, int count, float* dst_device, int dst_h,
                          int dst_w, bool swap_rb, gpuStream_t stream);

    // doc: long the UV offset, and the padded surface that made it a parameter
    // NV12 (as NVDEC and the RTSP path produce) straight to a letterboxed float NCHW row,
    // without an intermediate BGR image. The Python path could not do this without a host round
    // trip, and at 1000 frames a second a 1080p BGR temporary is 6 MB of pure waste per frame.
    //
    // `uv_offset` is where the interleaved chroma plane starts, in BYTES from `nv12_device`. A
    // PARAMETER and not a derivation, because the CALLER knows and this kernel cannot: the
    // plane's offset is a property of the buffer it was decoded into, and a kernel that guessed
    // it would read luma rows as chroma -- correct brightness, wrong colour, on every frame,
    // looking exactly like a model problem. `stride * src_h` for every producer in this tree;
    // `ingest/frame.h` states that once, with the measurement behind it.
    LetterboxMap nv12_letterbox_into(const uint8_t* nv12_device, int src_h, int src_w,
                                     int stride, size_t uv_offset, float* dst_device, int dst_h,
                                     int dst_w, bool swap_rb, float pad_value,
                                     gpuStream_t stream);

    // doc: long the crop's NV12 twin, and why converting once would be worse
    // `crop_resize_into`'s NV12 twin: N boxes out of a decoder surface, each resized into a
    // normalised NCHW row, with no BGR image in between.
    //
    // WITHOUT THIS the NV12 path stops at the detector. `crop_resize_into` indexes an HWC BGR
    // array, so a graph fed an NVDEC surface would have to convert the whole frame once --
    // ~6 MB of device temporary per 1080p frame, 6 GB/s at the design load, which is the cost
    // `nv12_letterbox_into` exists to avoid one stage earlier. Sampling NV12 in place costs
    // four chroma loads per output pixel and no allocation at all.
    //
    // Same conventions as the BGR twin, deliberately, so the two cannot disagree about a box:
    // boxes are `[x1, y1, x2, y2]` in FRAME pixels, clipped then truncated; a degenerate box
    // yields a black crop rather than a launch failure; sampling is bilinear with
    // `align_corners=False` in patch coordinates. `uv_offset` is the caller's, as above.
    void nv12_crop_resize_into(const uint8_t* nv12_device, int src_h, int src_w, int stride,
                               size_t uv_offset, const float* boxes_device, int count,
                               float* dst_device, int dst_h, int dst_w, bool swap_rb,
                               gpuStream_t stream);

    // doc: long what the fold is, why it belongs here, and what it costs on the host today
    // ONE AREA PER CROP, from a segmentation engine's two outputs, without them leaving the
    // device. `graph/mask_area.cpp` is the readable twin and stays: it is what the offline tier
    // and the cross-plane golden check, and a kernel is only trustworthy if a readable
    // implementation agrees with it.
    //
    // WHY IT IS WORTH A KERNEL, profiled 11 Sep on the whole chain: the prototype bank is
    // `(32, 160, 160)` floats -- 3.1 MB per crop -- and `TrtEngine::execute` copies it to the
    // host so a loop can reduce it to ONE float. That copy is the bulk of the run's 39.6 GiB of
    // device-to-host traffic (73.7% of all GPU memory-op time), and the loop costs 1.44 ms of
    // CPU per crop, which one core sustains 693 of.
    //
    // The arithmetic is `mask_area`'s, kept identical on purpose: the strongest candidate row
    // per crop by score, its `coefficients` mask weights against the prototype planes, a
    // comparison against the LOGIT of `mask_threshold` (so no sigmoid is computed), and the
    // count of cells above it scaled by the crop's pixels per cell. A crop whose best row
    // scores below `score_threshold` is area 0, which is "found nothing" rather than an error.
    //
    // One block per crop, threads striding the cells, a block reduction at the end.
    // `areas_device` holds `count` floats.
    //
    // THE ROW LAYOUT IT ASSUMES, because two of these are not parameters: column 4 of a
    // detection row is its SCORE, and `stride - prefix` must EQUAL `channels` -- the kernel
    // reads `channels` coefficients starting at `prefix`, so a shorter row would read the next
    // candidate's box and, for the last crop, walk off the allocation. Refused rather than
    // trusted, with the same reason `mask_area.cpp` gives.
    void mask_area_into(const float* rows_device, int candidates, int stride, int prefix,
                        const float* protos_device, int channels, int cells, int count,
                        float score_threshold, float mask_threshold, int crop_height,
                        int crop_width, float* areas_device, gpuStream_t stream);

}  // namespace shipinfer
