// THE ONE PLACE A STAGE'S PIXELS ARE TOLD APART, so no stage has to ask.
//
// A frame reaches the graph in one of two representations, and which one is a property of the
// camera's decoder rather than of the chain: an OpenCV or GStreamer source hands over host BGR
// that the worker uploads, and the NVDEC source hands over an NV12 surface that never left VRAM
// (V156's route). Both are pixels on one device; both support exactly two operations, because
// the graph only ever letterboxes a whole frame or cuts boxes out of it.
//
// So the branch lives here, twice, instead of in `DetectStage` and `CropStage` and in whatever
// stage needs pixels next. A third representation -- P016 for a 10-bit camera, or a decoder
// that hands back RGBA -- is then one file's problem, and cannot arrive in the graph
// half-wired.
#pragma once

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/runtime/ops.h"

namespace shipinfer {

    // `state`'s whole frame, letterboxed into one float NCHW row of `dst_device`.
    //
    // Returns what was applied, which the caller stores on the state: the decode must undo
    // exactly this transform, and the geometry is the same either way -- `letterbox_fit` of the
    // display extent -- so a chain cannot see which decoder a camera used.
    LetterboxMap letterbox_frame(const FrameState& state, float* dst_device, int dst_h,
                                 int dst_w, bool swap_rb, float pad_value, gpuStream_t stream);

    // `count` boxes out of `state`'s frame, each resized into a float NCHW row.
    //
    // `boxes_device` is x1,y1,x2,y2 in FRAME pixels, as the detector's decoded output is. Both
    // implementations clip then truncate and give a degenerate box a black crop, which is why
    // the caller needs no case of its own for either.
    void crop_frame(const FrameState& state, const float* boxes_device, int count,
                    float* dst_device, int dst_h, int dst_w, bool swap_rb, gpuStream_t stream);

}  // namespace shipinfer
