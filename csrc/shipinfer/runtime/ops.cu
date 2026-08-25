// Letterbox, crop-resize and NV12 letterbox. Bilinear, one launch per frame.
//
// These are the kernels the whole port turns on, so the arithmetic is written out rather than
// hidden: a resize that is subtly different from the Python reference produces detections that
// are subtly wrong, and nothing downstream would notice.

#include <algorithm>
#include <cmath>

#include "shipinfer/core/platform.h"
#include "shipinfer/runtime/ops.h"

namespace shipinfer {
    namespace {

        // Bilinear sample of a uint8 HWC BGR image, clamped at the edges. `__forceinline__`
        // because this is the innermost operation of the innermost loop of the whole system.
        __device__ __forceinline__ float sample_bilinear(const uint8_t* src, int src_h,
                                                         int src_w, int channel, float y,
                                                         float x) {
            const int x0 = max(0, min(src_w - 1, static_cast<int>(floorf(x))));
            const int y0 = max(0, min(src_h - 1, static_cast<int>(floorf(y))));
            const int x1 = min(src_w - 1, x0 + 1);
            const int y1 = min(src_h - 1, y0 + 1);
            const float wx = x - static_cast<float>(x0);
            const float wy = y - static_cast<float>(y0);

            const float p00 = static_cast<float>(src[(y0 * src_w + x0) * 3 + channel]);
            const float p01 = static_cast<float>(src[(y0 * src_w + x1) * 3 + channel]);
            const float p10 = static_cast<float>(src[(y1 * src_w + x0) * 3 + channel]);
            const float p11 = static_cast<float>(src[(y1 * src_w + x1) * 3 + channel]);
            return (p00 * (1.f - wx) + p01 * wx) * (1.f - wy) +
                   (p10 * (1.f - wx) + p11 * wx) * wy;
        }

        // `TorchImageOps._letterbox`: the frame is resized to `(new_h, new_w)` with
        // `interpolate(align_corners=False)` and pasted at `(pad_y, pad_x)`; everything else on
        // the canvas is the pad value. align_corners=False maps a destination pixel to the
        // source coordinate `(d + 0.5) * (in / out) - 0.5`, clamped at zero — pixel centres,
        // not corners. The first version sampled `(x - pad) / scale`, which at 1920 -> 640 is a
        // constant one-source-pixel offset across the whole detector input; every box, and so
        // every crop, was shifted against the Python plane's. `test_dataplane.cpp`'s reference
        // now transcribes torch_ops.py rather than this kernel, so a disagreement is visible
        // there.
        __device__ __forceinline__ float source_coordinate(int dst, int pad, int in, int out) {
            const float real = (static_cast<float>(dst - pad) + 0.5f) * static_cast<float>(in) /
                                   static_cast<float>(out) -
                               0.5f;
            return fmaxf(0.f, real);
        }

        __global__ void letterbox_kernel(const uint8_t* src, int src_h, int src_w, float* dst,
                                         int dst_h, int dst_w, int pad_x, int pad_y, int new_w,
                                         int new_h, bool swap_rb, float pad_value) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= dst_w || y >= dst_h) return;

            const int plane = dst_h * dst_w;
            // Inside the padding bars there is no source pixel. Writing the pad value rather
            // than clamping the edge matters: clamping smears the border into the bar and the
            // detector learns to see an object there.
            const bool inside =
                x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h;
            const float sx = source_coordinate(x, pad_x, src_w, new_w);
            const float sy = source_coordinate(y, pad_y, src_h, new_h);

            for (int c = 0; c < 3; ++c) {
                // The source is BGR; `swap_rb` names the *destination* order, which is what the
                // engine's own preprocessing expects. Getting this backwards is invisible until
                // the detections are quietly worse.
                const int src_c = swap_rb ? (2 - c) : c;
                const float value =
                    inside ? sample_bilinear(src, src_h, src_w, src_c, sy, sx) / 255.f
                           : pad_value;
                dst[c * plane + y * dst_w + x] = value;
            }
        }

        // The readable implementation this must agree with is `TorchImageOps.crop_batch`
        // (`src/shipinfer/runtime/ops/torch_ops.py`): clip the box to the image, truncate it to
        // integers, slice the patch `[y1:y2, x1:x2]`, and resize it with
        // `interpolate(align_corners=False)` — whose source coordinate is
        // `(dst + 0.5) * (in / out) - 0.5`, clamped at zero, with the right/bottom neighbour
        // clamped *inside the patch*. The first version skipped the `- 0.5` and clamped to the
        // image instead of the patch, so every crop reaching the embedder was offset by half a
        // source pixel relative to the Python plane's and the two planes' embeddings were not
        // comparable. `test_dataplane.cpp` carries the same arithmetic as a readable reference.
        __global__ void crop_resize_kernel(const uint8_t* src, int src_h, int src_w,
                                           const float* boxes, int count, float* dst, int dst_h,
                                           int dst_w, bool swap_rb) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const int n = blockIdx.z;
            if (x >= dst_w || y >= dst_h || n >= count) return;

            // Clip to the image, then truncate: `np.clip(...)` into an integer array.
            const int x1 = static_cast<int>(fminf(fmaxf(boxes[n * 4 + 0], 0.f), src_w - 1.f));
            const int y1 = static_cast<int>(fminf(fmaxf(boxes[n * 4 + 1], 0.f), src_h - 1.f));
            const int x2 = static_cast<int>(fminf(fmaxf(boxes[n * 4 + 2], 0.f), src_w - 1.f));
            const int y2 = static_cast<int>(fminf(fmaxf(boxes[n * 4 + 3], 0.f), src_h - 1.f));
            const int box_w = x2 - x1;
            const int box_h = y2 - y1;

            const int plane = dst_h * dst_w;
            float* out = dst + static_cast<size_t>(n) * 3 * plane;

            // A degenerate box is data, not a bug: a zero-area detection can come out of any
            // detector, and the answer is a black crop rather than a launch that reads out of
            // bounds. The Python implementation yields zeros for the same box.
            if (box_w <= 0 || box_h <= 0) {
                for (int c = 0; c < 3; ++c) out[c * plane + y * dst_w + x] = 0.f;
                return;
            }

            // align_corners=False, in patch coordinates, clamped at zero like torch does.
            const float lx =
                fmaxf(0.f, (static_cast<float>(x) + 0.5f) * static_cast<float>(box_w) /
                                   static_cast<float>(dst_w) -
                               0.5f);
            const float ly =
                fmaxf(0.f, (static_cast<float>(y) + 0.5f) * static_cast<float>(box_h) /
                                   static_cast<float>(dst_h) -
                               0.5f);
            const int px0 = min(static_cast<int>(lx), box_w - 1);
            const int py0 = min(static_cast<int>(ly), box_h - 1);
            const int px1 = min(px0 + 1, box_w - 1);
            const int py1 = min(py0 + 1, box_h - 1);
            const float wx = lx - static_cast<float>(px0);
            const float wy = ly - static_cast<float>(py0);
            const int sx0 = x1 + px0, sx1 = x1 + px1, sy0 = y1 + py0, sy1 = y1 + py1;

            for (int c = 0; c < 3; ++c) {
                const int src_c = swap_rb ? (2 - c) : c;
                const float p00 = static_cast<float>(src[(sy0 * src_w + sx0) * 3 + src_c]);
                const float p01 = static_cast<float>(src[(sy0 * src_w + sx1) * 3 + src_c]);
                const float p10 = static_cast<float>(src[(sy1 * src_w + sx0) * 3 + src_c]);
                const float p11 = static_cast<float>(src[(sy1 * src_w + sx1) * 3 + src_c]);
                const float value = (p00 * (1.f - wx) + p01 * wx) * (1.f - wy) +
                                    (p10 * (1.f - wx) + p11 * wx) * wy;
                out[c * plane + y * dst_w + x] = value / 255.f;
            }
        }

        // NV12: a full-resolution Y plane followed by a half-resolution interleaved UV plane.
        // This is what NVDEC hands back, and converting it to BGR first would cost a 6 MB
        // temporary per 1080p frame — at 1000 frames a second that is 6 GB/s of pure waste.
        __device__ __forceinline__ void nv12_to_bgr(const uint8_t* nv12, int src_h, int src_w,
                                                    int stride, float y_f, float x_f,
                                                    float* bgr) {
            const int xi = max(0, min(src_w - 1, static_cast<int>(x_f)));
            const int yi = max(0, min(src_h - 1, static_cast<int>(y_f)));
            const uint8_t* uv = nv12 + static_cast<size_t>(stride) * src_h;

            // BT.601 limited range, which is what H.264 from an IP camera carries.
            const float Y = (static_cast<float>(nv12[yi * stride + xi]) - 16.f) * 1.164383f;
            const int cx = (xi / 2) * 2;
            const int cy = yi / 2;
            const float U = static_cast<float>(uv[cy * stride + cx + 0]) - 128.f;
            const float V = static_cast<float>(uv[cy * stride + cx + 1]) - 128.f;

            bgr[0] = fminf(255.f, fmaxf(0.f, Y + 2.017232f * U));                  // B
            bgr[1] = fminf(255.f, fmaxf(0.f, Y - 0.391762f * U - 0.812968f * V));  // G
            bgr[2] = fminf(255.f, fmaxf(0.f, Y + 1.596027f * V));                  // R
        }

        // Same placement and the same centre mapping as `letterbox_kernel`, sampled nearest:
        // the chroma plane is half resolution and NVDEC's output is what a camera sent. There
        // is no Python twin for NV12 input (the Python plane decodes to BGR first), so the
        // readable reference for this one lives in `test_dataplane.cpp` and says so.
        __global__ void nv12_letterbox_kernel(const uint8_t* nv12, int src_h, int src_w,
                                              int stride, float* dst, int dst_h, int dst_w,
                                              int pad_x, int pad_y, int new_w, int new_h,
                                              bool swap_rb, float pad_value) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= dst_w || y >= dst_h) return;

            const int plane = dst_h * dst_w;
            const bool inside =
                x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h;
            const float sx = source_coordinate(x, pad_x, src_w, new_w);
            const float sy = source_coordinate(y, pad_y, src_h, new_h);

            float bgr[3] = {0.f, 0.f, 0.f};
            if (inside) nv12_to_bgr(nv12, src_h, src_w, stride, sy, sx, bgr);
            for (int c = 0; c < 3; ++c) {
                const int src_c = swap_rb ? (2 - c) : c;
                dst[c * plane + y * dst_w + x] = inside ? bgr[src_c] / 255.f : pad_value;
            }
        }

        LetterboxMap fit(int src_h, int src_w, int dst_h, int dst_w) {
            LetterboxMap map;
            map.scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                                 static_cast<float>(dst_h) / static_cast<float>(src_h));
            // `max(1, round(src * scale))` — Python's `round` is half-to-even, which is what
            // `nearbyint` does under the default rounding mode. The first version truncated,
            // which moved a bar by a pixel for some source sizes.
            map.new_w = std::max(
                1, static_cast<int>(std::nearbyint(static_cast<float>(src_w) * map.scale)));
            map.new_h = std::max(
                1, static_cast<int>(std::nearbyint(static_cast<float>(src_h) * map.scale)));
            map.pad_x = (dst_w - map.new_w) / 2;
            map.pad_y = (dst_h - map.new_h) / 2;
            return map;
        }

    }  // namespace

    LetterboxMap letterbox_fit(int src_h, int src_w, int dst_h, int dst_w) {
        return fit(src_h, src_w, dst_h, dst_w);
    }

    LetterboxMap letterbox_into(const uint8_t* src_device, int src_h, int src_w,
                                float* dst_device, int dst_h, int dst_w, bool swap_rb,
                                float pad_value, gpuStream_t stream) {
        const LetterboxMap map = fit(src_h, src_w, dst_h, dst_w);
        const dim3 block(16, 16);
        const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
        letterbox_kernel<<<grid, block, 0, stream>>>(src_device, src_h, src_w, dst_device,
                                                     dst_h, dst_w, map.pad_x, map.pad_y,
                                                     map.new_w, map.new_h, swap_rb, pad_value);
        GPU_CHECK(gpuGetLastError());
        return map;
    }

    void crop_resize_into(const uint8_t* src_device, int src_h, int src_w,
                          const float* boxes_device, int count, float* dst_device, int dst_h,
                          int dst_w, bool swap_rb, gpuStream_t stream) {
        if (count <= 0) return;
        const dim3 block(16, 16);
        const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y,
                        static_cast<unsigned>(count));
        crop_resize_kernel<<<grid, block, 0, stream>>>(
            src_device, src_h, src_w, boxes_device, count, dst_device, dst_h, dst_w, swap_rb);
        GPU_CHECK(gpuGetLastError());
    }

    LetterboxMap nv12_letterbox_into(const uint8_t* nv12_device, int src_h, int src_w,
                                     int stride, float* dst_device, int dst_h, int dst_w,
                                     bool swap_rb, float pad_value, gpuStream_t stream) {
        const LetterboxMap map = fit(src_h, src_w, dst_h, dst_w);
        const dim3 block(16, 16);
        const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
        nv12_letterbox_kernel<<<grid, block, 0, stream>>>(
            nv12_device, src_h, src_w, stride, dst_device, dst_h, dst_w, map.pad_x, map.pad_y,
            map.new_w, map.new_h, swap_rb, pad_value);
        GPU_CHECK(gpuGetLastError());
        return map;
    }

}  // namespace shipinfer
