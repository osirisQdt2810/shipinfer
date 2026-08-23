// Fused pre-processing and device-side NMS.
//
// These are the operations that justify a custom kernel at all. Everything else
// this project needs from a GPU — allocation, streams, resize on its own,
// matrix multiply — torch already does better than hand-written code would
// (ADR-003). What torch cannot do is run four memory-bound passes as one.

#pragma once

#include <cstdint>
#include <vector>

#include "shipinfer/platform.hpp"

namespace shipinfer {

  /// Mean/std normalisation in the source pixel scale (0-255), plus channel
  /// order.
  struct NormalizeParams {
      float mean[3] = {0.f, 0.f, 0.f};
      float std[3] = {255.f, 255.f, 255.f};
      bool swap_rb = true; ///< OpenCV hands out BGR; most checkpoints want RGB
  };

  /// Where a letterboxed image ended up, so post-processing can invert it
  /// exactly.
  ///
  /// Returned rather than recomputed downstream because box coordinates must be
  /// un-mapped with the *same* numbers that mapped them. Recomputing from the
  /// shapes is where off-by-one box drift comes from.
  struct LetterboxGeometry {
      float scale;
      float pad_x;
      float pad_y;
  };

  /// Batched letterbox: resize + pad + colour convert + normalise + NHWC->NCHW,
  /// in one pass.
  ///
  /// The fusion is the point. Each of those four steps is memory-bound, so
  /// running them separately reads and writes a 1080p frame four times for a
  /// result identical to reading and writing it once. One thread per *output*
  /// pixel, gathering from the source, means the expensive tensor is touched
  /// exactly once.
  ///
  /// Takes the descriptor table as a device pointer the CALLER owns, and neither
  /// allocates nor synchronises. An earlier version allocated that table per call, which
  /// forced a gpuStreamSynchronize before the temporary died — so every "async" call
  /// blocked until the kernel retired and batch n+1's upload could never overlap batch
  /// n's compute. A fast path that waits is not a fast path.
  ///
  /// @param views_device device array of `batch` ImageView, uploaded by the caller
  /// @param batch        number of images
  /// @param out          device buffer of at least batch * 3 * dst_h * dst_w floats
  /// @param dst_h,dst_w  model input extent
  /// @param pad_value    fill for the letterbox bars (114 by YOLO convention)
  void letterbox_batch(const ImageView* views_device, int batch, float* out, int dst_h,
                       int dst_w, const NormalizeParams& params, unsigned char pad_value,
                       gpuStream_t stream);

  /// Extract N boxes from one frame and resize each into a normalised NCHW
  /// tensor.
  ///
  /// The embedding stage's hot path, and the reason detect->crop stays on one
  /// GPU: the frame is megabytes, the crops are kilobytes, and only the crops
  /// need to travel (ADR-004).
  ///
  /// @param boxes  device array of N * 4 floats, [x1, y1, x2, y2] in frame pixels
  void crop_batch(const ImageView& frame, const float* boxes, int num_boxes, float* out,
                  int dst_h, int dst_w, const NormalizeParams& params, gpuStream_t stream);

  /// Class-agnostic NMS on the device; returns the kept indices, highest score
  /// first.
  ///
  /// On the device because the numbers say so: 25 000 candidate boxes is ~800 KB
  /// that never needs to cross PCIe when 20 survive. Copying them back to filter
  /// on the host is the most common self-inflicted bottleneck in this kind of
  /// pipeline.
  ///
  /// Uses the standard block-bitmask formulation: one bit per (box, box) pair
  /// packed into 64-bit words, so the O(n^2) overlap test runs entirely in
  /// parallel and only the tiny bitmask crosses back for the sequential sweep.
  /// This one DOES synchronise, and legitimately: the sweep over the bitmask is
  /// inherently sequential and runs on the host, so the mask must arrive first. What it
  /// no longer does is allocate — the scratch belongs to the caller and is reused.
  struct NmsScratch {
      float* boxes = nullptr;             ///< >= num_boxes * 4 floats, on the device
      unsigned long long* mask = nullptr; ///< >= nms_mask_words(num_boxes) words
      size_t box_floats = 0;
      size_t mask_words = 0;
  };

  /// Bitmask words needed for `num_boxes`, so a caller can size its scratch.
  size_t nms_mask_words(int num_boxes);

  std::vector<int64_t> nms(const float* boxes_host, const float* scores_host, int num_boxes,
                           float iou_threshold, float score_threshold, int max_output,
                           const NmsScratch& scratch, gpuStream_t stream);

  /// True when the build has GPU kernels compiled in.
  bool gpu_available();

  /// Number of visible devices, or 0.
  int device_count();

} // namespace shipinfer
