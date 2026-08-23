// The pybind11 surface: `shipinfer._C`.
//
// THE BOUNDARY THIS FILE EXISTS TO HOLD
//
// `include/` and `src/` are a plain C++/CUDA library: transformations and kernels that run
// on a CPU or a GPU and have never heard of Python. This file is the only one that knows
// an interpreter exists, and therefore the only one that may touch the GIL.
//
// Within it the rule is one transition per public entry point, in one shape:
//
//     1. GIL HELD    — read numpy buffers, validate, compute geometry, allocate results.
//                      Everything that needs Python happens here and produces plain PODs.
//     2. GIL RELEASED — one scoped release around the whole compute. The code inside must
//                      not touch a py:: object; it works from the PODs prepared above.
//     3. GIL HELD    — wrap the result.
//
// The previous version released the GIL in five places across nested helpers, so one call
// crossed the boundary three times and a reader could not tell from a helper's body whether
// the lock was held. Scattered releases are how a `py::` access ends up on the wrong side of
// one, and that failure is a hard interpreter crash rather than an exception.
//
// Two entry-point forms per operation:
//
//     letterbox_into(ptr)  -> writes into a caller-owned device buffer. The production path.
//     letterbox_batch(...) -> returns numpy, paying a device-to-host copy. Convenience and
//                             parity testing.
//
// Preprocessing feeds an engine on the same device, so round-tripping its output through
// host memory would undo most of what the fused kernel saved (ADR-007).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <vector>

#include "shipinfer/buffers.hpp"
#include "shipinfer/image_ops.hpp"

namespace py = pybind11;
using shipinfer::ImageView;
using shipinfer::NormalizeParams;
using shipinfer::StagingRing;
using shipinfer::StagingSlot;

namespace {

  using U8Array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;
  using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

  NormalizeParams make_params(const std::vector<float>& mean, const std::vector<float>& std,
                              bool swap_rb) {
    if (mean.size() != 3 || std.size() != 3) {
      throw std::invalid_argument("mean and std must each have three entries");
    }
    NormalizeParams params;
    for (int i = 0; i < 3; ++i) {
      if (std[i] == 0.f)
        throw std::invalid_argument("normalisation std must be non-zero");
      params.mean[i] = mean[i];
      params.std[i] = std[i];
    }
    params.swap_rb = swap_rb;
    return params;
  }

  /// One source frame, resolved to a raw pointer and its letterbox geometry.
  ///
  /// Produced while the GIL is held and consumed after it is released — which is the only
  /// reason it exists. It deliberately holds no py:: object: the numpy arrays are kept
  /// alive by the caller's argument vector for the duration of the call.
  struct FramePlan {
      const void* data;
      int height;
      int width;
      size_t bytes;
      float scale;
      int pad_x;
      int pad_y;
      int out_h;
      int out_w;
  };

  /// Fused pre/post-processing bound to one device.
  class ImageOps {
    public:
      explicit ImageOps(int device_index) : device_index_(device_index) {
        shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
      }

      int device_index() const { return device_index_; }

      py::dict scratch_bytes() const {
        py::dict out;
        out["staging_ring"] = ring_.bytes();
        out["pinned_download"] = pinned_download_.capacity();
        out["output"] = output_.capacity();
        out["nms"] = nms_boxes_.capacity() + nms_mask_.capacity();
        return out;
      }

      // -- letterbox -------------------------------------------------------------------

      /// Preprocess into a caller-owned device buffer. The production path.
      py::tuple letterbox_into(const std::vector<U8Array>& images, uintptr_t out_ptr,
                               size_t out_bytes, int dst_h, int dst_w,
                               const std::vector<float>& mean, const std::vector<float>& std,
                               bool swap_rb, int pad_value, uintptr_t stream_handle) {
        auto scales = py::array_t<float>(static_cast<py::ssize_t>(images.size()));
        auto pads = py::array_t<float>(
            {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(2)});
        const auto params = make_params(mean, std, swap_rb);
        const auto plans = plan_frames(images, dst_h, dst_w, out_bytes, scales.mutable_data(),
                                       pads.mutable_data());
        {
          py::gil_scoped_release release;
          run_letterbox(plans, reinterpret_cast<float*>(out_ptr), dst_h, dst_w, params,
                        static_cast<unsigned char>(pad_value),
                        reinterpret_cast<gpuStream_t>(stream_handle));
        }
        return py::make_tuple(scales, pads);
      }

      /// Preprocess and bring the result back to the host. Convenience and parity testing.
      py::tuple letterbox_batch(const std::vector<U8Array>& images, int dst_h, int dst_w,
                                const std::vector<float>& mean, const std::vector<float>& std,
                                bool swap_rb, int pad_value, uintptr_t stream_handle) {
        const size_t elems = images.size() * 3 * static_cast<size_t>(dst_h) * dst_w;
        const size_t bytes = elems * sizeof(float);

        auto result = py::array_t<float>(
            {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(3),
             static_cast<py::ssize_t>(dst_h), static_cast<py::ssize_t>(dst_w)});
        auto scales = py::array_t<float>(static_cast<py::ssize_t>(images.size()));
        auto pads = py::array_t<float>(
            {static_cast<py::ssize_t>(images.size()), static_cast<py::ssize_t>(2)});
        const auto params = make_params(mean, std, swap_rb);
        const auto plans = plan_frames(images, dst_h, dst_w, bytes, scales.mutable_data(),
                                       pads.mutable_data());
        auto* host_out = result.mutable_data();
        {
          py::gil_scoped_release release;
          const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
          auto* device_out = static_cast<float*>(output_.reserve(bytes));
          run_letterbox(plans, device_out, dst_h, dst_w, params,
                        static_cast<unsigned char>(pad_value), stream);
          download(device_out, host_out, bytes, stream);
        }
        return py::make_tuple(result, scales, pads);
      }

      // -- crops -----------------------------------------------------------------------

      void crop_into(const U8Array& image, const F32Array& boxes, uintptr_t out_ptr,
                     size_t out_bytes, int dst_h, int dst_w, const std::vector<float>& mean,
                     const std::vector<float>& std, bool swap_rb, uintptr_t stream_handle) {
        const auto params = make_params(mean, std, swap_rb);
        const auto plan = plan_crop(image, boxes, dst_h, dst_w, out_bytes);
        if (plan.num_boxes == 0)
          return;
        {
          py::gil_scoped_release release;
          run_crop(plan, reinterpret_cast<float*>(out_ptr), dst_h, dst_w, params,
                   reinterpret_cast<gpuStream_t>(stream_handle));
        }
      }

      F32Array crop_batch(const U8Array& image, const F32Array& boxes, int dst_h, int dst_w,
                          const std::vector<float>& mean, const std::vector<float>& std,
                          bool swap_rb, uintptr_t stream_handle) {
        const auto params = make_params(mean, std, swap_rb);
        const auto plan = plan_crop(image, boxes, dst_h, dst_w, /*out_bytes=*/SIZE_MAX);

        auto result = py::array_t<float>(
            {static_cast<py::ssize_t>(plan.num_boxes), static_cast<py::ssize_t>(3),
             static_cast<py::ssize_t>(dst_h), static_cast<py::ssize_t>(dst_w)});
        if (plan.num_boxes == 0)
          return result;

        const size_t bytes =
            static_cast<size_t>(plan.num_boxes) * 3 * dst_h * dst_w * sizeof(float);
        auto* host_out = result.mutable_data();
        {
          py::gil_scoped_release release;
          const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
          auto* device_out = static_cast<float*>(output_.reserve(bytes));
          run_crop(plan, device_out, dst_h, dst_w, params, stream);
          download(device_out, host_out, bytes, stream);
        }
        return result;
      }

      // -- nms -------------------------------------------------------------------------

      py::array_t<int64_t> nms(const F32Array& boxes, const F32Array& scores,
                               float iou_threshold, float score_threshold, int max_output,
                               uintptr_t stream_handle) {
        const auto box_info = boxes.request();
        const auto score_info = scores.request();
        if (box_info.ndim != 2 || box_info.shape[1] != 4) {
          throw std::invalid_argument("boxes must be (N, 4) float32");
        }
        if (score_info.ndim != 1 || score_info.shape[0] != box_info.shape[0]) {
          throw std::invalid_argument("scores must be (N,) float32 matching boxes");
        }
        const int n = static_cast<int>(box_info.shape[0]);
        const auto* box_data = static_cast<const float*>(box_info.ptr);
        const auto* score_data = static_cast<const float*>(score_info.ptr);

        std::vector<int64_t> kept;
        {
          py::gil_scoped_release release;
          kept = run_nms(box_data, score_data, n, iou_threshold, score_threshold, max_output,
                         reinterpret_cast<gpuStream_t>(stream_handle));
        }

        auto result = py::array_t<int64_t>(static_cast<py::ssize_t>(kept.size()));
        if (!kept.empty()) {
          std::memcpy(result.mutable_data(), kept.data(), kept.size() * sizeof(int64_t));
        }
        return result;
      }

    private:
      // ================================================================================
      // GIL HELD. These read Python objects and produce plain PODs.
      // ================================================================================

      /// Resolve a ragged batch to raw pointers plus its letterbox geometry.
      ///
      /// The geometry is three multiplies per image and is done here rather than in the
      /// kernel: it keeps the kernel free of divergence, and the scales and pads have to
      /// reach numpy anyway so that post-processing can invert the letterbox with exactly
      /// the numbers that applied it.
      std::vector<FramePlan> plan_frames(const std::vector<U8Array>& images, int dst_h,
                                         int dst_w, size_t out_bytes, float* scales_out,
                                         float* pads_out) const {
        if (images.empty())
          throw std::invalid_argument("letterbox needs at least one image");
        const size_t required =
            images.size() * 3 * static_cast<size_t>(dst_h) * dst_w * sizeof(float);
        if (out_bytes < required) {
          throw std::invalid_argument("output buffer is too small for this batch");
        }

        std::vector<FramePlan> plans;
        plans.reserve(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
          const auto info = images[i].request();
          if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::invalid_argument("each image must be (H, W, 3) uint8");
          }
          const int h = static_cast<int>(info.shape[0]);
          const int w = static_cast<int>(info.shape[1]);
          const float scale =
              std::min(static_cast<float>(dst_h) / h, static_cast<float>(dst_w) / w);
          const int out_h = std::max(1, static_cast<int>(lroundf(h * scale)));
          const int out_w = std::max(1, static_cast<int>(lroundf(w * scale)));
          const int pad_y = (dst_h - out_h) / 2;
          const int pad_x = (dst_w - out_w) / 2;

          plans.push_back(FramePlan{info.ptr, h, w, static_cast<size_t>(h) * w * 3, scale,
                                    pad_x, pad_y, out_h, out_w});
          scales_out[i] = scale;
          pads_out[i * 2 + 0] = static_cast<float>(pad_x);
          pads_out[i * 2 + 1] = static_cast<float>(pad_y);
        }
        return plans;
      }

      struct CropPlan {
          const void* frame;
          const void* boxes;
          int height;
          int width;
          int num_boxes;
      };

      CropPlan plan_crop(const U8Array& image, const F32Array& boxes, int dst_h, int dst_w,
                         size_t out_bytes) const {
        const auto image_info = image.request();
        const auto box_info = boxes.request();
        if (image_info.ndim != 3 || image_info.shape[2] != 3) {
          throw std::invalid_argument("image must be (H, W, 3) uint8");
        }
        if (box_info.ndim != 2 || box_info.shape[1] != 4) {
          throw std::invalid_argument("boxes must be (N, 4) float32");
        }
        const int num_boxes = static_cast<int>(box_info.shape[0]);
        const size_t required =
            static_cast<size_t>(num_boxes) * 3 * dst_h * dst_w * sizeof(float);
        if (out_bytes != SIZE_MAX && out_bytes < required) {
          throw std::invalid_argument("output buffer is too small");
        }
        return CropPlan{image_info.ptr, box_info.ptr, static_cast<int>(image_info.shape[0]),
                        static_cast<int>(image_info.shape[1]), num_boxes};
      }

      // ================================================================================
      // GIL RELEASED. Nothing below may touch a py:: object.
      // ================================================================================

      void run_letterbox(const std::vector<FramePlan>& plans, float* out, int dst_h, int dst_w,
                         const NormalizeParams& params, unsigned char pad_value,
                         gpuStream_t stream) {
        shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");

        // A slot the previous few batches have finished with. Rotating is what makes the
        // asynchronous path safe: with one set of buffers, this call's host memcpy would
        // overwrite pinned bytes a previous DMA is still reading, and its upload would
        // overwrite device frames a previous kernel is still sampling. Both produce
        // plausible output, and neither is visible to a benchmark that submits the same
        // image twice.
        auto& slot = ring_.acquire();

        size_t total = 0;
        for (const auto& plan : plans)
          total += plan.bytes;

        auto* pinned = slot.pinned().reserve(total);
        auto* device = static_cast<unsigned char*>(slot.frames().reserve(total));

        views_.clear();
        views_.reserve(plans.size());
        size_t offset = 0;
        for (const auto& plan : plans) {
          std::memcpy(pinned + offset, plan.data, plan.bytes);
          views_.push_back(ImageView{device + offset, plan.height, plan.width, plan.scale,
                                     plan.pad_x, plan.pad_y, plan.out_h, plan.out_w});
          offset += plan.bytes;
        }

        // One transfer, not one per frame: eight 6 MB copies cost eight driver round
        // trips and eight chances to serialise the stream.
        shipinfer::check(gpuMemcpyAsync(device, pinned, total, gpuMemcpyHostToDevice, stream),
                         "upload frames");

        auto* device_views =
            static_cast<ImageView*>(slot.views().reserve(views_.size() * sizeof(ImageView)));
        shipinfer::check(gpuMemcpyAsync(device_views, views_.data(),
                                        views_.size() * sizeof(ImageView),
                                        gpuMemcpyHostToDevice, stream),
                         "upload image views");

        shipinfer::letterbox_batch(device_views, static_cast<int>(plans.size()), out, dst_h,
                                   dst_w, params, pad_value, stream);
        // Not a synchronise: it marks when this slot's buffers stop being read, so the
        // rotation can reuse them without anybody waiting here.
        slot.record(stream);
      }

      void run_crop(const CropPlan& plan, float* out, int dst_h, int dst_w,
                    const NormalizeParams& params, gpuStream_t stream) {
        shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");

        const size_t frame_bytes = static_cast<size_t>(plan.height) * plan.width * 3;
        const size_t box_bytes = static_cast<size_t>(plan.num_boxes) * 4 * sizeof(float);
        // The boxes share one allocation with the frame, and h*w*3 is a multiple of 4
        // only by luck — a 1079x1919 frame is 6,211,803 bytes, so a float* placed
        // straight after it is misaligned. CUDA then raises cudaErrorMisalignedAddress,
        // which is STICKY: it poisons the context for the life of the process, so one
        // odd-sized camera would kill that worker permanently.
        const size_t box_offset = shipinfer::align_up(frame_bytes);
        const size_t packed = box_offset + box_bytes;

        auto& slot = ring_.acquire();
        auto* pinned = slot.pinned().reserve(packed);
        std::memcpy(pinned, plan.frame, frame_bytes);
        std::memcpy(pinned + box_offset, plan.boxes, box_bytes);

        auto* device = static_cast<unsigned char*>(slot.frames().reserve(packed));
        shipinfer::check(gpuMemcpyAsync(device, pinned, packed, gpuMemcpyHostToDevice, stream),
                         "upload frame and boxes");

        const ImageView view{device, plan.height, plan.width,  1.f,
                             0,      0,           plan.height, plan.width};
        shipinfer::crop_batch(view, reinterpret_cast<const float*>(device + box_offset),
                              plan.num_boxes, out, dst_h, dst_w, params, stream);
        slot.record(stream);
      }

      std::vector<int64_t> run_nms(const float* boxes, const float* scores, int n,
                                   float iou_threshold, float score_threshold, int max_output,
                                   gpuStream_t stream) {
        shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
        const size_t mask_words = shipinfer::nms_mask_words(n);

        shipinfer::NmsScratch scratch;
        scratch.box_floats = static_cast<size_t>(n) * 4;
        scratch.mask_words = mask_words;
        scratch.boxes =
            static_cast<float*>(nms_boxes_.reserve(scratch.box_floats * sizeof(float)));
        scratch.mask = static_cast<unsigned long long*>(
            nms_mask_.reserve(mask_words * sizeof(unsigned long long)));

        return shipinfer::nms(boxes, scores, n, iou_threshold, score_threshold, max_output,
                              scratch, stream);
      }

      /// Device to host, through pinned memory.
      ///
      /// A direct copy into a freshly allocated numpy array is startlingly slow: the
      /// destination is pageable *and* untouched, so the driver stages it through a small
      /// bounce buffer while the kernel faults in tens of megabytes of new pages. Going
      /// through pinned memory and then doing a plain memcpy is an order of magnitude
      /// faster, and it is why the convenience entry points are merely slower than the
      /// `_into` ones rather than unusable.
      void download(const void* device_src, void* host_dst, size_t bytes, gpuStream_t stream) {
        auto* staging = pinned_download_.reserve(bytes);
        shipinfer::check(
            gpuMemcpyAsync(staging, device_src, bytes, gpuMemcpyDeviceToHost, stream),
            "download");
        shipinfer::check(gpuStreamSynchronize(stream), "download synchronize");
        std::memcpy(host_dst, staging, bytes);
      }

      int device_index_;
      StagingRing ring_;                ///< rotated per call, so reuse never races a live copy
      shipinfer::DeviceScratch output_; ///< host-returning entry points only
      shipinfer::DeviceScratch nms_boxes_; ///< score-sorted boxes for one NMS call
      shipinfer::DeviceScratch nms_mask_;  ///< the (box, box) overlap bitmask
      shipinfer::PinnedScratch pinned_download_;
      std::vector<ImageView> views_; ///< host-side descriptor table, rebuilt per call
  };

} // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "shipinfer native data plane: fused preprocessing and device-side NMS";
  m.attr("__version__") = "0.1.0";

#if defined(SHIPINFER_WITH_HIP)
  m.attr("platform") = "hip";
#else
  m.attr("platform") = "cuda";
#endif

  py::register_exception<shipinfer::GpuError>(m, "GpuError", PyExc_RuntimeError);

  m.def("cuda_available", &shipinfer::gpu_available,
        "True when this build has GPU kernels and a visible device.");
  m.def("device_count", &shipinfer::device_count, "Number of visible devices.");

  py::class_<ImageOps>(m, "ImageOps", "Fused pre/post-processing kernels bound to one device.")
      .def(py::init<int>(), py::arg("device_index") = 0)
      .def_property_readonly("device_index", &ImageOps::device_index)
      .def("scratch_bytes", &ImageOps::scratch_bytes,
           "Persistent scratch held by this instance, in bytes.")
      .def("letterbox_batch", &ImageOps::letterbox_batch, py::arg("images"), py::arg("dst_h"),
           py::arg("dst_w"), py::arg("mean"), py::arg("std"), py::arg("swap_rb"),
           py::arg("pad_value") = 114, py::arg("stream") = 0,
           "Fused resize+pad+convert+normalise+NCHW, returned as numpy.")
      .def("letterbox_into", &ImageOps::letterbox_into, py::arg("images"), py::arg("out_ptr"),
           py::arg("out_bytes"), py::arg("dst_h"), py::arg("dst_w"), py::arg("mean"),
           py::arg("std"), py::arg("swap_rb"), py::arg("pad_value") = 114,
           py::arg("stream") = 0,
           "Same, written straight into a caller-owned device buffer. The fast "
           "path.")
      .def("crop_batch", &ImageOps::crop_batch, py::arg("image"), py::arg("boxes"),
           py::arg("dst_h"), py::arg("dst_w"), py::arg("mean"), py::arg("std"),
           py::arg("swap_rb"), py::arg("stream") = 0,
           "Extract and resize N boxes, returned as numpy.")
      .def("crop_into", &ImageOps::crop_into, py::arg("image"), py::arg("boxes"),
           py::arg("out_ptr"), py::arg("out_bytes"), py::arg("dst_h"), py::arg("dst_w"),
           py::arg("mean"), py::arg("std"), py::arg("swap_rb"), py::arg("stream") = 0,
           "Same, written straight into a caller-owned device buffer.")
      .def("nms", &ImageOps::nms, py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"),
           py::arg("score_threshold"), py::arg("max_output"), py::arg("stream") = 0,
           "Class-agnostic NMS on the device; returns kept indices.");
}
