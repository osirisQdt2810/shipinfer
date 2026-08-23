// The pybind11 surface: `shipinfer._C`.
//
// Two entry points per operation, and the difference between them is the whole
// point:
//
//   letterbox_batch(...)       -> numpy.  Convenient, and it pays a
//   device-to-host copy. letterbox_batch_into(ptr)  -> nothing. Writes straight
//   into a caller-owned device
//                                 buffer (in practice a torch CUDA tensor), so
//                                 the result never leaves the GPU on its way to
//                                 the model.
//
// The `_into` form is the one the pipeline uses. Preprocessing exists to feed
// an engine that lives on the same device; round-tripping its output through
// host memory would undo most of what the fused kernel saved (ADR-007).
//
// Scratch buffers live on the `ImageOps` object rather than per call, because
// `cudaMalloc` and `cudaFree` are synchronising and a pageable copy runs at
// half bandwidth.
//
// The GIL is released around every launch: without that, one thread in the
// preprocess would block every other worker in the process — exactly the
// bottleneck this removes.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <vector>

#include "shipinfer/buffers.hpp"
#include "shipinfer/image_ops.hpp"

namespace py = pybind11;
using shipinfer::DeviceScratch;
using shipinfer::ImageView;
using shipinfer::NormalizeParams;
using shipinfer::PinnedScratch;

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

  /// Fused pre/post-processing bound to one device, holding its own scratch.
  class ImageOps {
    public:
      explicit ImageOps(int device_index) : device_index_(device_index) {
        shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
      }

      int device_index() const { return device_index_; }

      py::dict scratch_bytes() const {
        py::dict out;
        out["frames"] = frames_.capacity();
        out["pinned"] = pinned_.capacity();
        out["pinned_download"] = pinned_download_.capacity();
        out["views"] = views_.capacity();
        out["output"] = output_.capacity();
        return out;
      }

      // -- letterbox
      // ------------------------------------------------------------------------

      /// Preprocess into a caller-owned device buffer. The fast path.
      py::tuple letterbox_into(const std::vector<U8Array>& images, uintptr_t out_ptr,
                               size_t out_bytes, int dst_h, int dst_w,
                               const std::vector<float>& mean, const std::vector<float>& std,
                               bool swap_rb, int pad_value, uintptr_t stream_handle) {
        const int batch = static_cast<int>(images.size());
        if (batch == 0)
          throw std::invalid_argument("letterbox needs at least one image");
        const size_t required = static_cast<size_t>(batch) * 3 * dst_h * dst_w * sizeof(float);
        if (out_bytes < required) {
          throw std::invalid_argument("output buffer is too small for this batch");
        }

        auto scales = py::array_t<float>(batch);
        auto pads = py::array_t<float>({batch, 2});
        stage_frames(images, dst_h, dst_w, scales.mutable_data(), pads.mutable_data());

        const auto params = make_params(mean, std, swap_rb);
        const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
        {
          py::gil_scoped_release release;
          shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
          shipinfer::letterbox_batch(views_host_, reinterpret_cast<float*>(out_ptr), dst_h,
                                     dst_w, params, static_cast<unsigned char>(pad_value),
                                     stream);
        }
        return py::make_tuple(scales, pads);
      }

      /// Preprocess and bring the result back to the host. Convenience and parity
      /// testing.
      py::tuple letterbox_batch(const std::vector<U8Array>& images, int dst_h, int dst_w,
                                const std::vector<float>& mean, const std::vector<float>& std,
                                bool swap_rb, int pad_value, uintptr_t stream_handle) {
        const int batch = static_cast<int>(images.size());
        if (batch == 0)
          throw std::invalid_argument("letterbox needs at least one image");
        const size_t elems = static_cast<size_t>(batch) * 3 * dst_h * dst_w;

        auto result = py::array_t<float>({batch, 3, dst_h, dst_w});
        auto* device_out = static_cast<float*>(output_.reserve(elems * sizeof(float)));
        auto extras = letterbox_into(images, reinterpret_cast<uintptr_t>(device_out),
                                     elems * sizeof(float), dst_h, dst_w, mean, std, swap_rb,
                                     pad_value, stream_handle);
        download(device_out, result.mutable_data(), elems * sizeof(float), stream_handle);
        return py::make_tuple(result, extras[0], extras[1]);
      }

      // -- crops
      // ----------------------------------------------------------------------------

      void crop_into(const U8Array& image, const F32Array& boxes, uintptr_t out_ptr,
                     size_t out_bytes, int dst_h, int dst_w, const std::vector<float>& mean,
                     const std::vector<float>& std, bool swap_rb, uintptr_t stream_handle) {
        const auto image_info = image.request();
        const auto box_info = boxes.request();
        if (image_info.ndim != 3 || image_info.shape[2] != 3) {
          throw std::invalid_argument("image must be (H, W, 3) uint8");
        }
        if (box_info.ndim != 2 || box_info.shape[1] != 4) {
          throw std::invalid_argument("boxes must be (N, 4) float32");
        }
        const int num_boxes = static_cast<int>(box_info.shape[0]);
        if (num_boxes == 0)
          return;

        const size_t required =
            static_cast<size_t>(num_boxes) * 3 * dst_h * dst_w * sizeof(float);
        if (out_bytes < required)
          throw std::invalid_argument("output buffer is too small");

        const int h = static_cast<int>(image_info.shape[0]);
        const int w = static_cast<int>(image_info.shape[1]);
        const size_t frame_bytes = static_cast<size_t>(h) * w * 3;
        const size_t box_bytes = static_cast<size_t>(num_boxes) * 4 * sizeof(float);

        auto* pinned = pinned_.reserve(frame_bytes + box_bytes);
        std::memcpy(pinned, image_info.ptr, frame_bytes);
        std::memcpy(pinned + frame_bytes, box_info.ptr, box_bytes);

        auto* device = static_cast<unsigned char*>(frames_.reserve(frame_bytes + box_bytes));
        const auto params = make_params(mean, std, swap_rb);
        const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
        {
          py::gil_scoped_release release;
          shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
          shipinfer::check(gpuMemcpyAsync(device, pinned, frame_bytes + box_bytes,
                                          gpuMemcpyHostToDevice, stream),
                           "upload frame and boxes");
          const ImageView view{device, h, w, 1.f, 0, 0, h, w};
          shipinfer::crop_batch(view, reinterpret_cast<const float*>(device + frame_bytes),
                                num_boxes, reinterpret_cast<float*>(out_ptr), dst_h, dst_w,
                                params, stream);
          shipinfer::check(gpuStreamSynchronize(stream), "crop synchronize");
        }
      }

      F32Array crop_batch(const U8Array& image, const F32Array& boxes, int dst_h, int dst_w,
                          const std::vector<float>& mean, const std::vector<float>& std,
                          bool swap_rb, uintptr_t stream_handle) {
        const auto box_info = boxes.request();
        if (box_info.ndim != 2 || box_info.shape[1] != 4) {
          throw std::invalid_argument("boxes must be (N, 4) float32");
        }
        const int num_boxes = static_cast<int>(box_info.shape[0]);
        auto result = py::array_t<float>({num_boxes, 3, dst_h, dst_w});
        if (num_boxes == 0)
          return result;

        const size_t elems = static_cast<size_t>(num_boxes) * 3 * dst_h * dst_w;
        auto* device_out = static_cast<float*>(output_.reserve(elems * sizeof(float)));
        crop_into(image, boxes, reinterpret_cast<uintptr_t>(device_out), elems * sizeof(float),
                  dst_h, dst_w, mean, std, swap_rb, stream_handle);
        download(device_out, result.mutable_data(), elems * sizeof(float), stream_handle);
        return result;
      }

      // -- nms
      // ------------------------------------------------------------------------------

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

        std::vector<int64_t> kept;
        {
          py::gil_scoped_release release;
          shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
          kept = shipinfer::nms(static_cast<const float*>(box_info.ptr),
                                static_cast<const float*>(score_info.ptr), n, iou_threshold,
                                score_threshold, max_output,
                                reinterpret_cast<gpuStream_t>(stream_handle));
        }
        auto result = py::array_t<int64_t>(static_cast<py::ssize_t>(kept.size()));
        if (!kept.empty()) {
          std::memcpy(result.mutable_data(), kept.data(), kept.size() * sizeof(int64_t));
        }
        return result;
      }

    private:
      /// Device-to-host through the pinned staging block.
      ///
      /// A direct DeviceToHost copy into a freshly allocated numpy array is
      /// startlingly slow: the destination is pageable *and* untouched, so the
      /// driver stages it through a small internal bounce buffer while the kernel
      /// faults in 39 MB of new pages. Going through pinned memory and then doing a
      /// plain `memcpy` is an order of magnitude faster, and it is why the
      /// convenience entry points are merely slower than the `_into` ones rather
      /// than unusable.
      void download(const void* device_src, void* host_dst, size_t bytes,
                    uintptr_t stream_handle) {
        auto* staging = pinned_download_.reserve(bytes);
        {
          py::gil_scoped_release release;
          const auto stream = reinterpret_cast<gpuStream_t>(stream_handle);
          shipinfer::check(
              gpuMemcpyAsync(staging, device_src, bytes, gpuMemcpyDeviceToHost, stream),
              "download");
          shipinfer::check(gpuStreamSynchronize(stream), "download synchronize");
          std::memcpy(host_dst, staging, bytes);
        }
      }

      /// Pack every frame into one pinned staging block, upload it in a single
      /// transfer, and build the device-side descriptor table.
      ///
      /// One copy instead of N. Eight separate 6 MB transfers cost eight driver
      /// round trips and eight chances to serialise the stream; one 50 MB transfer
      /// costs one.
      void stage_frames(const std::vector<U8Array>& images, int dst_h, int dst_w,
                        float* scales_out, float* pads_out) {
        const int batch = static_cast<int>(images.size());
        std::vector<py::buffer_info> infos;
        infos.reserve(batch);
        size_t total = 0;
        for (const auto& image : images) {
          auto info = image.request();
          if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::invalid_argument("each image must be (H, W, 3) uint8");
          }
          total += static_cast<size_t>(info.shape[0]) * info.shape[1] * 3;
          infos.push_back(std::move(info));
        }

        auto* pinned = pinned_.reserve(total);
        auto* device = static_cast<unsigned char*>(frames_.reserve(total));

        views_host_.clear();
        views_host_.reserve(batch);
        size_t offset = 0;
        for (int i = 0; i < batch; ++i) {
          const auto& info = infos[i];
          const int h = static_cast<int>(info.shape[0]);
          const int w = static_cast<int>(info.shape[1]);
          const size_t bytes = static_cast<size_t>(h) * w * 3;
          std::memcpy(pinned + offset, info.ptr, bytes);

          const float scale =
              std::min(static_cast<float>(dst_h) / h, static_cast<float>(dst_w) / w);
          const int out_h = std::max(1, static_cast<int>(lroundf(h * scale)));
          const int out_w = std::max(1, static_cast<int>(lroundf(w * scale)));
          const int pad_y = (dst_h - out_h) / 2;
          const int pad_x = (dst_w - out_w) / 2;

          views_host_.push_back(
              ImageView{device + offset, h, w, scale, pad_x, pad_y, out_h, out_w});
          scales_out[i] = scale;
          pads_out[i * 2 + 0] = static_cast<float>(pad_x);
          pads_out[i * 2 + 1] = static_cast<float>(pad_y);
          offset += bytes;
        }

        py::gil_scoped_release release;
        shipinfer::check(gpuSetDevice(device_index_), "gpuSetDevice");
        shipinfer::check(gpuMemcpy(device, pinned, total, gpuMemcpyHostToDevice),
                         "upload frames");
      }

      int device_index_;
      DeviceScratch frames_; ///< uploaded source frames
      DeviceScratch views_;  ///< reserved for the descriptor table (kernel-side)
      DeviceScratch output_; ///< only used by the host-returning convenience entry points
      PinnedScratch pinned_; ///< host staging for uploads
      PinnedScratch pinned_download_; ///< host staging for the convenience downloads
      std::vector<ImageView> views_host_;
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
