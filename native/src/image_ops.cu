#include <algorithm>
#include <cstring>
#include <numeric>

#include "shipinfer/image_ops.hpp"

namespace shipinfer {
  namespace {

    /// Bilinear sample from an HWC uint8 image, clamped at the border.
    ///
    /// `__forceinline__` because it is called once per output pixel per channel: at
    /// 1000 fps x 640x640 that is ~1.2 billion calls a second, and a real call
    /// would dominate.
    __device__ __forceinline__ float sample_bilinear(const unsigned char* img, int h, int w,
                                                     float y, float x, int channel) {
      const int x0 = static_cast<int>(floorf(x));
      const int y0 = static_cast<int>(floorf(y));
      const int x1 = min(x0 + 1, w - 1);
      const int y1 = min(y0 + 1, h - 1);
      const int xc = max(x0, 0);
      const int yc = max(y0, 0);
      const float wx = x - static_cast<float>(x0);
      const float wy = y - static_cast<float>(y0);

      const float v00 = img[(yc * w + xc) * 3 + channel];
      const float v01 = img[(yc * w + x1) * 3 + channel];
      const float v10 = img[(y1 * w + xc) * 3 + channel];
      const float v11 = img[(y1 * w + x1) * 3 + channel];
      return (v00 * (1.f - wx) + v01 * wx) * (1.f - wy) + (v10 * (1.f - wx) + v11 * wx) * wy;
    }

    /// One thread per output pixel; each writes all three channels.
    ///
    /// Writing three channels per thread rather than launching 3x the threads keeps
    /// the source coordinate arithmetic (a division and two multiplies) done once
    /// instead of three times, and the three destination writes are 4 bytes each
    /// into three separate channel planes — which is the price of NCHW and is paid
    /// either way.
    __global__ void letterbox_kernel(const ImageView* views, int batch, float* __restrict__ out,
                                     int dst_h, int dst_w, float m0, float m1, float m2,
                                     float s0, float s1, float s2, bool swap_rb,
                                     float pad_value) {
      const int total = batch * dst_h * dst_w;
      for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
           idx += blockDim.x * gridDim.x) {
        const int x = idx % dst_w;
        const int y = (idx / dst_w) % dst_h;
        const int n = idx / (dst_w * dst_h);

        const ImageView view = views[n];
        const int plane = dst_h * dst_w;
        float* dst = out + static_cast<size_t>(n) * 3 * plane + y * dst_w + x;

        const int local_x = x - view.pad_x;
        const int local_y = y - view.pad_y;

        float v[3];
        if (local_x < 0 || local_y < 0 || local_x >= view.out_w || local_y >= view.out_h) {
          v[0] = v[1] = v[2] = pad_value;
        } else {
          // Half-pixel centres: the same convention as OpenCV and torch's
          // `align_corners=False`, so a model trained with either preprocessing
          // sees the pixels it expects.
          const float src_y =
              (static_cast<float>(local_y) + 0.5f) * view.height / view.out_h - 0.5f;
          const float src_x =
              (static_cast<float>(local_x) + 0.5f) * view.width / view.out_w - 0.5f;
          v[0] = sample_bilinear(view.data, view.height, view.width, src_y, src_x, 0);
          v[1] = sample_bilinear(view.data, view.height, view.width, src_y, src_x, 1);
          v[2] = sample_bilinear(view.data, view.height, view.width, src_y, src_x, 2);
        }

        const int c0 = swap_rb ? 2 : 0;
        const int c2 = swap_rb ? 0 : 2;
        dst[0 * plane] = (v[c0] - m0) / s0;
        dst[1 * plane] = (v[1] - m1) / s1;
        dst[2 * plane] = (v[c2] - m2) / s2;
      }
    }

    __global__ void crop_kernel(ImageView frame, const float* __restrict__ boxes, int num_boxes,
                                float* __restrict__ out, int dst_h, int dst_w, float m0,
                                float m1, float m2, float s0, float s1, float s2,
                                bool swap_rb) {
      const int total = num_boxes * dst_h * dst_w;
      for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
           idx += blockDim.x * gridDim.x) {
        const int x = idx % dst_w;
        const int y = (idx / dst_w) % dst_h;
        const int n = idx / (dst_w * dst_h);

        const float x1 =
            fminf(fmaxf(boxes[n * 4 + 0], 0.f), static_cast<float>(frame.width - 1));
        const float y1 =
            fminf(fmaxf(boxes[n * 4 + 1], 0.f), static_cast<float>(frame.height - 1));
        const float x2 =
            fminf(fmaxf(boxes[n * 4 + 2], 0.f), static_cast<float>(frame.width - 1));
        const float y2 =
            fminf(fmaxf(boxes[n * 4 + 3], 0.f), static_cast<float>(frame.height - 1));

        const int plane = dst_h * dst_w;
        float* dst = out + static_cast<size_t>(n) * 3 * plane + y * dst_w + x;

        float v[3] = {0.f, 0.f, 0.f};
        // A degenerate box yields a black crop rather than an out-of-bounds read.
        // Detectors do emit them, and killing a whole batch over one bad box is the
        // wrong trade.
        if (x2 > x1 && y2 > y1) {
          const float src_y = y1 + (static_cast<float>(y) + 0.5f) * (y2 - y1) / dst_h - 0.5f;
          const float src_x = x1 + (static_cast<float>(x) + 0.5f) * (x2 - x1) / dst_w - 0.5f;
          v[0] = sample_bilinear(frame.data, frame.height, frame.width, src_y, src_x, 0);
          v[1] = sample_bilinear(frame.data, frame.height, frame.width, src_y, src_x, 1);
          v[2] = sample_bilinear(frame.data, frame.height, frame.width, src_y, src_x, 2);
        }

        const int c0 = swap_rb ? 2 : 0;
        const int c2 = swap_rb ? 0 : 2;
        dst[0 * plane] = (v[c0] - m0) / s0;
        dst[1 * plane] = (v[1] - m1) / s1;
        dst[2 * plane] = (v[c2] - m2) / s2;
      }
    }

    constexpr int kNmsBlock = 64; // one 64-bit mask word per block of candidates

    __device__ __forceinline__ float iou(const float* a, const float* b) {
      const float inter_w = fmaxf(0.f, fminf(a[2], b[2]) - fmaxf(a[0], b[0]));
      const float inter_h = fmaxf(0.f, fminf(a[3], b[3]) - fmaxf(a[1], b[1]));
      const float inter = inter_w * inter_h;
      const float area_a = fmaxf(0.f, a[2] - a[0]) * fmaxf(0.f, a[3] - a[1]);
      const float area_b = fmaxf(0.f, b[2] - b[0]) * fmaxf(0.f, b[3] - b[1]);
      return inter / fmaxf(area_a + area_b - inter, 1e-9f);
    }

    /// Overlap bitmask: bit (i, j) is set when box i suppresses box j.
    ///
    /// The standard formulation, and the reason it is standard: the O(n^2) IoU work
    /// is embarrassingly parallel while the *decision* is inherently sequential.
    /// Computing the mask on the device and sweeping it on the host moves 25 000
    /// boxes' worth of comparisons onto the GPU and brings back only n^2/64 words.
    __global__ void nms_mask_kernel(const float* __restrict__ boxes, int n, float threshold,
                                    unsigned long long* __restrict__ mask) {
      const int row_start = blockIdx.y;
      const int col_start = blockIdx.x;
      const int row_size = min(n - row_start * kNmsBlock, kNmsBlock);
      const int col_size = min(n - col_start * kNmsBlock, kNmsBlock);

      __shared__ float block_boxes[kNmsBlock * 4];
      if (threadIdx.x < col_size) {
        const int src = (col_start * kNmsBlock + threadIdx.x) * 4;
        block_boxes[threadIdx.x * 4 + 0] = boxes[src + 0];
        block_boxes[threadIdx.x * 4 + 1] = boxes[src + 1];
        block_boxes[threadIdx.x * 4 + 2] = boxes[src + 2];
        block_boxes[threadIdx.x * 4 + 3] = boxes[src + 3];
      }
      __syncthreads();

      if (threadIdx.x < row_size) {
        const int row = row_start * kNmsBlock + threadIdx.x;
        const float* current = boxes + row * 4;
        unsigned long long bits = 0ULL;
        const int start = (row_start == col_start) ? threadIdx.x + 1 : 0;
        for (int i = start; i < col_size; ++i) {
          if (iou(current, block_boxes + i * 4) > threshold) {
            bits |= 1ULL << i;
          }
        }
        mask[row * gridDim.x + col_start] = bits;
      }
    }

    /// RAII for a scratch device allocation.
    ///
    /// Scoped because these are per-call temporaries. Anything with a lifetime
    /// longer than one call belongs to torch's caching allocator on the Python
    /// side, not here.
    template <typename T> class DeviceBuffer {
      public:
        explicit DeviceBuffer(size_t count) : count_(count) {
          if (count_ > 0)
            check(gpuMalloc(&ptr_, count_ * sizeof(T)), "gpuMalloc");
        }
        ~DeviceBuffer() {
          if (ptr_ != nullptr)
            gpuFree(ptr_);
        }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* get() { return ptr_; }
        const T* get() const { return ptr_; }
        size_t size() const { return count_; }

      private:
        T* ptr_ = nullptr;
        size_t count_ = 0;
    };

  } // namespace

  void letterbox_batch(const std::vector<ImageView>& images, float* out, int dst_h, int dst_w,
                       const NormalizeParams& params, unsigned char pad_value,
                       gpuStream_t stream) {
    if (images.empty())
      return;
    const int batch = static_cast<int>(images.size());

    DeviceBuffer<ImageView> views(images.size());
    check(gpuMemcpyAsync(views.get(), images.data(), images.size() * sizeof(ImageView),
                         gpuMemcpyHostToDevice, stream),
          "upload image views");

    const int total = batch * dst_h * dst_w;
    const int blocks = std::min(ceil_div(total, kBlockSize), 65535);
    letterbox_kernel<<<blocks, kBlockSize, 0, stream>>>(
        views.get(), batch, out, dst_h, dst_w, params.mean[0], params.mean[1], params.mean[2],
        params.std[0], params.std[1], params.std[2], params.swap_rb,
        static_cast<float>(pad_value));
    check_launch("letterbox_kernel");
    // The descriptor buffer dies at the end of this scope, so the copy that reads
    // it must have completed. This is the one place a synchronise is not
    // avoidable overhead.
    check(gpuStreamSynchronize(stream), "letterbox synchronize");
  }

  void crop_batch(const ImageView& frame, const float* boxes, int num_boxes, float* out,
                  int dst_h, int dst_w, const NormalizeParams& params, gpuStream_t stream) {
    if (num_boxes <= 0)
      return;
    const int total = num_boxes * dst_h * dst_w;
    const int blocks = std::min(ceil_div(total, kBlockSize), 65535);
    crop_kernel<<<blocks, kBlockSize, 0, stream>>>(
        frame, boxes, num_boxes, out, dst_h, dst_w, params.mean[0], params.mean[1],
        params.mean[2], params.std[0], params.std[1], params.std[2], params.swap_rb);
    check_launch("crop_kernel");
  }

  std::vector<int64_t> nms(const float* boxes_host, const float* scores_host, int num_boxes,
                           float iou_threshold, float score_threshold, int max_output,
                           gpuStream_t stream) {
    std::vector<int64_t> order;
    order.reserve(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {
      if (scores_host[i] >= score_threshold)
        order.push_back(i);
    }
    if (order.empty())
      return {};

    // Stable sort so equal scores keep input order: an unstable sort makes the
    // same input produce different output between runs, which turns a tracking
    // regression into a heisenbug.
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return scores_host[a] > scores_host[b]; });

    const int n = static_cast<int>(order.size());
    std::vector<float> sorted(static_cast<size_t>(n) * 4);
    for (int i = 0; i < n; ++i) {
      std::memcpy(&sorted[static_cast<size_t>(i) * 4], &boxes_host[order[i] * 4],
                  4 * sizeof(float));
    }

    DeviceBuffer<float> d_boxes(sorted.size());
    check(gpuMemcpyAsync(d_boxes.get(), sorted.data(), sorted.size() * sizeof(float),
                         gpuMemcpyHostToDevice, stream),
          "upload nms boxes");

    const int col_blocks = ceil_div(n, kNmsBlock);
    DeviceBuffer<unsigned long long> d_mask(static_cast<size_t>(n) * col_blocks);
    dim3 grid(col_blocks, col_blocks);
    nms_mask_kernel<<<grid, kNmsBlock, 0, stream>>>(d_boxes.get(), n, iou_threshold,
                                                    d_mask.get());
    check_launch("nms_mask_kernel");

    std::vector<unsigned long long> mask(static_cast<size_t>(n) * col_blocks);
    check(gpuMemcpyAsync(mask.data(), d_mask.get(), mask.size() * sizeof(unsigned long long),
                         gpuMemcpyDeviceToHost, stream),
          "download nms mask");
    check(gpuStreamSynchronize(stream), "nms synchronize");

    std::vector<unsigned long long> removed(col_blocks, 0ULL);
    std::vector<int64_t> keep;
    keep.reserve(std::min(n, max_output));
    for (int i = 0; i < n && static_cast<int>(keep.size()) < max_output; ++i) {
      if (removed[i / kNmsBlock] & (1ULL << (i % kNmsBlock)))
        continue;
      keep.push_back(order[i]);
      const unsigned long long* row = mask.data() + static_cast<size_t>(i) * col_blocks;
      for (int j = i / kNmsBlock; j < col_blocks; ++j)
        removed[j] |= row[j];
    }
    return keep;
  }

  bool gpu_available() {
    return device_count() > 0;
  }

  int device_count() {
    int count = 0;
    if (gpuGetDeviceCount(&count) != gpuSuccess)
      return 0;
    return count;
  }

} // namespace shipinfer
