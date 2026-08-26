// Tests for the C++ data plane.
//
// No framework: a test binary that returns non-zero and says what failed is the whole contract,
// and adding gtest would mean the build needs a dependency the container does not have.
//
// Two kinds of test here, and the split is deliberate:
//
//  * **Pure** — the fair queue and the collector. No device, so these run anywhere and they
//    cover the invariants this project exists to hold: a quiet camera is not starved by a busy
//    one, and every frame that is opened is reported exactly once.
//  * **Parity** — the CUDA kernels against a readable CPU implementation written directly
//    below them. `CLAUDE.md`: "A fused kernel is only trustworthy if a readable implementation
//    agrees with it". The readable one lives here rather than in Python because the thing being
//    checked is this translation unit's arithmetic.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/pipeline/graph/shapes.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/runtime/ops.h"

using namespace shipinfer;

namespace {

    int failures = 0;
    int checks = 0;
    int skips = 0;

    // A parity test that cannot run must say so and be counted — a device-less run that prints
    // "N checks, 0 failure(s)" and reads as green is a test that fails open.
    void skip(const std::string& why) {
        ++skips;
        std::fprintf(stderr, "SKIP: %s\n", why.c_str());
    }

    // "expected:", because the message states the property that was *supposed* to hold.
    // Printing it bare after "FAIL:" reads as though the good thing happened — review caught
    // exactly that: a regression printed "FAIL: the degenerate box produced a black crop",
    // which is the opposite of the news. Fixing the frame rather than rewording thirty
    // messages keeps every one of them readable as the property it asserts.
    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: expected: %s\n", what.c_str());
        }
    }

    void check_near(double actual, double expected, double tolerance, const std::string& what) {
        ++checks;
        if (std::fabs(actual - expected) > tolerance) {
            ++failures;
            std::fprintf(stderr, "FAIL: expected: %s (got %g, wanted %g +- %g)\n", what.c_str(),
                         actual, expected, tolerance);
        }
    }

    // The queue tests live in test_scheduling.cpp, next to the seam they mirror.

    // -- the fair queue ---------------------------------------------------------------------

    // -- the collector ----------------------------------------------------------------------

    std::shared_ptr<FrameState> a_frame(const std::string& camera, int64_t id) {
        FrameTag tag;
        tag.camera_id = camera;
        tag.frame_id = id;
        return std::make_shared<FrameState>(tag, 8, 8, 0.0f);
    }

    void test_a_frame_with_more_objects_than_the_batch_keeps_every_chunk() {
        // 17 people against an engine built at 16: two chunks. The first version attached a new
        // ObjectBatch per chunk under one name and `attach` assigned, so the frame reached the
        // sink with the last chunk's single row — sealed Complete. The assembly is a pure
        // function now, so this needs no engine and no device.
        const int width = 4, limit = 16;
        std::vector<int> indices;
        for (int i = 0; i < 17; ++i) indices.push_back(i);
        ObjectBatch out;
        out.name = "person_embedder_out";
        for (size_t start = 0; start < indices.size(); start += limit) {
            const int count = static_cast<int>(std::min<size_t>(limit, indices.size() - start));
            std::vector<float> rows(static_cast<size_t>(count) * width);
            for (int r = 0; r < count; ++r)
                rows[static_cast<size_t>(r) * width] = static_cast<float>(start + r);
            out.append(rows.data(), count, width, indices, start);
        }
        check(out.rows() == 17, "every object has a row, not only the last chunk's");
        check(out.data.size() == 17u * width, "the rows of both chunks are kept");
        check(out.object_indices == indices,
              "indices are in the frame's own order across chunks");
        check(out.row(16)[0] == 16.f, "the second chunk's row is the second chunk's data");

        bool refused = false;
        try {
            out.append(std::vector<float>(3).data(), 1, 3, indices, 0);
        } catch (
            const BackendError&) {  // the house vocabulary, caught by the per-frame handler
            refused = true;
        }
        check(refused,
              "a chunk of a different width is a bug, refused rather than concatenated");
    }

    void test_a_plan_whose_shape_disagrees_with_the_config_is_refused_at_construction() {
        TensorSpec seg;
        seg.name = "images";
        seg.dims = {3, 512, 512};
        bool refused = false;
        std::string message;
        try {
            expect_input_row(seg, {3, 640, 640}, "ship_segmenter");
        } catch (const ConfigError& error) {
            refused = true;
            message = error.what();
        }
        check(refused, "a 512 plan fed 640 rows is refused before any device write");
        check(message.find("[3, 512, 512]") != std::string::npos &&
                  message.find("[3, 640, 640]") != std::string::npos,
              "the refusal names both shapes");
        TensorSpec dynamic;
        dynamic.name = "images";
        dynamic.dims = {3, -1, -1};
        bool accepted = true;
        try {
            expect_input_row(dynamic, {3, 640, 640}, "detector");
        } catch (const ConfigError&) {
            accepted = false;
        }
        check(accepted, "a dynamic plan dimension matches what the graph feeds");
    }

    void test_a_plan_with_non_float32_io_is_refused() {
        TensorSpec half;
        half.name = "output0";
        half.element_size = 2;
        bool refused = false;
        try {
            expect_float32(half, "ship_detector");
        } catch (const ConfigError&) {
            refused = true;
        }
        check(refused,
              "an FP16 binding would be reinterpreted as float32 garbage; refused instead");
    }

    void test_every_opened_frame_is_reported_exactly_once() {
        int reported = 0;
        FrameCollector collector([&reported](FrameResult&&) { ++reported; }, 64, 1500);

        for (int i = 0; i < 10; ++i) {
            auto state = a_frame("cam0", i);
            check(collector.open(state, {"detect"}), "opened");
            collector.deliver(state->tag(), "detect");
            collector.seal(state->tag());
        }
        check(reported == 10, "ten opened, ten reported");
        check(collector.pending() == 0, "nothing left pending");
    }

    void test_a_duplicate_tag_is_refused_rather_than_clobbering() {
        FrameCollector collector([](FrameResult&&) {}, 64, 1500);
        auto first = a_frame("cam0", 7);
        auto second = a_frame("cam0", 7);

        check(collector.open(first, {"detect"}), "the first is opened");
        check(!collector.open(second, {"detect"}),
              "the duplicate is refused, so the first frame's caller is not stranded");
    }

    void test_a_timed_out_frame_is_still_reported() {
        int reported = 0;
        FinishReason reason = FinishReason::Complete;
        FrameCollector collector(
            [&](FrameResult&& result) {
                ++reported;
                reason = result.reason;
            },
            64, /*timeout_ms=*/1);

        auto state = a_frame("cam0", 0);
        collector.open(state, {"detect", "never_answers"});
        collector.deliver(state->tag(), "detect");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        check(collector.sweep() == 1, "the sweeper finished it");
        check(reported == 1, "and reported it");
        check(reason == FinishReason::Timeout, "as a timeout");
    }

    void test_the_missing_stages_are_named() {
        std::vector<std::string> missing;
        FrameCollector collector([&missing](FrameResult&& result) { missing = result.missing; },
                                 64, 1);

        auto state = a_frame("cam0", 0);
        collector.open(state, {"detect", "ship_segmenter"});
        collector.deliver(state->tag(), "detect");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        collector.sweep();

        check(missing.size() == 1 && missing[0] == "ship_segmenter",
              "the event says which stage did not answer");
    }

    void test_the_capture_is_not_affected_by_later_writes() {
        // ADR-002: the emitter must not read a state whose worker is still running. The capture
        // is taken when the frame is finished; anything the worker does afterwards is
        // invisible.
        EmissionInputs captured;
        FrameCollector collector(
            [&captured](FrameResult&& result) { captured = result.inputs; }, 64, 1500);

        auto state = a_frame("cam0", 0);
        state->set_detections({Detection{0, 0, 4, 4, 0.9f, 0, 0}});
        collector.open(state, {"detect"});
        collector.deliver(state->tag(), "detect");
        collector.seal(state->tag());

        state->set_detections(std::vector<Detection>(12));

        check(captured.detections.size() == 1, "the emitter saw the worker's later write");
    }

    void test_a_shutdown_reports_everything_still_pending() {
        int reported = 0;
        FrameCollector collector([&reported](FrameResult&&) { ++reported; }, 64, 100000);
        for (int i = 0; i < 5; ++i) collector.open(a_frame("cam0", i), {"detect"});

        check(collector.drain() == 5, "the drain finished all five");
        check(reported == 5, "and none was silently discarded");
    }

    void test_a_full_buffer_evicts_the_greediest_camera() {
        FrameCollector collector([](FrameResult&&) {}, 2, 100000);
        collector.open(a_frame("busy", 0), {"detect"});
        collector.open(a_frame("busy", 1), {"detect"});
        check(collector.open(a_frame("quiet", 0), {"detect"}),
              "the quiet camera's frame is admitted by evicting the greedy one's");
        check(collector.evicted() == 1, "and the eviction is counted");
    }

    // -- kernel parity ------------------------------------------------------------------------

    // The readable implementation, transcribed from `TorchImageOps._letterbox`
    // (`src/shipinfer/runtime/ops/torch_ops.py`), NOT from the kernel: scale, `max(1, round(src
    // * scale))`, bars of `(dst - new) // 2`, and `interpolate(align_corners=False)` — source
    // coordinate `(d + 0.5) * (in / out) - 0.5` clamped at zero, neighbour clamped at `in - 1`.
    // The first reference restated the kernel's own `(x - pad) / scale` and so agreed with any
    // mapping the kernel chose; this one measures the gap to the Python plane instead of hiding
    // it.
    float torch_source(int dst, int pad, int in, int out) {
        return std::max(0.f, (static_cast<float>(dst - pad) + 0.5f) * static_cast<float>(in) /
                                     static_cast<float>(out) -
                                 0.5f);
    }

    std::vector<float> letterbox_reference(const std::vector<uint8_t>& src, int src_h,
                                           int src_w, int dst_h, int dst_w, bool swap_rb,
                                           float pad_value, LetterboxMap map) {
        std::vector<float> dst(static_cast<size_t>(3) * dst_h * dst_w, pad_value);
        const int plane = dst_h * dst_w;
        for (int y = map.pad_y; y < map.pad_y + map.new_h; ++y) {
            for (int x = map.pad_x; x < map.pad_x + map.new_w; ++x) {
                const float sx = torch_source(x, map.pad_x, src_w, map.new_w);
                const float sy = torch_source(y, map.pad_y, src_h, map.new_h);
                const int x0 = std::min(src_w - 1, static_cast<int>(sx));
                const int y0 = std::min(src_h - 1, static_cast<int>(sy));
                const int x1 = std::min(src_w - 1, x0 + 1);
                const int y1 = std::min(src_h - 1, y0 + 1);
                const float wx = sx - x0, wy = sy - y0;
                for (int c = 0; c < 3; ++c) {
                    const int sc = swap_rb ? (2 - c) : c;
                    const auto at = [&](int yy, int xx) {
                        return static_cast<float>(src[(yy * src_w + xx) * 3 + sc]);
                    };
                    dst[c * plane + y * dst_w + x] =
                        ((at(y0, x0) * (1 - wx) + at(y0, x1) * wx) * (1 - wy) +
                         (at(y1, x0) * (1 - wx) + at(y1, x1) * wx) * wy) /
                        255.f;
                }
            }
        }
        return dst;
    }

    void test_the_letterbox_placement_is_pythons() {
        // 1920x1080 onto 640: scale 1/3, new 640x360, bars of 140 — what torch_ops.py computes.
        const LetterboxMap map = letterbox_fit(1080, 1920, 640, 640);
        check(map.new_w == 640 && map.new_h == 360, "the resized extent is round(src * scale)");
        check(map.pad_x == 0 && map.pad_y == 140, "the bars are (dst - new) // 2");
        // A half case: 251 * 0.5 = 125.5 rounds to even, 126 — Python's round, not truncation.
        const LetterboxMap half = letterbox_fit(251, 500, 250, 250);
        check(half.new_h == 126,
              "round half to even, as Python's round (251 * 0.5 = 125.5 -> 126)");
    }

    void test_the_letterbox_kernel_agrees_with_the_reference() {
        const int src_h = 90, src_w = 160, dst = 64;
        std::vector<uint8_t> host(static_cast<size_t>(src_h) * src_w * 3);
        for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<uint8_t>((i * 37) % 251);

        uint8_t* device_src = nullptr;
        float* device_dst = nullptr;
        if (gpuMalloc(&device_src, host.size()) != gpuSuccess) {
            skip("no CUDA device for the letterbox parity test");
            return;
        }
        gpuMalloc(&device_dst, static_cast<size_t>(3) * dst * dst * sizeof(float));
        gpuMemcpy(device_src, host.data(), host.size(), gpuMemcpyHostToDevice);

        const LetterboxMap map = letterbox_into(device_src, src_h, src_w, device_dst, dst, dst,
                                                true, 114.f / 255.f, nullptr);
        gpuDeviceSynchronize();

        std::vector<float> got(static_cast<size_t>(3) * dst * dst);
        gpuMemcpy(got.data(), device_dst, got.size() * sizeof(float), gpuMemcpyDeviceToHost);
        const auto want =
            letterbox_reference(host, src_h, src_w, dst, dst, true, 114.f / 255.f, map);

        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want[i])));
        // 1e-4: the kernel uses `floorf` and the reference an int cast, which agree for the
        // non-negative coordinates a letterbox produces, and the rest is float association
        // order.
        check_near(worst, 0.0, 1e-4,
                   "the letterbox kernel matches the readable implementation");

        // The bars carry the pad value, not a smeared edge pixel. Clamping instead would give
        // the detector an object to find in the padding.
        if (map.pad_y > 0)
            check_near(got[0], 114.f / 255.f, 1e-6,
                       "the top bar is 114/255, the Python plane's pad");
        check(map.new_w == dst && map.new_h == 36 && map.pad_y == 14,
              "160x90 onto 64: 64x36 at (0, 14)");
        gpuFree(device_src);
        gpuFree(device_dst);
    }

    // The readable crop: `TorchImageOps.crop_batch` in nested loops — clip, truncate, slice the
    // patch, `interpolate(align_corners=False)`. If this and the kernel disagree, the kernel is
    // wrong; and if this and the torch implementation disagree, the *Python* parity test in the
    // parent (`tests/runtime/test_ops_parity.py`) is where that shows.
    std::vector<float> crop_reference(const std::vector<uint8_t>& src, int src_h, int src_w,
                                      const float* box, int dst_h, int dst_w, bool swap_rb) {
        std::vector<float> dst(static_cast<size_t>(3) * dst_h * dst_w, 0.f);
        const int plane = dst_h * dst_w;
        auto clip = [](float v, float hi) {
            return static_cast<int>(std::min(std::max(v, 0.f), hi));
        };
        const int x1 = clip(box[0], src_w - 1.f), y1 = clip(box[1], src_h - 1.f);
        const int x2 = clip(box[2], src_w - 1.f), y2 = clip(box[3], src_h - 1.f);
        const int bw = x2 - x1, bh = y2 - y1;
        if (bw <= 0 || bh <= 0) return dst;
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                const float lx =
                    std::max(0.f, (x + 0.5f) * bw / static_cast<float>(dst_w) - 0.5f);
                const float ly =
                    std::max(0.f, (y + 0.5f) * bh / static_cast<float>(dst_h) - 0.5f);
                const int px0 = std::min(static_cast<int>(lx), bw - 1);
                const int py0 = std::min(static_cast<int>(ly), bh - 1);
                const int px1 = std::min(px0 + 1, bw - 1), py1 = std::min(py0 + 1, bh - 1);
                const float wx = lx - px0, wy = ly - py0;
                for (int c = 0; c < 3; ++c) {
                    const int sc = swap_rb ? (2 - c) : c;
                    const auto at = [&](int yy, int xx) {
                        return static_cast<float>(
                            src[((y1 + yy) * src_w + (x1 + xx)) * 3 + sc]);
                    };
                    dst[c * plane + y * dst_w + x] =
                        ((at(py0, px0) * (1 - wx) + at(py0, px1) * wx) * (1 - wy) +
                         (at(py1, px0) * (1 - wx) + at(py1, px1) * wx) * wy) /
                        255.f;
                }
            }
        }
        return dst;
    }

    void test_the_crop_kernel_agrees_with_the_reference() {
        const int src_h = 72, src_w = 128, ch = 24, cw = 16;
        std::vector<uint8_t> host(static_cast<size_t>(src_h) * src_w * 3);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<uint8_t>((i * 53 + 7) % 249);
        // Fractional boxes, one touching the right/bottom edge, one partly outside the image:
        // the cases where a half-pixel offset or an image-clamp instead of a patch-clamp shows.
        const float boxes[12] = {10.3f, 5.7f, 57.9f, 40.2f, 100.f, 30.f,
                                 127.f, 71.f, -4.f,  -2.f,  20.f,  9.f};
        const int count = 3;

        uint8_t* device_src = nullptr;
        if (gpuMalloc(&device_src, host.size()) != gpuSuccess) {
            skip("no CUDA device for the crop parity test");
            return;
        }
        float* device_boxes = nullptr;
        float* device_dst = nullptr;
        gpuMalloc(&device_boxes, sizeof(boxes));
        gpuMalloc(&device_dst, static_cast<size_t>(count) * 3 * ch * cw * sizeof(float));
        gpuMemcpy(device_src, host.data(), host.size(), gpuMemcpyHostToDevice);
        gpuMemcpy(device_boxes, boxes, sizeof(boxes), gpuMemcpyHostToDevice);

        crop_resize_into(device_src, src_h, src_w, device_boxes, count, device_dst, ch, cw,
                         true, nullptr);
        gpuDeviceSynchronize();
        std::vector<float> got(static_cast<size_t>(count) * 3 * ch * cw);
        gpuMemcpy(got.data(), device_dst, got.size() * sizeof(float), gpuMemcpyDeviceToHost);

        const size_t per_crop = static_cast<size_t>(3) * ch * cw;
        for (int n = 0; n < count; ++n) {
            const auto want = crop_reference(host, src_h, src_w, boxes + n * 4, ch, cw, true);
            double worst = 0;
            for (size_t i = 0; i < per_crop; ++i) {
                worst = std::max(
                    worst, static_cast<double>(std::fabs(got[n * per_crop + i] - want[i])));
            }
            check_near(worst, 0.0, 1e-4,
                       "crop " + std::to_string(n) + " matches the readable implementation");
        }
        gpuFree(device_src);
        gpuFree(device_boxes);
        gpuFree(device_dst);
    }

    // The readable NV12 letterbox: torch's placement and centre mapping (above), sampled
    // nearest, BT.601 limited range with the kernel's constants. There is no Python twin for
    // NV12 input — the Python plane decodes to BGR first — so this reference is the readable
    // statement of what the kernel is meant to do, and says so.
    std::vector<float> nv12_letterbox_reference(const std::vector<uint8_t>& nv12, int src_h,
                                                int src_w, int stride, int dst_h, int dst_w,
                                                bool swap_rb, float pad_value,
                                                LetterboxMap map) {
        std::vector<float> dst(static_cast<size_t>(3) * dst_h * dst_w, pad_value);
        const int plane = dst_h * dst_w;
        const uint8_t* uv = nv12.data() + static_cast<size_t>(stride) * src_h;
        for (int y = map.pad_y; y < map.pad_y + map.new_h; ++y) {
            for (int x = map.pad_x; x < map.pad_x + map.new_w; ++x) {
                const float sx = torch_source(x, map.pad_x, src_w, map.new_w);
                const float sy = torch_source(y, map.pad_y, src_h, map.new_h);
                const int xi = std::max(0, std::min(src_w - 1, static_cast<int>(sx)));
                const int yi = std::max(0, std::min(src_h - 1, static_cast<int>(sy)));
                const float Y = (static_cast<float>(nv12[yi * stride + xi]) - 16.f) * 1.164383f;
                const int cx = (xi / 2) * 2, cy = yi / 2;
                const float U = static_cast<float>(uv[cy * stride + cx + 0]) - 128.f;
                const float V = static_cast<float>(uv[cy * stride + cx + 1]) - 128.f;
                const float bgr[3] = {
                    std::min(255.f, std::max(0.f, Y + 2.017232f * U)),
                    std::min(255.f, std::max(0.f, Y - 0.391762f * U - 0.812968f * V)),
                    std::min(255.f, std::max(0.f, Y + 1.596027f * V))};
                for (int c = 0; c < 3; ++c) {
                    const int sc = swap_rb ? (2 - c) : c;
                    dst[c * plane + y * dst_w + x] = bgr[sc] / 255.f;
                }
            }
        }
        return dst;
    }

    void test_the_nv12_letterbox_kernel_agrees_with_the_reference() {
        const int src_h = 90, src_w = 160, stride = 192,
                  dst = 64;  // a padded stride, as NVDEC gives
        std::vector<uint8_t> host(static_cast<size_t>(stride) * src_h * 3 / 2);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<uint8_t>(16 + (i * 29) % 220);

        uint8_t* device_src = nullptr;
        float* device_dst = nullptr;
        if (gpuMalloc(&device_src, host.size()) != gpuSuccess) {
            skip("no CUDA device for the NV12 parity test");
            return;
        }
        gpuMalloc(&device_dst, static_cast<size_t>(3) * dst * dst * sizeof(float));
        gpuMemcpy(device_src, host.data(), host.size(), gpuMemcpyHostToDevice);

        const LetterboxMap map = nv12_letterbox_into(device_src, src_h, src_w, stride,
                                                     device_dst, dst, dst, true, 0.5f, nullptr);
        gpuDeviceSynchronize();
        std::vector<float> got(static_cast<size_t>(3) * dst * dst);
        gpuMemcpy(got.data(), device_dst, got.size() * sizeof(float), gpuMemcpyDeviceToHost);
        const auto want =
            nv12_letterbox_reference(host, src_h, src_w, stride, dst, dst, true, 0.5f, map);

        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want[i])));
        check_near(worst, 0.0, 1e-4,
                   "the NV12 letterbox kernel matches the readable implementation");
        if (map.pad_y > 0) check_near(got[0], 0.5f, 1e-6, "the NV12 top bar is the pad value");
        gpuFree(device_src);
        gpuFree(device_dst);
    }

    void test_a_capture_during_attach_is_a_consistent_snapshot() {
        // The sweeper captures a frame whose worker is mid-`attach` — the designed timeout
        // path. Without the lock this is two threads on one std::map, and this hammer crashes
        // or reads a torn map; with it, every snapshot is a batch set that existed at some
        // instant. A hammer proves a negative only weakly, so the check is on the invariant,
        // not on timing.
        FrameState state(FrameTag{"cam", 1}, 8, 8, 20.f);
        std::atomic<bool> stop{false};
        std::atomic<int> bad{0};
        std::thread writer([&]() {
            for (int i = 0; i < 20000; ++i) {
                ObjectBatch batch;
                batch.name = (i % 2 == 0) ? "person_embedder_out" : "ship_embedder_out";
                batch.width = 4;
                batch.object_indices = {i};
                batch.data = {1.f, 2.f, 3.f, 4.f};
                state.attach(std::move(batch));
                if (i % 1000 == 0) state.drop("ship_embedder_out");
            }
            stop.store(true);
        });
        while (!stop.load()) {
            const EmissionInputs snapshot = state.capture();
            for (const auto& [name, batch] : snapshot.batches) {
                if (batch.name != name || batch.data.size() != 4u) bad.fetch_add(1);
            }
        }
        writer.join();
        check(bad.load() == 0, "every capture during attach was a consistent snapshot");
    }

    void test_a_degenerate_box_yields_a_black_crop() {
        // A zero-area detection is data, not a bug. The alternative is a launch that reads out
        // of bounds, and the Python parity test pins the same behaviour.
        const int src_h = 32, src_w = 32, ch = 8, cw = 8;
        std::vector<uint8_t> host(static_cast<size_t>(src_h) * src_w * 3, 200);
        uint8_t* device_src = nullptr;
        if (gpuMalloc(&device_src, host.size()) != gpuSuccess) {
            skip("no CUDA device for the degenerate-box test");
            return;
        }
        gpuMemcpy(device_src, host.data(), host.size(), gpuMemcpyHostToDevice);

        const float boxes[8] = {4, 4, 12, 12, /* degenerate */ 5, 5, 5, 5};
        float* device_boxes = nullptr;
        float* device_dst = nullptr;
        gpuMalloc(&device_boxes, sizeof(boxes));
        gpuMalloc(&device_dst, static_cast<size_t>(2) * 3 * ch * cw * sizeof(float));
        gpuMemcpy(device_boxes, boxes, sizeof(boxes), gpuMemcpyHostToDevice);

        crop_resize_into(device_src, src_h, src_w, device_boxes, 2, device_dst, ch, cw, true,
                         nullptr);
        gpuDeviceSynchronize();

        std::vector<float> got(static_cast<size_t>(2) * 3 * ch * cw);
        gpuMemcpy(got.data(), device_dst, got.size() * sizeof(float), gpuMemcpyDeviceToHost);

        const size_t per_crop = static_cast<size_t>(3) * ch * cw;
        double first_sum = 0, second_sum = 0;
        for (size_t i = 0; i < per_crop; ++i) first_sum += got[i];
        for (size_t i = 0; i < per_crop; ++i) second_sum += got[per_crop + i];
        check(first_sum > 0, "the real box produced pixels");
        check_near(second_sum, 0.0, 1e-6, "the degenerate box produced a black crop");

        gpuFree(device_src);
        gpuFree(device_boxes);
        gpuFree(device_dst);
    }

}  // namespace

int main() {
    test_a_plan_whose_shape_disagrees_with_the_config_is_refused_at_construction();
    test_a_plan_with_non_float32_io_is_refused();
    test_a_frame_with_more_objects_than_the_batch_keeps_every_chunk();
    test_every_opened_frame_is_reported_exactly_once();
    test_a_duplicate_tag_is_refused_rather_than_clobbering();
    test_a_timed_out_frame_is_still_reported();
    test_the_missing_stages_are_named();
    test_the_capture_is_not_affected_by_later_writes();
    test_a_shutdown_reports_everything_still_pending();
    test_a_full_buffer_evicts_the_greediest_camera();

    test_the_letterbox_placement_is_pythons();
    test_the_letterbox_kernel_agrees_with_the_reference();
    test_the_crop_kernel_agrees_with_the_reference();
    test_the_nv12_letterbox_kernel_agrees_with_the_reference();
    test_a_degenerate_box_yields_a_black_crop();
    test_a_capture_during_attach_is_a_consistent_snapshot();

    std::printf("%d checks, %d failure(s), %d skipped\n", checks, failures, skips);
    return failures == 0 ? 0 : 1;
}
