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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/runtime/ops.h"
#include "shipinfer/scheduling/queues/fair.h"

using namespace shipinfer;

namespace {

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }

    void check_near(double actual, double expected, double tolerance, const std::string& what) {
        ++checks;
        if (std::fabs(actual - expected) > tolerance) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s (got %g, expected %g +- %g)\n", what.c_str(), actual,
                         expected, tolerance);
        }
    }

    struct Item {
        std::string cam;
        int id = 0;
        size_t row_count = 1;

        size_t rows() const { return row_count; }
        std::string camera() const { return cam; }
    };

    // -- the fair queue ---------------------------------------------------------------------

    void test_a_busy_camera_cannot_starve_a_quiet_one() {
        // The failure this project exists to fix, in its smallest form: one camera sends twenty
        // frames and another sends one, and the quiet camera's frame must not be at the back of
        // twenty.
        FairQueue<Item> queue(64, Overflow::Reject);
        for (int i = 0; i < 20; ++i) queue.put(Item{"busy", i});
        queue.put(Item{"quiet", 0});

        const auto batch = queue.drain(4, 0);
        bool saw_quiet = false;
        for (const auto& item : batch) saw_quiet |= item.cam == "quiet";
        check(saw_quiet, "the quiet camera appears in the first drain of four");
    }

    void test_the_drain_counts_rows_not_items() {
        // A per-object request carries one row per crop. Counting items against a row budget
        // overfills the batch: sixteen requests of a frame's crops each assembled 24 rows against
        // max_batch_size 16, the assembler refused it, and every request in it failed.
        FairQueue<Item> queue(64, Overflow::Reject);
        queue.put(Item{"a", 0, 6});
        queue.put(Item{"b", 1, 6});
        queue.put(Item{"c", 2, 6});

        const auto batch = queue.drain(8, 0);
        size_t rows = 0;
        for (const auto& item : batch) rows += item.rows();
        check(rows <= 8, "the drain respects the row budget");
        check(batch.size() == 1, "two six-row items do not fit an eight-row budget");
    }

    void test_an_oversized_item_is_returned_alone_not_refused() {
        // Refusing it would park it at the head of its lane forever and stall the model. Letting it
        // through gives the assembler a chance to name the real problem.
        FairQueue<Item> queue(64, Overflow::Reject);
        queue.put(Item{"a", 0, 99});

        const auto batch = queue.drain(8, 0);
        check(batch.size() == 1, "an item larger than the budget is still dequeued");
    }

    void test_a_full_queue_refuses_rather_than_dropping_silently() {
        FairQueue<Item> queue(2, Overflow::Reject);
        check(queue.put(Item{"a", 0}), "first accepted");
        check(queue.put(Item{"a", 1}), "second accepted");
        check(!queue.put(Item{"a", 2}), "third refused");
        check(queue.stats().rejected == 1, "the refusal is counted");
        check(queue.stats().rejected_by_camera["a"] == 1, "and attributed to the camera");
    }

    void test_eviction_charges_the_greediest_camera() {
        // ADR-005: the victim of an eviction should be the cause of the pressure. The previous
        // generation evicted the *oldest* entry, so a crowded camera pushed out a quiet one's work.
        FairQueue<Item> queue(4, Overflow::EvictGreediest);
        queue.put(Item{"quiet", 0});
        for (int i = 0; i < 3; ++i) queue.put(Item{"busy", i});
        queue.put(Item{"busy", 99});  // full: something must go

        const auto stats = queue.stats();
        check(stats.evicted == 1, "exactly one eviction");
        check(stats.evicted_by_camera.count("busy") == 1, "the greedy camera lost the frame");
        check(stats.evicted_by_camera.count("quiet") == 0, "the quiet camera did not");
    }

    void test_a_blocked_producer_wakes_when_a_slot_frees() {
        // Not "eventually succeeds" — *when*. The Python version regressed here and its test could
        // not tell, because it only asserted the outcome: the producer slept the whole 500 ms
        // timeout instead of waking at 50, and a drop was then charged to a camera that had done
        // nothing wrong.
        FairQueue<Item> queue(1, Overflow::Block, 2000);
        queue.put(Item{"a", 0});

        std::thread drainer([&queue] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            queue.drain(1, 0);
        });
        const auto start = std::chrono::steady_clock::now();
        check(queue.put(Item{"a", 1}), "the blocked put eventually succeeds");
        const double waited_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        drainer.join();
        check(waited_ms < 500, "it woke on the drain, not on the timeout (waited " +
                                   std::to_string(static_cast<int>(waited_ms)) + " ms)");
    }

    // -- the collector ----------------------------------------------------------------------

    std::shared_ptr<FrameState> a_frame(const std::string& camera, int64_t id) {
        FrameTag tag;
        tag.camera_id = camera;
        tag.frame_id = id;
        return std::make_shared<FrameState>(tag, 8, 8, 0.0f);
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
        FrameCollector collector([&missing](FrameResult&& result) { missing = result.missing; }, 64,
                                 1);

        auto state = a_frame("cam0", 0);
        collector.open(state, {"detect", "ship_segmenter"});
        collector.deliver(state->tag(), "detect");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        collector.sweep();

        check(missing.size() == 1 && missing[0] == "ship_segmenter",
              "the event says which stage did not answer");
    }

    void test_the_capture_is_not_affected_by_later_writes() {
        // ADR-002: the emitter must not read a state whose worker is still running. The capture is
        // taken when the frame is finished; anything the worker does afterwards is invisible.
        EmissionInputs captured;
        FrameCollector collector([&captured](FrameResult&& result) { captured = result.inputs; },
                                 64, 1500);

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

    // The readable implementation. Deliberately the slow, obvious version: nested loops, one
    // bilinear sample at a time, no cleverness. If this and the kernel disagree, the kernel is
    // wrong — that is the whole point of writing it twice.
    std::vector<float> letterbox_reference(const std::vector<uint8_t>& src, int src_h, int src_w,
                                           int dst_h, int dst_w, bool swap_rb, float pad_value,
                                           LetterboxMap map) {
        std::vector<float> dst(static_cast<size_t>(3) * dst_h * dst_w, 0.f);
        const int plane = dst_h * dst_w;
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                const float sx = (static_cast<float>(x) - map.pad_x) / map.scale;
                const float sy = (static_cast<float>(y) - map.pad_y) / map.scale;
                const bool inside = sx >= 0.f && sy >= 0.f && sx <= src_w - 1 && sy <= src_h - 1;
                for (int c = 0; c < 3; ++c) {
                    const int sc = swap_rb ? (2 - c) : c;
                    float value = pad_value;
                    if (inside) {
                        const int x0 = std::max(0, std::min(src_w - 1, static_cast<int>(sx)));
                        const int y0 = std::max(0, std::min(src_h - 1, static_cast<int>(sy)));
                        const int x1 = std::min(src_w - 1, x0 + 1);
                        const int y1 = std::min(src_h - 1, y0 + 1);
                        const float wx = sx - x0;
                        const float wy = sy - y0;
                        const auto at = [&](int yy, int xx) {
                            return static_cast<float>(src[(yy * src_w + xx) * 3 + sc]);
                        };
                        value = ((at(y0, x0) * (1 - wx) + at(y0, x1) * wx) * (1 - wy) +
                                 (at(y1, x0) * (1 - wx) + at(y1, x1) * wx) * wy) /
                                255.f;
                    }
                    dst[c * plane + y * dst_w + x] = value;
                }
            }
        }
        return dst;
    }

    void test_the_letterbox_kernel_agrees_with_the_reference() {
        const int src_h = 90, src_w = 160, dst = 64;
        std::vector<uint8_t> host(static_cast<size_t>(src_h) * src_w * 3);
        for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<uint8_t>((i * 37) % 251);

        uint8_t* device_src = nullptr;
        float* device_dst = nullptr;
        if (gpuMalloc(&device_src, host.size()) != gpuSuccess) {
            std::fprintf(stderr, "SKIP: no CUDA device for the parity tests\n");
            return;
        }
        gpuMalloc(&device_dst, static_cast<size_t>(3) * dst * dst * sizeof(float));
        gpuMemcpy(device_src, host.data(), host.size(), gpuMemcpyHostToDevice);

        const LetterboxMap map =
            letterbox_into(device_src, src_h, src_w, device_dst, dst, dst, true, 0.5f, nullptr);
        gpuDeviceSynchronize();

        std::vector<float> got(static_cast<size_t>(3) * dst * dst);
        gpuMemcpy(got.data(), device_dst, got.size() * sizeof(float), gpuMemcpyDeviceToHost);
        const auto want = letterbox_reference(host, src_h, src_w, dst, dst, true, 0.5f, map);

        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want[i])));
        // 1e-4: the kernel uses `floorf` and the reference an int cast, which agree for the
        // non-negative coordinates a letterbox produces, and the rest is float association order.
        check_near(worst, 0.0, 1e-4, "the letterbox kernel matches the readable implementation");

        // The bars carry the pad value, not a smeared edge pixel. Clamping instead would give the
        // detector an object to find in the padding.
        const bool wide = src_w * dst / src_h > dst;
        if (wide && map.pad_y > 0) {
            check_near(got[0], 0.5f, 1e-6, "the top bar is the pad value");
        }
        gpuFree(device_src);
        gpuFree(device_dst);
    }

    void test_a_degenerate_box_yields_a_black_crop() {
        // A zero-area detection is data, not a bug. The alternative is a launch that reads out of
        // bounds, and the Python parity test pins the same behaviour.
        const int src_h = 32, src_w = 32, ch = 8, cw = 8;
        std::vector<uint8_t> host(static_cast<size_t>(src_h) * src_w * 3, 200);
        uint8_t* device_src = nullptr;
        if (gpuMalloc(&device_src, host.size()) != gpuSuccess) return;
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
    test_a_busy_camera_cannot_starve_a_quiet_one();
    test_the_drain_counts_rows_not_items();
    test_an_oversized_item_is_returned_alone_not_refused();
    test_a_full_queue_refuses_rather_than_dropping_silently();
    test_eviction_charges_the_greediest_camera();
    test_a_blocked_producer_wakes_when_a_slot_frees();

    test_every_opened_frame_is_reported_exactly_once();
    test_a_duplicate_tag_is_refused_rather_than_clobbering();
    test_a_timed_out_frame_is_still_reported();
    test_the_missing_stages_are_named();
    test_the_capture_is_not_affected_by_later_writes();
    test_a_shutdown_reports_everything_still_pending();
    test_a_full_buffer_evicts_the_greediest_camera();

    test_the_letterbox_kernel_agrees_with_the_reference();
    test_a_degenerate_box_yields_a_black_crop();

    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
