// The graph's planning and reassembly wiring — `tests/pipeline/test_graph.py`'s claims, with no
// device: fake stages that mark names on the state, and the real Dag, FrameState and collector.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/pixels.h"
#include "shipinfer/pipeline/graph/stage.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/queue_sink.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/pipeline/surface_intake.h"
#include "shipinfer/runtime/containment.h"
#include "shipinfer/runtime/ops.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;
    int skips = 0;

    // A test that cannot run must say so and be counted — a device-less run that prints
    // "N checks, 0 failure(s)" and reads as green is a test that fails open.
    void skip(const std::string& why) {
        ++skips;
        std::fprintf(stderr, "SKIP: %s\n", why.c_str());
    }

    bool has_device() {
        void* probe = nullptr;
        if (gpuMalloc(&probe, 256) != gpuSuccess) return false;
        gpuFree(probe);
        return true;
    }

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }

    // A stage that produces a payload of `rows` rows under `produces[0]`, or throws.
    class FakeStage : public Stage {
      public:
        FakeStage(std::string name, std::vector<std::string> consumes,
                  std::vector<std::string> needs, std::string produces, size_t rows,
                  bool fail = false)
            : Stage(std::move(name), std::move(consumes), std::move(needs), {produces}),
              produces_(std::move(produces)),
              rows_(rows),
              fail_(fail) {}
        int runs = 0;

      protected:
        size_t do_run(FrameState& state) override {
            ++runs;
            if (fail_) throw BackendError("injected");
            if (produces_ == DETECTIONS) {
                std::vector<Detection> dets(rows_);
                state.set_detections(std::move(dets));
                state.set_detected(true);
            } else {
                DevicePayload payload;
                payload.name = produces_;
                payload.rows = rows_;
                std::vector<int> idx(rows_);
                payload.object_indices = idx;
                state.attach_payload(std::move(payload));
            }
            return rows_;
        }

      private:
        std::string produces_;
        size_t rows_;
        bool fail_;
    };

    class RecordingObserver : public StageObserver {
      public:
        std::vector<std::vector<std::string>> planned_calls;
        std::vector<StageOutcome> finished_calls;
        void planned(const std::vector<std::string>& stages) override {
            planned_calls.push_back(stages);
        }
        void finished(const StageOutcome& outcome) override {
            finished_calls.push_back(outcome);
        }
    };

    std::shared_ptr<FrameState> a_frame() {
        auto state = std::make_shared<FrameState>(FrameTag{"cam", 1, 0}, 8, 8, 20.f);
        state->set_image(std::make_shared<DeviceBuffer>(), 0);  // present, never read here
        return state;
    }

    // A surface-only frame satisfies `FRAME_INPUT`. No device: this is the PLANNER's half, and
    // it is the half that fails silently -- a chain whose detector never becomes runnable
    // produces complete frames with no detections, and every count downstream agrees with
    // itself.
    void test_a_frame_carrying_only_a_surface_still_has_pixels() {
        FrameState state(FrameTag{"cam", 1, 0}, 250, 320, 20.f);
        check(state.available().empty(), "a frame with neither representation offers nothing");

        // A non-null pointer is all `DeviceSurface::empty()` asks about; nothing here reads it.
        uint8_t byte = 0;
        DeviceSurface surface;
        surface.nv12 = &byte;
        surface.stride = 384;
        surface.uv_offset = static_cast<size_t>(384) * 256;
        state.set_surface(surface, 3);

        const std::vector<std::string> available = state.available();
        check(available.size() == 1 && available.front() == FRAME_INPUT,
              "an NV12 surface satisfies FRAME_INPUT, so the detector becomes runnable");
        const std::vector<std::string> non_empty = state.non_empty();
        // Spelled out rather than compared against `available`: both come from `has_pixels()`,
        // so comparing them passes on exactly the drift this test exists to catch.
        check(non_empty.size() == 1 && non_empty.front() == FRAME_INPUT,
              "and is non-empty, so a `needs` on it is met too");
        check(state.device() == 3, "on the device it was decoded on (ADR-004)");
        check(state.surface().uv_offset == static_cast<size_t>(384) * 256,
              "with the uv_offset CARRIED rather than derived (`ingest/frame.h`)");
        check(state.image() == nullptr, "and no host image: exactly one representation");

        state.release_image();
        check(state.available().empty(),
              "and releasing the pixels releases BOTH -- a surface left behind would keep an "
              "NVDEC slot out of a pool of four for the life of the frame");
    }

    // The seam picks by representation, and the fixture is built so that picking wrong is
    // VISIBLE rather than merely wrong-looking: the surface is padded -- stride above the
    // width, and a chroma offset above the luma the reader would derive -- with every padding
    // byte a sentinel. A read that ignored the stride, which is what routing an NV12 frame
    // through the BGR entry point does, pulls the sentinel into the output.
    void test_the_pixel_seam_reads_a_padded_surface_as_a_surface() {
        const int src_h = 90, src_w = 160, stride = 192, plane_h = 96, dst = 32;
        const size_t uv_offset = static_cast<size_t>(stride) * plane_h;
        std::vector<uint8_t> host(uv_offset + static_cast<size_t>(stride) * src_h / 2, 0xFF);
        // The image proper: a mid-grey ramp well away from the sentinel, chroma neutral.
        for (int y = 0; y < src_h; ++y) {
            for (int x = 0; x < src_w; ++x)
                host[y * stride + x] = static_cast<uint8_t>(40 + x % 60);
        }
        for (int y = 0; y < src_h / 2; ++y) {
            for (int x = 0; x < src_w; ++x) host[uv_offset + y * stride + x] = 128;
        }

        DeviceBuffer pixels(host.size());
        GPU_CHECK(gpuMemcpy(pixels.get(), host.data(), host.size(), gpuMemcpyHostToDevice));
        FrameState state(FrameTag{"cam", 1, 0}, src_h, src_w, 20.f);
        DeviceSurface surface;
        surface.nv12 = pixels.as<uint8_t>();
        surface.stride = stride;
        surface.uv_offset = uv_offset;
        state.set_surface(surface, 0);

        DeviceBuffer out(static_cast<size_t>(3) * dst * dst * sizeof(float));
        const LetterboxMap map = letterbox_frame(state, out.as<float>(), dst, dst,
                                                 /*swap_rb=*/true, 0.f, nullptr);
        GPU_CHECK(gpuStreamSynchronize(nullptr));
        const LetterboxMap want = letterbox_fit(src_h, src_w, dst, dst);
        check(map.scale == want.scale && map.pad_x == want.pad_x && map.pad_y == want.pad_y,
              "the geometry is the display extent's, whichever representation carried it");

        std::vector<float> got(static_cast<size_t>(3) * dst * dst);
        GPU_CHECK(gpuMemcpy(got.data(), out.get(), out.bytes(), gpuMemcpyDeviceToHost));
        float brightest = 0.f;
        for (int y = 0; y < map.new_h; ++y) {
            for (int x = 0; x < map.new_w; ++x) {
                const size_t at = static_cast<size_t>(map.pad_y + y) * dst + map.pad_x + x;
                for (int c = 0; c < 3; ++c) {
                    brightest =
                        std::max(brightest, got[static_cast<size_t>(c) * dst * dst + at]);
                }
            }
        }
        // 0xFF luma with neutral chroma converts to ~1.0; the ramp tops out at 99, which is
        // ~0.11 of full scale. So the sentinel cannot hide inside the image's own range.
        check(brightest > 0.02f,
              "the image was read at all: brightest inside the letterbox is " +
                  std::to_string(brightest));
        check(brightest < 0.5f,
              "and no padding byte reached the output -- a stride-blind read would put 0xFF "
              "there: brightest is " +
                  std::to_string(brightest));

        // The crop half of the same seam, on the same surface. An ordinary box and a degenerate
        // one, because "agrees with the reference" can mean "both produced nothing".
        const std::vector<float> boxes{20.f, 10.f, 120.f, 70.f, 5.f, 5.f, 5.f, 5.f};
        DeviceBuffer boxes_device(boxes.size() * sizeof(float));
        GPU_CHECK(gpuMemcpy(boxes_device.get(), boxes.data(), boxes_device.bytes(),
                            gpuMemcpyHostToDevice));
        const size_t row = static_cast<size_t>(3) * 16 * 16;
        DeviceBuffer crops(2 * row * sizeof(float));
        crop_frame(state, boxes_device.as<float>(), 2, crops.as<float>(), 16, 16,
                   /*swap_rb=*/true, nullptr);
        GPU_CHECK(gpuStreamSynchronize(nullptr));
        std::vector<float> cropped(2 * row);
        GPU_CHECK(gpuMemcpy(cropped.data(), crops.get(), crops.bytes(), gpuMemcpyDeviceToHost));
        float ordinary = 0.f, degenerate = 0.f;
        for (size_t i = 0; i < row; ++i) {
            ordinary = std::max(ordinary, cropped[i]);
            degenerate = std::max(degenerate, cropped[row + i]);
        }
        check(ordinary > 0.02f && ordinary < 0.5f,
              "a crop out of the surface carries the image and not the padding: " +
                  std::to_string(ordinary));
        check(degenerate == 0.f, "and a zero-area box is black rather than a launch failure");
    }

    // One padded NV12 surface on the device, plus its host bytes. Every byte the image does not
    // occupy is 0xFF -- the stride padding, and the gap between the last image row and the
    // plane's end -- so a copy that carried the padding into the middle of its buffer, or read
    // the chroma from the wrong offset, shows up as a bright pixel rather than a subtle one.
    struct PaddedSurface {
        static constexpr int height = 90, width = 160, stride = 192, plane = 96;
        static size_t uv_offset() { return static_cast<size_t>(stride) * plane; }
        static size_t bytes() { return uv_offset() + static_cast<size_t>(stride) * height / 2; }

        std::vector<uint8_t> host;
        DeviceBuffer device;

        PaddedSurface() : host(bytes(), 0xFF), device(bytes()) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    host[static_cast<size_t>(y) * stride + x] =
                        static_cast<uint8_t>(30 + (x * 3 + y) % 60);
                }
            }
            for (int y = 0; y < height / 2; ++y) {
                for (int x = 0; x < width; ++x) {
                    host[uv_offset() + static_cast<size_t>(y) * stride + x] =
                        static_cast<uint8_t>(x % 2 ? 120 : 136);
                }
            }
            GPU_CHECK(gpuMemcpy(device.get(), host.data(), host.size(), gpuMemcpyHostToDevice));
        }
    };

    // The `DeviceImage` a decoder would hand over for `surface`, geometry and all. A helper
    // because three tests fill the same eight fields and a typo in one of them is a test that
    // passes for the wrong reason.
    DeviceImage a_device_image(const PaddedSurface& surface) {
        DeviceImage image;
        image.nv12 = surface.device.get();
        image.height = PaddedSurface::height;
        image.width = PaddedSurface::width;
        image.pitch = PaddedSurface::stride;
        image.uv_offset = PaddedSurface::uv_offset();
        image.device = 0;
        return image;
    }

    // `nv12_letterbox_into` run over one surface, read back. What the KERNELS see, which is the
    // only definition of "the copy preserved the frame" that matters here.
    std::vector<float> letterboxed(const uint8_t* nv12, int stride, size_t uv_offset, int dst) {
        DeviceBuffer out(static_cast<size_t>(3) * dst * dst * sizeof(float));
        nv12_letterbox_into(nv12, PaddedSurface::height, PaddedSurface::width, stride,
                            uv_offset, out.as<float>(), dst, dst, /*swap_rb=*/true, 0.f,
                            nullptr);
        GPU_CHECK(gpuStreamSynchronize(nullptr));
        std::vector<float> got(static_cast<size_t>(3) * dst * dst);
        GPU_CHECK(gpuMemcpy(got.data(), out.get(), out.bytes(), gpuMemcpyDeviceToHost));
        return got;
    }

    // The intake's whole job, in the order it matters: the decoder's slot goes back, the pixels
    // do not change, and the buffer is reused instead of allocated per frame.
    void test_the_intake_frees_the_decoders_slot_and_keeps_the_pixels() {
        PaddedSurface source;
        bool slot_returned = false;

        DeviceImage image;
        image.nv12 = source.device.get();
        image.height = PaddedSurface::height;
        image.width = PaddedSurface::width;
        image.pitch = PaddedSurface::stride;
        image.uv_offset = PaddedSurface::uv_offset();
        image.device = 0;
        // Stands in for `cuvidUnmapVideoFrame`: what a real surface's release does is give a
        // slot out of a pool of two back, and the point of the intake is that it happens before
        // the frame is queued rather than after a worker is done with it.
        image.owner = std::shared_ptr<const void>(source.device.get(),
                                                  [&](const void*) { slot_returned = true; });

        auto intake = std::make_shared<SurfaceIntake>(0, /*max_pooled=*/2);
        DeviceSurface taken = SurfaceIntake::take(intake, image, "cam");
        image.owner.reset();
        check(slot_returned, "taking a surface lets the decoder's slot go back at once");
        check(taken.uv_offset ==
                  static_cast<size_t>(PaddedSurface::stride) * PaddedSurface::height,
              "and what comes out is TIGHT -- the source padding is not carried along: " +
                  std::to_string(taken.uv_offset));

        const int dst = 32;
        const std::vector<float> before =
            letterboxed(source.device.as<uint8_t>(), PaddedSurface::stride,
                        PaddedSurface::uv_offset(), dst);
        const std::vector<float> after =
            letterboxed(taken.nv12, taken.stride, taken.uv_offset, dst);
        check(before == after,
              "and the kernels see exactly the same frame through the copy -- same kernel, "
              "same inputs, so anything but bit-identical is a copy that lost or moved bytes");
        float brightest = 0.f;
        for (float value : after) brightest = std::max(brightest, value);
        check(brightest < 0.5f,
              "with no 0xFF padding byte anywhere in the output: brightest is " +
                  std::to_string(brightest));

        check(intake->pooled() == 0, "a surface in hand is not in the pool");
        const void* reused = taken.nv12;
        taken.owner.reset();
        taken.nv12 = nullptr;
        check(intake->pooled() == 1, "releasing it returns the buffer rather than freeing it");
        DeviceSurface again = SurfaceIntake::take(intake, image, "cam");
        check(again.nv12 == reused && intake->pooled() == 0,
              "and the next frame gets that buffer back -- a cudaMalloc per frame at a "
              "thousand frames a second is what this class exists to avoid");

        // The cap, which is the difference between a pool and a leak: a consumer that stalls
        // must not turn into unbounded VRAM.
        DeviceSurface second = SurfaceIntake::take(intake, image, "cam");
        DeviceSurface third = SurfaceIntake::take(intake, image, "cam");
        again.owner.reset();
        second.owner.reset();
        third.owner.reset();
        check(intake->pooled() == 2, "and the pool holds at most what it was sized for: " +
                                         std::to_string(intake->pooled()));
    }

    // TWO RESOLUTIONS ON ONE GPU, which is an ordinary maritime fleet and what the first
    // version could not do: it tracked ONE size, so an alternating pair retired the whole free
    // list every frame -- a `cudaFree` per pooled buffer inside the lock and then a
    // `cudaMalloc`, both device-synchronising, on the ingest thread. The class's own reuse
    // check passed throughout, because it only ever handed it one size.
    void test_two_resolutions_on_one_gpu_each_get_pool_hits() {
        PaddedSurface big;
        PaddedSurface small;  // same fixture; a smaller GEOMETRY is what makes it another size
        DeviceImage wide = a_device_image(big);
        DeviceImage narrow = a_device_image(small);
        narrow.height = PaddedSurface::height / 2;  // half the plane, so half the buffer
        narrow.uv_offset = static_cast<size_t>(PaddedSurface::stride) * narrow.height;

        auto intake = std::make_shared<SurfaceIntake>(0, /*max_pooled=*/2);
        DeviceSurface a = SurfaceIntake::take(intake, wide, "cam-wide");
        DeviceSurface b = SurfaceIntake::take(intake, narrow, "cam-narrow");
        const void* first_wide = a.nv12;
        const void* first_narrow = b.nv12;
        a.owner.reset();
        b.owner.reset();
        check(intake->pooled() == 2, "both sizes are held, not one at the other's expense: " +
                                         std::to_string(intake->pooled()));

        // ALTERNATING, which is the pattern that used to thrash. Both must be pool hits.
        DeviceSurface again_wide = SurfaceIntake::take(intake, wide, "cam-wide");
        DeviceSurface again_narrow = SurfaceIntake::take(intake, narrow, "cam-narrow");
        check(again_wide.nv12 == first_wide && again_narrow.nv12 == first_narrow,
              "and each resolution gets ITS OWN buffer back rather than a fresh allocation");
        check(intake->pooled() == 0, "with both buckets now empty");
        again_wide.owner.reset();
        again_narrow.owner.reset();
    }

    // THE CONTRACT THE SINK'S LIFETIME RESTS ON, asserted where it lives rather than inferred
    // from a run that did not crash. A raw `this` in the deleter does not crash on release:
    // the freed pool still looks intact, `give_back` locks a destroyed mutex and returns, and
    // every gate stays green -- which is exactly how round 1 shipped it. What IS observable is
    // whether anything holds the pool, so that is what this checks.
    void test_a_surface_holds_its_pool_alive() {
        PaddedSurface decoded;
        auto intake = std::make_shared<SurfaceIntake>(0, /*max_pooled=*/2);
        std::weak_ptr<SurfaceIntake> watch = intake;
        DeviceSurface surface = SurfaceIntake::take(intake, a_device_image(decoded), "cam");

        intake.reset();  // the sink's reference goes, as it does on the unwind path
        check(!watch.expired(),
              "a live surface holds its pool alive -- with a raw pointer in the deleter this "
              "is already gone and the release below locks a destroyed mutex");
        surface.owner.reset();
        check(watch.expired(), "and the last surface releasing lets the pool go");
    }

    // The pool must outlive the sink, because the sink is destroyed BEFORE the guard that stops
    // the workers (`bench.cpp` declares `JoinOnUnwind` first and `QueueSink` last), so an
    // unwind between `manager.start()` and `queue.close()` leaves worker threads holding
    // surfaces whose pool has gone. The first version captured `this` and argued the opposite;
    // with it, this test locks a destroyed mutex.
    void test_a_surface_outlives_the_sink_that_made_it() {
        PaddedSurface decoded;
        DeviceSurface held;
        {
            FairPriorityQueue<FrameWork> queue("pipeline", 8, Overflow::Reject);
            QueueSink sink(queue, /*pooled=*/2, /*devices=*/{0});
            Frame frame;
            frame.tag = FrameTag{"cam", 1, 0};
            frame.device = a_device_image(decoded);
            frame.device.owner =
                std::shared_ptr<const void>(decoded.device.get(), [](const void*) {});
            sink.put(std::move(frame));
            std::vector<FrameWork> batch = queue.get_batch(BatchWindow{4, 0}, 10);
            check(batch.size() == 1, "one work item, carrying the sink's copy");
            held = batch[0].surface;
        }
        // The sink, its queue and the work item are all destroyed now; this surface is not.
        check(held.nv12 != nullptr, "the surface survives its sink");
        held.owner.reset();  // returns the buffer to a pool that must still exist
        check(true, "and releasing it afterwards runs the deleter against a live pool");
    }

    // The sink is where a frame becomes scheduled work, and it is the one place that decides
    // between the two pixel representations. Three questions, and the third is the one an
    // operator meets.
    void test_the_sink_turns_either_representation_into_one_work_item() {
        FairPriorityQueue<FrameWork> queue("pipeline", 16, Overflow::Reject);
        QueueSink sink(queue, /*pooled=*/4, /*devices=*/{0});

        // -- a host frame: the pixels stay on the host and the worker uploads them ----------
        std::vector<uint8_t> pixels(8 * 8 * 3, 7);
        Frame host;
        host.tag = FrameTag{"cam", 1, 0};
        host.image.pixels = pixels.data();
        host.image.height = 8;
        host.image.width = 8;
        sink.put(std::move(host));
        std::vector<FrameWork> batch = queue.get_batch(BatchWindow{4, 0}, 10);
        check(batch.size() == 1 && batch[0].device < 0,
              "a host frame becomes a work item with no device, which any worker may take");
        check(batch[0].frame.pixels == pixels.data() && batch[0].surface.empty(),
              "carrying the library's pixels and no surface");
        check(batch[0].state != nullptr && batch[0].state->available().empty(),
              "and NO pixels on the state yet -- the worker attaches them after uploading");

        // -- a device frame: copied out of the decoder's pool, and on the state at once -----
        PaddedSurface decoded;
        bool slot_returned = false;
        Frame device;
        device.tag = FrameTag{"cam", 2, 0};
        device.device.nv12 = decoded.device.get();
        device.device.height = PaddedSurface::height;
        device.device.width = PaddedSurface::width;
        device.device.pitch = PaddedSurface::stride;
        device.device.uv_offset = PaddedSurface::uv_offset();
        device.device.device = 0;
        device.device.owner = std::shared_ptr<const void>(
            decoded.device.get(), [&](const void*) { slot_returned = true; });
        sink.put(std::move(device));
        check(slot_returned,
              "a device frame gives the decode slot back BEFORE it is queued -- a pool of two "
              "output surfaces cannot survive a queue that holds frames");
        batch = queue.get_batch(BatchWindow{4, 0}, 10);
        check(batch.size() == 1 && batch[0].device == 0 && !batch[0].surface.empty(),
              "and becomes a work item bound to the device it was decoded on");
        check(batch[0].state != nullptr && batch[0].state->available().size() == 1,
              "with the pixels already on the state: there is nothing to upload");
        check(batch[0].surface.nv12 != decoded.device.as<uint8_t>(),
              "and they are the sink's copy, not the decoder's surface");
    }

    // NO DEVICE NEEDED, and that is worth the dummy pointer: the refusal happens before
    // `SurfaceIntake::take` and therefore before any CUDA call, so this runs in the offline
    // tier where a wrong predicate would otherwise only show up on a GPU box.
    void test_a_device_frame_from_a_gpu_this_process_lacks_is_refused_by_name() {
        FairPriorityQueue<FrameWork> queue("pipeline", 16, Overflow::Reject);
        QueueSink sink(queue, /*pooled=*/4, /*devices=*/{1, 2});
        uint8_t nothing = 0;  // never read: `empty()` asks only that it is non-null
        Frame frame;
        frame.tag = FrameTag{"cam09", 3, 0};
        frame.device.nv12 = &nothing;
        frame.device.height = PaddedSurface::height;
        frame.device.width = PaddedSurface::width;
        frame.device.pitch = PaddedSurface::stride;
        frame.device.uv_offset = PaddedSurface::uv_offset();
        frame.device.device = 0;  // a process driving {1, 2} has no worker for it
        frame.device.owner = std::shared_ptr<const void>(&nothing, [](const void*) {});
        std::string reason;
        try {
            sink.put(std::move(frame));
        } catch (const ConfigError& error) {
            reason = error.what();
        }
        // A `ConfigError` and not a drop, because `CameraActor` refuses one fatally: the camera
        // stops with this text in its health line rather than failing every frame forever.
        // THE SET AND NOT THE COUNT is what this pins. `devices > 1` was the first predicate
        // and it accepts exactly this frame -- one GPU, and the wrong one -- which is the
        // `--devices 3 --source nvdec` case where every frame then failed in the worker.
        check(reason.find("cam09") != std::string::npos &&
                  reason.find("gpu0") != std::string::npos &&
                  reason.find("1,2") != std::string::npos,
              "a GPU this process does not drive is refused, naming the camera, its GPU and "
              "the ones there are: " +
                  reason);
        check(queue.stats().depth == 0, "and nothing is queued");
    }

    // NO DEVICE NEEDED: the refusal happens before `gpuSetDevice` is reached, which is what
    // lets this one run in the offline tier where the rest of the intake's gates cannot.
    void test_an_incomplete_surface_never_becomes_a_buffer() {
        DeviceImage image;  // no pointer, no geometry
        auto intake = std::make_shared<SurfaceIntake>(0);
        std::string reason;
        try {
            SurfaceIntake::take(intake, image, "cam42");
        } catch (const ConfigError& error) {
            reason = error.what();
        }
        check(reason.find("incomplete") != std::string::npos &&
                  reason.find("cam42") != std::string::npos,
              "an incomplete surface is refused rather than sized, naming the camera an "
              "operator would then go and look at: " +
                  reason);
    }

    // Neither representation attached. A stage's `needs` is `FRAME_INPUT`, so the planner
    // should never run it -- which makes this a wiring fault, and it must arrive as one rather
    // than as a black frame that reads as a camera pointing at a wall.
    void test_a_frame_with_no_pixels_is_refused_by_name() {
        FrameState state(FrameTag{"cam", 7, 0}, 8, 8, 20.f);
        std::string reason;
        try {
            letterbox_frame(state, nullptr, 4, 4, true, 0.f, nullptr);
        } catch (const ConfigError& error) {
            reason = error.what();
        }
        check(reason.find(state.tag().key()) != std::string::npos &&
                  reason.find("no pixels") != std::string::npos,
              "letterboxing a frame with no pixels names the frame and the fault: " + reason);
        reason.clear();
        try {
            crop_frame(state, nullptr, 1, nullptr, 4, 4, true, nullptr);
        } catch (const ConfigError& error) {
            reason = error.what();
        }
        check(reason.find("no pixels") != std::string::npos,
              "and so does cropping one: " + reason);
    }

    void test_both_clocks_survive_the_capture() {
        // ADR-002 in the reassembly path: the tag travels unchanged from decode to emission,
        // and the tag now carries two clocks. `captured_ns` is what a latency measurement
        // subtracts from; `captured_unix_ns` is what an event a human reads is stamped with.
        // Dropping either here is invisible until somebody reports a latency of minus
        // fifty-four years.
        FrameTag tag;
        tag.camera_id = "cam";
        tag.frame_id = 9;
        tag.captured_ns = monotonic_ns();
        tag.captured_unix_ns = unix_ns();
        FrameState state(tag, 8, 8, 20.f);
        const EmissionInputs captured = state.capture();
        check(captured.tag.camera_id == "cam" && captured.tag.frame_id == 9,
              "the identity round-trips through capture()");
        check(captured.tag.captured_ns == tag.captured_ns &&
                  captured.tag.captured_unix_ns == tag.captured_unix_ns,
              "and so do BOTH clocks — the steady one for latency, the wall one for humans");
    }

    void test_a_stage_whose_input_is_empty_is_skipped_not_failed() {
        Dag dag;
        auto* detect = new FakeStage("detect", {FRAME_INPUT}, {FRAME_INPUT}, DETECTIONS, 2);
        auto* crop =
            new FakeStage("crop", {DETECTIONS, FRAME_INPUT}, {DETECTIONS}, "ship_crops", 0);
        auto* seg = new FakeStage("ship_segmenter", {"ship_crops"}, {"ship_crops"}, "masks", 1);
        dag.add(std::unique_ptr<Stage>(detect));
        dag.add(std::unique_ptr<Stage>(crop));
        dag.add(std::unique_ptr<Stage>(seg));
        auto state = a_frame();
        RecordingObserver observer;
        const auto outcomes = dag.execute(*state, observer);
        check(outcomes[0].status == StageStatus::Ran && outcomes[1].status == StageStatus::Ran,
              "detect and crop ran");
        check(outcomes[2].status == StageStatus::Skipped && seg->runs == 0,
              "no ship crops: the segmenter is skipped, never called");
        bool seg_planned = false;
        for (const auto& call : observer.planned_calls) {
            for (const auto& name : call) seg_planned = seg_planned || name == "ship_segmenter";
        }
        check(!seg_planned, "and it was never announced, so reassembly does not wait for it");
    }

    void test_a_failing_stage_does_not_end_the_frame() {
        Dag dag;
        dag.add(std::make_unique<FakeStage>("detect", std::vector<std::string>{FRAME_INPUT},
                                            std::vector<std::string>{FRAME_INPUT}, DETECTIONS,
                                            2));
        dag.add(std::make_unique<FakeStage>(
            "crop", std::vector<std::string>{DETECTIONS, FRAME_INPUT},
            std::vector<std::string>{DETECTIONS}, "person_crops", 2));
        dag.add(std::make_unique<FakeStage>(
            "person_embedder", std::vector<std::string>{"person_crops"},
            std::vector<std::string>{"person_crops"}, "embeddings", 2,
            /*fail=*/true));
        auto* seg =
            new FakeStage("ship_segmenter", {"person_crops"}, {"person_crops"}, "masks", 2);
        dag.add(std::unique_ptr<Stage>(seg));
        auto state = a_frame();
        RecordingObserver observer;
        const auto outcomes = dag.execute(*state, observer);
        check(outcomes[2].status == StageStatus::Failed && outcomes[2].error == "injected",
              "the embedder failed, and says why");
        check(outcomes[3].status == StageStatus::Ran && seg->runs == 1,
              "the other branch continued");
    }

    void test_the_collector_sees_planned_delivered_and_missing() {
        std::vector<FrameResult> results;
        FrameCollector collector([&](FrameResult&& r) { results.push_back(std::move(r)); }, 16,
                                 1500);
        Dag dag;
        dag.add(std::make_unique<FakeStage>("detect", std::vector<std::string>{FRAME_INPUT},
                                            std::vector<std::string>{FRAME_INPUT}, DETECTIONS,
                                            1));
        dag.add(std::make_unique<FakeStage>(
            "crop", std::vector<std::string>{DETECTIONS, FRAME_INPUT},
            std::vector<std::string>{DETECTIONS}, "person_crops", 1));
        dag.add(std::make_unique<FakeStage>(
            "person_embedder", std::vector<std::string>{"person_crops"},
            std::vector<std::string>{"person_crops"}, "embeddings", 1,
            /*fail=*/true));
        auto state = a_frame();
        check(collector.open(state, {"detect", "crop"}),
              "opened with the unconditional stages");
        CollectorObserver observer(collector, state->tag());
        dag.execute(*state, observer);
        collector.seal(state->tag());
        check(results.size() == 1 && results[0].reason == FinishReason::Incomplete,
              "a planned stage that failed makes the frame Incomplete");
        check(results[0].missing == std::vector<std::string>{"person_embedder"},
              "and the missing stage is named — failed, not skipped, not timed out");
    }

    void test_a_skipped_branch_is_a_complete_frame() {
        std::vector<FrameResult> results;
        FrameCollector collector([&](FrameResult&& r) { results.push_back(std::move(r)); }, 16,
                                 1500);
        Dag dag;
        dag.add(std::make_unique<FakeStage>("detect", std::vector<std::string>{FRAME_INPUT},
                                            std::vector<std::string>{FRAME_INPUT}, DETECTIONS,
                                            1));
        dag.add(std::make_unique<FakeStage>(
            "crop", std::vector<std::string>{DETECTIONS, FRAME_INPUT},
            std::vector<std::string>{DETECTIONS}, "ship_crops", 0));
        dag.add(std::make_unique<FakeStage>(
            "ship_segmenter", std::vector<std::string>{"ship_crops"},
            std::vector<std::string>{"ship_crops"}, "masks", 1));
        auto state = a_frame();
        collector.open(state, {"detect", "crop"});
        CollectorObserver observer(collector, state->tag());
        dag.execute(*state, observer);
        collector.seal(state->tag());
        check(results.size() == 1 && results[0].reason == FinishReason::Complete &&
                  results[0].missing.empty(),
              "a frame with no ships is Complete: the segmenter was a skip, not a failure");
    }

}  // namespace

// -- WorkerScratch: a buffer is reused only once nobody else holds it -----------------
void scratch_pool_reuses_only_released_buffers() {
    WorkerScratch scratch(Device::cuda(0));
    std::shared_ptr<DeviceBuffer> first = scratch.acquire("crops", 1024);
    std::shared_ptr<DeviceBuffer> second = scratch.acquire("crops", 1024);
    check(first.get() != second.get(),
          "a held buffer is not handed out again (the timed-out request still points at it)");
    check(scratch.held("crops") == 2, "both are held");
    const DeviceBuffer* released = first.get();
    first.reset();
    std::shared_ptr<DeviceBuffer> third = scratch.acquire("crops", 1024);
    check(third.get() == released, "a released buffer is the one reused");
    check(scratch.held("crops") == 2, "still two held: the reused one and `second`");
    std::shared_ptr<DeviceBuffer> larger = scratch.acquire("crops", 4096);
    check(
        larger->bytes() >= 4096 && larger.get() != second.get() && larger.get() != third.get(),
        "a request for more bytes than any free buffer holds gets a new one");
}

void scratch_pool_refuses_unbounded_growth() {
    WorkerScratch scratch(Device::cuda(0));
    std::vector<std::shared_ptr<DeviceBuffer>> held;
    for (size_t i = 0; i < WorkerScratch::kMaxHeldPerName; ++i) {
        held.push_back(scratch.acquire("frames", 256));
    }
    bool refused = false;
    try {
        scratch.acquire("frames", 256);
    } catch (const ServerStateError& error) {
        refused = std::string(error.what()).find("still held") != std::string::npos;
    }
    check(refused, "the pool refuses past its cap, naming the held payloads");
    held.pop_back();
    check(scratch.acquire("frames", 256) != nullptr, "one release, one more acquire");
}

int main() {
    // This binary opens devices, so it consults the container rule itself — the hook
    // knows its name, but a rule only the hook enforces is not a rule (CLAUDE.md).
    shipinfer::runtime::require_container("csrc test_pipeline");
    // The graph tests need no device and run first; the scratch tests allocate device
    // memory and skip — counted, on stderr — where there is none, so a device-less run
    // still exercises what this binary exists for instead of terminating on the way in.
    test_both_clocks_survive_the_capture();
    test_a_stage_whose_input_is_empty_is_skipped_not_failed();
    test_a_failing_stage_does_not_end_the_frame();
    test_the_collector_sees_planned_delivered_and_missing();
    test_a_skipped_branch_is_a_complete_frame();
    test_a_frame_carrying_only_a_surface_still_has_pixels();
    test_a_frame_with_no_pixels_is_refused_by_name();
    test_an_incomplete_surface_never_becomes_a_buffer();
    test_a_device_frame_from_a_gpu_this_process_lacks_is_refused_by_name();
    if (has_device()) {
        test_the_pixel_seam_reads_a_padded_surface_as_a_surface();
        test_the_intake_frees_the_decoders_slot_and_keeps_the_pixels();
        test_two_resolutions_on_one_gpu_each_get_pool_hits();
        test_a_surface_holds_its_pool_alive();
        test_a_surface_outlives_the_sink_that_made_it();
        test_the_sink_turns_either_representation_into_one_work_item();
        scratch_pool_reuses_only_released_buffers();
        scratch_pool_refuses_unbounded_growth();
    } else {
        skip("no CUDA device for the worker-scratch pool or padded-surface tests");
    }
    std::printf("%d checks, %d failure(s), %d skipped\n", checks, failures, skips);
    return failures == 0 ? 0 : 1;
}
