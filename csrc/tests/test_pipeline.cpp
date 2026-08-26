// The graph's planning and reassembly wiring — `tests/pipeline/test_graph.py`'s claims, with no
// device: fake stages that mark names on the state, and the real Dag, FrameState and collector.

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/stage.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/runtime/containment.h"

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
    test_a_stage_whose_input_is_empty_is_skipped_not_failed();
    test_a_failing_stage_does_not_end_the_frame();
    test_the_collector_sees_planned_delivered_and_missing();
    test_a_skipped_branch_is_a_complete_frame();
    if (has_device()) {
        scratch_pool_reuses_only_released_buffers();
        scratch_pool_refuses_unbounded_growth();
    } else {
        skip("no CUDA device for the worker-scratch pool tests");
    }
    std::printf("%d checks, %d failure(s), %d skipped\n", checks, failures, skips);
    return failures == 0 ? 0 : 1;
}
