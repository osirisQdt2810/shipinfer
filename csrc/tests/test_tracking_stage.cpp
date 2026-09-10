// `pipeline/tracking/stage.cpp` -- the stage that puts a track id on a detection's row.
//
// What is worth testing here is not the tracker (that is `test_tracking_shard.cpp` and
// shipvision's own suite) but the SCATTER: which rows get an id, which get none, and that the
// row indices are the detections' own so `events/records.cpp` can join them.
//
// A DETECTION THE TRACKER DID NOT CONFIRM MUST GET NO ROW. `-1` in an `ObjectBatch` would
// arrive in an event as `track_id: -1`, which is a lie a consumer cannot tell from an id.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/emission.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/tracking/registry.h"
#include "shipinfer/pipeline/tracking/stage.h"

namespace {

    using namespace shipinfer;
    // `TrackerShard` is in `shipinfer` (#169 put it there); the stage and its registry are in
    // `shipinfer::tracking`, like `pipeline::events`.
    using shipinfer::tracking::TrackStage;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    Detection box(float x, float y, float score, int class_id) {
        Detection det;
        det.x1 = x;
        det.y1 = y;
        det.x2 = x + 40.0f;
        det.y2 = y + 80.0f;
        det.score = score;
        det.class_id = class_id;
        return det;
    }

    // A pointer, because `FrameState` holds a mutex and is not copyable -- which is the
    // ADR-002 fact that one worker owns a frame for its whole life.
    std::unique_ptr<FrameState> frame_with(const std::string& camera, int64_t frame_id,
                                           std::vector<Detection> detections) {
        FrameTag tag;
        tag.camera_id = camera;
        tag.frame_id = frame_id;
        auto state = std::make_unique<FrameState>(tag, 1080, 1920, 20.0f);
        state->set_detections(std::move(detections));
        return state;
    }

    void a_confirmed_track_puts_its_id_on_the_rows_own_index() {
        TrackerShard shard;
        TrackStage stage("track", "track_out", shard);
        // Two frames, because ByteTrack confirms on the SECOND hit: one frame alone gives a
        // tentative track and no id, which is the tracker's business and not this stage's.
        for (int64_t id = 1; id <= 3; ++id) {
            auto state = frame_with("cam0", id, {box(100, 100, 0.9f, 0)});
            stage.run(*state);
            const ObjectBatch* batch = state->batch("track_out");
            check(batch != nullptr, "the batch is attached under the output name");
            if (batch == nullptr) return;
            check(batch->width == 1, "one id per row");
            if (!batch->empty()) {
                check(batch->object_indices[0] == 0, "the row index is the detection's own");
                check(batch->row(0)[0] >= 1.0f, "a confirmed id is positive");
            }
        }
    }

    void an_unconfirmed_detection_gets_no_row() {
        TrackerShard shard;
        TrackStage stage("track", "track_out", shard);
        // One frame only: nothing can be confirmed yet, so the batch must be EMPTY rather
        // than carrying a -1.
        auto state = frame_with("cam1", 1, {box(10, 10, 0.9f, 0)});
        stage.run(*state);
        const ObjectBatch* batch = state->batch("track_out");

        check(batch != nullptr, "an empty batch is still attached");
        if (batch != nullptr) check(batch->empty(), "no row for an unconfirmed detection");
    }

    void a_frame_with_no_detections_attaches_an_empty_batch() {
        TrackerShard shard;
        TrackStage stage("track", "track_out", shard);
        auto state = frame_with("cam2", 1, {});
        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 0, "no rows");
        check(batch != nullptr && batch->empty(), "the name exists and is empty");
    }

    void the_registry_builds_it_by_the_plans_impl_name() {
        // The graph asks by name and includes no tracker header; this is that call.
        TrackStageSpec spec;
        spec.slot = "track";
        spec.output = "track_out";
        spec.impl = "shipvision";
        std::unique_ptr<Stage> stage = tracking::create_track_stage(spec.impl, spec);

        check(stage != nullptr, "the registrar in stage.cpp ran");
        check(stage->name() == "track", "the stage takes the plan's slot name");
    }

    void an_unknown_impl_is_refused_by_name() {
        TrackStageSpec spec;
        spec.slot = "track";
        spec.output = "track_out";
        spec.impl = "no_such_tracker";
        bool refused = false;
        try {
            tracking::create_track_stage(spec.impl, spec);
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("unknown tracker") != std::string::npos;
        }

        check(refused, "an unknown impl is a ConfigError naming it");
    }

}  // namespace

int main() {
    a_confirmed_track_puts_its_id_on_the_rows_own_index();
    an_unconfirmed_detection_gets_no_row();
    a_frame_with_no_detections_attaches_an_empty_batch();
    the_registry_builds_it_by_the_plans_impl_name();
    an_unknown_impl_is_refused_by_name();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
