// `TrackStage`'s SCATTER — which rows get an id, which get none, and whose index they carry.
//
// AGAINST A FAKE ASSOCIATOR, deliberately. The tracker's own behaviour is
// `test_tracking_associator.cpp` and shipvision's suite; what this stage owns is the mapping
// from ids to `ObjectBatch` rows, and a fake makes `-1` reachable on demand instead of hoping
// ByteTrack declines to confirm. It also keeps this test out of the external lane -- it reaches
// `core/platform.h` through `graph/state.h`, so it could not live in the lane's offline job.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/emission.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/tracking/associator.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // Answers whatever it was told to, so `-1` is reachable without a real tracker declining.
    class ScriptedAssociator : public tracking::Associator {
      public:
        explicit ScriptedAssociator(std::vector<int> answer) : answer_(std::move(answer)) {}

        std::vector<int> ids(const std::string& camera_id, int64_t frame_id,
                             const std::vector<Detection>& detections) override {
            seen_camera = camera_id;
            seen_frame = frame_id;
            seen_detections = detections.size();
            return answer_;
        }

        std::string seen_camera;
        int64_t seen_frame = 0;
        size_t seen_detections = 0;

      private:
        std::vector<int> answer_;
    };

    // Refuses every frame the way `TrackerShard` refuses a reordered one, so the stage's
    // handling of a ROUTINE refusal is reachable without a real tracker and a race.
    class RefusingAssociator : public tracking::Associator {
      public:
        std::vector<int> ids(const std::string& camera_id, int64_t frame_id,
                             const std::vector<Detection>&) override {
            throw InferenceError("camera '" + camera_id + "': frame " +
                                 std::to_string(frame_id) + " reached the tracker after 99");
        }
    };

    // A tracker whose configuration is wrong fails on every frame, and that IS a fault -- so
    // the stage must not turn it into a quiet untracked frame.
    class MisconfiguredAssociator : public tracking::Associator {
      public:
        std::vector<int> ids(const std::string&, int64_t,
                             const std::vector<Detection>&) override {
            throw ConfigError("tracker 'shipvision': max_age must be positive");
        }
    };

    Detection box(float x) {
        Detection det;
        det.x1 = x;
        det.y1 = 10.0f;
        det.x2 = x + 40.0f;
        det.y2 = 90.0f;
        det.score = 0.9f;
        return det;
    }

    std::unique_ptr<FrameState> frame_with(const std::string& camera, int64_t frame_id,
                                           std::vector<Detection> detections) {
        FrameTag tag;
        tag.camera_id = camera;
        tag.frame_id = frame_id;
        auto state = std::make_unique<FrameState>(tag, 1080, 1920, 20.0f);
        state->set_detections(std::move(detections));
        // BOTH, the way `DetectStage::do_run` does it: `set_detections` fills the vector and
        // `set_detected` records that the detector ANSWERED, which is the flag `available()`
        // reads. A frame with an empty vector and this flag set is "no objects"; without it,
        // it is "the detector never ran", and the two are different events for a tracker.
        state->set_detected(true);
        return state;
    }

    void every_confirmed_id_lands_on_its_own_detections_index() {
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{7, 8, 9});
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 42, {box(0), box(100), box(200)});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 3, "three ids, three rows");
        check(batch != nullptr, "attached under the output name");
        if (batch == nullptr) return;
        check(batch->width == 1, "one id per row");
        check(batch->object_indices == std::vector<int>({0, 1, 2}), "the detections' indices");
        check(batch->row(0)[0] == 7.0f && batch->row(2)[0] == 9.0f, "in order");
    }

    void an_unmatched_detection_gets_no_row_at_all() {
        // The middle one matched nothing. A `-1` in the batch would arrive in an event as
        // `track_id: -1`, which is a lie a consumer cannot tell from an id.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{7, -1, 9});
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 1, {box(0), box(100), box(200)});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 2, "two rows for three detections");
        check(batch != nullptr && batch->object_indices == std::vector<int>({0, 2}),
              "and the surviving indices are 0 and 2, not 0 and 1");
    }

    void the_camera_and_frame_reach_the_associator() {
        // The sharding is per CAMERA and the ordering guard is per FRAME, so both have to
        // arrive -- a stage that passed the wrong camera would merge two views silently.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{1});
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam-seven", 99, {box(0)});

        stage.run(*state);

        check(associator->seen_camera == "cam-seven", "the frame's camera");
        check(associator->seen_frame == 99, "the frame's id");
        check(associator->seen_detections == 1, "and its detections");
    }

    Detection classed(float x, int class_id) {
        Detection det = box(x);
        det.class_id = class_id;
        return det;
    }

    void a_selection_tracks_only_its_own_rows() {
        // The cross-plane finding: the Python element states `selects_rows = True` and feeds
        // its tracker only the declared rows, so a plane that tracked every row for a
        // `classes: [ship]` slot would emit ids the other never does AND different ids for the
        // ships, because association and the per-camera counter saw the people too.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{5, 6});
        TrackStage stage("track", "track_out", /*class_id=*/8, associator);
        auto state = frame_with(
            "cam0", 1, {classed(0, 0), classed(100, 8), classed(200, 0), classed(300, 8)});

        stage.run(*state);
        const ObjectBatch* batch = state->batch("track_out");

        check(associator->seen_detections == 2,
              "only the two class-8 rows reached the tracker");
        check(batch != nullptr, "attached");
        if (batch == nullptr) return;
        // 1 and 3 are the class-8 detections' own indices, not 0 and 1.
        check(batch->object_indices == std::vector<int>({1, 3}), "the selected rows' indices");
    }

    void a_declared_empty_selection_tracks_nothing() {
        // `kNoClass` is negative like `kAnyClass`, so a `>= 0` test would read a declared EMPTY
        // selection as every row -- the opposite of what it means, and the same trap
        // `CropStage`'s comment records.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{1, 2});
        TrackStage stage("track", "track_out", CropSpec::kNoClass, associator);
        auto state = frame_with("cam0", 1, {classed(0, 0), classed(100, 8)});

        const size_t rows = stage.run(*state).rows;

        check(associator->seen_detections == 0, "no rows reached the tracker");
        check(rows == 0, "and none came back");
    }

    void a_frame_with_no_detections_attaches_an_empty_batch() {
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{});
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 1, {});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 0, "no rows");
        check(batch != nullptr && batch->empty(), "the name exists and is empty");
    }

    void a_refused_frame_runs_with_no_ids_rather_than_failing() {
        // THE COST OF GETTING THIS WRONG is not a missing id, it is a TIMEOUT. A Failed stage
        // never delivers its slot to the collector, so the frame waits in `pending_` for
        // `sweep()` and is retired at `timeout_ms` in the fault channel -- one reassembly
        // window late, for a frame whose only problem is that it has no ids.
        auto associator = std::make_shared<RefusingAssociator>();
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 41, {box(0), box(100)});

        const StageOutcome outcome = stage.run(*state);
        const ObjectBatch* batch = state->batch("track_out");

        check(outcome.ran(), "the stage RAN, so the collector gets its slot on time");
        check(outcome.error.empty(), "and a reordering is not reported as a stage fault");
        check(outcome.rows == 0, "with no rows");
        check(batch != nullptr && batch->empty(), "the output name exists and is empty");
        check(associator->untracked_frames() == 1, "and the frame is counted, not swallowed");
    }

    void an_ordinary_frame_counts_no_untracked() {
        // The counter's other half: a number that only goes up is not evidence.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{7});
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 1, {box(0)});

        stage.run(*state);

        check(associator->untracked_frames() == 0, "a tracked frame is not an untracked one");
    }

    void a_misconfigured_tracker_still_fails_the_stage() {
        // Only `InferenceError` is the routine refusal. A ConfigError means the deploy is
        // wrong on every frame, and burying it as "untracked" would hide it behind a counter.
        auto associator = std::make_shared<MisconfiguredAssociator>();
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 1, {box(0)});

        const StageOutcome outcome = stage.run(*state);

        check(outcome.status == StageStatus::Failed, "a configuration fault still fails");
        check(outcome.error.find("max_age") != std::string::npos, "carrying its message");
        check(associator->untracked_frames() == 0, "and is not counted as a reordering");
    }

    // The Dag as the graph runs it, because `stage.run` called directly cannot see the defect
    // below: `Dag::runnable` is what decides whether a stage is offered a frame at all.
    class SilentObserver : public StageObserver {
      public:
        void planned(const std::vector<std::string>&) override {}
        void finished(const StageOutcome& outcome) override { outcomes.push_back(outcome); }

        std::vector<StageOutcome> outcomes;
    };

    // Counts calls and answers nothing, which is all this needs: the question is whether the
    // associator was ASKED.
    class CountingAssociator : public tracking::Associator {
      public:
        std::vector<int> ids(const std::string&, int64_t,
                             const std::vector<Detection>& dets) override {
            ++calls;
            last_size = dets.size();
            return {};
        }

        int calls = 0;
        size_t last_size = 0;
    };

    void a_zero_detection_frame_still_reaches_the_tracker() {
        // THE COST OF GETTING THIS WRONG is an invented identity. `needs` is "present and
        // non-empty", so a stage that NEEDED the detections was skipped on every frame the
        // detector answered with no boxes -- and a tracker that is not advanced does not age,
        // so a ship that left frame stays lost-but-alive and the next object near its last
        // box is published with its id. `track.py` advances on an empty `Detections` for
        // exactly this reason.
        auto associator = std::make_shared<CountingAssociator>();
        Dag dag;
        dag.add(std::make_unique<TrackStage>("track", "track_out", CropSpec::kAnyClass,
                                             associator));
        auto state = frame_with("cam0", 7, {});
        SilentObserver observer;

        dag.execute(*state, observer);

        check(associator->calls == 1, "the tracker was advanced by the empty frame");
        check(associator->last_size == 0, "with no detections");
        check(observer.outcomes.size() == 1 && observer.outcomes[0].ran(),
              "and the stage RAN, so the collector is not left waiting");
    }

    void a_frame_the_detector_never_answered_for_is_still_skipped() {
        // The OTHER arm, and it is not the same event: `track.py` takes `_untracked` when
        // `meta["detections"] is None`. `consumes` gates on `available()`, which is
        // `detected_`, so this stage is still absent from that frame's plan.
        auto associator = std::make_shared<CountingAssociator>();
        Dag dag;
        dag.add(std::make_unique<TrackStage>("track", "track_out", CropSpec::kAnyClass,
                                             associator));
        FrameTag tag;
        tag.camera_id = "cam0";
        tag.frame_id = 8;
        // NO `set_detected`, which is the whole difference from the test above.
        FrameState state(tag, 1080, 1920, 20.0f);
        SilentObserver observer;

        dag.execute(state, observer);

        check(associator->calls == 0, "no detector answer, no tracker call");
        check(observer.outcomes.size() == 1 &&
                  observer.outcomes[0].status == StageStatus::Skipped,
              "and the stage is skipped rather than run or failed");
    }

    void more_ids_than_detections_cannot_walk_off_the_end() {
        // Not a caller this tree has; a tracker that answered long would be a crash otherwise.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{1, 2, 3, 4});
        TrackStage stage("track", "track_out", CropSpec::kAnyClass, associator);
        auto state = frame_with("cam0", 1, {box(0)});

        const size_t rows = stage.run(*state).rows;

        check(rows == 1, "bounded by the detections, not by the answer");
    }

}  // namespace

int main() {
    every_confirmed_id_lands_on_its_own_detections_index();
    an_unmatched_detection_gets_no_row_at_all();
    the_camera_and_frame_reach_the_associator();
    a_selection_tracks_only_its_own_rows();
    a_declared_empty_selection_tracks_nothing();
    a_frame_with_no_detections_attaches_an_empty_batch();
    a_refused_frame_runs_with_no_ids_rather_than_failing();
    an_ordinary_frame_counts_no_untracked();
    a_misconfigured_tracker_still_fails_the_stage();
    a_zero_detection_frame_still_reaches_the_tracker();
    a_frame_the_detector_never_answered_for_is_still_skipped();
    more_ids_than_detections_cannot_walk_off_the_end();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
