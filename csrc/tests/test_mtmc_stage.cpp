// `MtmcStage`'s JOIN: which rows it can contribute, and where the answer lands.
//
// AGAINST A FAKE TRACKER, deliberately. The tracker's own answers are
// `test_cluster_parity.cpp`'s business and the barrier's are `test_mtmc_barrier.cpp`'s; what
// this stage owns is reading each detection's track id and embedding out of the frame's
// batches, handing its camera's rows to the barrier, and scattering the group's answer back
// onto the detector's own indices.
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/emission.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/graph/state.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::ClusterObservation;
    using shipinfer::mtmc::IdentitySizes;
    using shipinfer::mtmc::InstantBarrier;
    using shipinfer::mtmc::TrackKey;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // Hands every track a global id equal to its track id times ten, and records what it saw.
    // Enough to tell "the stage gave me this row" from "the stage gave me the wrong row".
    class ScriptedTracker : public mtmc::ClusterTracker {
      public:
        std::map<TrackKey, int64_t> ids(
            const std::vector<ClusterObservation>& instant) override {
            seen = instant;
            std::map<TrackKey, int64_t> out;
            for (const ClusterObservation& observation : instant) {
                out[observation.key] = observation.key.track_id * 10;
            }
            return out;
        }

        IdentitySizes sizes() const override { return IdentitySizes{}; }

        std::vector<ClusterObservation> seen;
    };

    // A TRACKER THAT ANSWERS `-1`, which is what the seam does for an observation the gate
    // did not admit: "no identity yet", not an identity called -1.
    class GatingTracker : public mtmc::ClusterTracker {
      public:
        std::map<TrackKey, int64_t> ids(
            const std::vector<ClusterObservation>& instant) override {
            std::map<TrackKey, int64_t> out;
            for (const ClusterObservation& observation : instant) {
                out[observation.key] = mtmc::kUnidentified;
            }
            return out;
        }

        IdentitySizes sizes() const override { return IdentitySizes{}; }
    };

    // A TRACKER THAT REFUSES the way real data makes the real one refuse: a duplicate
    // (camera, track) in one frame, a zero embedding, a second embedding width.
    class RefusingTracker : public mtmc::ClusterTracker {
      public:
        std::map<TrackKey, int64_t> ids(const std::vector<ClusterObservation>&) override {
            throw InferenceError("cam0#7 appears twice in one instant");
        }

        IdentitySizes sizes() const override { return IdentitySizes{}; }
    };

    Detection box(float x, int class_id, int index) {
        Detection det;
        det.x1 = x;
        det.y1 = 10.0f;
        det.x2 = x + 40.0f;
        det.y2 = 300.0f;
        det.score = 0.9f;
        det.class_id = class_id;
        det.index = index;
        return det;
    }

    std::unique_ptr<FrameState> frame_with(const std::string& camera, int64_t frame_id,
                                           std::vector<Detection> detections) {
        FrameTag tag;
        tag.camera_id = camera;
        tag.frame_id = frame_id;
        tag.captured_ns = 1'000'000'000LL * frame_id;
        // BOTH STAMPS, the way `ingest/frame.h` sets them: the instant is keyed on the
        // CAPTURE (wall) clock, because the other plane keys the same barrier on
        // `captured_unix_ns` and two planes bucketing one clip differently is two different
        // sets of global ids.
        tag.captured_unix_ns = 1'700'000'000'000'000'000LL + 1'000'000'000LL * frame_id;
        auto state = std::make_unique<FrameState>(tag, 1080, 1920, 20.0f);
        state->set_detections(std::move(detections));
        state->set_detected(true);
        return state;
    }

    void attach(FrameState& state, const std::string& name, int width,
                const std::vector<int>& indices, const std::vector<float>& data) {
        ObjectBatch batch;
        batch.name = name;
        batch.width = width;
        batch.object_indices = indices;
        batch.data = data;
        state.attach(std::move(batch));
    }

    InstantBarrier::Options options(int workers = 1) {
        InstantBarrier::Options built;
        built.sync_window_s = 0.06;
        built.workers = workers;
        return built;
    }

    void a_tracked_and_embedded_row_gets_its_groups_id() {
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0)});
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0}, {1.0f, 0.0f});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* out = state->batch("mtmc_out");

        check(rows == 1, "one row in, one row out");
        check(out != nullptr && out->object_indices == std::vector<int>({0}),
              "on the DETECTION's index");
        check(out != nullptr && out->rows() == 1 && out->row(0)[0] == 70.0f,
              "carrying the group's id for that track");
    }

    void the_key_is_camera_and_TRACK_not_the_detection_row() {
        // Identity is keyed by (camera, track) because a detection is one frame's observation
        // and an identity has to persist across frames. Row 0 holding track 7 must reach the
        // tracker as `cam0#7`, not `cam0#0`.
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0)});
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0}, {1.0f, 0.0f});

        stage.run(*state);

        check(tracker->seen.size() == 1, "one observation reached the tracker");
        check(!tracker->seen.empty() && tracker->seen[0].key == TrackKey{"cam0", 7},
              "keyed by the TRACK id");
        check(!tracker->seen.empty() && tracker->seen[0].box_index == 0,
              "and it carries the detection index for the scatter back");
    }

    void an_untracked_row_is_passed_over_rather_than_given_somebody_elses_id() {
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0), box(100, 0, 1)});
        // Only row 0 was tracked; row 1 has an embedding but no track id.
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0, 1}, {1.0f, 0.0f, 0.0f, 1.0f});

        stage.run(*state);
        const ObjectBatch* out = state->batch("mtmc_out");

        check(tracker->seen.size() == 1, "only the tracked row reached the tracker");
        check(out != nullptr && out->object_indices == std::vector<int>({0}),
              "and only it got an id -- an untracked row has nothing an identity can hold");
    }

    void a_row_with_no_embedding_is_passed_over_too() {
        // Cross-camera identity is decided on APPEARANCE, so a row no embedder answered for
        // cannot take part. `records.cpp` leaves its global id null.
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0), box(100, 8, 1)});
        attach(*state, "track_out", 1, {0, 1}, {7.0f, 8.0f});
        attach(*state, "embed_out", 2, {1}, {0.0f, 1.0f});  // only row 1 embedded

        stage.run(*state);
        const ObjectBatch* out = state->batch("mtmc_out");

        check(tracker->seen.size() == 1, "only the embedded row reached the tracker");
        check(out != nullptr && out->object_indices == std::vector<int>({1}), "and only it");
    }

    void a_row_is_found_in_whichever_embedder_holds_it() {
        // A chain embeds people and ships separately, and one row is in exactly one of them.
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_person_out", "embed_ship_out"},
                        barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0), box(100, 8, 1)});
        attach(*state, "track_out", 1, {0, 1}, {7.0f, 8.0f});
        attach(*state, "embed_person_out", 2, {0}, {1.0f, 0.0f});
        attach(*state, "embed_ship_out", 2, {1}, {0.0f, 1.0f});

        stage.run(*state);
        const ObjectBatch* out = state->batch("mtmc_out");

        check(tracker->seen.size() == 2, "both rows reached the tracker");
        check(out != nullptr && out->object_indices == std::vector<int>({0, 1}),
              "and both got ids, each from its own embedder");
    }

    void a_camera_with_nothing_to_report_still_reports() {
        // Otherwise the instant it belongs to waits for it until the window runs out, and
        // every other camera in the group is answered late. `needs` is empty for that reason.
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0)});
        attach(*state, "track_out", 1, {}, {});  // present and EMPTY
        attach(*state, "embed_out", 2, {}, {});

        const StageOutcome outcome = stage.run(*state);
        const ObjectBatch* out = state->batch("mtmc_out");

        check(outcome.ran(), "the stage ran");
        check(outcome.rows == 0, "with no rows");
        check(out != nullptr && out->empty(), "and the output name exists and is empty");
        check(barrier->instant_stats().count(mtmc::kClosedComplete) == 1,
              "the instant closed on this camera's report rather than waiting for it");
    }

    void the_answer_is_read_out_by_key_and_never_by_position() {
        // The group's answer covers every camera, so this frame must read ITS rows out by the
        // key it chose. A positional read would hand cam0 cam1's id, plausibly.
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        // Track id 3 on detection row 1: if the scatter used the row or the position, the id
        // would be 0 or 70 rather than 30.
        auto state = frame_with("cam0", 1, {box(0, 0, 0), box(100, 0, 1)});
        attach(*state, "track_out", 1, {1}, {3.0f});
        attach(*state, "embed_out", 2, {0, 1}, {1.0f, 0.0f, 0.0f, 1.0f});

        stage.run(*state);
        const ObjectBatch* out = state->batch("mtmc_out");

        check(out != nullptr && out->object_indices == std::vector<int>({1}),
              "the detection index, which is 1");
        check(out != nullptr && out->rows() == 1 && out->row(0)[0] == 30.0f,
              "and the id for TRACK 3, which is 30");
    }

    // doc: long the policy, the frame it protects, and the counter that keeps it visible
    void a_frame_with_no_capture_stamp_is_refused_rather_than_bucketed_at_zero() {
        // A source that never stamps would put every camera's every frame into ONE instant,
        // which closes once and makes everything after it late for the life of the process --
        // read as clock skew rather than as the wiring fault it is. The Python element's
        // validator refuses it; so does this (#222's review asked for the same zero refusal
        // on both planes).
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<ScriptedTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        FrameTag tag;
        tag.camera_id = "cam0";
        tag.frame_id = 1;
        tag.captured_ns = 1'000'000'000LL;  // the STEADY stamp is set; the capture one is not
        auto state = std::make_unique<FrameState>(tag, 1080, 1920, 20.0f);
        state->set_detections({box(0, 0, 0)});
        state->set_detected(true);
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0}, {1.0f, 0.0f});
        // THROUGH THE OUTCOME, not a `catch`: `Stage::run` is a template method that turns
        // any exception into a `Failed` outcome carrying its message and never lets one
        // escape into the graph. A test that caught would be testing the wrong seam.
        const StageOutcome outcome = stage.run(*state);

        check(outcome.status == StageStatus::Failed,
              "the frame fails its stage rather than being bucketed at zero");
        check(outcome.error.find("no capture stamp") != std::string::npos,
              "with the reason, got: " + (outcome.error.empty() ? "(none)" : outcome.error));
        check(outcome.error.find("cam0") != std::string::npos,
              "and the camera, because 'some frame' sends the reader to fifty of them");
    }

    void an_unidentified_row_is_not_published_as_identity_minus_one() {
        // The gate admits nothing for a track that is too small or too new, and the seam
        // answers `kUnidentified` for it rather than dropping the entry -- "unidentifiable"
        // and "absent" are different facts. What must NOT happen is that sentinel reaching an
        // event as a global id, which is a -1 on somebody's screen. Measured on real footage
        // that this is the common case, not the rare one: at the gate's defaults every
        // observation comes back unidentified (`CSRC-MTMC-GATE-OPTIONS`).
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<GatingTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0)});
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0}, {1.0f, 0.0f});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* out = state->batch("mtmc_out");

        check(rows == 0, "an unidentified row publishes nothing");
        check(out != nullptr && out->rows() == 0,
              "and the payload is attached and empty rather than carrying -1 as an id");
    }

    void a_refused_instant_costs_the_GROUP_its_ids_and_not_the_closing_frame() {
        // THE POLICY, settled here rather than by whichever `catch` is nearest (#221 round 3
        // asked for exactly that). `barrier.h`: a throwing association releases every waiter
        // with `kDroppedFailed` AND rethrows into the thread that closed the bucket -- whose
        // frame would then fail its stage and be retired as a reassembly TIMEOUT, published a
        // window late in the fault channel for a frame whose only problem is somebody else's
        // duplicate row. That is #215's lesson one level up.
        //
        // So the group loses its ids for that instant -- every camera publishes a null
        // global_id, which the event schema already means -- and the frame completes. Counted
        // on the tracker, because one tracker serves every worker on a slot.
        auto barrier = std::make_shared<InstantBarrier>(options());
        auto tracker = std::make_shared<RefusingTracker>();
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0)});
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0}, {1.0f, 0.0f});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* out = state->batch("mtmc_out");

        check(rows == 0, "the refused instant publishes no ids");
        check(out != nullptr,
              "and the payload is still attached, so a reader joins on a name "
              "that exists rather than a missing key");
        check(tracker->refused_instants() == 1,
              "and the refusal is COUNTED, not swallowed: a run with data the gate will not "
              "take says so, and the bench prints it per slot");
    }

    void a_missed_instant_publishes_no_ids_rather_than_wrong_ones() {
        // `workers = 1` is zero permits, so the never-starve guard fires and this frame is
        // emitted with an honest gap. Its rows still took part in the group's association --
        // only the ANSWER is not delivered to it.
        auto barrier = std::make_shared<InstantBarrier>(options(/*workers=*/1));
        auto tracker = std::make_shared<ScriptedTracker>();
        barrier->camera_added("cam0");
        barrier->camera_added("cam-absent");
        MtmcStage stage("mtmc", "mtmc_out", "track_out", {"embed_out"}, barrier, tracker);
        auto state = frame_with("cam0", 1, {box(0, 0, 0)});
        attach(*state, "track_out", 1, {0}, {7.0f});
        attach(*state, "embed_out", 2, {0}, {1.0f, 0.0f});

        const StageOutcome outcome = stage.run(*state);
        const ObjectBatch* out = state->batch("mtmc_out");

        check(outcome.ran(), "the stage RAN, so the collector gets its slot on time");
        check(outcome.rows == 0, "with no ids");
        check(out != nullptr && out->empty(), "and an empty batch rather than a missing key");
        check(barrier->frame_stats().count(mtmc::kMissedWouldStarve) == 1,
              "counted as the gap it is");
    }

}  // namespace

int main() {
    a_tracked_and_embedded_row_gets_its_groups_id();
    the_key_is_camera_and_TRACK_not_the_detection_row();
    an_untracked_row_is_passed_over_rather_than_given_somebody_elses_id();
    a_row_with_no_embedding_is_passed_over_too();
    a_row_is_found_in_whichever_embedder_holds_it();
    a_camera_with_nothing_to_report_still_reports();
    the_answer_is_read_out_by_key_and_never_by_position();
    a_frame_with_no_capture_stamp_is_refused_rather_than_bucketed_at_zero();
    an_unidentified_row_is_not_published_as_identity_minus_one();
    a_refused_instant_costs_the_GROUP_its_ids_and_not_the_closing_frame();
    a_missed_instant_publishes_no_ids_rather_than_wrong_ones();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
