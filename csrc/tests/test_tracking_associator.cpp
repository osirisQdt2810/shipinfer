// The `Associator` seam and its registry — the LANE side, and offline by design.
//
// `tracking/associator.h` explains why this file exists apart from the stage's: a tracker
// reaches an external lane and a stage reaches `core/platform.h`, and the lane's CI job builds
// with g++ alone. So the tracker is tested here, without a `FrameState` anywhere in sight, and
// the stage's scatter is tested in `test_tracking_stage.cpp` against a fake.
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/pipeline/tracking/associator.h"

namespace {

    using namespace shipinfer;
    using shipinfer::tracking::Associator;

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

    void the_lanes_tracker_registers_itself() {
        // This binary exists only when the lane is in the build, so the registrar HAS run.
        check(tracking::ASSOCIATORS().has("shipvision"), "shipvision is registered");
        check(tracking::create_associator("shipvision", "reg") != nullptr, "and it builds");
    }

    void one_slot_is_one_associator_however_many_callers() {
        // `bench.cpp` builds one Dag per worker, so every worker asks for this slot's tracker.
        // Two associators would be two identity spaces for one camera on one slot.
        const auto first = tracking::create_associator("shipvision", "track_person");
        const auto second = tracking::create_associator("shipvision", "track_person");

        check(first.get() == second.get(), "the same associator, not a second identity space");
    }

    void two_slots_are_two_associators() {
        // The chain this plane now accepts: two track slots with DISJOINT selections over one
        // camera (`test_chain.py::test_disjoint_selections_are_permitted`). One associator is
        // one shard keyed by camera, so sharing it would have the two slots take turns
        // refusing each other's frames -- the Python plane builds one shard per element.
        const auto persons = tracking::create_associator("shipvision", "slot_persons");
        const auto ships = tracking::create_associator("shipvision", "slot_ships");

        check(persons.get() != ships.get(), "two slots, two identity spaces");
    }

    void two_slots_track_the_same_frame_of_one_camera() {
        // The defect this is really about, and it is not about pointers: BOTH slots see
        // `(cam0, 42)` -- one frame, two selections -- so a shared ordering guard would refuse
        // whichever asked second, on every frame, forever.
        const auto persons = tracking::create_associator("shipvision", "pair_persons");
        const auto ships = tracking::create_associator("shipvision", "pair_ships");

        // CAUGHT, so a regression prints a FAIL rather than aborting the binary on an
        // uncaught throw -- which is what the shared-shard version does here, verbatim:
        // "frame 1 reached the tracker after frame 1".
        try {
            for (int64_t frame = 1; frame <= 3; ++frame) {
                check(persons->ids("cam-pair", frame, {box(100, 100, 0.9f, 0)}).size() == 1,
                      "the person slot answers");
                check(ships->ids("cam-pair", frame, {box(400, 100, 0.9f, 8)}).size() == 1,
                      "and the ship slot answers the SAME frame id");
            }
        } catch (const InferenceError& error) {
            check(false, std::string("one slot refused the other's frame: ") + error.what());
        }
        check(persons->untracked_frames() == 0 && ships->untracked_frames() == 0,
              "and neither refused a frame");
    }

    void a_refused_frame_throws_so_the_caller_can_publish_it_untracked() {
        // The seam's one exception, documented on `Associator::ids`: a frame that does not
        // advance the camera's stream. `TrackStage` catches exactly this and publishes the
        // frame with no ids (`test_tracking_stage.cpp`).
        const auto associator = tracking::create_associator("shipvision", "slot_reorder");
        associator->ids("cam-reorder", 7, {box(100, 100, 0.9f, 0)});

        bool refused = false;
        try {
            associator->ids("cam-reorder", 6, {box(100, 100, 0.9f, 0)});
        } catch (const InferenceError& error) {
            refused = std::string(error.what()).find("after frame 7") != std::string::npos;
        }

        check(refused, "a frame that does not advance the stream is refused by InferenceError");
    }

    void a_confirmed_track_gets_a_positive_id() {
        const auto associator = tracking::create_associator("shipvision", "slot_confirm");
        std::vector<int> last;
        // ByteTrack confirms on a later hit, so one frame is not enough -- which is the
        // tracker's business and is exactly why the STAGE is tested against a fake instead.
        for (int64_t frame = 1; frame <= 4; ++frame) {
            last = associator->ids("cam-assoc", frame, {box(100, 100, 0.9f, 0)});
            check(last.size() == 1, "one id per detection, always");
        }
        check(!last.empty() && last[0] >= 1, "a repeated detection is eventually confirmed");
    }

    void a_frame_with_no_detections_answers_nothing() {
        const auto associator = tracking::create_associator("shipvision", "slot_empty");

        check(associator->ids("cam-empty", 1, {}).empty(), "no detections, no ids");
    }

    void the_registry_refuses_an_absent_name_with_its_own_error() {
        // `AssociatorRegistry::create` is public, so it cannot answer `std::out_of_range` for
        // a caller that did not ask `has` first -- the registry's vocabulary is ConfigError.
        bool refused = false;
        try {
            tracking::ASSOCIATORS().create("no_such_tracker");
        } catch (const ConfigError& error) {
            refused =
                std::string(error.what()).find("no tracker is registered") != std::string::npos;
        }

        check(refused, "create() on an absent name is a ConfigError, not out_of_range");
    }

    void made_associators_lists_what_was_built() {
        // The run's read path for per-slot counters: `bench.cpp` has no way into a worker's
        // Dag, so it asks here.
        tracking::create_associator("shipvision", "slot_listed");
        bool found = false;
        for (const tracking::MadeAssociator& made : tracking::made_associators()) {
            if (made.slot != "slot_listed") continue;
            found = made.impl == "shipvision" && made.associator != nullptr;
        }

        check(found, "a built associator is listed by impl and slot");
    }

    void an_unknown_impl_is_refused_by_name() {
        bool refused = false;
        try {
            tracking::create_associator("no_such_tracker", "slot_unknown");
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("unknown tracker") != std::string::npos;
        }

        check(refused, "an unknown impl is a ConfigError naming it");
    }

}  // namespace

int main() {
    the_lanes_tracker_registers_itself();
    one_slot_is_one_associator_however_many_callers();
    two_slots_are_two_associators();
    two_slots_track_the_same_frame_of_one_camera();
    a_refused_frame_throws_so_the_caller_can_publish_it_untracked();
    a_confirmed_track_gets_a_positive_id();
    a_frame_with_no_detections_answers_nothing();
    the_registry_refuses_an_absent_name_with_its_own_error();
    made_associators_lists_what_was_built();
    an_unknown_impl_is_refused_by_name();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
