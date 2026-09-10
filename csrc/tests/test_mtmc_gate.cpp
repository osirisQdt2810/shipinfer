// The two gates before the expensive part: too small, and too new to trust yet.
//
// Both are about one failure -- a bad crop produces a CONFIDENT embedding -- so what these
// tests pin is not arithmetic but the order (height first, then age) and the boundedness of the
// one piece of state. Pure, offline, no submodule.
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/pipeline/mtmc/gate.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::ClusterObservation;
    using shipinfer::mtmc::ObservationGate;
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

    // `height` is the box's height in pixels of a 1080-tall frame, so the fraction is readable:
    // 200 is well over a ninth, 50 is well under.
    ClusterObservation seen(const std::string& camera, int64_t track, float height) {
        ClusterObservation out;
        out.key = TrackKey{camera, track};
        out.embedding = {1.0f, 0.0f};
        out.box[0] = 0.0f;
        out.box[1] = 0.0f;
        out.box[2] = 40.0f;
        out.box[3] = height;
        out.frame_width = 1920;
        out.frame_height = 1080;
        return out;
    }

    ObservationGate::Options options(int min_hits = 3, double min_height = 1.0 / 9.0) {
        ObservationGate::Options built;
        built.min_hits = min_hits;
        built.min_height_fraction = min_height;
        return built;
    }

    void the_options_are_refused_when_they_would_admit_anything() {
        bool hits = false;
        bool height = false;
        try {
            ObservationGate bad(options(0));
        } catch (const ConfigError& error) {
            hits = std::string(error.what()).find("min_hits must be at least 1") !=
                   std::string::npos;
        }
        try {
            ObservationGate bad(options(3, 1.0));
        } catch (const ConfigError& error) {
            height = std::string(error.what()).find("must be in [0, 1)") != std::string::npos;
        }

        check(hits, "min_hits 0 would admit a track on the frame it was first seen");
        check(height, "and a fraction of 1.0 admits nothing at all, which is not a threshold");
    }

    void a_track_must_be_seen_min_hits_times_before_it_is_admitted() {
        ObservationGate gate(options(3));

        check(gate.filter({seen("cam0", 1, 200)}).empty(), "not on the first frame");
        check(gate.filter({seen("cam0", 1, 200)}).empty(), "not on the second");
        check(gate.filter({seen("cam0", 1, 200)}).size() == 1, "admitted on the third");
        check(gate.hits(TrackKey{"cam0", 1}) == 3, "and its run is three");
    }

    void a_track_that_is_too_small_never_accrues_age() {
        // The order that is load-bearing: age counts consecutive frames in which the track was
        // ALSO large enough. Otherwise a figure walking in from the distance banks three
        // frames while unusably small and enters the matrix on its first usable frame with the
        // gate already satisfied.
        ObservationGate gate(options(3));
        for (int frame = 0; frame < 5; ++frame) {
            check(gate.filter({seen("cam0", 1, 50)}).empty(),
                  "a small track is never admitted");
        }
        check(gate.hits(TrackKey{"cam0", 1}) == 0, "and banks no age at all");

        // Now it comes close: it still needs its three qualifying frames.
        check(gate.filter({seen("cam0", 1, 200)}).empty(), "the first usable frame is hit 1");
        check(gate.filter({seen("cam0", 1, 200)}).empty(), "then 2");
        check(gate.filter({seen("cam0", 1, 200)}).size() == 1, "and only then admitted");
    }

    void a_missed_instant_restarts_the_run_and_bounds_the_map() {
        // Replacing the map rather than pruning it is what enforces "consecutive", and it is
        // also what keeps the map bounded by the tracks in flight instead of by uptime.
        ObservationGate gate(options(3));
        gate.filter({seen("cam0", 1, 200)});
        gate.filter({seen("cam0", 1, 200)});
        check(gate.hits(TrackKey{"cam0", 1}) == 2, "two consecutive");

        gate.filter({seen("cam1", 9, 200)});  // cam0#1 not reported

        check(gate.hits(TrackKey{"cam0", 1}) == 0, "the run restarted from nothing");
        check(gate.size() == 1, "and the map holds only the tracks in flight");
    }

    void the_height_test_is_strictly_greater_than_the_threshold() {
        // `> min_height_fraction`, the reference's comparison. A box exactly at the threshold
        // is not admitted, which matters because 1/9 of 1080 is exactly 120.
        ObservationGate gate(options(1, 120.0 / 1080.0));

        check(gate.filter({seen("cam0", 1, 120)}).empty(), "exactly at the threshold: out");
        check(gate.filter({seen("cam0", 2, 121)}).size() == 1, "a pixel over: in");
    }

    void a_frame_with_no_extent_admits_nothing_rather_than_dividing_by_zero() {
        ObservationGate gate(options(1));
        ClusterObservation broken = seen("cam0", 1, 200);
        broken.frame_height = 0;

        check(gate.filter({broken}).empty(),
              "a frame whose extent nobody filled in is a wiring fault, and admitting every "
              "track through it is the failure this gate exists for");
        check(gate.hits(TrackKey{"cam0", 1}) == 0, "and it accrues no age either");
    }

    void the_admitted_keep_their_input_order() {
        // The caller zips the gate's answer against the clusterer's labels, so order is part
        // of the contract rather than an accident of the container.
        ObservationGate gate(options(1));

        const auto admitted =
            gate.filter({seen("cam2", 1, 200), seen("cam0", 1, 200), seen("cam1", 1, 200)});

        check(admitted.size() == 3, "all three qualify");
        check(admitted[0].key.camera_id == "cam2" && admitted[2].key.camera_id == "cam1",
              "and they come back in INPUT order, not sorted");
    }

    void reset_forgets_every_run() {
        ObservationGate gate(options(3));
        gate.filter({seen("cam0", 1, 200)});
        gate.filter({seen("cam0", 1, 200)});

        gate.reset();

        check(gate.size() == 0, "nothing held");
        check(gate.filter({seen("cam0", 1, 200)}).empty(), "and the run starts again from one");
    }

}  // namespace

int main() {
    the_options_are_refused_when_they_would_admit_anything();
    a_track_must_be_seen_min_hits_times_before_it_is_admitted();
    a_track_that_is_too_small_never_accrues_age();
    a_missed_instant_restarts_the_run_and_bounds_the_map();
    the_height_test_is_strictly_greater_than_the_threshold();
    a_frame_with_no_extent_admits_nothing_rather_than_dividing_by_zero();
    the_admitted_keep_their_input_order();
    reset_forgets_every_run();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
