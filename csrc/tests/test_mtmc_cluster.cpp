// The cross-camera seam and its registry -- the lane-free half, so this runs offline.
//
// A fake tracker rather than the real one, deliberately: what this file owns is the registry's
// behaviour (one per slot, typed refusals, the lane-absent message) and the seam's contract.
// The real tracker's answers are `test_identity_parity.cpp`'s business.
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/ingest/omitted_lanes.h"
#include "shipinfer/pipeline/mtmc/cluster.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::ClusterObservation;
    using shipinfer::mtmc::ClusterTracker;
    using shipinfer::mtmc::IdentitySizes;
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

    // Answers one identity per camera, counting calls. Enough to see WHICH tracker a slot got.
    class CountingTracker : public ClusterTracker {
      public:
        std::map<TrackKey, int64_t> ids(const std::vector<ClusterObservation>& instant,
                                        const std::vector<std::string>& cameras) override {
            ++calls;
            rostered = cameras;
            std::map<TrackKey, int64_t> out;
            int64_t next = 0;
            for (const ClusterObservation& observation : instant) {
                out[observation.key] = next++;
            }
            return out;
        }

        //: The roster the seam was handed, so a test can see it arrive rather than assume it.
        std::vector<std::string> rostered;

        IdentitySizes sizes() const override {
            IdentitySizes out;
            out.identities = static_cast<size_t>(calls);
            return out;
        }

        int calls = 0;
    };

    const mtmc::ClusterRegistrar kFake("fake", [](const mtmc::ClusterOptions&) {
        return std::make_shared<CountingTracker>();
    });

    // A FACTORY THAT FAILS, which is what makes the null-cache fix testable. Registered here
    // rather than in the lane because the shape is the seam's, not the algorithm's: the lane's
    // real tracker is simply the first factory in the tree that CAN throw.
    // WHAT THE FACTORY WAS HANDED, so a test can check the chain's numbers arrived rather
    // than only that something was built.
    mtmc::ClusterOptions kSeen;

    const mtmc::ClusterRegistrar kOptions("options", [](const mtmc::ClusterOptions& options) {
        kSeen = options;
        return std::make_shared<CountingTracker>();
    });

    const mtmc::ClusterRegistrar kThrows(
        "throws", [](const mtmc::ClusterOptions&) -> std::shared_ptr<ClusterTracker> {
            throw ConfigError("bad algorithm config");
        });

    void the_registry_hands_out_one_tracker_per_slot() {
        // A cross-camera tracker IS the identity space for a group, so two instances would
        // issue two contradictory sets of global ids for the same objects.
        const auto first = mtmc::create_cluster_tracker("fake", "quay", {});
        const auto second = mtmc::create_cluster_tracker("fake", "quay", {});
        const auto other = mtmc::create_cluster_tracker("fake", "jetty", {});

        check(first.get() == second.get(),
              "one slot, one identity space, however many callers");
        check(first.get() != other.get(), "two slots are two groups and two identity spaces");
    }

    void an_unknown_impl_is_refused_by_name() {
        bool refused = false;
        try {
            mtmc::create_cluster_tracker("no_such", "quay", {});
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("unknown cross-camera tracker") !=
                      std::string::npos;
        }

        check(refused, "an unknown impl is a ConfigError naming it");
    }

    void the_registry_refuses_an_absent_name_with_its_own_error() {
        // `create` is public, so it cannot answer `std::out_of_range` for a caller that did
        // not ask `has` first -- the hole `AssociatorRegistry::create` was reviewed for.
        bool refused = false;
        try {
            mtmc::CLUSTERERS().create("no_such", {});
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("no cross-camera tracker is registered") !=
                      std::string::npos;
        }

        check(refused, "create() on an absent name is a ConfigError, not out_of_range");
    }

    void a_lane_less_build_blames_the_LANE_and_not_the_name() {
        // Only meaningful when the lane really is absent, which is the offline tier's normal
        // state -- so the assertion is on the MESSAGE, conditioned on the build.
        const std::string omitted = "," + omitted_lanes() + ",";
        if (omitted.find(",shipvision,") == std::string::npos) {
            check(true, "the shipvision lane is in this build, so there is nothing to blame");
            return;
        }
        std::string message;
        try {
            mtmc::create_cluster_tracker("shipvision", "quay", {});
        } catch (const ConfigError& error) {
            message = error.what();
        }

        check(message.find("was not compiled into this binary") != std::string::npos,
              "a lane-less build says the LANE is absent");
        check(message.find("unknown") == std::string::npos,
              "and does not call a correctly-spelled name unknown");
    }

    void the_chain_s_gate_options_reach_the_implementation() {
        // The knob this whole item exists for: the reference's production floor admits nothing
        // on footage whose subjects are small in frame, and until now no chain could say so.
        mtmc::ClusterOptions options;
        options.min_hits = 1;
        options.min_height_fraction = 0.02;
        const auto tracker = mtmc::create_cluster_tracker("options", "quay", options);

        check(tracker != nullptr, "the tracker is built with the chain's thresholds");
        check(kSeen.min_hits && *kSeen.min_hits == 1, "min_hits reached the factory");
        check(kSeen.min_height_fraction && *kSeen.min_height_fraction == 0.02,
              "and so did the height floor");
    }

    void one_slot_is_one_gate_and_a_second_answer_is_refused() {
        // The tracker is CACHED per (impl, slot), so a second caller asking for different
        // thresholds would silently get the first caller's gate -- a running configuration
        // that is in no file. One slot is one camera group is one gate.
        mtmc::ClusterOptions first;
        first.min_hits = 2;
        mtmc::create_cluster_tracker("options", "contested", first);
        mtmc::ClusterOptions second;
        second.min_hits = 5;
        std::string message;

        try {
            mtmc::create_cluster_tracker("options", "contested", second);
        } catch (const ConfigError& error) {
            message = error.what();
        }

        check(message.find("different gate options") != std::string::npos,
              "the second set is refused, got: " + (message.empty() ? "(nothing)" : message));
        check(mtmc::create_cluster_tracker("options", "contested", first) != nullptr,
              "and asking again with the SAME options is the ordinary cache hit");
    }

    void a_factory_that_throws_caches_nothing() {
        // `made()[{impl, slot}]` default-inserted BEFORE the factory ran, so a constructor that
        // threw left a null `shared_ptr` under that key -- and `made_cluster_trackers()` copies
        // every entry unfiltered, so the bench's per-slot report would dereference it: a
        // segfault in the REPORTING path, minutes after and nowhere near the configuration
        // error that caused it. Twice, because the second attempt is what would find the
        // cached null and answer it as a tracker.
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool refused = false;
            try {
                mtmc::create_cluster_tracker("throws", "quay", {});
            } catch (const ConfigError&) {
                refused = true;
            }
            check(refused, "a throwing factory is refused, and again on the retry");
        }

        for (const mtmc::MadeClusterTracker& made : mtmc::made_cluster_trackers()) {
            check(made.tracker != nullptr, "no null tracker is ever listed: " + made.slot);
        }
    }

    void what_was_built_is_listable() {
        mtmc::create_cluster_tracker("fake", "listed", {});
        bool found = false;
        for (const mtmc::MadeClusterTracker& made : mtmc::made_cluster_trackers()) {
            if (made.slot == "listed") found = made.impl == "fake" && made.tracker != nullptr;
        }

        check(found, "a built tracker is listed by impl and slot, for the run's report");
    }

    void the_seam_takes_a_whole_instant() {
        // Handing a cross-camera tracker one camera at a time turns cross-camera association
        // into within-camera deduplication, which is why the argument is a vector.
        const auto tracker = mtmc::create_cluster_tracker("fake", "instant", {});
        ClusterObservation first{
            TrackKey{"cam0", 1}, {1.f, 0.f}, {0.f, 0.f, 10.f, 10.f}, 1920, 1080};
        ClusterObservation second{
            TrackKey{"cam1", 4}, {1.f, 0.f}, {0.f, 0.f, 10.f, 10.f}, 1920, 1080};

        const auto out = tracker->ids({first, second}, {"cam0", "cam1"});

        check(out.size() == 2, "every observation of the instant is answered");
        check(out.count(TrackKey{"cam0", 1}) == 1 && out.count(TrackKey{"cam1", 4}) == 1,
              "keyed by (camera, track) and never by position");
    }

}  // namespace

int main() {
    the_registry_hands_out_one_tracker_per_slot();
    an_unknown_impl_is_refused_by_name();
    the_registry_refuses_an_absent_name_with_its_own_error();
    a_lane_less_build_blames_the_LANE_and_not_the_name();
    the_chain_s_gate_options_reach_the_implementation();
    one_slot_is_one_gate_and_a_second_answer_is_refused();
    a_factory_that_throws_caches_nothing();
    what_was_built_is_listable();
    the_seam_takes_a_whole_instant();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
