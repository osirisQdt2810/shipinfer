#include "shipinfer/pipeline/mtmc/cluster.h"

#include <map>
#include <mutex>
#include <sstream>
#include <utility>

#include "shipinfer/ingest/omitted_lanes.h"

namespace shipinfer::mtmc {

    void ClusterRegistry::add(const std::string& impl, ClusterTrackerFactory factory) {
        entries_[impl] = std::move(factory);
    }

    bool ClusterRegistry::has(const std::string& impl) const {
        return entries_.count(impl) != 0;
    }

    std::vector<std::string> ClusterRegistry::names() const {
        std::vector<std::string> out;
        for (const auto& [name, _] : entries_) out.push_back(name);
        return out;
    }

    std::shared_ptr<ClusterTracker> ClusterRegistry::create(
        const std::string& impl, const ClusterOptions& options) const {
        // `find` and a TYPED refusal, not `at`: this is public, so a caller that had not asked
        // `has` first would get `std::out_of_range` out of a registry whose whole vocabulary is
        // `ConfigError`. The same hole `AssociatorRegistry::create` was reviewed for.
        const auto entry = entries_.find(impl);
        if (entry == entries_.end()) {
            throw ConfigError("no cross-camera tracker is registered as '" + impl + "'");
        }
        return entry->second(options);
    }

    ClusterRegistry& CLUSTERERS() {
        static ClusterRegistry registry;
        return registry;
    }

    namespace {

        // (impl, slot) -> the one tracker that pair gets. Never cleared, which is right for
        // `bench` (one chain, one process) and is a thing to fix when a runner gains
        // `UpdateTopology`: a swapped chain reusing a slot name would inherit the old chain's
        // identities, global ids and all.
        std::mutex& made_lock() {
            static std::mutex lock;
            return lock;
        }

        // The OPTIONS are cached beside the tracker, because a second caller asking for
        // different ones on the same slot has to be refused rather than handed the first
        // caller's gate -- which would be a running configuration that is in no file.
        struct Made {
            std::shared_ptr<ClusterTracker> tracker;
            ClusterOptions options;
        };

        std::map<std::pair<std::string, std::string>, Made>& made() {
            static std::map<std::pair<std::string, std::string>, Made> cache;
            return cache;
        }

    }  // namespace

    std::vector<MadeClusterTracker> made_cluster_trackers() {
        std::lock_guard<std::mutex> held(made_lock());
        std::vector<MadeClusterTracker> out;
        for (const auto& [key, made] : made()) {
            out.push_back(MadeClusterTracker{key.first, key.second, made.tracker});
        }
        return out;
    }

    std::shared_ptr<ClusterTracker> create_cluster_tracker(const std::string& impl,
                                                           const std::string& slot,
                                                           const ClusterOptions& options) {
        // `has` OUTSIDE the lock and the factory INSIDE it, both deliberate and both narrow:
        // registrars are file-scope statics that run before main, so the name table is not
        // written concurrently; and holding the lock across a trivial factory is what makes
        // "one tracker per (impl, slot)" true rather than likely. The mutex is not recursive,
        // so a factory that called this again would deadlock -- no impl does, and one that
        // needed to would take a second lock rather than this one.
        if (CLUSTERERS().has(impl)) {
            std::lock_guard<std::mutex> held(made_lock());
            const auto key = std::make_pair(impl, slot);
            const auto found = made().find(key);
            if (found != made().end()) {
                // REFUSED, not silently shared. The tracker is cached per (impl, slot), so a
                // second caller asking for different thresholds would get the FIRST caller's
                // gate -- a running configuration that is in no file. One slot is one group
                // is one gate; two chains disagreeing about it is a start-up fault.
                if (!(found->second.options == options)) {
                    throw ConfigError(
                        "cross-camera slot '" + slot +
                        "' was already built with different gate options; one slot is one "
                        "camera group and one gate, so two callers cannot configure it twice "
                        "-- say it once in the chain");
                }
                return found->second.tracker;
            }
            // BUILT INTO A LOCAL FIRST. `made()[key]` default-inserts before the factory runs,
            // so a constructor that threw left a NULL shared_ptr cached under that key --
            // `made_cluster_trackers()` would then report an entry whose `tracker` is null and
            // the bench's per-slot report dereferences it. Not reachable while every factory
            // was trivial; the lane's tracker is the first one that can fail on a bad
            // algorithm config, which is this PR. (`tracking/associator.cpp` has the same
            // shape and the same fix is owed there -- `MTMC-TWO-SLOT-CACHED-REGISTRIES`.)
            std::shared_ptr<ClusterTracker> built = CLUSTERERS().create(impl, options);
            made().emplace(key, Made{built, options});
            return built;
        }
        std::ostringstream known;
        for (const std::string& name : CLUSTERERS().names()) {
            known << (known.tellp() > 0 ? ", " : "") << name;
        }
        const std::string listed = known.tellp() > 0 ? known.str() : "(none)";
        // CHECKED BEFORE "unknown", the order `SourceRegistry::canonical` and
        // `create_associator` both use: an operator whose chain says `impl: shipvision` has
        // written the right name, and sending them to re-read it is the least useful answer.
        const std::string omitted = "," + omitted_lanes() + ",";
        if (impl == "shipvision" && omitted.find(",shipvision,") != std::string::npos) {
            throw ConfigError(
                "cross-camera tracker '" + impl +
                "' exists but was not compiled into this binary (the 'shipvision' external "
                "lane was not part of this build — see scripts/build_csrc.py --with-external, "
                "and `git submodule update --init 3rdparty/shipvision`); known: " +
                listed);
        }
        throw ConfigError("unknown cross-camera tracker '" + impl + "'; known: " + listed);
    }

}  // namespace shipinfer::mtmc
