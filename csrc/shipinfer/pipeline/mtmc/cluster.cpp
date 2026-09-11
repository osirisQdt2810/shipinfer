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

    std::shared_ptr<ClusterTracker> ClusterRegistry::create(const std::string& impl) const {
        // `find` and a TYPED refusal, not `at`: this is public, so a caller that had not asked
        // `has` first would get `std::out_of_range` out of a registry whose whole vocabulary is
        // `ConfigError`. The same hole `AssociatorRegistry::create` was reviewed for.
        const auto entry = entries_.find(impl);
        if (entry == entries_.end()) {
            throw ConfigError("no cross-camera tracker is registered as '" + impl + "'");
        }
        return entry->second();
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

        std::map<std::pair<std::string, std::string>, std::shared_ptr<ClusterTracker>>& made() {
            static std::map<std::pair<std::string, std::string>,
                            std::shared_ptr<ClusterTracker>>
                cache;
            return cache;
        }

    }  // namespace

    std::vector<MadeClusterTracker> made_cluster_trackers() {
        std::lock_guard<std::mutex> held(made_lock());
        std::vector<MadeClusterTracker> out;
        for (const auto& [key, tracker] : made()) {
            out.push_back(MadeClusterTracker{key.first, key.second, tracker});
        }
        return out;
    }

    std::shared_ptr<ClusterTracker> create_cluster_tracker(const std::string& impl,
                                                           const std::string& slot) {
        // `has` OUTSIDE the lock and the factory INSIDE it, both deliberate and both narrow:
        // registrars are file-scope statics that run before main, so the name table is not
        // written concurrently; and holding the lock across a trivial factory is what makes
        // "one tracker per (impl, slot)" true rather than likely. The mutex is not recursive,
        // so a factory that called this again would deadlock -- no impl does, and one that
        // needed to would take a second lock rather than this one.
        if (CLUSTERERS().has(impl)) {
            std::lock_guard<std::mutex> held(made_lock());
            std::shared_ptr<ClusterTracker>& cached = made()[{impl, slot}];
            if (!cached) cached = CLUSTERERS().create(impl);
            return cached;
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
