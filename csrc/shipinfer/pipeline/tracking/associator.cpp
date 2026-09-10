#include "shipinfer/pipeline/tracking/associator.h"

#include <map>
#include <mutex>
#include <sstream>
#include <utility>

#include "shipinfer/ingest/omitted_lanes.h"

namespace shipinfer::tracking {

    void AssociatorRegistry::add(const std::string& impl, AssociatorFactory factory) {
        entries_[impl] = std::move(factory);
    }

    bool AssociatorRegistry::has(const std::string& impl) const {
        return entries_.count(impl) != 0;
    }

    std::vector<std::string> AssociatorRegistry::names() const {
        std::vector<std::string> out;
        for (const auto& [name, _] : entries_) out.push_back(name);
        return out;
    }

    std::shared_ptr<Associator> AssociatorRegistry::create(const std::string& impl) const {
        // `find` and a TYPED refusal, not `at`: this is public, so a second caller that had
        // not asked `has` first would get `std::out_of_range` out of a registry whose whole
        // vocabulary is `ConfigError`.
        const auto entry = entries_.find(impl);
        if (entry == entries_.end()) {
            throw ConfigError("no tracker is registered as '" + impl + "'");
        }
        return entry->second();
    }

    AssociatorRegistry& ASSOCIATORS() {
        static AssociatorRegistry registry;
        return registry;
    }

    namespace {

        // (impl, slot) -> the one associator that pair gets. See `create_associator`'s header
        // comment for why the key has two halves and why the cache is process-wide.
        //
        // NEVER CLEARED, which is right for `bench` (one chain, one process) and is a thing to
        // fix when a runner gains `UpdateTopology`: a swapped chain reusing a slot name would
        // be handed the old chain's tracker state, ids and all.
        std::mutex& made_lock() {
            static std::mutex lock;
            return lock;
        }

        std::map<std::pair<std::string, std::string>, std::shared_ptr<Associator>>& made() {
            static std::map<std::pair<std::string, std::string>, std::shared_ptr<Associator>>
                cache;
            return cache;
        }

    }  // namespace

    std::vector<MadeAssociator> made_associators() {
        std::lock_guard<std::mutex> held(made_lock());
        std::vector<MadeAssociator> out;
        for (const auto& [key, associator] : made()) {
            out.push_back(MadeAssociator{key.first, key.second, associator});
        }
        return out;
    }

    std::shared_ptr<Associator> create_associator(const std::string& impl,
                                                  const std::string& slot) {
        if (ASSOCIATORS().has(impl)) {
            std::lock_guard<std::mutex> held(made_lock());
            std::shared_ptr<Associator>& cached = made()[{impl, slot}];
            if (!cached) cached = ASSOCIATORS().create(impl);
            return cached;
        }
        std::ostringstream known;
        for (const std::string& name : ASSOCIATORS().names()) {
            known << (known.tellp() > 0 ? ", " : "") << name;
        }
        const std::string listed = known.tellp() > 0 ? known.str() : "(none)";
        // CHECKED BEFORE "unknown", the same order and reason as `SourceRegistry::canonical`:
        // an operator whose chain says `impl: shipvision` has written the right name, and
        // sending them to re-read it is the least useful answer.
        //
        // A DIRECT CHECK AND NOT A SECOND TABLE. `ingest/omitted_lanes.h` keeps a lane -> names
        // table because several lanes register several sources, and its own comment warns that
        // two tables are what drift. Here there is one lane and one impl, and that table's row
        // for this lane is deliberately EMPTY because the lane registers no video SOURCE.
        const std::string omitted = "," + omitted_lanes() + ",";
        if (impl == "shipvision" && omitted.find(",shipvision,") != std::string::npos) {
            throw ConfigError(
                "tracker '" + impl +
                "' exists but was not compiled into this binary (the 'shipvision' external "
                "lane was not part of this build — see scripts/build_csrc.py --with-external, "
                "and `git submodule update --init 3rdparty/shipvision`); known trackers: " +
                listed);
        }
        throw ConfigError("unknown tracker '" + impl + "'; known trackers: " + listed);
    }

}  // namespace shipinfer::tracking
