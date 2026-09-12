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

    std::shared_ptr<Associator> AssociatorRegistry::create(
        const std::string& impl, const TrackerOptions& options) const {
        // `find` and a TYPED refusal, not `at`: this is public, so a second caller that had
        // not asked `has` first would get `std::out_of_range` out of a registry whose whole
        // vocabulary is `ConfigError`.
        const auto entry = entries_.find(impl);
        if (entry == entries_.end()) {
            throw ConfigError("no tracker is registered as '" + impl + "'");
        }
        return entry->second(options);
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

        //: What each cached (impl, slot) was BUILT with, so a second caller asking for the
        //: same pair with different options is refused rather than silently handed the first
        //: caller's tracker. The cache is the whole reason: whoever calls first wins, and
        //: without this the loser runs a configuration no chain states.
        std::map<std::pair<std::string, std::string>, TrackerOptions>& made_options() {
            static std::map<std::pair<std::string, std::string>, TrackerOptions> cache;
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
                                                  const std::string& slot,
                                                  const TrackerOptions& options) {
        if (ASSOCIATORS().has(impl)) {
            std::lock_guard<std::mutex> held(made_lock());
            const std::pair<std::string, std::string> key{impl, slot};
            std::shared_ptr<Associator>& cached = made()[key];
            if (!cached) {
                cached = ASSOCIATORS().create(impl, options);
                made_options()[key] = options;
            } else if (!(made_options()[key] == options)) {
                throw ConfigError(
                    "tracker slot '" + slot + "' was already built for impl '" + impl +
                    "' with different options; whoever calls first wins this cache, so the "
                    "second set would be silently ignored and this slot would run a "
                    "configuration no chain states");
            }
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
