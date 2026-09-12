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

        //: The tracker AND what it was built with, in one entry rather than two maps keyed
        //: alike: a second caller asking for the same pair with different options is refused
        //: rather than silently handed the first caller's tracker. `mtmc/cluster.cpp` holds
        //: its pair the same way, and two parallel maps is a second chance to insert into one
        //: and not the other.
        struct Made {
            std::shared_ptr<Associator> associator;
            TrackerOptions options;
        };

        std::map<std::pair<std::string, std::string>, Made>& made() {
            static std::map<std::pair<std::string, std::string>, Made> cache;
            return cache;
        }

    }  // namespace

    std::vector<MadeAssociator> made_associators() {
        std::lock_guard<std::mutex> held(made_lock());
        std::vector<MadeAssociator> out;
        for (const auto& [key, entry] : made()) {
            out.push_back(MadeAssociator{key.first, key.second, entry.associator});
        }
        return out;
    }

    std::shared_ptr<Associator> create_associator(const std::string& impl,
                                                  const std::string& slot,
                                                  const TrackerOptions& options) {
        if (ASSOCIATORS().has(impl)) {
            std::lock_guard<std::mutex> held(made_lock());
            const std::pair<std::string, std::string> key{impl, slot};
            const auto found = made().find(key);
            if (found != made().end()) {
                if (!(found->second.options == options)) {
                    throw ConfigError(
                        "tracker slot '" + slot + "' was already built for impl '" + impl +
                        "' with different options; whoever calls first wins this cache, so "
                        "the second set would be silently ignored and this slot would run a "
                        "configuration no chain states");
                }
                return found->second.associator;
            }
            // BUILT INTO A LOCAL FIRST. `made()[key]` default-inserts before the factory runs,
            // so a constructor that threw left a NULL under that key -- `made_associators()`
            // publishes it and `bench.cpp`'s per-slot report dereferences it, losing the whole
            // run's output including the message naming the bad key. Harmless while the
            // factory could not fail; this PR's key table is what made it reachable, exactly
            // as `mtmc/cluster.cpp` predicted when it took the same fix.
            std::shared_ptr<Associator> built = ASSOCIATORS().create(impl, options);
            made().emplace(key, Made{built, options});
            return built;
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
