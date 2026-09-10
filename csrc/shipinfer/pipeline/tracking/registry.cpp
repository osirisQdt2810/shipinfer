#include "shipinfer/pipeline/tracking/registry.h"

#include <sstream>

#include "shipinfer/core/types.h"
#include "shipinfer/ingest/omitted_lanes.h"

namespace shipinfer::tracking {

    void TrackerRegistry::add(const std::string& impl, TrackStageFactory factory) {
        entries_[impl] = std::move(factory);
    }

    bool TrackerRegistry::has(const std::string& impl) const {
        return entries_.count(impl) != 0;
    }

    std::vector<std::string> TrackerRegistry::names() const {
        std::vector<std::string> out;
        for (const auto& [name, _] : entries_) out.push_back(name);
        return out;
    }

    std::unique_ptr<Stage> TrackerRegistry::create(const std::string& impl,
                                                   const TrackStageSpec& spec) const {
        return entries_.at(impl)(spec);
    }

    TrackerRegistry& TRACKERS() {
        static TrackerRegistry registry;
        return registry;
    }

    std::unique_ptr<Stage> create_track_stage(const std::string& impl,
                                              const TrackStageSpec& spec) {
        if (TRACKERS().has(impl)) return TRACKERS().create(impl, spec);
        std::ostringstream known;
        for (const std::string& name : TRACKERS().names()) {
            known << (known.tellp() > 0 ? ", " : "") << name;
        }
        const std::string listed = known.tellp() > 0 ? known.str() : "(none)";
        // CHECKED BEFORE "unknown", the same order and for the same reason as
        // `SourceRegistry::canonical`: an operator whose chain says `impl: shipvision` has
        // written the right name, and sending them to re-read it is the least useful answer.
        //
        // A DIRECT CHECK AND NOT A SECOND TABLE. `ingest/omitted_lanes.h` keeps a lane -> names
        // table because several lanes register several sources, and its own comment warns that
        // two tables are what drift. Here there is one lane and one impl, and the lane's row in
        // that table is deliberately EMPTY because the lane registers no video SOURCE -- so a
        // table here would be one row that has to agree with a row asserted to be empty.
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
