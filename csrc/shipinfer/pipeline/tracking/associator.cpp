#include "shipinfer/pipeline/tracking/associator.h"

#include <sstream>

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
        return entries_.at(impl)();
    }

    AssociatorRegistry& ASSOCIATORS() {
        static AssociatorRegistry registry;
        return registry;
    }

    std::shared_ptr<Associator> create_associator(const std::string& impl) {
        if (ASSOCIATORS().has(impl)) return ASSOCIATORS().create(impl);
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
