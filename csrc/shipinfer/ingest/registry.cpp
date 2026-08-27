// THE OFFLINE-CLOSURE INVARIANT — read before adding an include to any unit under `ingest/`.
//
// `scripts/build_csrc.py` decides what a binary needs by walking `#include "shipinfer/..."`
// lines and following every header to the `.cpp` beside it. Two units under `sources/` reach
// outside this tree, and **no unit under `ingest/` other than each one's own `.cpp` may include
// its header**:
//
//   sources/replay.h      -> `core/platform.h` (the driver) and OpenCV
//   sources/gstreamer.h   -> `libgstreamer-1.0` / `libgstreamer-app-1.0`
//
// Including either here, in the registry every other unit reaches, would pull those
// prerequisites into the closure of the whole ingest plane, and the offline C++ tier would stop
// building on a machine with no CUDA and no `-dev` package. (`sources/gstreamer_pipeline.h` is
// the pure sibling of the second one and is free of all of it — anything may include that.)
//
// That is why this registry is populated by file-scope registrars in the source files
// themselves and knows the name of none of them. The visible consequence, and it is the right
// one: an offline binary's registry legitimately contains no *real* source, because both of the
// ones shipped today live in units an offline build does not compile. `--with-external
// gstreamer` opts the second one back in for the container that has GStreamer, which is how its
// registry tests get to run at all.
#include "shipinfer/ingest/registry.h"

#include <sstream>

#include "shipinfer/core/types.h"

namespace shipinfer {

    void SourceRegistry::add(const std::string& name, const std::vector<std::string>& aliases,
                             const std::string& description, SourceFactory factory) {
        if (entries_.count(name) != 0 || by_alias_.count(name) != 0) {
            throw ConfigError("video source " + name + " is registered twice");
        }
        entries_[name] = Entry{description, std::move(factory)};
        for (const std::string& alias : aliases) {
            if (entries_.count(alias) != 0 || by_alias_.count(alias) != 0) {
                throw ConfigError("video source alias " + alias + " is already taken");
            }
            by_alias_[alias] = name;
        }
    }

    std::string SourceRegistry::canonical(const std::string& name) const {
        if (entries_.count(name) != 0) return name;
        auto alias = by_alias_.find(name);
        if (alias != by_alias_.end()) return alias->second;
        std::ostringstream known;
        for (const auto& [n, _] : entries_) known << (known.tellp() > 0 ? ", " : "") << n;
        throw ConfigError("unknown video source '" + name +
                          "'; known sources: " + (known.tellp() > 0 ? known.str() : "(none)"));
    }

    bool SourceRegistry::contains(const std::string& name) const {
        return entries_.count(name) != 0 || by_alias_.count(name) != 0;
    }

    std::unique_ptr<FrameSource> SourceRegistry::build(const std::string& name,
                                                       const IngestConfig& config,
                                                       FrameCounter& counter,
                                                       StopSignal& stop) const {
        return entries_.at(canonical(name)).factory(config, counter, stop);
    }

    std::vector<std::string> SourceRegistry::names() const {
        std::vector<std::string> out;
        for (const auto& [name, _] : entries_) out.push_back(name);
        return out;
    }

    std::vector<std::pair<std::string, std::string>> SourceRegistry::describe() const {
        std::vector<std::pair<std::string, std::string>> out;
        for (const auto& [name, entry] : entries_) out.emplace_back(name, entry.description);
        return out;
    }

    SourceRegistry& SOURCES() {
        static SourceRegistry registry;
        return registry;
    }

    std::unique_ptr<FrameSource> create_source(const IngestConfig& config,
                                               FrameCounter& counter, StopSignal& stop) {
        if (config.source.empty()) {
            // Named rather than defaulted: this plane has no environment layer yet, so a
            // guessed backend would report "unknown video source 'gstreamer'" for a problem
            // that is actually "nobody said which one".
            throw ConfigError("camera '" + config.camera_id +
                              "': no video source named; set the camera's `source`");
        }
        return SOURCES().build(config.source, config, counter, stop);
    }

}  // namespace shipinfer
