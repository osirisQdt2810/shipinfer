// `SOURCES`, the video-source registry — `ingest/registry.py`.
//
// A new source is a new file plus one `SourceRegistrar` at the bottom of it; nothing else in
// the tree changes. That is the registry rule the Python plane runs on (CLAUDE.md, seam 1), and
// the reason a deployment picks a backend by *name* in its settings rather than by a branch.
//
// Registration is eager, and that is the point: a source module is import-safe — it names
// nothing of its decode runtime until `do_open` — so this registry can list a backend on a host
// that cannot run it and still fail usefully, at `open()`, with a `SourceUnavailableError`
// naming what to install. Lazy registration would hide the name until the load succeeded, so a
// misconfigured deployment would be told "unknown video source" when the truth is "that library
// is not installed". Those are different problems with different fixes.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/stop_signal.h"
#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/config.h"
#include "shipinfer/ingest/frame.h"

namespace shipinfer {

    // How a caller substitutes a source. The counter is passed in, never created by the
    // factory, because frame ids must survive the reconnect that replaces the source.
    using SourceFactory = std::function<std::unique_ptr<FrameSource>(
        const IngestConfig&, FrameCounter&, StopSignal&)>;

    class SourceRegistry {
      public:
        void add(const std::string& name, const std::vector<std::string>& aliases,
                 const std::string& description, SourceFactory factory);
        // The canonical name for a name or alias; throws `ConfigError` naming the alternatives,
        // because "unknown video source 'gstremaer'" with no list is a twenty-minute detour.
        std::string canonical(const std::string& name) const;
        std::unique_ptr<FrameSource> build(const std::string& name, const IngestConfig& config,
                                           FrameCounter& counter, StopSignal& stop) const;
        std::vector<std::string> names() const;
        std::vector<std::pair<std::string, std::string>> describe() const;
        bool contains(const std::string& name) const;

      private:
        struct Entry {
            std::string description;
            SourceFactory factory;
        };
        std::map<std::string, Entry> entries_;
        std::map<std::string, std::string> by_alias_;
    };

    // Function-local static: every translation unit's registrar can run before `main`, in any
    // order, and still find the one registry.
    SourceRegistry& SOURCES();

    // Build the source this camera asks for.
    //
    // Throws ConfigError when `config.source` is empty or is not registered.
    std::unique_ptr<FrameSource> create_source(const IngestConfig& config,
                                               FrameCounter& counter, StopSignal& stop);

    // One of these at the bottom of each source file is the `@SOURCES.register(...)`.
    struct SourceRegistrar {
        SourceRegistrar(const std::string& name, const std::vector<std::string>& aliases,
                        const std::string& description, SourceFactory factory) {
            SOURCES().add(name, aliases, description, std::move(factory));
        }
    };

}  // namespace shipinfer
