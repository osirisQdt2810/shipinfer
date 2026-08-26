#include "shipinfer/scheduling/policies/locality_spillover.h"

#include <memory>

#include "shipinfer/core/types.h"
#include "shipinfer/scheduling/policies/registry.h"

namespace shipinfer {
    namespace {
        const PolicyRegistrar kRegistrar(
            "locality_spillover", {"locality"},
            "stay on the resident GPU until its queue backs up, then spill",
            [](const PolicyOptions& options) -> std::unique_ptr<PlacementPolicy> {
                refuse_unknown_options("locality_spillover", options, {"spill_threshold"});
                const int threshold = option_int(options, "spill_threshold", 4);
                if (threshold < 0) {
                    throw ConfigError("locality_spillover: spill_threshold must be >= 0");
                }
                return std::make_unique<LocalityAwareSpilloverPolicy>(threshold);
            });
    }  // namespace
}  // namespace shipinfer
