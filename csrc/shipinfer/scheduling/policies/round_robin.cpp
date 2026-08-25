#include "shipinfer/scheduling/policies/round_robin.h"

#include <memory>

#include "shipinfer/scheduling/policies/registry.h"

namespace shipinfer {
    namespace {
        const PolicyRegistrar kRegistrar(
            "round_robin", {"rr"}, "rotate in order, load-blind (baseline)",
            [](const PolicyOptions& options) -> std::unique_ptr<PlacementPolicy> {
                refuse_unknown_options("round_robin", options, {});
                return std::make_unique<RoundRobinPolicy>();
            });
    }  // namespace
}  // namespace shipinfer
