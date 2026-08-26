#include "shipinfer/scheduling/policies/join_shortest_queue.h"

#include <memory>

#include "shipinfer/scheduling/policies/registry.h"

namespace shipinfer {
    namespace {
        const PolicyRegistrar kRegistrar(
            "join_shortest_queue", {"jsq"},
            "shortest queue over all instances (optimal for small pools)",
            [](const PolicyOptions& options) -> std::unique_ptr<PlacementPolicy> {
                refuse_unknown_options("join_shortest_queue", options, {});
                return std::make_unique<JoinShortestQueuePolicy>();
            });
    }  // namespace
}  // namespace shipinfer
