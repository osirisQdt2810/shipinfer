#include "shipinfer/scheduling/policies/sequence_affinity.h"

#include <memory>

#include "shipinfer/scheduling/policies/registry.h"

namespace shipinfer {
    namespace {
        const PolicyRegistrar kRegistrar(
            "sequence_affinity", {"sticky", "sequence"},
            "pin each camera to one instance; re-pin when it dies",
            [](const PolicyOptions& options) -> std::unique_ptr<PlacementPolicy> {
                refuse_unknown_options("sequence_affinity", options, {"max_sequences"});
                return std::make_unique<SequenceAffinityPolicy>(
                    nullptr, static_cast<size_t>(option_int("placement policy", options,
                                                            "max_sequences", 4096)));
            });
    }  // namespace
}  // namespace shipinfer
