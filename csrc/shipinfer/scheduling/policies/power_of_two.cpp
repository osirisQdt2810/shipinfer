#include "shipinfer/scheduling/policies/power_of_two.h"

#include <memory>

#include "shipinfer/scheduling/policies/registry.h"

namespace shipinfer {
    namespace {
        const PolicyRegistrar kRegistrar(
            "power_of_two", {"p2c"},
            "two random probes, shorter queue wins (scales, no herding)",
            [](const PolicyOptions& options) -> std::unique_ptr<PlacementPolicy> {
                refuse_unknown_options("power_of_two", options, {"seed"});
                auto seed = options.find("seed");
                return seed == options.end()
                           ? std::make_unique<PowerOfTwoChoicesPolicy>()
                           : std::make_unique<PowerOfTwoChoicesPolicy>(
                                 static_cast<unsigned>(option_int(options, "seed", 0)));
            });
    }  // namespace
}  // namespace shipinfer
