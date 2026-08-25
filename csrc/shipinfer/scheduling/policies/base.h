// The placement-policy contract — `scheduling/policies/base.py`.
//
// A policy answers exactly one question — *which instance runs this request* — and is given
// exactly enough information to answer it. Anything richer would tempt an implementation into
// asking a backend something expensive on the critical path.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "shipinfer/core/device.h"

namespace shipinfer {

    // The slice of a model instance a policy may see (`Placeable`). Four plain reads. `depth`
    // is read without a lock on purpose: the policy consults it thousands of times a second,
    // and a slightly stale value changes which of two near-equal GPUs wins, nothing more.
    class Placeable {
      public:
        virtual ~Placeable() = default;
        virtual Device device() const = 0;
        virtual size_t depth() const = 0;
        virtual double ewma_latency_us() const = 0;
        virtual bool is_ready() const = 0;
    };

    // The slice of a request a policy may see: the locality hint and the sequence key.
    struct PlacementRequest {
        std::optional<Device> resident_device;
        std::string camera_id;
    };

    // Chooses one instance out of the ready candidates for a model. Subclasses live one per
    // file in this directory and register themselves with `POLICIES()` (registry.h).
    class PlacementPolicy {
      public:
        virtual ~PlacementPolicy() = default;
        // `candidates` are ready instances and never empty — the dispatcher filters and throws.
        // Returns one element of `candidates`; returning anything else is a programming error.
        virtual Placeable* select(const std::vector<Placeable*>& candidates,
                                  const PlacementRequest& request) = 0;
        virtual std::string name() const = 0;
        virtual std::string describe() const { return name(); }
    };

}  // namespace shipinfer
