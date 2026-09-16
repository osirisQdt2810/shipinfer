// What a run calls each of its devices in the output. Reporting, so `obs/`.
//
// Narrowing the container's visible set is worth ~9.5 s of CUDA enumeration on a shared box,
// and it RENUMBERS the devices from 0 -- while every archived `per_device*` line in
// `benchmarks/RESULTS.md` reports HOST ids. So a run is given ordinals and a parallel list of
// labels, and everything printed goes through `label_of`.
//
// EXTRACTED from `cli/bench.cpp`'s anonymous namespace, a `main()` translation unit no gate can
// link, for the reason `graph/bench_models.h` was: #294 round 1 mapped labels POSITIONALLY and
// printed host 2's counters under `6:`, an inverted table in the one file whose value is
// cross-run comparison, and nothing in the tree failed. Asserted rather than correct by
// reading.
//
// Offline: no CUDA, no TensorRT.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace shipinfer {

    // The name device `ordinal` goes by. Its own number when no labels were given -- which is
    // every run that narrows nothing -- and otherwise the label in the same position.
    inline int label_of(const std::vector<int>& ordinals, const std::vector<int>& labels,
                        int ordinal) {
        for (std::size_t i = 0; i < ordinals.size(); ++i)
            if (ordinals[i] == ordinal) return i < labels.size() ? labels[i] : ordinal;
        // NOT an error: a device outside the run's own list can reach a printf on an error
        // path, and mangling it into a label would be worse than reporting it as it is.
        return ordinal;
    }

    // Empty, or exactly one label per device. A SHORT list is the dangerous case -- it would
    // relabel the leading devices and leave the rest reporting ordinals, which is the one
    // output a reader cannot tell from a correct one. Returns the complaint, or "".
    inline std::string labels_refusal(const std::vector<int>& ordinals,
                                      const std::vector<int>& labels) {
        if (labels.empty() || labels.size() == ordinals.size()) return "";
        return "--gpu-labels has " + std::to_string(labels.size()) +
               " entries and --gpu-ids has " + std::to_string(ordinals.size()) +
               "; a label per device, in the same order, or none at all";
    }

}  // namespace shipinfer
