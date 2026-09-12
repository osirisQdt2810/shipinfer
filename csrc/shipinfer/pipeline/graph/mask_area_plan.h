// The mask fold, resolved against an engine's declared output shapes — and nothing else.
//
// PURE, and that is the point: every refusal below is a statement about a configuration, so it
// belongs where the rest of the plan's validation lives and has to be testable with two
// `(name, dims)` pairs and no driver. `backends/tensorrt/fold.h` turns the answer into the
// closure the instance runs; `graph/mask_area.cpp` does the same arithmetic on the host and
// stays, because a fused kernel is only trustworthy if a readable implementation agrees.
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/pipeline/graph/mask_area.h"

namespace shipinfer {

    // One engine output, as the artefact declares it: its name and its per-row shape.
    using DeclaredOutput = std::pair<std::string, std::vector<int64_t>>;

    // Everything the fold kernel needs, resolved from names to positions and sizes.
    struct MaskAreaPlan {
        size_t detections_index = 0;
        size_t prototypes_index = 0;
        int candidates = 0;
        int stride = 0;
        int prefix = 0;
        int channels = 0;
        int cells = 0;
        float score_threshold = 0.f;
        float mask_threshold = 0.f;
        int crop_height = 0;
        int crop_width = 0;
        std::string name;
    };

    // Resolve `spec`'s two output NAMES against what the engine declares.
    //
    // Throws `ConfigError` when an output is missing and `BackendError` when a shape is not
    // the rank the fold needs or the two disagree about the coefficient count -- the same
    // refusals, in the same words, as `mask_area` gives, because they say the same thing: the
    // engine is not the one this slot was configured for, and combining its outputs anyway
    // would build a plausible mask from the wrong planes.
    MaskAreaPlan resolve_mask_area(const std::vector<DeclaredOutput>& outputs,
                                   const MaskAreaSpec& spec);

}  // namespace shipinfer
