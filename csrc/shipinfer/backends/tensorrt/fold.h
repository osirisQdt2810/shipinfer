// The mask fold as an engine attachment — `graph/mask_area_plan.h` resolved into a closure.
//
// Thin on purpose: every refusal is in the pure half, and what is left here is the one thing
// that cannot be — a call into the kernel with the instance's own device pointers, on the
// instance's own stream, while it still owns the batch.
#pragma once

#include "shipinfer/backends/tensorrt/adapter.h"
#include "shipinfer/pipeline/graph/mask_area.h"

namespace shipinfer {

    // Resolve `spec` against `engine` and build the attachment. Throws what
    // `resolve_mask_area` throws.
    AdapterFold mask_area_fold(const TrtEngine& engine, const MaskAreaSpec& spec);

}  // namespace shipinfer
