#include "shipinfer/backends/tensorrt/fold.h"

#include <utility>
#include <vector>

#include "shipinfer/pipeline/graph/mask_area_plan.h"
#include "shipinfer/runtime/ops.h"

namespace shipinfer {

    AdapterFold mask_area_fold(const TrtEngine& engine, const MaskAreaSpec& spec) {
        std::vector<DeclaredOutput> declared;
        declared.reserve(engine.outputs().size());
        for (const TensorSpec& output : engine.outputs()) {
            declared.emplace_back(output.name, output.dims);
        }
        const MaskAreaPlan plan = resolve_mask_area(declared, spec);

        AdapterFold attached;
        attached.prototypes_index = plan.prototypes_index;
        attached.name = plan.name;
        // BY VALUE: the closure outlives this call and runs on every batch, so it must not
        // hold a reference to the engine's vectors or to `plan`.
        attached.fold = [plan](const TrtInstance& instance, int rows, float* areas) {
            mask_area_into(instance.output_device(plan.detections_index), plan.candidates,
                           plan.stride, plan.prefix,
                           instance.output_device(plan.prototypes_index), plan.channels,
                           plan.cells, rows, plan.score_threshold, plan.mask_threshold,
                           plan.crop_height, plan.crop_width, areas, instance.stream());
        };
        return attached;
    }

}  // namespace shipinfer
