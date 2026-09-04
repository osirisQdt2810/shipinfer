#include "shipinfer/pipeline/graph/from_plan.h"

namespace shipinfer {

    std::set<std::string> loaded_names(const ModelMap& models) {
        std::set<std::string> names;
        for (const auto& [name, model] : models) names.insert(name);
        return names;
    }

    Dag build_dag(const PlanStages& planned, const ModelMap& models, WorkerScratch& scratch,
                  std::chrono::milliseconds timeout) {
        Dag dag;
        dag.add(std::make_unique<DetectStage>(planned.detect_slot,
                                              *models.at(planned.detect_model), planned.detect,
                                              scratch, timeout));
        // One crop pass for every payload, which is this plane's own choice and the reason a
        // crop element is not a stage: N classes cost one walk of the frame, not N.
        if (!planned.crops.empty()) {
            dag.add(std::make_unique<CropStage>("crop", planned.crops,
                                                planned.detect.max_objects, scratch));
        }
        for (const std::array<std::string, 4>& object : planned.objects) {
            dag.add(std::make_unique<ObjectStage>(object[0], *models.at(object[1]), object[2],
                                                  object[3], timeout));
        }
        return dag;
    }

}  // namespace shipinfer
