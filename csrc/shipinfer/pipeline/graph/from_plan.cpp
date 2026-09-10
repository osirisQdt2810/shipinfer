#include "shipinfer/pipeline/graph/from_plan.h"

#include "shipinfer/pipeline/tracking/registry.h"

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
        for (const ObjectStageSpec& object : planned.objects) {
            ObjectCombine combine;
            if (object.fold) {
                combine = [spec = *object.fold](const InferenceResponse& response) {
                    return mask_area(response, spec);
                };
            }
            dag.add(std::make_unique<ObjectStage>(object.slot, *models.at(object.model),
                                                  object.source, object.output, timeout,
                                                  std::move(combine)));
        }
        // LAST, and after the croppers for the reason the chain gives itself (`after:
        // [embed_ship, embed_person]`): the ids are scattered onto the same rows their vectors
        // are, and a tracker that ran first would be tracking boxes nothing had embedded.
        //
        // BY NAME, and this file includes no tracker header: `tracking/registry.h` explains
        // that including one would drag an external lane into the closure of the whole graph
        // plane -- measured, as a binary that would not link at all.
        if (planned.track) {
            dag.add(tracking::create_track_stage(planned.track->impl, *planned.track));
        }
        return dag;
    }

}  // namespace shipinfer
