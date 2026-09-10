#include "shipinfer/pipeline/graph/from_plan.h"

#include "shipinfer/pipeline/tracking/associator.h"

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
        // The ASSOCIATOR by name, and the stage built here: `tracking/associator.h` explains
        // the two build lines that put the tracker in another unit -- one measured as a binary
        // that would not link, the other as a lane whose CI job is g++ alone.
        for (const TrackStageSpec& track : planned.tracks) {
            // ONE ASSOCIATOR PER SLOT, and the registry decides whether two slots share one:
            // `bytetrack.cpp` hands out the same shard for every caller, which is right for
            // two trackers with disjoint selections over one camera -- they are one identity
            // space seeing different rows, not two spaces seeing the same ones.
            dag.add(std::make_unique<TrackStage>(track.slot, track.output, track.class_id,
                                                 tracking::create_associator(track.impl)));
        }
        return dag;
    }

}  // namespace shipinfer
