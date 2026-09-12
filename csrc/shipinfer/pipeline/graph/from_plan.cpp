#include "shipinfer/pipeline/graph/from_plan.h"

#include <algorithm>

#include "shipinfer/pipeline/mtmc/cluster.h"
#include "shipinfer/pipeline/tracking/associator.h"

namespace shipinfer {

    std::set<std::string> loaded_names(const ModelMap& models) {
        std::set<std::string> names;
        for (const auto& [name, model] : models) names.insert(name);
        return names;
    }

    MtmcRuntime mtmc_runtime(const PlanStages& planned, const PlanSettings& settings) {
        MtmcRuntime runtime;
        if (planned.mtmcs.empty()) return runtime;
        // ONE BUDGET FOR THE PROCESS, sized from the plan's own worker count, which is what
        // makes the never-starve invariant hold for any number of barriers (`mtmc/barrier.h`).
        // `workers - 1`: the last worker must never be the one parked.
        runtime.budget =
            std::make_shared<mtmc::WaiterBudget>(std::max(0, settings.workers - 1));
        for (const MtmcStageSpec& spec : planned.mtmcs) {
            mtmc::BarrierOptions options;
            options.workers = settings.workers;
            // FROM THE PLAN when the chain says so. `ship_person_cpu.yaml` has stated
            // `sync_window_ms: 60` all along and this plane ran its own default, so the two
            // bucketed instants differently for one chain file -- and the window is what the
            // whole chain's throughput turns on, so an unstated one measures a configuration
            // nobody chose.
            if (spec.sync_window_ms) options.sync_window_s = *spec.sync_window_ms / 1000.0;
            if (spec.max_instants) options.max_instants = *spec.max_instants;
            const auto barrier =
                std::make_shared<mtmc::InstantBarrier>(options, runtime.budget);
            // THE ROSTER, ANNOUNCED BEFORE ANY WORKER STARTS, which is the other half of
            // reading it from the plan. `barrier.h`: an announced camera wins over a
            // merely-seen one the moment anything announces, so a chain that names its four
            // cameras forms instants over those four -- and without this the barrier accreted
            // every camera the shard saw while the other plane waited for the declared list,
            // which is two instant memberships for one chain file (#222's review). The
            // Python element does the same, per member, in `mtmc.py`.
            for (const std::string& camera : spec.cameras) barrier->camera_added(camera);
            runtime.barriers[spec.slot] = barrier;
        }
        return runtime;
    }

    Dag build_dag(const PlanStages& planned, const ModelMap& models, WorkerScratch& scratch,
                  std::chrono::milliseconds timeout, const MtmcRuntime& mtmc_shared) {
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
                // ALREADY FOLDED IS NOT FOLDED AGAIN: when the model's engine carries the
                // device fold (`graph/mask_area_device.h`), the area arrives as an ordinary
                // named output and the prototype bank never came home -- so there is nothing
                // here to fold and `mask_area` would refuse for a missing output. Asked per
                // response rather than per stage because the fold is the ENGINE's property
                // and a non-TensorRT backend for the same slot still folds on the host.
                combine = [spec = *object.fold](const InferenceResponse& response) {
                    const OutputTensor* folded = response.named(spec.name);
                    return folded != nullptr ? *folded : mask_area(response, spec);
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
            // ONE ASSOCIATOR PER SLOT, shared by every worker -- `create_associator` keys its
            // cache by (impl, slot) and says why both halves are needed. Two slots with
            // disjoint selections over one camera are TWO identity spaces, the way
            // `track.py::_do_open` builds a shard per element instance; sharing one would
            // have each slot refuse the other's frames as out of order, forever.
            dag.add(std::make_unique<TrackStage>(
                track.slot, track.output, track.class_id,
                tracking::create_associator(track.impl, track.slot)));
        }
        // LAST, after the trackers whose ids it consumes and the embedders whose vectors
        // decide it -- the order the chain declares for itself (`after: [track]`).
        for (const MtmcStageSpec& spec : planned.mtmcs) {
            const auto barrier = mtmc_shared.barriers.find(spec.slot);
            if (barrier == mtmc_shared.barriers.end() || !barrier->second) {
                // REFUSED rather than built here. A barrier per Dag is a barrier per WORKER,
                // and a cross-camera tracker that saw one worker's frames would be doing
                // within-camera deduplication -- the failure `mtmc/barrier.h` exists to
                // prevent. Whoever owns the fleet calls `mtmc_runtime` once and passes it in.
                throw ConfigError(
                    "plan runs mtmc slot '" + spec.slot +
                    "' but no barrier was handed to build_dag for it. One barrier per slot is "
                    "shared by every worker: call `mtmc_runtime(planned, settings)` once for "
                    "the process and pass the result in, or a per-worker barrier would turn "
                    "cross-camera association into within-camera deduplication");
            }
            dag.add(std::make_unique<MtmcStage>(
                spec.slot, spec.output, spec.track_source, spec.embedding_sources,
                barrier->second,
                mtmc::create_cluster_tracker(spec.impl, spec.slot, spec.gate)));
        }
        return dag;
    }

}  // namespace shipinfer
