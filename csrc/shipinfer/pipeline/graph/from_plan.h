// A resolved plan becomes stages — the only place this plane decides what to run.
//
// Split from `plan.h` for the reason `emission.h` was split from `state.h`: the READER must
// stay CUDA-free so the offline tier can hold it to the golden, and building a stage needs
// `WorkerScratch`, which is device memory. One header per side of that line.
//
// What was here before was an `if (models.count(...))` ladder in `cli/bench.cpp` with the
// crop extents, the payload names, the output names, the class ids and the label table as
// literals — and its own comments said so ("these names are the graph's config"). One of
// those literals was wrong for months: the label table said a ship was class 1 while the
// crop specs said 8, so every ship reached the event writer as `unknown`.
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/engine/model.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/plan.h"
#include "shipinfer/pipeline/graph/stages.h"

namespace shipinfer {

    // What a plan says this process will run, decided WITHOUT touching a device: the stage
    // names the collector expects, the two tables the event writer needs, and the slots this
    // plane has no stage for. One decision, so a label table that disagrees with the crop
    // specs — the defect this replaces — cannot come back.
    //
    // Separate from the Dag because the Dag is per worker thread (its own `WorkerScratch`)
    // while these are computed once for the run.
    struct PlanTables {
        std::vector<std::string> stage_names;
        pipeline::events::ClassLabels labels;
        pipeline::events::FieldMap fields;
        // decode, track, mtmc, recognize, output — and any slot whose model this process did
        // not load. Named rather than dropped: a run that silently executes four of nine
        // slots is the quiet half-pipeline `missing_stages` exists to report.
        std::vector<std::string> unsupported;
    };

    // The payload a crop element consumes, and the batch its model publishes. Derived once
    // here so `<slot>_crops` and `<slot>_out` cannot drift between the two readers.
    std::string crop_payload_of(const std::string& slot);
    std::string output_of(const std::string& slot);

    using ModelMap = std::map<std::string, std::unique_ptr<Model>>;

    // What will run, from the plan and the models this process loaded. A slot naming a model
    // that is absent is SKIPPED and named in `unsupported`, which is what makes a partial
    // engine set runnable — the ladder's one virtue, kept.
    PlanTables plan_tables(const ResolvedPlan& plan, const ModelMap& models);

    // The same decision, as stages on one worker's scratch.
    //
    // Throws ConfigError when the plan asks for something this plane cannot express: a
    // non-square letterbox, a crop element with two classes, a class label with no id, or a
    // crop with no extent.
    Dag build_dag(const ResolvedPlan& plan, const ModelMap& models, WorkerScratch& scratch,
                  std::chrono::milliseconds timeout);

}  // namespace shipinfer
