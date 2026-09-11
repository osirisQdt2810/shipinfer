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
//
// The DECISION is `plan_stages.h`, which is CUDA-free and has its own offline gate. What is
// left here is the part that cannot be: turning it into stages on a worker's scratch.
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "shipinfer/engine/model.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/plan_stages.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/mtmc/barrier.h"
#include "shipinfer/pipeline/mtmc/cluster.h"

namespace shipinfer {

    using ModelMap = std::map<std::string, std::unique_ptr<Model>>;

    //: The names of the models this process loaded -- what `plan_stages` needs, which is
    //: strictly less than the pool itself.
    std::set<std::string> loaded_names(const ModelMap& models);

    // The plan's decision, as stages on one worker's scratch. Refusals are `plan_stages`'s.
    // doc: long why the barrier and its budget are the CALLER's and not built here
    //: The cross-camera pieces one process shares, because a Dag is per WORKER and these are
    //: not. The barrier turns per-camera frames into instants and must see every worker's
    //: frames to do it; the budget caps the waiters across every barrier in the process, which
    //: is its whole reason to exist (`mtmc/barrier.h`: two barriers each admitting
    //: `workers - 1` would park every worker between them). Built once by whoever owns the
    //: fleet -- `cli/bench.cpp` -- and handed to every worker's `build_dag`. Absent means the
    //: plan runs no `mtmc` slot, and one that does with these empty is a refusal.
    struct MtmcRuntime {
        std::shared_ptr<mtmc::WaiterBudget> budget;
        //: Slot -> its barrier. One per slot, because one barrier is one camera group.
        std::map<std::string, std::shared_ptr<mtmc::InstantBarrier>> barriers;
    };

    // Every `mtmc` slot the plan runs, with its barrier and tracker built. Called ONCE per
    // process, before the workers start: two calls would be two identity spaces for one group.
    MtmcRuntime mtmc_runtime(const PlanStages& planned, const PlanSettings& settings);

    Dag build_dag(const PlanStages& planned, const ModelMap& models, WorkerScratch& scratch,
                  std::chrono::milliseconds timeout, const MtmcRuntime& mtmc = {});

}  // namespace shipinfer
