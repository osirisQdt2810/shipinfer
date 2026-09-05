// The plan's model-bearing slots as a list of engines to load: plain values in, plain values
// out, so it is asserted rather than "correct by reading".
//
// Extracted from `cli/bench.cpp` for the reason `plan_stages.h`'s own header gives: it lived
// in an anonymous namespace inside a `main()` translation unit, so none of the C++ gates could
// touch it -- and three of its four refusals were added in response to review with nothing
// checking any of them.
//
// CUDA-free, like the rest of the graph's decision half.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/plan.h"

namespace shipinfer {

    // Where an engine's path comes from when the plan does not carry one: the per-model flags
    // `cli/bench.cpp` has always had, kept for a run with an engine and no repository.
    struct BenchEngines {
        //: The model repository root the plan's `artefact` paths hang off. Set means the plan
        //: is authoritative and a slot it cannot resolve is a refusal, not a skip.
        std::string repository;
        std::string detector;
        std::string segmenter;
        std::string person_embedder;
        std::string ship_embedder;
        //: The batch window for a run that has no plan runtime to read one from.
        int batch_delay_us = 2000;
        int detector_instances = 2;
        int segmenter_instances = 1;
        int person_embedder_instances = 2;
        int ship_embedder_instances = 1;
    };

    // One model to load: what the plan says, resolved to a path this process can open.
    struct BenchModel {
        std::string name;
        std::string engine;
        int per_device = 1;
        int queue_delay_us = 0;
        //: `(3, h, w)` -- one row of the model's input, from the slot's own geometry.
        std::vector<int64_t> fed_row;
    };

    // Every model-bearing slot THIS PLANE RUNS, deduplicated by model.
    //
    // Deduplicated because one engine is loaded per device and its weights are paid for once;
    // skipping a kind this plane does not run (`plane_runs`) because a `recognize: {impl:
    // pool}` slot is a valid chain that `plan_stages` reports as "not run here", and aborting
    // the whole measurement over it would take the detector and both embedders down with it.
    //
    // Throws `ConfigError` when: the plan names an artefact this process cannot resolve under
    // `--repository`; two slots feed one model rows of different extents (one engine, one
    // input shape); a slot this plane runs states no extent at all; or `--repository` is set
    // and the plan carries no `instances` for a model, which would otherwise fall back to a
    // flag's default for a model the repository asked no device instances of.
    std::vector<BenchModel> bench_models(const ResolvedPlan& plan, const BenchEngines& engines);

}  // namespace shipinfer
