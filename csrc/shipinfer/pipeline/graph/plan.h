// The resolved chain this plane was handed — `topology/plan.py`, verb for verb.
//
// ADR-014 says the control plane "hands this plane a resolved configuration", and ADR-017
// makes `Topology.from_spec` the single door a chain becomes trustworthy through. So this is
// a READER, never a validator: no YAML, no element registry, no second copy of the chain
// rules. What arrives has already been refused or accepted on the Python side.
//
// CUDA-free on purpose, like `emission.h`: the reader is then compiled and run by the offline
// C++ tier on a machine with no driver, which is where a format divergence should be caught.
//
// `namespace shipinfer` and not `shipinfer::pipeline::graph`, because that is what every
// other header in this directory uses and one file's taste is not worth a qualified name at
// each of its call sites.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // Two positive integers, `(height, width)` — the order `pipeline.ship_reid_crop` uses.
    struct Extent {
        int height = 0;
        int width = 0;
    };

    // One resolved slot. Everything optional is optional in the plan text too, and absent
    // means "the chain file said nothing", never "zero".
    struct PlanNode {
        std::string slot;
        std::string kind;
        std::string impl;
        std::string model;
        // ABSENT is "no selection declared" -- every row. PRESENT AND EMPTY is a declared
        // empty selection -- no rows. `elements/detections.py` names the failure of
        // conflating them: "a typo silently select[s] everything -- at `track` a wrong
        // answer, at an embedder a doubled GPU bill". The plan spells both, so this must too.
        std::optional<std::vector<std::string>> classes;
        std::optional<Extent> crop;
        std::optional<Extent> letterbox;
        std::optional<double> score_threshold;
        std::optional<int> max_detections;
        // The runtime the model repository resolved: instances PER DEVICE, the batch window
        // in microseconds (0 is "no window", which is dynamic batching off), and the artefact
        // relative to the repository root. Absent where the plan was resolved without a
        // repository, which is why every consumer refuses by name rather than defaulting --
        // `bench.cpp` used to restate all three on its command line and its numbers disagreed
        // with the repository's on three of four models.
        std::optional<int> instances;
        std::optional<int> queue_delay_us;
        std::string artefact;
        // A segmentation fold's two cuts. Both are defaulted on both planes, which is why the
        // plan carries them: an omission agrees by luck today and diverges the moment a chain
        // file states one. `fold_mask` is strictly inside (0, 1) -- the cut is
        // `log(m / (1 - m))`, which is -inf at 0 and a division by zero at 1.
        std::optional<double> fold_score;
        std::optional<double> fold_mask;
        // Which of the engine's outputs the fold reads. Which slot a YOLO-seg export puts its
        // prototypes in is the export's choice, so assuming `output0`/`output1` here refused
        // an engine that names them anything else -- loudly, but from the wrong plane.
        std::string fold_detections;
        std::string fold_prototypes;
        std::string when;
        std::string per;
        std::string scope;
    };

    struct PlanEdge {
        std::string producer;
        std::string consumer;
        std::string caps;  // `<format>@<location>`, already negotiated
    };

    // The run configuration a chain does not declare and a repository does not hold --
    // `topology/plan.py`'s `PlanSettings`, field for field. Every one of these was a default
    // in `cli/bench.cpp` and a flag on its command line, defaulted a SECOND time there, so the
    // two planes ran configurations neither settings file described (P5-C).
    //
    // Two capacities and not one: `instance_queue` bounds each model instance's own queue
    // (`scheduler.max_queue_size`, 64 -- deliberately small, because a deep queue converts a
    // throughput problem into a latency problem and then hides it) and `pipeline_queue` bounds
    // the frames ingest may hand forward (`pipeline.queue_capacity`, 256). `bench.cpp` used
    // ONE number, 65536, for both -- 1024x the first, which is a per-instance queue that can
    // never reject and therefore a backpressure measurement whose zero was guaranteed.
    struct PlanSettings {
        int workers = 0;
        int pipeline_queue = 0;
        int instance_queue = 0;
        int enqueue_block_timeout_ms = 0;
        int stage_timeout_ms = 0;
        int reassembly_capacity = 0;
        int reassembly_timeout_ms = 0;
        int reassembly_sweep_ms = 0;
    };

    // The `setting` keys in WRITTEN order, paired with the member each names. One table, so
    // the reader, the writer and the completeness check cannot disagree about the set --
    // which is the failure the closed set exists to prevent.
    const std::vector<std::pair<std::string, int PlanSettings::*>>& setting_keys();

    struct ResolvedPlan {
        int version = 0;
        std::string name;
        // Absent where the plan states no `setting` line: the shape of the chain alone. A
        // consumer that NEEDS them refuses then rather than substituting its own defaults,
        // because a default only one plane holds is exactly what carrying them replaced.
        std::optional<PlanSettings> settings;
        std::vector<PlanNode> nodes;
        std::vector<PlanEdge> edges;
        std::map<int, std::string> labels;                       // class id -> label
        std::map<std::string, std::vector<std::string>> fields;  // event field -> slots
        const PlanNode* node(const std::string& slot) const;
        // The class id a label was given, or nothing. An `optional` and not -1, which is a
        // legal declared id: `class_of` happens to throw on any negative, so the sentinel
        // was safe by accident rather than by construction. The ids are the CHECKPOINT's --
        // this demo detector calls a ship 8, and a plane that assumed 1 cropped the right
        // rows and labelled every ship `unknown` in its events.
        std::optional<int> class_id(const std::string& label) const;
    };

    // The version this reader knows. A plan that says anything else is refused, because a
    // plan half understood is a chain running something other than what was declared.
    inline constexpr int kPlanVersion = 2;

    // Refusals name the line: `<source>:<n>: ...`. A plan one plane reads and the other
    // refuses is the worst outcome this seam has, so the messages are worth matching.
    ResolvedPlan parse_plan(const std::string& text, const std::string& source = "<string>");
    ResolvedPlan read_plan(const std::string& path);

    // Back to text, byte-identical to what `topology/plan.py` wrote. This is what makes the
    // gate a gate: re-serialising a golden and comparing is a stronger claim than a parse
    // that discards what it did not understand.
    std::string plan_text(const ResolvedPlan& plan);

}  // namespace shipinfer
