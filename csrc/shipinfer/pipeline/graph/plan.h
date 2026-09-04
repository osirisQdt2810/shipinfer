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
        std::string when;
        std::string per;
        std::string scope;
    };

    struct PlanEdge {
        std::string producer;
        std::string consumer;
        std::string caps;  // `<format>@<location>`, already negotiated
    };

    struct ResolvedPlan {
        int version = 0;
        std::string name;
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
    inline constexpr int kPlanVersion = 1;

    // Refusals name the line: `<source>:<n>: ...`. A plan one plane reads and the other
    // refuses is the worst outcome this seam has, so the messages are worth matching.
    ResolvedPlan parse_plan(const std::string& text, const std::string& source = "<string>");
    ResolvedPlan read_plan(const std::string& path);

    // Back to text, byte-identical to what `topology/plan.py` wrote. This is what makes the
    // gate a gate: re-serialising a golden and comparing is a stronger claim than a parse
    // that discards what it did not understand.
    std::string plan_text(const ResolvedPlan& plan);

}  // namespace shipinfer
