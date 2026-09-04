// What a resolved plan says this process will run — decided WITHOUT touching a device.
//
// Split out of `from_plan.h` because that header includes `engine/model.h` and `stages.h`,
// which reach CUDA, so the *decision* was untestable offline while the reader beside it had a
// gate with 68 checks. Three defects lived here and all three survived review of the reader:
// a declared-empty selection that cropped everything, a second `detect` slot silently
// swallowed, and a null dereference on a `field` naming an undeclared slot.
//
// So the decision takes model NAMES rather than a `ModelMap`, `CropSpec` and `DetectConfig`
// move here (plain values, no device in them), and `csrc/tests/test_plan_stages.cpp` asserts
// the lot with g++ alone. Same split, and the same reason, as `emission.h` out of `state.h`.
#pragma once

#include <array>
#include <set>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/plan.h"

namespace shipinfer {

    struct DetectConfig {
        int size = 640;
        float score_threshold = 0.25f;
        int max_objects = 64;
        float pad_value = 114.f / 255.f;  // TorchImageOps: fill 114, normalise mean 0 / std 255
    };

    struct CropSpec {
        //: Every row -- what a crop element with no `classes:` means on the Python plane.
        static constexpr int kAnyClass = -1;
        //: No row: a DECLARED empty selection (`classes -` in a plan). An id no detector
        //: emits, so the payload is produced with zero rows and the branch is skipped, which
        //: is what "select nothing" has to mean on this side too.
        static constexpr int kNoClass = -2;

        std::string name;        // the payload's name, e.g. "person_crops"
        std::string class_name;  // "person" / "ship", for the event builder
        int class_id = kAnyClass;
        int height = 256;
        int width = 128;
    };

    // What a plan says this process will run: the stage names the collector expects, the two
    // tables the event writer needs, the crop sets, and the slots this plane has no stage for.
    // One decision, so a label table that disagrees with the crop specs -- the defect ADR-020
    // cites -- cannot come back.
    struct PlanStages {
        std::string detect_slot;
        std::string detect_model;
        DetectConfig detect;
        std::vector<CropSpec> crops;
        //: `<slot, model, source payload, output batch>`, in the plan's order.
        std::vector<std::array<std::string, 4>> objects;
        std::vector<std::string> stage_names;
        pipeline::events::ClassLabels labels;
        pipeline::events::FieldMap fields;
        // decode, track, mtmc, recognize, output -- and any slot whose model this process did
        // not load. Named rather than dropped: a run that silently executes four of nine
        // slots is the quiet half-pipeline `missing_stages` exists to report.
        std::vector<std::string> unsupported;
    };

    // The payload a crop element consumes, and the batch its model publishes. Derived once so
    // `<slot>_crops` and `<slot>_out` cannot drift between the two readers.
    std::string crop_payload_of(const std::string& slot);
    std::string output_of(const std::string& slot);

    // Decide, from the plan and the names of the models this process loaded.
    //
    // Throws ConfigError when the plan asks for something this plane cannot express or would
    // have to guess at: no runnable detect slot, TWO of them, a non-square letterbox, a crop
    // element with two classes or with no extent, a class the label table does not name, or a
    // `field` naming a slot no `node` declares.
    PlanStages plan_stages(const ResolvedPlan& plan, const std::set<std::string>& loaded);

}  // namespace shipinfer
