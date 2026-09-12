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
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/mask_area.h"
#include "shipinfer/pipeline/graph/plan.h"
#include "shipinfer/pipeline/mtmc/cluster.h"
#include "shipinfer/pipeline/tracking/associator.h"

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
    // One per-object stage: which model runs on which payload, and the fold that turns its
    // response into one row per crop. `fold` is set only for a SEGMENT slot -- an embedder's
    // engine already answers one vector per crop, so its stage scatters the response as it is.
    struct ObjectStageSpec {
        std::string slot;
        std::string model;
        std::string source;  // the crop payload it consumes
        std::string output;  // the `ObjectBatch` it attaches
        std::optional<MaskAreaSpec> fold;
    };

    //: The cross-camera slot. `track_source` and `embedding_sources` are OUTPUT names and not
    //: slot names, because an `ObjectBatch` is keyed by a stage's output (`stages.cpp`:
    //: `out.name = output_`) -- the same derivation `fields` uses, so one spelling serves both.
    struct MtmcStageSpec {
        std::string slot;
        std::string output;
        //: The chain's `impl:`, which is `mtmc/cluster.h`'s registry key.
        std::string impl;
        std::string track_source;
        //: The barrier's knobs as the CHAIN states them, absent when it does not. Absent
        //: means the barrier's own defaults, which is the only sense in which "not stated"
        //: and "stated as the default" differ here -- and the reason they are carried at all
        //: is that `ship_person_cpu.yaml` states the window and this plane ignored it.
        //: The group's name and its DECLARED ROSTER, announced to the barrier before any
        //: worker starts. `barrier.h`: an announced camera wins over a merely-seen one the
        //: moment anything announces, so a chain that names its four cameras forms instants
        //: over those four -- which is what the other plane does, and what this plane did not
        //: until #222's review found the two rosters differing.
        std::string group;
        std::vector<std::string> cameras;
        //: The gate's thresholds as the chain states them, absent when it does not -- the
        //: implementation's own defaults then stand, rather than a second copy of them here.
        mtmc::ClusterOptions gate;
        std::optional<double> sync_window_ms;
        std::optional<int> max_instants;
        //: EVERY embedder's output. Several, because a chain embeds people and ships
        //: separately and one row is in exactly one of them.
        std::vector<std::string> embedding_sources;
    };

    //: The tracker's slot, when the plan declares one this plane can run. `output` is the
    //: `ObjectBatch` name its ids arrive under, derived by `output_of` like every other
    //: stage's, so the plan's `field track_id <slot>` line finds them.
    struct TrackStageSpec {
        std::string slot;
        std::string output;
        //: The chain's `impl:`, which is the registry key -- `tracking/associator.h`. The
        //: algorithm inside an impl (`params: algorithm: bytetrack`) is that lane's own
        //: business and the plan does not carry it.
        std::string impl;
        //: WHICH ROWS THIS TRACKER SEES, on `CropSpec`'s convention and for the same reason:
        //: the Python element states `selects_rows = True` and feeds its tracker only the
        //: declared rows, so a plane that tracked every row would emit ids the other never
        //: does AND different ids for the rows they share -- association and the per-camera
        //: counter would have seen boxes the other tracker never got.
        int class_id = CropSpec::kAnyClass;
        //: What the chain said about this tracker, carried to `create_associator` untouched.
        //: The lane converts and refuses a key it does not have.
        tracking::TrackerOptions options;
    };

    struct PlanStages {
        std::string detect_slot;
        std::string detect_model;
        DetectConfig detect;
        std::vector<CropSpec> crops;
        //: The per-object stages, in the plan's order.
        std::vector<ObjectStageSpec> objects;
        //: SEVERAL, because the chain permits it: `_check_one_filler_per_row` refuses two
        //: trackers whose selections overlap and allows two that are disjoint, so a plane that
        //: refused the second outright would throw on a chain the other plane loads.
        std::vector<TrackStageSpec> tracks;
        //: The cross-camera slot, when the plan declares one this plane can run. At most one
        //: is a REFUSAL rather than a convention (`plan_stages.cpp`): two `mtmc` slots are two
        //: camera GROUPS, which the Python plane supports and this plane's barrier budget is
        //: sized for -- but the two would need their group memberships from the chain, and no
        //: chain states them yet, so both would silently be the whole fleet.
        std::vector<MtmcStageSpec> mtmcs;
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
    // Which KINDS this plane runs at all: a detector, and the two that crop per detection.
    // Exported because `cli/bench.cpp` builds its model list from the same plan and asked the
    // same question -- and a second copy of the answer is how one door runs a slot the other
    // reports as "not run here".
    bool plane_runs(const std::string& kind);

    // ...and which of those kinds runs an ENGINE, which is a different question now that one
    // of them does not. `bench_models.cpp` asks this one: a `track` slot with a stray `model:`
    // is a loadable chain (`chain.py`: a model on an element that needs none is "meaningless,
    // but accepted"), and asking `plane_runs` there sent it to `fed_row_of`, which found no
    // crop or letterbox extent and aborted the WHOLE run over a slot that needs no engine.
    bool plane_runs_a_model(const std::string& kind);

    std::string crop_payload_of(const std::string& slot);
    std::string output_of(const std::string& slot);

    // The chain `cli/bench.cpp` runs when it is given no `--plan`, as a VALUE rather than
    // as code in an app nothing compiles: `bench.cpp` is the file `CSRC-BENCH-UNCOMPILED` is
    // about, so its label table -- the literal that said a ship was 1 -- would otherwise be
    // "correct by reading" again. `test_plan_stages.cpp` asserts it instead.
    //
    // The ids are `pipeline.class_labels`'s and the extents are `ship_mask_crop` /
    // `ship_reid_crop` / `person_reid_crop` (`core/settings/pipeline.py`). Slots are named
    // after their models, which is what the log lines and the collector already expect.
    ResolvedPlan default_bench_plan();

    // Decide, from the plan and the names of the models this process loaded.
    //
    // Throws ConfigError when the plan asks for something this plane cannot express or would
    // have to guess at: no runnable detect slot, TWO of them, a non-square letterbox, a crop
    // element with two classes or with no extent, a class the label table does not name, or a
    // `field` naming a slot no `node` declares.
    // The fold a slot's engine needs, or nothing when its answers need none.
    //
    // Exposed because TWO callers need the same answer at different times: `plan_stages`
    // builds the stage that folds on the host, and the model loader attaches the same fold to
    // the ENGINE so the prototype bank never comes home. The second runs before `plan_stages`
    // can (it needs the loaded models), so the two cannot share through `PlanStages`.
    std::optional<MaskAreaSpec> fold_of(const ResolvedPlan& plan, const PlanNode& node);

    PlanStages plan_stages(const ResolvedPlan& plan, const std::set<std::string>& loaded);

}  // namespace shipinfer
