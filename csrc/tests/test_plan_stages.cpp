// What a plan says this process will RUN — the decision, gated offline.
//
// This file exists because three defects hid behind an arrangement rather than behind a hard
// question. `test_plan_parity` covers the reader thoroughly (68 checks) and the reader was
// fine; the behaviour lives in the decision next to it, and that was untestable with g++
// alone because its header reached CUDA. Review found all three by reading:
//
//   * a DECLARED EMPTY selection cropped every row (`>= 0` skips nothing for -2), which is
//     the opposite of what it means and "at an embedder a doubled GPU bill";
//   * a second `detect` slot was silently swallowed, so the run detected at the wrong
//     threshold and the declared slot vanished from both the Dag and `unsupported`;
//   * a `field` naming an undeclared slot dereferenced null before any worker started.
//
// So `plan_stages.h` is CUDA-free and takes model NAMES, and every one of those is a check
// below. Offline: g++ alone, no CUDA, no GStreamer.

#include <cstdio>
#include <set>
#include <string>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/plan_stages.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    const std::set<std::string> kLoaded = {"ship_detector", "ship_embedder", "person_embedder"};

    ResolvedPlan plan_of(const std::string& body) {
        return parse_plan("plan 1 probe\nlabel 0 person\nlabel 8 ship\n" + body, "probe");
    }

    const std::string kDetect =
        "node detect detect pool\nmodel ship_detector\nletterbox 640 640\nscore 0.4\n"
        "max_detections 30\n";

    bool refused(const std::string& body) {
        try {
            plan_stages(plan_of(body), kLoaded);
            return false;
        } catch (const ConfigError&) {
            return true;
        }
    }

    void the_detect_slot_carries_the_plans_numbers() {
        const PlanStages built = plan_stages(plan_of(kDetect), kLoaded);

        check(built.detect_slot == "detect", "the detect slot");
        check(built.detect_model == "ship_detector", "and its model");
        check(built.detect.size == 640, "the letterbox extent");
        check(built.detect.score_threshold == 0.4f, "the threshold, from the plan");
        check(built.detect.max_objects == 30, "and the cap");
        check(built.stage_names == std::vector<std::string>{"detect"}, "one stage, no crop");
        check(built.crops.empty() && built.objects.empty(), "and nothing to crop");
    }

    // THE defect: `kNoClass` is negative, so a `>= 0` guard skipped nothing and a slot that
    // declared `classes: []` was handed every row.
    void a_declared_empty_selection_matches_no_row() {
        const PlanStages built = plan_stages(
            plan_of(kDetect + "node embed_none embed pool\nmodel ship_embedder\nclasses -\n"
                              "crop 256 128\n"),
            kLoaded);

        check(built.crops.size() == 1, "one crop set");
        check(built.crops[0].class_id == CropSpec::kNoClass, "a declared-empty selection");
        check(built.crops[0].class_id != CropSpec::kAnyClass,
              "and it is NOT the every-row sentinel, which is the whole distinction");
        check(built.crops[0].class_name.empty(), "no class name to put in an event");
        // The consequence, spelled out: the two sentinels must not be interchangeable under
        // any comparison a filter is likely to use.
        check(CropSpec::kNoClass < 0 && CropSpec::kAnyClass < 0,
              "both are negative, which is why `>= 0` was the wrong test");
        check(CropSpec::kNoClass != CropSpec::kAnyClass, "and they are different values");
    }

    void no_selection_at_all_matches_every_row() {
        const PlanStages built = plan_stages(
            plan_of(kDetect + "node embed_all embed pool\nmodel ship_embedder\ncrop 256 128\n"),
            kLoaded);

        check(built.crops.size() == 1 && built.crops[0].class_id == CropSpec::kAnyClass,
              "no `classes` line: every row, as `pool.py` defaults");
    }

    void a_named_class_resolves_through_the_label_table() {
        const PlanStages built = plan_stages(
            plan_of(kDetect + "node embed_ship embed pool\nmodel ship_embedder\nclasses ship\n"
                              "crop 256 128\n"),
            kLoaded);

        check(built.crops[0].class_id == 8, "a ship is 8 in this plan's table");
        check(built.crops[0].class_name == "ship", "and the event gets the name");
        check(built.crops[0].name == "embed_ship_crops", "the payload name is derived");
        check(built.objects.size() == 1 && built.objects[0][3] == "embed_ship_out",
              "and so is the output batch, which is how a batch is keyed");
    }

    void two_detect_slots_are_refused_rather_than_one_dropped() {
        const std::string second =
            "node detect_small detect pool\nmodel ship_detector\nletterbox 640 640\n"
            "score 1e-05\nmax_detections 7\n";

        check(refused(kDetect + second), "two runnable detect slots: refused, not last-wins");
        // Non-vacuity: the same plan with the second slot's model absent is fine, because
        // then only one is runnable and the other is reported.
        const PlanStages built =
            plan_stages(parse_plan("plan 1 probe\nlabel 8 ship\n" + kDetect +
                                       "node detect_other detect pool\nmodel absent_model\n",
                                   "probe"),
                        kLoaded);
        check(built.detect_slot == "detect", "the runnable one runs");
        check(built.unsupported == std::vector<std::string>{"detect_other"},
              "and the other is NAMED rather than dropped in silence");
    }

    void a_field_naming_an_undeclared_slot_is_refused() {
        check(refused(kDetect + "field embedding embed_ship\n"),
              "a `field` naming a slot no `node` declares: refused, not a null dereference");
        // And one that names a declared-but-not-runnable slot is fine: it simply contributes
        // nothing to the table, which is what a partial engine set means.
        const PlanStages built =
            plan_stages(plan_of(kDetect + "node embed_ship embed pool\nmodel absent_model\n"
                                          "field embedding embed_ship\n"),
                        kLoaded);
        check(built.fields.empty(), "a slot this process cannot run fills no field");
    }

    void the_refusals_this_plane_owes_the_operator() {
        // The reader's own refusals (a missing header, an unknown verb) are
        // `test_plan_parity`'s; these are the ones this decision owns.
        check(refused("plan 1 probe\nnode e embed pool\nmodel ship_embedder\ncrop 256 128\n"),
              "no runnable detect slot: everything else consumes its boxes");
        check(refused(kDetect + "node e embed pool\nmodel ship_embedder\nclasses ship\n"),
              "a crop slot with no extent");
        check(refused(kDetect + "node e embed pool\nmodel ship_embedder\n"
                                "classes ship,person\ncrop 256 128\n"),
              "two classes in one CropSpec");
        check(refused(kDetect + "node e embed pool\nmodel ship_embedder\nclasses vessel\n"
                                "crop 256 128\n"),
              "a class the label table does not name");
        check(refused("plan 1 probe\nnode detect detect pool\nmodel ship_detector\n"
                      "letterbox 640 480\n"),
              "a non-square letterbox, which DetectConfig cannot express");
    }

    void the_tables_the_event_writer_needs() {
        const PlanStages built = plan_stages(
            plan_of(kDetect + "node embed_ship embed pool\nmodel ship_embedder\nclasses ship\n"
                              "crop 256 128\n"
                              "node embed_person embed pool\nmodel person_embedder\n"
                              "classes person\ncrop 256 128\n"
                              "field embedding embed_ship embed_person\n"),
            kLoaded);

        check(built.labels.size() == 2 && built.labels.at(8) == "ship", "the label table");
        check(built.stage_names ==
                  std::vector<std::string>{"detect", "crop", "embed_ship", "embed_person"},
              "the stage names the collector expects, in the plan's order");
        check(built.crops.size() == 2, "one crop set per slot");
    }

}  // namespace

int main() {
    try {
        the_detect_slot_carries_the_plans_numbers();
        a_declared_empty_selection_matches_no_row();
        no_selection_at_all_matches_every_row();
        a_named_class_resolves_through_the_label_table();
        two_detect_slots_are_refused_rather_than_one_dropped();
        a_field_naming_an_undeclared_slot_is_refused();
        the_refusals_this_plane_owes_the_operator();
        the_tables_the_event_writer_needs();
    } catch (const std::exception& error) {
        std::printf("FAIL: unexpected exception: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
