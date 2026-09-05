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
        return parse_plan("plan 2 probe\nlabel 0 person\nlabel 8 ship\n" + body, "probe");
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

        // A SEGMENT slot reads the same way, which it did not between #132 and
        // P6-SEGMENT-CROP: this plane REFUSED one with no `classes:` while the Python loader
        // accepted it, so one chain file had two answers. The refusal's argument -- "every
        // row is a 640x640 crop per person" -- is equally true of the embedder above, which
        // has always been allowed to say it, and both planes now do the same work for it.
        // Its own `loaded` set: `kLoaded` has no segmenter, and an unrunnable slot never
        // reaches `class_of` at all -- which is how the check this replaces once passed
        // while asserting nothing.
        const PlanStages segment = plan_stages(
            plan_of(kDetect + "node segment_all segment pool\nmodel ship_segmenter\n"
                              "crop 640 640\n"),
            {"ship_detector", "ship_segmenter"});

        check(segment.crops.size() == 1 && segment.crops[0].class_id == CropSpec::kAnyClass,
              "and a segment slot with no `classes` is every row too, not a refusal");
        check(segment.objects.size() == 1 && segment.objects[0].fold.has_value(),
              "with its fold attached, because it is still a segment slot");
        // The plan's cuts, not this plane's constants: `MaskAreaSpec`'s defaults happen to
        // agree with the Python fold's, so a plan that omitted them was right BY LUCK.
        check(segment.objects[0].fold->score_threshold == 0.25f &&
                  segment.objects[0].fold->mask_threshold == 0.5f,
              "a plan stating no cut leaves the shared defaults");

        const PlanStages stated = plan_stages(
            plan_of(kDetect + "node segment_all segment pool\nmodel ship_segmenter\n"
                              "crop 640 640\nfold_score 0.4\nfold_mask 0.6\n"),
            {"ship_detector", "ship_segmenter"});

        check(stated.objects.at(0).fold->score_threshold == 0.4f &&
                  stated.objects.at(0).fold->mask_threshold == 0.6f,
              "and a plan that states them is what the fold uses");

        // The engine's output NAMES likewise: a YOLO-seg export chooses which slot holds its
        // prototypes, and assuming `output0`/`output1` refused a valid engine from here.
        check(segment.objects.at(0).fold->detections == "output0" &&
                  segment.objects.at(0).fold->prototypes == "output1",
              "a plan stating no output names leaves the export's usual pair");

        const PlanStages named = plan_stages(
            plan_of(kDetect + "node segment_all segment pool\nmodel ship_segmenter\n"
                              "crop 640 640\nfold_detections boxes\nfold_prototypes protos\n"),
            {"ship_detector", "ship_segmenter"});

        check(named.objects.at(0).fold->detections == "boxes" &&
                  named.objects.at(0).fold->prototypes == "protos",
              "and a plan that names them is what the fold reads");
    }

    void a_named_class_resolves_through_the_label_table() {
        const PlanStages built = plan_stages(
            plan_of(kDetect + "node embed_ship embed pool\nmodel ship_embedder\nclasses ship\n"
                              "crop 256 128\n"),
            kLoaded);

        check(built.crops[0].class_id == 8, "a ship is 8 in this plan's table");
        check(built.crops[0].class_name == "ship", "and the event gets the name");
        check(built.crops[0].name == "embed_ship_crops", "the payload name is derived");
        check(built.objects.size() == 1 && built.objects[0].output == "embed_ship_out",
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
            plan_stages(parse_plan("plan 2 probe\nlabel 8 ship\n" + kDetect +
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
        // NO `plan 2 probe` prefix here: `plan_of` prepends the header, and a body carrying
        // its own was refused on a DUPLICATE HEADER before `plan_stages` ran at all -- three
        // checks passing for the wrong reason, and "no runnable detect slot" then had no
        // coverage anywhere (found by #132's round 3).
        check(refused("node e embed pool\nmodel ship_embedder\ncrop 256 128\n"),
              "no runnable detect slot: everything else consumes its boxes");
        check(refused(kDetect + "node e embed pool\nmodel ship_embedder\nclasses ship\n"),
              "a crop slot with no extent");
        check(refused(kDetect + "node e embed pool\nmodel ship_embedder\n"
                                "classes ship,person\ncrop 256 128\n"),
              "two classes in one CropSpec");
        check(refused(kDetect + "node e embed pool\nmodel ship_embedder\nclasses vessel\n"
                                "crop 256 128\n"),
              "a class the label table does not name");
        check(refused("node detect detect pool\nmodel ship_detector\nletterbox 640 480\n"),
              "a non-square letterbox, which DetectConfig cannot express");
        // A cap of `-1` is the same sign class as `kNoClass` one function over: the
        // comparison is `detections.size() < static_cast<size_t>(max_objects)`, so a
        // negative bound is no bound. Refused here rather than surviving to the loop.
        check(refused("node detect detect pool\nmodel ship_detector\nmax_detections -1\n"),
              "a declared cap of -1, which `static_cast<size_t>` turns into no cap");
        // A segment slot with no `classes:` is NOT refused here any more -- see
        // `no_selection_at_all_matches_every_row`, which asserts the reading it gets now.
        // It was, between #132 and P6-SEGMENT-CROP, while the Python plane accepted the same
        // file: one chain file with two answers.
    }

    // A plan built in CODE never passes through `parse_plan`, so the reader's refusals do
    // not protect it -- and `default_bench_plan()` is exactly such a plan. These assert the
    // DECISION's own guards, which a `refused("...")` above cannot: it would be satisfied by
    // the reader and stay green if this layer dropped its check.
    void a_plan_built_in_code_is_still_checked() {
        ResolvedPlan plan = default_bench_plan();
        PlanNode* detect = nullptr;
        for (PlanNode& node : plan.nodes) {
            if (node.kind == "detect") detect = &node;
        }
        check(detect != nullptr, "the default plan has a detect slot");

        const std::set<std::string> loaded = {"ship_detector"};
        detect->max_detections = -1;
        bool refused_cap = false;
        try {
            plan_stages(plan, loaded);
        } catch (const ConfigError&) {
            refused_cap = true;
        }
        check(refused_cap, "a cap of -1 in a code-built plan: refused by the decision itself");

        detect->max_detections = 64;
        detect->letterbox = Extent{640, 480};
        bool refused_letterbox = false;
        try {
            plan_stages(plan, loaded);
        } catch (const ConfigError&) {
            refused_letterbox = true;
        }
        check(refused_letterbox, "and so is a non-square letterbox");
    }

    // `bench.cpp` is the file `CSRC-BENCH-UNCOMPILED` is about, so the chain it runs without
    // a `--plan` used to be "correct by reading" -- and the literal that was WRONG for months
    // was exactly this table. It is a value now, and these are the assertions.
    void the_default_bench_chain_is_what_the_ladder_built() {
        const ResolvedPlan plan = default_bench_plan();
        const PlanStages built = plan_stages(
            plan, {"ship_detector", "ship_segmenter", "person_embedder", "ship_embedder"});

        check(plan.class_id("ship") == 8, "a ship is 8 -- the literal that said 1");
        check(plan.class_id("person") == 0, "and a person is 0");
        check(built.detect.size == 640, "the letterbox");
        check(built.crops.size() == 3, "three crop sets, as the ladder built");
        for (const CropSpec& spec : built.crops) {
            if (spec.name == "ship_segmenter_crops") {
                check(spec.height == 640 && spec.width == 640, "the segmenter's crop");
                check(spec.class_id == 8, "on ships");
            } else if (spec.name == "person_embedder_crops") {
                check(spec.height == 256 && spec.width == 128,
                      "a person crop: tall, not square");
                check(spec.class_id == 0, "on people");
            } else {
                check(spec.name == "ship_embedder_crops", "and the ship embedder's");
                check(spec.class_id == 8 && spec.height == 256, "on ships, same extent");
            }
        }
        check(built.fields.at("embedding").size() == 2, "both embedders fill `embedding`");
        check(built.fields.at("mask_area_px").size() == 1, "the segmenter fills mask_area_px");
        check(built.stage_names.size() == 5, "detect, crop, and one per cropper");
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
        the_default_bench_chain_is_what_the_ladder_built();
        a_plan_built_in_code_is_still_checked();
    } catch (const std::exception& error) {
        std::printf("FAIL: unexpected exception: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
