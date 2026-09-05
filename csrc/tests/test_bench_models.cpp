// The plan's model-bearing slots as a list of engines to load: `graph/bench_models.cpp`.
//
// It lived in an anonymous namespace inside `cli/bench.cpp`, a `main()` translation unit no
// gate can link, and three of its four refusals were added in response to review with nothing
// checking any of them. `plan_stages.h`'s own header makes the argument this file acts on:
// extracted so it is asserted rather than "correct by reading".
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/bench_models.h"

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

    PlanNode node_of(const std::string& slot, const std::string& kind,
                     const std::string& model) {
        PlanNode node;
        node.slot = slot;
        node.kind = kind;
        node.impl = "pool";
        node.model = model;
        return node;
    }

    ResolvedPlan plan_of(std::vector<PlanNode> nodes) {
        ResolvedPlan plan;
        plan.version = 1;
        plan.name = "test";
        plan.nodes = std::move(nodes);
        return plan;
    }

    // A detector and a ship embedder, both resolved as a repository resolves them.
    ResolvedPlan a_repository_plan() {
        PlanNode detect = node_of("detect", "detect", "ship_detector");
        detect.letterbox = Extent{640, 640};
        detect.instances = 2;
        detect.queue_delay_us = 5000;
        detect.artefact = "ship_detector/1/model.plan";
        PlanNode embed = node_of("embed_ship", "embed", "ship_embedder");
        embed.crop = Extent{256, 128};
        embed.instances = 1;
        embed.queue_delay_us = 8000;
        embed.artefact = "ship_embedder/1/model.plan";
        return plan_of({detect, embed});
    }

    BenchEngines with_repository() {
        BenchEngines engines;
        engines.repository = "/work/model_repository";
        return engines;
    }

    bool refused(const ResolvedPlan& plan, const BenchEngines& engines, std::string& message) {
        try {
            bench_models(plan, engines);
            return false;
        } catch (const std::exception& error) {
            message = error.what();
            return true;
        }
    }

    void a_repository_plan_resolves_every_number_from_the_plan() {
        const std::vector<BenchModel> models =
            bench_models(a_repository_plan(), with_repository());

        check(models.size() == 2, "one entry per model-bearing slot");
        check(models[0].engine == "/work/model_repository/ship_detector/1/model.plan",
              "the artefact is joined to the repository root, not to this box's filesystem");
        check(models[0].per_device == 2 && models[1].per_device == 1,
              "the instance counts are the plan's, not a flag's default of 2/1");
        check(models[0].queue_delay_us == 5000 && models[1].queue_delay_us == 8000,
              "and the batch windows are per model rather than one global number");
        check(models[0].fed_row == std::vector<int64_t>{3, 640, 640},
              "the detector is fed its letterbox extent");
        check(models[1].fed_row == std::vector<int64_t>{3, 256, 128},
              "and the embedder its crop -- the table of four hard-coded shapes is gone");
    }

    void a_kind_this_plane_does_not_run_is_skipped_not_refused() {
        // `recognize: {impl: pool}` is a valid chain -- `plan_stages` reports it as "not run
        // here" -- and `resolve_plan` fills `crop` only for embed/segment, so it arrives with
        // a model, an artefact and NO extent. Refusing it aborted the whole measurement
        // before a single engine loaded, taking the detector and the embedders down with it.
        std::vector<PlanNode> nodes = a_repository_plan().nodes;
        PlanNode recognize = node_of("recognize", "recognize", "ship_recognizer");
        recognize.instances = 1;
        recognize.artefact = "ship_recognizer/1/model.plan";
        nodes.push_back(recognize);

        const std::vector<BenchModel> models = bench_models(plan_of(nodes), with_repository());

        check(models.size() == 2, "the recognizer is not in the list");
        check(models[0].name == "ship_detector" && models[1].name == "ship_embedder",
              "and the models this plane does run are still all there");
    }

    void a_slot_this_plane_runs_with_no_extent_is_refused() {
        // The other half: a kind this plane DOES run has to state its geometry, because the
        // engine's input shape is what the crop is resized to.
        PlanNode detect = node_of("detect", "detect", "ship_detector");
        detect.instances = 2;
        detect.artefact = "ship_detector/1/model.plan";  // no letterbox
        std::string message;

        check(refused(plan_of({detect}), with_repository(), message),
              "a runnable slot with no extent is refused");
        check(message.find("no crop or letterbox") != std::string::npos,
              "and the message says which geometry is missing");
    }

    void two_slots_feeding_one_model_at_different_extents_are_refused() {
        // Expressible: `branching.yaml` crops `ship_segmenter` at 512 where
        // `ship_person_cpu.yaml` crops it at 640. One engine is loaded per model, so its
        // input shape has to be one shape.
        PlanNode first = node_of("seg_a", "segment", "ship_segmenter");
        first.crop = Extent{640, 640};
        first.instances = 2;
        first.artefact = "ship_segmenter/1/model.plan";
        PlanNode second = first;
        second.slot = "seg_b";
        second.crop = Extent{512, 512};
        std::string message;

        check(refused(plan_of({first, second}), with_repository(), message),
              "two extents for one model are refused");
        check(message.find("seg_b") != std::string::npos, "naming the second slot");

        // ...and the same model at the SAME extent is the ordinary dedup: one engine, once.
        PlanNode same = first;
        same.slot = "seg_c";
        const std::vector<BenchModel> models =
            bench_models(plan_of({first, same}), with_repository());
        check(models.size() == 1, "agreeing slots load the engine once, weights paid once");
    }

    void a_missing_artefact_or_instances_is_refused_under_a_repository() {
        PlanNode detect = node_of("detect", "detect", "ship_detector");
        detect.letterbox = Extent{640, 640};
        detect.instances = 2;
        std::string message;

        check(refused(plan_of({detect}), with_repository(), message),
              "a plan with no artefact cannot be resolved against a repository");
        check(message.find("Re-run") != std::string::npos,
              "and the message says how to fix it");

        // No `instances` is what a repository says for a model it asks NO device instance of
        // (a CPU-only one). Falling back to a flag's default would run GPU instances of it.
        PlanNode cpu_only = detect;
        cpu_only.artefact = "ship_detector/1/model.plan";
        cpu_only.instances.reset();
        check(refused(plan_of({cpu_only}), with_repository(), message),
              "and a plan with no instances is refused rather than defaulted");
        check(message.find("no DEVICE instance") != std::string::npos,
              "naming what the repository actually said");
    }

    void the_flag_path_still_works_without_a_repository() {
        // The run with an engine and no repository: the flags answer, and a model no flag
        // names is skipped rather than refused.
        BenchEngines engines;
        engines.detector = "/work/models/yolo26n.engine";
        engines.detector_instances = 3;
        engines.batch_delay_us = 2000;

        const std::vector<BenchModel> models = bench_models(a_repository_plan(), engines);

        check(models.size() == 1 && models[0].name == "ship_detector",
              "only the model a flag names is loaded");
        check(models[0].engine == "/work/models/yolo26n.engine", "at the flag's path");
        check(models[0].per_device == 2,
              "and the PLAN's instance count still wins where it states one -- the flag is a "
              "fallback, not an override");
        check(models[0].queue_delay_us == 5000, "as does its batch window");
    }

}  // namespace

int main() {
    try {
        a_repository_plan_resolves_every_number_from_the_plan();
        a_kind_this_plane_does_not_run_is_skipped_not_refused();
        a_slot_this_plane_runs_with_no_extent_is_refused();
        two_slots_feeding_one_model_at_different_extents_are_refused();
        a_missing_artefact_or_instances_is_refused_under_a_repository();
        the_flag_path_still_works_without_a_repository();
    } catch (const std::exception& error) {
        std::printf("FAIL: uncaught: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
