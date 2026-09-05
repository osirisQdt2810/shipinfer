#include "shipinfer/pipeline/graph/bench_models.h"

#include <algorithm>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/plan_stages.h"

namespace shipinfer {

    namespace {

        // The four flags, matched by MODEL name. Only reached when no repository was given.
        std::string flag_engine_of(const std::string& model, const BenchEngines& engines) {
            if (model == "ship_detector") return engines.detector;
            if (model == "ship_segmenter") return engines.segmenter;
            if (model == "person_embedder") return engines.person_embedder;
            if (model == "ship_embedder") return engines.ship_embedder;
            return "";
        }

        int flag_instances_of(const std::string& model, const BenchEngines& engines) {
            if (model == "ship_detector") return engines.detector_instances;
            if (model == "ship_segmenter") return engines.segmenter_instances;
            if (model == "person_embedder") return engines.person_embedder_instances;
            if (model == "ship_embedder") return engines.ship_embedder_instances;
            return 1;
        }

        // What the model is fed, per row: `(3, h, w)` from the slot's own geometry. The plan
        // carries it, which is what deleted the hard-coded table of four names with their
        // extents.
        std::vector<int64_t> fed_row_of(const PlanNode& node) {
            const std::optional<Extent>& extent = node.crop ? node.crop : node.letterbox;
            if (!extent) {
                throw ConfigError("slot '" + node.slot + "' runs model '" + node.model +
                                  "' on this plane and the plan states no crop or letterbox "
                                  "extent for it, so this process cannot say what the engine "
                                  "is fed");
            }
            return {3, extent->height, extent->width};
        }

    }  // namespace

    std::vector<BenchModel> bench_models(const ResolvedPlan& plan,
                                         const BenchEngines& engines) {
        const bool from_plan = !engines.repository.empty();
        std::vector<BenchModel> models;
        for (const PlanNode& node : plan.nodes) {
            // A kind this plane does not run is not this list's business, even when it names a
            // model: `recognize: {impl: pool}` is a valid chain and `plan_stages` reports it
            // as "not run here". Asked BEFORE any refusal, so such a slot cannot abort a run
            // that was never going to include it.
            if (node.model.empty() || !plane_runs(node.kind)) continue;

            const std::vector<int64_t> fed_row = fed_row_of(node);
            const auto seen =
                std::find_if(models.begin(), models.end(),
                             [&node](const BenchModel& m) { return m.name == node.model; });
            if (seen != models.end()) {
                // The dedup is deliberate -- one engine per device, weights paid once -- but
                // the geometry is per SLOT, and two slots may name one model at different
                // extents (`branching.yaml` crops `ship_segmenter` at 512 where
                // `ship_person_cpu.yaml` crops it at 640). Feeding the second slot's rows at
                // the first's shape is a silent wrong answer.
                if (seen->fed_row != fed_row) {
                    throw ConfigError("slot '" + node.slot + "' feeds model '" + node.model +
                                      "' rows of a different extent than an earlier slot "
                                      "does; one engine is loaded per model, so its input "
                                      "shape has to be one shape");
                }
                continue;
            }

            BenchModel model;
            model.name = node.model;
            model.engine =
                from_plan ? (node.artefact.empty() ? std::string()
                                                   : engines.repository + "/" + node.artefact)
                          : flag_engine_of(node.model, engines);
            if (model.engine.empty()) {
                if (from_plan) {
                    throw ConfigError("slot '" + node.slot + "' runs model '" + node.model +
                                      "' and the plan states no `artefact` for it, so a "
                                      "repository cannot resolve an engine. Re-run "
                                      "`shipinfer plan` against the repository");
                }
                continue;  // no flag named it: this run does not run it
            }
            if (from_plan && !node.instances) {
                // A plan resolved against a repository states `instances` for every model it
                // asks device instances of -- and states NONE for a CPU-only model. Falling
                // back to a flag's default there would run GPU instances of a model the
                // repository asked for none of, with nothing said.
                throw ConfigError("slot '" + node.slot + "' runs model '" + node.model +
                                  "' and the plan states no `instances` for it, which is what "
                                  "a repository says when a model asks for no DEVICE instance "
                                  "at all. This process runs on devices");
            }
            model.per_device =
                node.instances ? *node.instances : flag_instances_of(node.model, engines);
            model.queue_delay_us =
                node.queue_delay_us ? *node.queue_delay_us : engines.batch_delay_us;
            model.fed_row = fed_row;
            models.push_back(std::move(model));
        }
        return models;
    }

}  // namespace shipinfer
