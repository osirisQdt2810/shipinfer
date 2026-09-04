#include "shipinfer/pipeline/graph/from_plan.h"

namespace shipinfer {

    namespace {

        // Which event field a kind fills is stated by the plan (`field` lines), so this file
        // needs no copy of that rule. What it does need is which kinds this plane runs.
        bool crops(const std::string& kind) {
            return kind == "embed" || kind == "segment";
        }

        bool runnable(const PlanNode& node, const ModelMap& models) {
            if (node.kind != "detect" && !crops(node.kind)) return false;
            return !node.model.empty() && models.count(node.model) != 0;
        }

        // The detect slot and the crop slots, in the plan's order. One walk, so the tables
        // and the Dag cannot disagree about what runs.
        struct Selected {
            const PlanNode* detect = nullptr;
            std::vector<const PlanNode*> croppers;
            std::vector<std::string> unsupported;
        };

        Selected select(const ResolvedPlan& plan, const ModelMap& models) {
            Selected chosen;
            for (const PlanNode& node : plan.nodes) {
                if (!runnable(node, models)) {
                    chosen.unsupported.push_back(node.slot);
                } else if (node.kind == "detect") {
                    chosen.detect = &node;
                } else {
                    chosen.croppers.push_back(&node);
                }
            }
            if (chosen.detect == nullptr) {
                throw ConfigError("plan '" + plan.name +
                                  "' has no runnable detect slot, and every other stage this "
                                  "plane runs consumes the detector's boxes");
            }
            return chosen;
        }

        // Three states, not two. No `classes` line is EVERY row, which is what a crop
        // element with no `classes:` means on the Python plane. A declared EMPTY selection is
        // no rows -- `kNoClass`, an id no detector emits, so `CropStage` produces the payload
        // with zero rows and the branch is skipped rather than silently run on everything.
        int class_of(const ResolvedPlan& plan, const PlanNode& node) {
            if (!node.classes) return CropSpec::kAnyClass;
            if (node.classes->empty()) return CropSpec::kNoClass;
            if (node.classes->size() > 1) {
                throw ConfigError("slot '" + node.slot + "' selects " +
                                  std::to_string(node.classes->size()) +
                                  " classes and a CropSpec carries one; split the slot");
            }
            const int id = plan.class_id(node.classes->front());
            if (id < 0) {
                throw ConfigError("slot '" + node.slot + "' selects class '" +
                                  node.classes->front() +
                                  "' and the plan's label table does not name it. A label "
                                  "nobody detects selects no rows and reports nothing wrong");
            }
            return id;
        }

        DetectConfig detect_config(const PlanNode& node) {
            DetectConfig config;
            if (node.letterbox) {
                if (node.letterbox->height != node.letterbox->width) {
                    throw ConfigError("slot '" + node.slot + "' letterboxes to " +
                                      std::to_string(node.letterbox->height) + "x" +
                                      std::to_string(node.letterbox->width) +
                                      " and this plane's DetectConfig carries one extent");
                }
                config.size = node.letterbox->height;
            }
            if (node.score_threshold) {
                config.score_threshold = static_cast<float>(*node.score_threshold);
            }
            if (node.max_detections) config.max_objects = *node.max_detections;
            return config;
        }

        CropSpec crop_spec_of(const ResolvedPlan& plan, const PlanNode& node) {
            if (!node.crop || node.crop->height <= 0 || node.crop->width <= 0) {
                throw ConfigError("slot '" + node.slot +
                                  "' crops and the plan states no extent for it; `shipinfer "
                                  "plan` resolves that from the model's config.yaml");
            }
            return CropSpec{crop_payload_of(node.slot),
                            !node.classes || node.classes->empty() ? "" : node.classes->front(),
                            class_of(plan, node), node.crop->height, node.crop->width};
        }

    }  // namespace

    std::string crop_payload_of(const std::string& slot) {
        return slot + "_crops";
    }

    std::string output_of(const std::string& slot) {
        return slot + "_out";
    }

    PlanTables plan_tables(const ResolvedPlan& plan, const ModelMap& models) {
        const Selected chosen = select(plan, models);
        PlanTables tables;
        tables.labels = plan.labels;
        tables.unsupported = chosen.unsupported;
        // An `ObjectBatch` is keyed by a stage's OUTPUT name and not its own
        // (`stages.cpp`: `out.name = output_`), and looking one up by the stage name found
        // nothing on every frame. Derived here so both readers use one spelling.
        for (const auto& [field, slots] : plan.fields) {
            for (const std::string& slot : slots) {
                if (runnable(*plan.node(slot), models)) {
                    tables.fields[field].push_back(output_of(slot));
                }
            }
        }
        tables.stage_names.push_back(chosen.detect->slot);
        if (!chosen.croppers.empty()) tables.stage_names.push_back("crop");
        for (const PlanNode* node : chosen.croppers) {
            tables.stage_names.push_back(node->slot);
        }
        return tables;
    }

    Dag build_dag(const ResolvedPlan& plan, const ModelMap& models, WorkerScratch& scratch,
                  std::chrono::milliseconds timeout) {
        const Selected chosen = select(plan, models);
        const DetectConfig config = detect_config(*chosen.detect);
        Dag dag;
        dag.add(std::make_unique<DetectStage>(
            chosen.detect->slot, *models.at(chosen.detect->model), config, scratch, timeout));
        // One crop pass for every payload, which is this plane's own choice and the reason a
        // crop element is not a stage: N classes cost one walk of the frame, not N.
        std::vector<CropSpec> crop_specs;
        for (const PlanNode* node : chosen.croppers) {
            crop_specs.push_back(crop_spec_of(plan, *node));
        }
        if (!crop_specs.empty()) {
            dag.add(
                std::make_unique<CropStage>("crop", crop_specs, config.max_objects, scratch));
        }
        for (const PlanNode* node : chosen.croppers) {
            dag.add(std::make_unique<ObjectStage>(node->slot, *models.at(node->model),
                                                  crop_payload_of(node->slot),
                                                  output_of(node->slot), timeout));
        }
        return dag;
    }

}  // namespace shipinfer
