#include "shipinfer/pipeline/graph/plan_stages.h"

namespace shipinfer {

    namespace {

        // Which event field a kind fills is stated by the plan (`field` lines), so this file
        // needs no copy of that rule. What it does need is which kinds this plane runs.
        bool crops(const std::string& kind) {
            return kind == "embed" || kind == "segment";
        }

        bool runnable(const PlanNode& node, const std::set<std::string>& loaded) {
            if (node.kind != "detect" && !crops(node.kind)) return false;
            return !node.model.empty() && loaded.count(node.model) != 0;
        }

        int class_of(const ResolvedPlan& plan, const PlanNode& node) {
            // Three states, not two. No `classes` line is EVERY row, which is what a crop
            // element with no `classes:` means on the Python plane. A declared EMPTY
            // selection is no rows.
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

    PlanStages plan_stages(const ResolvedPlan& plan, const std::set<std::string>& loaded) {
        PlanStages built;
        const PlanNode* detect = nullptr;
        std::vector<const PlanNode*> croppers;
        for (const PlanNode& node : plan.nodes) {
            if (!runnable(node, loaded)) {
                built.unsupported.push_back(node.slot);
            } else if (node.kind == "detect") {
                // REFUSED, not last-wins. Two detectors is a supported chain shape on the
                // Python plane (`topology/plan.py::_labels` unions their tables), so a plan
                // with two arrives here without a murmur -- and overwriting meant the run
                // detected at the second slot's threshold while the first vanished from both
                // the Dag and `unsupported`: "a chain running something other than what was
                // declared", which is the phrase the version gate exists for.
                if (detect != nullptr) {
                    throw ConfigError("plan '" + plan.name +
                                      "' has two runnable detect slots ('" + detect->slot +
                                      "' and '" + node.slot +
                                      "'); this plane runs one detector per shard, so say "
                                      "which in the chain rather than letting one be dropped");
                }
                detect = &node;
            } else {
                croppers.push_back(&node);
            }
        }
        if (detect == nullptr) {
            throw ConfigError("plan '" + plan.name +
                              "' has no runnable detect slot, and every other stage this "
                              "plane runs consumes the detector's boxes");
        }

        built.detect_slot = detect->slot;
        built.detect_model = detect->model;
        built.detect = detect_config(*detect);
        built.labels = plan.labels;
        built.stage_names.push_back(detect->slot);
        if (!croppers.empty()) built.stage_names.push_back("crop");
        for (const PlanNode* node : croppers) {
            built.crops.push_back(crop_spec_of(plan, *node));
            built.objects.push_back(
                {node->slot, node->model, crop_payload_of(node->slot), output_of(node->slot)});
            built.stage_names.push_back(node->slot);
        }
        // An `ObjectBatch` is keyed by a stage's OUTPUT name and not its own (`stages.cpp`:
        // `out.name = output_`), and looking one up by the stage name found nothing on every
        // frame. Derived here so both readers use one spelling.
        for (const auto& [field, slots] : plan.fields) {
            for (const std::string& slot : slots) {
                const PlanNode* target = plan.node(slot);
                if (target == nullptr) {
                    throw ConfigError("field '" + field + "' names slot '" + slot +
                                      "', which no `node` declares");
                }
                if (runnable(*target, loaded)) {
                    built.fields[field].push_back(output_of(slot));
                }
            }
        }
        return built;
    }

}  // namespace shipinfer
