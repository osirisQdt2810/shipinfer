#include "shipinfer/pipeline/graph/plan_stages.h"

#include "shipinfer/pipeline/mtmc/cluster.h"
#include "shipinfer/pipeline/tracking/associator.h"

namespace shipinfer {

    namespace {

        // Which event field a kind fills is stated by the plan (`field` lines), so this file
        // needs no copy of that rule. What it does need is which kinds this plane runs.
        bool crops(const std::string& kind) {
            return kind == "embed" || kind == "segment";
        }

        // A stage this plane implements IN TREE rather than through an engine, so `runnable`
        // below cannot ask it for a loaded model -- a tracker has none.
        bool in_tree(const std::string& kind) {
            return kind == "track" || kind == "mtmc";
        }

        bool runnable(const PlanNode& node, const std::set<std::string>& loaded) {
            if (!plane_runs(node.kind)) return false;
            // THE REGISTRY, not `true`. A tracker lives in an external lane, and a binary built
            // without that lane has no registrar -- so answering "runnable" there would put
            // the slot in `stage_names`, claim it in the run's own note, and then throw when
            // the Dag asked for it. Asking the registry makes a lane-less build report
            // `track` in `unsupported` instead, which is what it did before there was a
            // tracker at all and is the only coherent answer.
            if (node.kind == "mtmc") return mtmc::CLUSTERERS().has(node.impl);
            if (in_tree(node.kind)) return tracking::ASSOCIATORS().has(node.impl);
            return !node.model.empty() && loaded.count(node.model) != 0;
        }

        int class_of(const ResolvedPlan& plan, const PlanNode& node) {
            // Three states, not two. No `classes` line is EVERY row, which is what a crop
            // element with no `classes:` means on the Python plane. A declared EMPTY
            // selection is no rows.
            if (!node.classes) {
                // Every row, for a segment slot as for an embed one. This REFUSED a segment
                // slot with no `classes:` between #132 and here, on the argument that "every
                // row" was a 640x640 crop per person that nobody chose -- true, and equally
                // true of an embedder, which has always been allowed to say it. The reason
                // the two differed was that `PoolSegment` did not crop at all on the Python
                // plane, so a plan with no selection meant different work on each side. Since
                // P6-SEGMENT-CROP it means the same work on both, so one chain file gets one
                // answer and the cost is the chain author's to choose
                // (`SEGMENT-NO-CLASSES-ASYMMETRY`).
                return CropSpec::kAnyClass;
            }
            if (node.classes->empty()) return CropSpec::kNoClass;
            if (node.classes->size() > 1) {
                throw ConfigError("slot '" + node.slot + "' selects " +
                                  std::to_string(node.classes->size()) +
                                  " classes and a CropSpec carries one; split the slot");
            }
            const std::optional<int> id = plan.class_id(node.classes->front());
            if (!id) {
                throw ConfigError("slot '" + node.slot + "' selects class '" +
                                  node.classes->front() +
                                  "' and the plan's label table does not name it. A label "
                                  "nobody detects selects no rows and reports nothing wrong");
            }
            return *id;
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
            if (node.max_detections) {
                // Refused, not taken: `stages.cpp` compares `detections.size() <
                // static_cast<size_t>(max_objects)`, so a declared -1 becomes
                // 18446744073709551615
                // -- no bound at all -- while the Python plane's `keep[: -1]` drops one row.
                // One plan, two detection sets, and the unbounded one is on the plane whose
                // numbers are the deployment's. Same sign class as `kNoClass` last round: a
                // negative int slipping through a comparison never written for it.
                if (*node.max_detections <= 0) {
                    throw ConfigError("slot '" + node.slot + "' declares max_detections " +
                                      std::to_string(*node.max_detections) +
                                      "; a cap is a positive count on both planes, and `-1` "
                                      "for `no limit` is not a spelling either one has");
                }
                config.max_objects = *node.max_detections;
            }
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

    ResolvedPlan default_bench_plan() {
        ResolvedPlan plan;
        plan.version = kPlanVersion;
        plan.name = "bench-default";
        // `core/settings/`'s own defaults, so the no-`--plan` run and a plan written from an
        // unconfigured deployment are the SAME configuration. A different number here would be
        // a fifth set, which is what P5-C exists to stop: `pipeline.workers`,
        // `pipeline.queue_capacity`, `scheduler.max_queue_size`,
        // `scheduler.enqueue_block_timeout_ms`, `pipeline.stage_timeout_ms` and the three
        // `pipeline.reassembly` fields, in that order.
        plan.settings = PlanSettings{4, 256, 64, 50, 5000, 1024, 1500, 100};
        // `SchedulerSettings.placement_policy`'s own default, for the same reason: the
        // no-`--plan` run and a plan from an unconfigured deployment are one configuration.
        plan.policy = "locality_spillover";
        plan.labels = {{0, "person"}, {8, "ship"}};
        PlanNode detect;
        detect.slot = "detect";
        detect.kind = "detect";
        detect.impl = "pool";
        detect.model = "ship_detector";
        detect.letterbox = Extent{640, 640};
        plan.nodes.push_back(detect);
        const struct {
            const char* slot;
            const char* kind;
            const char* label;
            Extent crop;
        } croppers[] = {
            {"ship_segmenter", "segment", "ship", Extent{640, 640}},
            {"person_embedder", "embed", "person", Extent{256, 128}},
            {"ship_embedder", "embed", "ship", Extent{256, 128}},
        };
        for (const auto& spec : croppers) {
            PlanNode node;
            node.slot = spec.slot;
            node.kind = spec.kind;
            node.impl = "pool";
            node.model = spec.slot;
            node.classes = std::vector<std::string>{spec.label};
            node.crop = spec.crop;
            plan.nodes.push_back(node);
            plan.fields[spec.kind == std::string("segment") ? "mask_area_px" : "embedding"]
                .push_back(spec.slot);
        }
        return plan;
    }

    bool plane_runs(const std::string& kind) {
        return kind == "detect" || crops(kind) || in_tree(kind);
    }

    bool plane_runs_a_model(const std::string& kind) {
        return plane_runs(kind) && !in_tree(kind);
    }

    std::string crop_payload_of(const std::string& slot) {
        return slot + "_crops";
    }

    std::string output_of(const std::string& slot) {
        return slot + "_out";
    }

    std::optional<MaskAreaSpec> fold_of(const ResolvedPlan& plan, const PlanNode& node) {
        // A SEGMENT slot's engine answers detection rows and a prototype bank, never a mask,
        // so something has to fold the two into one area per crop -- `PoolSegment._reduced` on
        // the Python plane. An embedder's answers one vector per crop already, so it has no
        // fold and its response is scattered as it is.
        if (node.kind != "segment") return std::nullopt;
        const CropSpec crop = crop_spec_of(plan, node);
        MaskAreaSpec fold;
        fold.crop_height = crop.height;
        fold.crop_width = crop.width;
        // The plan's cuts where it states them. The defaults on `MaskAreaSpec` agree with the
        // Python fold's, which is exactly why an absent line was silent: both planes were
        // right by luck until a chain file said otherwise.
        if (!node.fold_detections.empty()) fold.detections = node.fold_detections;
        if (!node.fold_prototypes.empty()) fold.prototypes = node.fold_prototypes;
        if (node.fold_score) fold.score_threshold = static_cast<float>(*node.fold_score);
        if (node.fold_mask) fold.mask_threshold = static_cast<float>(*node.fold_mask);
        return fold;
    }

    PlanStages plan_stages(const ResolvedPlan& plan, const std::set<std::string>& loaded) {
        PlanStages built;
        const PlanNode* detect = nullptr;
        std::vector<const PlanNode*> trackers;
        std::vector<const PlanNode*> groups;
        std::vector<const PlanNode*> croppers;
        for (const PlanNode& node : plan.nodes) {
            if (!runnable(node, loaded)) {
                built.unsupported.push_back(node.slot);
            } else if (node.kind == "mtmc") {
                // REFUSED, not last-wins and not several -- and the REASON is narrower than
                // it was. The chain DOES state membership (`group:` and `cameras:`, which the
                // plan now carries), so what is missing is the ROUTING: this plane's stage
                // hands its camera's rows to the one barrier it was built with, and nothing
                // decides which of two barriers a frame belongs to. Until a stage picks its
                // group by roster, two slots would both be handed every camera this shard
                // sees and would issue two contradictory sets of global ids for the same
                // objects -- worse than refusing to start (`CSRC-MTMC-TWO-GROUPS-PER-SHARD`).
                if (!groups.empty()) {
                    throw ConfigError(
                        "plan '" + plan.name + "' has two runnable mtmc slots ('" +
                        groups.front()->slot + "' and '" + node.slot +
                        "'); the chain states each group's roster but this plane does not yet "
                        "route a camera to its group, so both slots would take every camera "
                        "this shard sees and issue contradictory global ids");
                }
                groups.push_back(&node);
            } else if (node.kind == "track") {
                // NOT refused for being a second one. Two trackers over one camera's rows is
                // refused at LOAD by `chain.py::_check_one_filler_per_row` when their
                // selections overlap, and permitted when they are disjoint -- so refusing here
                // would throw on a chain the Python plane runs.
                trackers.push_back(&node);
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
            const CropSpec crop = crop_spec_of(plan, *node);
            built.crops.push_back(crop);
            ObjectStageSpec object;
            object.slot = node->slot;
            object.model = node->model;
            object.source = crop_payload_of(node->slot);
            object.output = output_of(node->slot);
            // A SEGMENT slot's engine answers detection rows and a prototype bank, never a
            // mask, so its stage folds the two into one area per crop before scattering --
            // `PoolSegment._reduced` on the Python plane. An embedder's answers one vector per
            // crop already, so it has no fold and its response is scattered as it is.
            object.fold = fold_of(plan, *node);
            built.objects.push_back(std::move(object));
            built.stage_names.push_back(node->slot);
        }
        // AFTER the croppers, because a tracker consumes what the detector produced and its
        // ids are scattered onto the same rows the embedders' vectors are -- the chain says so
        // too (`after: [embed_ship, embed_person]`).
        for (const PlanNode* node : trackers) {
            TrackStageSpec spec;
            spec.slot = node->slot;
            spec.output = output_of(node->slot);
            spec.impl = node->impl;
            // `class_of`, the same reader the croppers use: no `classes:` is every row, a
            // declared empty selection is no rows, and a named one is that class's id.
            spec.class_id = class_of(plan, *node);
            built.tracks.push_back(std::move(spec));
            built.stage_names.push_back(node->slot);
        }
        // AFTER the trackers, because cross-camera identity consumes their ids -- the chain
        // says so too (`after: [track]`) -- and after the embedders because it is decided on
        // appearance. `stage_names` is the order the Dag runs them in.
        for (const PlanNode* node : groups) {
            MtmcStageSpec spec;
            spec.slot = node->slot;
            spec.output = output_of(node->slot);
            spec.impl = node->impl;
            if (built.tracks.empty()) {
                throw ConfigError(
                    "plan '" + plan.name + "' runs mtmc slot '" + node->slot +
                    "' with no runnable tracker. Cross-camera identity is keyed by (camera, "
                    "track), so without one there is nothing for an identity to hold on to "
                    "and every row would be published unidentified");
            }
            // THE FIRST tracker's output. A chain with two disjoint trackers is legal, and
            // choosing between them here would be a silent policy -- so it is the first in
            // plan order, stated rather than derived, and `MTMC-READS-ONE-TRACKER` holds the
            // question of what a two-tracker chain should feed a group.
            spec.track_source = built.tracks.front().output;
            spec.sync_window_ms = node->sync_window_ms;
            spec.max_instants = node->max_instants;
            // THE ROSTER, so the barrier waits for the cameras the CHAIN names rather than
            // for every camera this shard happens to see.
            spec.group = node->group.empty() ? node->slot : node->group;
            spec.cameras = node->cameras;
            spec.gate.min_hits = node->min_hits;
            spec.gate.min_height_fraction = node->min_height_fraction;
            for (const ObjectStageSpec& object : built.objects) {
                if (object.fold) continue;  // a segmenter's fold is an area, not an embedding
                spec.embedding_sources.push_back(object.output);
            }
            built.mtmcs.push_back(std::move(spec));
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
