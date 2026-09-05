#include "shipinfer/pipeline/events/records.h"

#include <algorithm>

namespace shipinfer::pipeline::events {

    namespace {

        // Which field a map entry fills, resolved ONCE per entry rather than per object.
        // `RECORD_CONVERTERS.get(name)` is hoisted out of the row loop on the Python plane
        // for the same reason: at ~15 000 objects a second, a string compare per object is a
        // cost with nothing to show for it.
        enum class Field { Embedding, ShipId, Similarity, MaskArea, TrackId };

        // `_as_float` on the Python plane is `row.reshape(-1)[0]` -- the FIRST element, not a
        // sum, and this matches it. What it does NOT do is reduce: whatever the segment slot
        // attached, `MaskArea` publishes its first float. That is wrong for a raw YOLO-seg
        // row and this plane attaches exactly that -- `ObjectStage::do_run` appends the
        // engine's rows with no combine, so `mask_area_px` here is `output0[crop][0][0]`, a
        // box coordinate. The Python plane folds (`InstanceMaskArea`, P6-SEGMENT-CROP) and
        // this plane has no port of it: `CSRC-SEGMENT-FOLD-MISSING` on the ledger. The record
        // parity gate cannot see it -- its scenarios state already-reduced `(N, 1)` rows.
        void set_field(ObjectRecord& record, Field field, const float* row, int width) {
            switch (field) {
                case Field::Embedding:
                    record.embedding.assign(row, row + width);
                    return;
                case Field::ShipId:
                    record.ship_id = static_cast<int64_t>(row[0]);
                    return;
                case Field::Similarity:
                    record.similarity = row[0];
                    return;
                case Field::MaskArea:
                    record.mask_area_px = row[0];
                    return;
                case Field::TrackId:
                    record.track_id = static_cast<int64_t>(row[0]);
                    return;
            }
        }

        // The field's own name back, for a refusal that has to name it. A `Field` is an
        // enum so the string is not on it, and a second table is cheaper than carrying the
        // name through every row.
        const char* field_name_of(Field field) {
            switch (field) {
                case Field::Embedding:
                    return "embedding";
                case Field::ShipId:
                    return "ship_id";
                case Field::Similarity:
                    return "similarity";
                case Field::MaskArea:
                    return "mask_area_px";
                case Field::TrackId:
                    return "track_id";
            }
            return "?";
        }

        // Named rather than skipped, and this is the same refusal the Python plane makes ("no
        // converter for ObjectRecord field"): a typo in a field map that silently filled
        // nothing would read as a stage that did not run.
        Field field_of(const std::string& name) {
            if (name == "embedding") return Field::Embedding;
            if (name == "ship_id") return Field::ShipId;
            if (name == "similarity") return Field::Similarity;
            if (name == "mask_area_px") return Field::MaskArea;
            if (name == "track_id") return Field::TrackId;
            throw ConfigError("no converter for ObjectRecord field '" + name +
                              "' on this plane; ObjectBatch carries floats, so a string "
                              "field such as track_state cannot travel in one here");
        }

    }  // namespace

    std::vector<ObjectRecord> build_records(const EmissionInputs& inputs,
                                            const ClassLabels& labels, const FieldMap& fields) {
        // Every entry resolved before a single row is touched, so a typo fails the call
        // rather than half-filling the frame it was given.
        std::vector<std::pair<Field, const std::vector<std::string>*>> resolved;
        resolved.reserve(fields.size());
        for (const auto& [name, candidates] : fields) {
            resolved.emplace_back(field_of(name), &candidates);
        }
        std::vector<ObjectRecord> records(inputs.detections.size());
        // By detection INDEX, because that is what a batch scatters against -- and sized from
        // the same capture, so a batch whose index runs past this list simply has nothing to
        // fill rather than reaching out of bounds.
        // A contested row is REFUSED, not picked. This is the decision the Python chain
        // plane already made and tested: `PoolEmbed._scatter` and `ChainWalk.inbound` both
        // raise when a second slot files a row an earlier one covered, and
        // `tests/runners/test_walk.py` states the reasoning -- "there is no answer to 'which
        // of these two vectors is this object's'. Silently keeping one would attach an
        // appearance vector chosen by declaration order."
        //
        // So the candidate list is a COVERAGE union, not a priority list: two slots covering
        // one detection means the chain file asked both of them for it, and both paid a GPU
        // for it. Per frame, and `cli/bench.cpp`'s sink counts it per camera -- which is why
        // this may throw at all (#129 round 2: nothing may throw past `seal()`).
        // `Topology.from_spec::_check_one_filler_per_row` refuses most such chains at LOAD;
        // what reaches here is a slot declaring `selects_rows = false` (`PoolSegment`, until
        // P6-SEGMENT-CROP) and a row contested at run time despite disjoint declarations.
        // Sized once, cleared per field. `assign` on the first iteration does the fill,
        // so the vector is constructed empty rather than filled twice.
        std::vector<bool> filled;
        for (const auto& [field, candidates] : resolved) {
            filled.assign(records.size(), false);
            for (const std::string& candidate : *candidates) {
                const auto found = inputs.batches.find(candidate);
                if (found == inputs.batches.end() || found->second.empty()) continue;
                const ObjectBatch& batch = found->second;
                for (size_t row = 0; row < batch.rows(); ++row) {
                    const int index = batch.object_indices[row];
                    if (index < 0 || static_cast<size_t>(index) >= records.size()) continue;
                    const size_t at = static_cast<size_t>(index);
                    if (filled[at]) {
                        throw InferenceError(
                            "two batches cover detection row " + std::to_string(index) +
                            " for the event's '" + field_name_of(field) + "': '" + candidate +
                            "' and an earlier candidate. Batches sharing a field merge their "
                            "coverage rather than one replacing the other, and two of them "
                            "covering one detection means the chain asked both for it -- "
                            "check their `params: classes:` do not overlap");
                    }
                    set_field(records[at], field, batch.row(row), batch.width);
                    filled[at] = true;
                }
            }
        }

        for (size_t i = 0; i < inputs.detections.size(); ++i) {
            const Detection& detection = inputs.detections[i];
            ObjectRecord& record = records[i];
            // `<camera>_<frame>_<index>`: unique across the fleet and derivable by a consumer,
            // so a tracker can key on it without a side channel.
            record.det_id = inputs.tag.camera_id + "_" + std::to_string(inputs.tag.frame_id) +
                            "_" + std::to_string(detection.index);
            // A class id past the configured labels is NAMED rather than silently relabelled:
            // a checkpoint that grew a class must not publish `person` for it.
            const auto label = labels.find(detection.class_id);
            record.class_name = label == labels.end() ? kUnknownLabel : label->second;
            record.score = detection.score;
            record.bbox[0] = detection.x1;
            record.bbox[1] = detection.y1;
            record.bbox[2] = detection.x2;
            record.bbox[3] = detection.y2;
        }
        return records;
    }

    PerceptionEvent event_of(const EmissionInputs& inputs, FinishReason reason,
                             const std::vector<std::string>& missing,
                             const std::string& source_id, const ClassLabels& labels,
                             const FieldMap& fields) {
        return shipinfer::events::build(
            inputs.tag.camera_id, static_cast<int64_t>(inputs.tag.frame_id), source_id,
            build_records(inputs, labels, fields), inputs.width, inputs.height, inputs.fps,
            inputs.tag.captured_ns, inputs.tag.captured_unix_ns, missing,
            // `to_string`, not a second vocabulary: the wire word IS the collector's word.
            // An earlier version mapped Incomplete and Evicted onto `failed`, following
            // `core/events/schema.py`'s docstring -- which does not describe what that plane
            // emits. `pipeline/runner.py:622` passes `result.reason` through verbatim, so a
            // Python shard publishes `evicted` where this one published `failed`, and a
            // consumer alarming on ADR-005's eviction signal saw nothing from a C++ shard.
            to_string(reason));
    }

}  // namespace shipinfer::pipeline::events
