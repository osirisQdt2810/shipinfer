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
        // sum. A `MaskArea` stage already reduces its mask to `(N, 1)`, so the two agree
        // today; summing would diverge silently the moment a segmenter attached a raw row.
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
        // FIRST candidate wins, matching `build_records`'s own "in priority order" -- which
        // neither plane did: both overwrote, so the LAST batch to mention a row set the
        // field. Two batches covering one row is meant to be impossible (a batch holds the
        // rows of its own class), and the resolved chain plan can now make it possible, so
        // `Topology.from_spec` refuses such a chain at load. This is the rule for whatever
        // gets past it: deterministic, and the same on both planes.
        std::vector<std::vector<bool>> filled(resolved.size(),
                                              std::vector<bool>(records.size(), false));
        for (size_t slot = 0; slot < resolved.size(); ++slot) {
            const auto& [field, candidates] = resolved[slot];
            for (const std::string& candidate : *candidates) {
                const auto found = inputs.batches.find(candidate);
                if (found == inputs.batches.end() || found->second.empty()) continue;
                const ObjectBatch& batch = found->second;
                for (size_t row = 0; row < batch.rows(); ++row) {
                    const int index = batch.object_indices[row];
                    if (index < 0 || static_cast<size_t>(index) >= records.size()) continue;
                    if (filled[slot][static_cast<size_t>(index)]) continue;
                    set_field(records[static_cast<size_t>(index)], field, batch.row(row),
                              batch.width);
                    filled[slot][static_cast<size_t>(index)] = true;
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
