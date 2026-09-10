#include "shipinfer/pipeline/tracking/stage.h"

#include "shipinfer/pipeline/graph/emission.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/tracking/registry.h"

namespace shipinfer::tracking {

    TrackStage::TrackStage(std::string name, std::string output, TrackerShard& shard)
        // CONSUMES the detections and NEEDS them: the tracker has nothing to associate on a
        // frame the detector produced no boxes for, and `needs` is what keeps the stage out of
        // the plan for that frame rather than running it on an empty vector.
        : Stage(std::move(name), {DETECTIONS}, {DETECTIONS}, {output}),
          output_(std::move(output)),
          shard_(shard) {}

    size_t TrackStage::do_run(FrameState& state) {
        const std::vector<Detection>& detections = state.detections();
        ObjectBatch batch;
        batch.name = output_;
        batch.width = 1;
        if (!detections.empty()) {
            const TrackUpdate update =
                shard_.update(state.tag().camera_id, state.tag().frame_id, detections);
            for (size_t row = 0; row < update.ids.size(); ++row) {
                // A row per CONFIRMED id only. `-1` is "matched no track", and an absent row is
                // how `events/records.cpp` leaves `track_id` null for that object.
                if (update.ids[row] < 0) continue;
                batch.object_indices.push_back(detections[row].index);
                // FLOAT, because `ObjectBatch::data` is one: exact for ids below 2^24, and a
                // monotonic per-camera counter reaches that after ~16.7M tracks. The Python
                // plane carries the same ids as `int64`, so the two planes differ THERE and the
                // ledger holds the item; nothing here can widen the carrier alone.
                batch.data.push_back(static_cast<float>(update.ids[row]));
            }
        }
        const size_t rows = batch.rows();
        // Attached even when EMPTY, like a crop payload: the name exists, so a reader that
        // joins on it finds an answer rather than a missing key.
        state.attach(std::move(batch));
        return rows;
    }

    TrackerShard& process_shard() {
        static TrackerShard shard;
        return shard;
    }

    namespace {

        // `impl: shipvision` in the chain, which is the name the plan carries -- the algorithm
        // inside it (`params: algorithm: bytetrack`) is the only one this lane has, so the plan
        // writer does not need to emit it and this does not need to read it.
        const TrackerRegistrar kShipvision("shipvision", [](const TrackStageSpec& spec) {
            return std::unique_ptr<Stage>(
                new TrackStage(spec.slot, spec.output, process_shard()));
        });

    }  // namespace

}  // namespace shipinfer::tracking
