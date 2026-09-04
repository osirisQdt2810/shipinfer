// `pipeline/events/records.cpp`, the one translation unit that runs in production.
//
// It had NO test, which is why its first version was broken in a way nothing could see: it
// looked `FrameResult::inputs.batches` up by STAGE name while an `ObjectBatch` is keyed by a
// stage's OUTPUT name (`graph/stages.cpp`: `out.name = output_`), so every embedding the
// pipeline exists to compute was dropped -- and dropped as `[]`, which is indistinguishable
// from "the embedder did not run". Its only caller was `cli/bench.cpp`, which needs TensorRT
// plans, so the gate the whole feature was built around could not reach it.
//
// It is testable at all because `ObjectBatch`, `EmissionInputs` and `FinishReason` were split
// out of `graph/state.h` into `graph/emission.h` -- host memory and a plain enum, where
// `DevicePayload` and `FrameState` next door need `core/device.h` and therefore CUDA.

#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/events/schema.h"
#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/emission.h"

namespace {

    using namespace shipinfer;
    using namespace shipinfer::pipeline::events;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // Two detections -- one person at index 0, one ship at index 1 -- and whatever batches a
    // case wants attached to them. An `EmissionInputs` and not a `FrameResult`: the collector's
    // header reaches `core/device.h`, and this is the CUDA-free half it was split from.
    EmissionInputs a_frame() {
        EmissionInputs inputs;
        inputs.tag.camera_id = "cam0";
        inputs.tag.frame_id = 7;
        inputs.width = 1920;
        inputs.height = 1080;
        inputs.fps = 20.f;
        Detection person;
        person.x1 = 10;
        person.y1 = 20;
        person.x2 = 30;
        person.y2 = 40;
        person.score = 0.5f;
        person.class_id = 0;
        person.index = 0;
        Detection ship;
        ship.x1 = 1;
        ship.y1 = 2;
        ship.x2 = 3;
        ship.y2 = 4;
        ship.score = 0.75f;
        ship.class_id = 1;
        ship.index = 1;
        inputs.detections = {person, ship};
        return inputs;
    }

    ObjectBatch a_batch(const std::string& name, std::vector<int> indices,
                        std::vector<float> data, int width) {
        ObjectBatch batch;
        batch.name = name;
        batch.object_indices = std::move(indices);
        batch.data = std::move(data);
        batch.width = width;
        return batch;
    }

    // Sparse on purpose: `{0: "person", 7: "ship"}` is what a real config looks like,
    // and a positional vector would need six fillers to say it.
    const ClassLabels kLabels = {{0, "person"}, {1, "ship"}};

    // -- the defect this file exists for ------------------------------------------------

    void test_a_batch_is_found_by_its_output_name() {
        EmissionInputs result = a_frame();
        result.batches["person_embedder_out"] =
            a_batch("person_embedder_out", {0}, {0.25f, 0.5f}, 2);
        const FieldMap fields = {{"embedding", {"ship_embedder_out", "person_embedder_out"}}};

        const std::vector<ObjectRecord> records = build_records(result, kLabels, fields);

        check(records.size() == 2, "one record per detection");
        check(records[0].embedding.size() == 2,
              "the person's embedding arrived; got " +
                  std::to_string(records[0].embedding.size()) +
                  " floats, which is the batch-name defect if it is 0");
        check(records[1].embedding.empty(),
              "the ship has none, because no ship batch was attached -- and empty means "
              "'the stage did not run', which is what the event should say");
    }

    void test_the_class_is_never_used_to_pick_a_batch() {
        // The two candidates cannot collide because a batch holds rows of its own class only,
        // which is what lets the field map replace a `class_name == "ship"` test -- the second
        // spelling of "which class is this row" that puts a ship's embedding on a person.
        EmissionInputs result = a_frame();
        result.batches["ship_embedder_out"] =
            a_batch("ship_embedder_out", {1}, {1.0f, 2.0f, 3.0f}, 3);
        result.batches["person_embedder_out"] =
            a_batch("person_embedder_out", {0}, {0.25f, 0.5f}, 2);
        const FieldMap fields = {{"embedding", {"ship_embedder_out", "person_embedder_out"}}};

        const std::vector<ObjectRecord> records = build_records(result, kLabels, fields);

        check(records[0].embedding.size() == 2, "the person got the person embedder's row");
        check(records[1].embedding.size() == 3, "the ship got the ship embedder's row");
    }

    void test_a_scalar_field_takes_the_rows_first_element() {
        // `_as_float` on the Python plane is `row.reshape(-1)[0]`, NOT a sum. An earlier
        // version summed the row and said in a comment that Python did too; they agree only
        // because `MaskArea` already reduces to `(N, 1)`.
        EmissionInputs result = a_frame();
        result.batches["ship_segmenter_out"] =
            a_batch("ship_segmenter_out", {1}, {100.f, 7.f}, 2);
        const FieldMap fields = {{"mask_area_px", {"ship_segmenter_out"}}};

        const std::vector<ObjectRecord> records = build_records(result, kLabels, fields);

        check(records[1].mask_area_px.has_value() && *records[1].mask_area_px == 100.0,
              "mask_area_px is the row's FIRST element, not its sum (107 would be the sum)");
        check(!records[0].mask_area_px.has_value(), "the person has no mask area");
    }

    void test_an_unfillable_field_is_refused_by_name() {
        std::string message;
        try {
            (void)build_records(a_frame(), kLabels, {{"track_state", {"tracker_out"}}});
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("track_state") != std::string::npos,
              "a field this plane cannot fill is named, not silently skipped: " + message);
    }

    void test_an_unmapped_class_id_is_unknown_not_relabelled() {
        EmissionInputs result = a_frame();
        result.detections[1].class_id = 9;

        const std::vector<ObjectRecord> records = build_records(result, kLabels, {});

        check(records[1].class_name == "unknown",
              "an unmapped class id is Python's UNKNOWN_LABEL, not an invented name and not "
              "`person`; got " +
                  records[1].class_name);
    }

    void test_a_batch_index_past_the_detections_is_dropped_not_dereferenced() {
        EmissionInputs result = a_frame();
        result.batches["person_embedder_out"] =
            a_batch("person_embedder_out", {0, 42}, {1.f, 2.f}, 1);
        const FieldMap fields = {{"embedding", {"person_embedder_out"}}};

        const std::vector<ObjectRecord> records = build_records(result, kLabels, fields);

        check(records.size() == 2 && records[0].embedding.size() == 1,
              "a row scattered against a shorter list has nothing to fill, and does not "
              "reach out of bounds");
    }

    void test_the_det_id_and_the_geometry_come_from_the_tag() {
        const std::vector<ObjectRecord> records = build_records(a_frame(), kLabels, {});

        check(records[0].det_id == "cam0_7_0" && records[1].det_id == "cam0_7_1",
              "`<camera>_<frame>_<index>`, derivable by a consumer");
        check(records[0].class_name == "person" && records[1].class_name == "ship",
              "the class id maps through the labels it was given");
        check(records[1].bbox[2] == 3.0 && records[1].score == 0.75,
              "the box and the score travel unchanged");
    }

    void test_every_finish_reason_has_a_wire_word() {
        // The COLLECTOR's five words, not a second vocabulary. Python's
        // `pipeline/runner.py` passes `result.reason` through verbatim, so these are the
        // strings that reach the wire there -- and an earlier version of this plane wrote
        // `failed` for two of them, so an operator alarming on ADR-005's eviction signal saw
        // nothing from a C++ shard. `core/events/schema.py`'s docstring said `failed`; it was
        // describing a plane that does not exist and is corrected in this branch.
        const std::vector<std::pair<FinishReason, std::string>> words = {
            {FinishReason::Complete, "complete"}, {FinishReason::Incomplete, "incomplete"},
            {FinishReason::Timeout, "timeout"},   {FinishReason::Shutdown, "shutdown"},
            {FinishReason::Evicted, "evicted"},
        };
        for (const auto& [reason, want] : words) {
            check(std::string(to_string(reason)) == want, "FinishReason -> '" +
                                                              std::string(to_string(reason)) +
                                                              "', want '" + want + "'");
        }
    }

    void test_the_whole_event_carries_the_frames_geometry_and_reason() {
        EmissionInputs result = a_frame();

        result.batches["person_embedder_out"] =
            a_batch("person_embedder_out", {0}, {0.25f, 0.5f}, 2);

        const PerceptionEvent event =
            event_of(result, FinishReason::Timeout, {"ship_segmenter"}, "shard-1", kLabels,
                     {{"embedding", {"person_embedder_out"}}});
        const std::string line = event.to_json();

        check(event.img_fps == 20, "fps 20 reaches the event");
        check(event.reason == "timeout" && event.is_partial(),
              "a timed-out frame says so, and is partial");
        check(line.find("\"body_feature_vec\":[[0.25,0.5]]") != std::string::npos,
              "the embedding reaches the JSON, which is what #1 broke: " + line.substr(0, 200));
        check(line.find("\"camera_id\":\"cam0\"") != std::string::npos, "the tag survives");
    }

}  // namespace

int main() {
    try {
        test_a_batch_is_found_by_its_output_name();
        test_the_class_is_never_used_to_pick_a_batch();
        test_a_scalar_field_takes_the_rows_first_element();
        test_an_unfillable_field_is_refused_by_name();
        test_an_unmapped_class_id_is_unknown_not_relabelled();
        test_a_batch_index_past_the_detections_is_dropped_not_dereferenced();
        test_the_det_id_and_the_geometry_come_from_the_tag();
        test_every_finish_reason_has_a_wire_word();
        test_the_whole_event_carries_the_frames_geometry_and_reason();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: the record builder's tests could not run: %s\n",
                     error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
