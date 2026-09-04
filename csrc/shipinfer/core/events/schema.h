// The perception event — `src/shipinfer/core/events/schema.py`, key for key.
//
// Schema v4: per-object parallel arrays split by class, plus frame identity, geometry and
// `missing_stages`, so a partial frame says so instead of reading as an empty complete one.
// Why every v1 key keeps its name, type and people-only meaning (a deployed `motservice`
// must need no rebuild): `docs/design/event-schema.md`.
//
// Before this existed the C++ plane emitted NOTHING: `cli/bench.cpp`'s collector sink took a
// `FrameResult&&`, counted it and dropped it, and the only JSON in the tree was the
// occupancy log. So "same events out" was a writer that did not exist rather than a port of
// one, and the sync rule had nothing to compare.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace shipinfer::events {

    //: The `type` field every consumer switches on. Unchanged from v1 on purpose.
    inline const char* const kMessageType = "Det2MOT";
    //: 1 was `DetectionMOTFrameData`; 2 adds ships, timing and completeness; 3 the track id
    //: and its state; 4 the cross-camera `global_id`. Additive, and bumped so a consumer can
    //: branch on it instead of probing for a key.
    constexpr int kSchemaVersion = 4;

    // One detected object and everything the DAG learned about it. An optional is empty when
    // the stage that fills it did not run -- a person has no `ship_id` and never will, and a
    // ship whose recogniser timed out has none *yet*. Both are distinguishable from 0, which
    // is a legitimate gallery id.
    struct ObjectRecord {
        //: `<camera>_<frame>_<index>`: unique across the fleet and derivable by a consumer.
        std::string det_id;
        //: The label from `pipeline.class_labels`, never a raw class index -- the index is a
        //: property of the checkpoint and changes when the model is retrained.
        std::string class_name;
        double score = 0.0;
        //: `(x1, y1, x2, y2)` in the SOURCE frame's pixels, letterbox already undone.
        double bbox[4] = {0.0, 0.0, 0.0, 0.0};
        //: Appearance embedding; empty when the embedder did not run for this object.
        std::vector<double> embedding;
        std::optional<int64_t> ship_id;
        std::optional<double> similarity;
        //: Mask AREA in pixels. The mask itself is deliberately not published: a 512x512
        //: float mask is 1 MB and this bus carries metadata, not pixels.
        std::optional<double> mask_area_px;
        //: Single-camera track identity from Plane 3, process-unique.
        std::optional<int64_t> track_id;
        std::optional<std::string> track_state;
        //: Fleet-wide identity for this object's tracklet (v4). The same object seen by two
        //: cameras carries one `global_id` and two `track_id`s -- that is the distinction.
        std::optional<int64_t> global_id;
    };

    // One frame's perception result, ready to serialise.
    struct PerceptionEvent {
        std::string camera_id;
        //: The frame id from the ingest tag. Named `image_id` on the wire, as in v1.
        int64_t frame_id = 0;
        //: Which perception process produced this -- v1's `sub_id`.
        std::string source_id;
        std::vector<ObjectRecord> objects;
        int64_t img_width = 0;
        int64_t img_height = 0;
        //: ROUNDED, because the Python plane rounds it: `img_fps=round(fps)`.
        int64_t img_fps = 0;
        int64_t captured_unix_ns = 0;
        int64_t emitted_unix_ns = 0;
        //: Capture to emission, from the MONOTONIC clock -- carried rather than derived from
        //: the two wall-clock stamps so an NTP step cannot become a negative latency.
        int64_t latency_us = 0;
        std::vector<std::string> missing_stages;
        //: `complete`, `timeout`, `failed` or `shutdown`.
        std::string reason = "complete";
        //: Free-form additions a deployment needs and the schema should not grow a field for.
        //:
        //: A VECTOR of pairs, not a map, and for the reason `to_json` gives about every other
        //: key: a map sorts, and Python's `extra` is an insertion-ordered `dict`, so two
        //: entries would come out in one order there and another here. Values are strings
        //: here where Python's are `Any` -- a single-key string extra (the only one anything
        //: sets today: `deepstream/probe.py`) matches exactly, and anything richer does not
        //: yet. Stated rather than discovered.
        std::vector<std::pair<std::string, std::string>> extra;

        bool is_partial() const { return !missing_stages.empty(); }
        std::vector<const ObjectRecord*> objects_of(const std::string& class_name) const;

        // One line of JSON, byte-identical to `PerceptionEvent.to_json()` on the same inputs.
        // Byte-identical and not merely equivalent: the parity gate is a string compare,
        // because this plane writes JSON and never parses it.
        std::string to_json() const;
    };

    // `PerceptionEvent.build`: the one constructor. Fills `emitted_unix_ns` from the wall
    // clock and `latency_us` from the monotonic pair, exactly as the Python classmethod does.
    PerceptionEvent build(const std::string& camera_id, int64_t frame_id,
                          const std::string& source_id, std::vector<ObjectRecord> objects,
                          int64_t width, int64_t height, double fps, int64_t captured_ns,
                          int64_t captured_unix_ns,
                          const std::vector<std::string>& missing_stages,
                          const std::string& reason, int64_t now_ns = 0,
                          int64_t now_unix_ns = 0);

}  // namespace shipinfer::events
