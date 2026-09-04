// A finished frame's stage outputs, as the event's per-object records.
//
// UNDER `pipeline/`, not `core/`, and that placement is the rule rather than taste: this
// needs `Detection` and `ObjectBatch`, so a header in `core/` would have to include
// `pipeline/` and `core/` imports nothing but `core/` (ADR-001). The Python plane drew the
// same line -- `build_records` lives in `pipeline/graph/state.py` while
// `core/events/convert.py` holds only `as_embedding` and imports numpy alone.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "shipinfer/core/events/schema.h"
#include "shipinfer/pipeline/graph/emission.h"

namespace shipinfer::pipeline::events {

    using shipinfer::events::ObjectRecord;
    using shipinfer::events::PerceptionEvent;

    // Which per-object batch fills which `ObjectRecord` field -- `DEFAULT_RECORD_FIELDS` in
    // `pipeline/graph/graph.py`. A table, not a chain of conditionals: adding a field is an
    // entry here and a field there.
    //
    // Two candidates per field is normal, and it is what removes the class check: a ship's
    // embedding comes from the ship embedder and a person's from the person embedder. An
    // earlier version picked the embedder with `class_name == "ship"` -- a second spelling of
    // "which class is this row", which is exactly how a ship's embedding ends up on a person.
    //
    // They CAN collide, and this comment said they could not. A resolved chain plan may carry
    // a crop slot with no `classes:`, which is every row, so two candidates can cover one
    // detection. This list is therefore a COVERAGE UNION and not a priority list: a row two
    // of them cover is REFUSED by `build_records`, which is the decision
    // `PoolEmbed._scatter` and `ChainWalk.inbound` already made on the other plane. Nothing
    // refuses such a chain at LOAD yet -- `RECORDS-COLLISION-AT-LOAD` says what that costs.
    using FieldMap = std::map<std::string, std::vector<std::string>>;

    //: The labels a class id maps to (`pipeline.class_labels`). Passed in, never hard-coded:
    //: a raw class index is a property of the checkpoint and changes when it is retrained.
    //:
    //: A MAP and not a positional vector, because the Python side is a `Mapping[int, str]`
    //: and is sparse -- `{0: "person", 7: "ship"}` is an ordinary configuration, and a vector
    //: would need six filler entries to express it. An id with no label becomes
    //: `UNKNOWN_LABEL`, which is `"unknown"` there (`topology/elements/detections.py`).
    using ClassLabels = std::map<int, std::string>;

    //: What an unmapped class id is called. The same word the Python plane uses: a checkpoint
    //: that grew a class must not publish `person` for it, and must not invent a name either.
    inline const char* const kUnknownLabel = "unknown";

    // One record per detection, in detection order -- which is the order every `*_vec` is
    // indexed by, so a consumer joining `body_bbox_vec[i]` to `body_track_id_vec[i]` gets the
    // object it asked for. A field left unset means the stage that fills it did not run,
    // which is what the event should say and is distinguishable from a zero.
    std::vector<ObjectRecord> build_records(const EmissionInputs& inputs,
                                            const ClassLabels& labels, const FieldMap& fields);

    // The whole event. `source_id` is which perception process produced it -- v1's `sub_id`.
    // Takes the three pieces of a `FrameResult` it needs rather than the result itself: the
    // collector's header reaches `core/device.h`, and depending on it would put this back
    // out of reach of every CUDA-free binary -- which is how its first version shipped with
    // no test.
    PerceptionEvent event_of(const EmissionInputs& inputs, FinishReason reason,
                             const std::vector<std::string>& missing,
                             const std::string& source_id, const ClassLabels& labels,
                             const FieldMap& fields);

}  // namespace shipinfer::pipeline::events
