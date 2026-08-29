# The perception event's compatibility rationale (v1 → v3)

The event keeps the old Kafka contract and extends it, because the downstream services
already exist: `motservice` consumes per-frame detections and `mtmcservice` consumes
tracklets (`references/bitbucket-subfaceid`). Replacing their input format would mean
rewriting both, so the rule from `docs/new-system-architecture.md` is *giữ contract cũ,
mở rộng schema cho ship* — keep the old contract, extend the schema for ships.

**v1** is `DetectionMOTFrameData` (`KafkaData/DetectionMOTFrameData.h`): one message type,
`Det2MOT`, as parallel arrays — `sub_id, det_id_vec, camera_id, image_id,
det_body_score_vec, body_bbox_vec, body_feature_vec, img_width, img_height, img_fps`.

**v2 adds ships, compatibly.** Every v1 key keeps its name, its type and its meaning, and
still carries **people only**, so a running `motservice` needs no change and no rebuild.
Ships get their own parallel arrays in the same idiom (`ship_bbox_vec`,
`ship_feature_vec`, `ship_id_vec`, …) — the existing consumer code's own style.
`schema_version` becomes explicit, so a consumer branches on the number instead of
guessing from a key's presence. Completeness becomes explicit (`partial`,
`missing_stages`): a frame that lost its embedder is not a frame with no people in it.

**v3 adds the tracklet.** With Plane 3 running in-process the identity is known when the
event is built, so it travels with the object — `body_track_id_vec` beside
`body_bbox_vec`, `ship_track_id_vec` beside `ship_bbox_vec`. Purely additive: `as_det2mot`
is untouched, so a deployed `motservice` that ignores the new keys keeps working, and one
that reads them can stop doing its own association.

The module itself (`src/shipinfer/core/events/schema.py`) is stdlib-only so a consumer may
copy it out wholesale; `TestTheSchemaIsPortable` enforces that.

## Which chain element fills which field

The `output` element (`topology/elements/output.py`) reads only `item.meta`, so this table is
its whole contract with the six elements ahead of it. A key nobody filed becomes a `None`
field — never a zero and never an omission, because a frame with no ships and a frame whose
embedder timed out have to be different events, which is what `missing_stages` is for.

| `meta` key | filed by | becomes |
|---|---|---|
| `detections` | `detect` (`elements/pool.py`) | one `ObjectRecord` each |
| `frame_hw` | `detect` | `img_width` / `img_height` |
| `fps` | the runner (`runners/frames.py`) | `img_fps` |
| `vectors` | `embed_*` (`elements/pool.py`) | `embedding` |
| `identities` | `recognize` (`elements/recognize.py`) | `ship_id` / `similarity` |
| `tracks` | `track` (`elements/track.py`) | `track_id` / `track_state` |
| `track_rows` | `track` | which detection row each track came from |
| `global_ids` | `mtmc` (`elements/mtmc.py`) | `global_id` |
| `missing_stages` | whichever stage had a gap | `missing_stages` |

The event's rows are the frame's *detections*: `det_id`, `bbox`, `score`, the embedding and
the identity fields are one row in the v1 payload `motservice` consumes, and every later
version added parallel arrays beside them rather than a second shape.
