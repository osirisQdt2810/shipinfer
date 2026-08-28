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
