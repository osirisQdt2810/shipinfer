# Feature Log

One entry per large feature or seam change. Append-only, newest on top. Skip it for tiny
edits, typo fixes and pure docs.

---

## 2026-09-05 — The segmenter crops (P6-SEGMENT-CROP)

`PoolSegment` letterboxed the whole frame; the C++ plane has always cut a `ship_crops_640`
set, so `mask_area_px` came from different pixels on each. It extends `_PoolCropElement` now.
What kept it waiting: a YOLO-seg engine emits rows plus a prototype bank and never a mask, so
one crop's area is two outputs multiplied and reduced — a fold no per-row scatter can express.
`_reduced` is that seam, run once per chunk before the scatter, and the ~3 MB prototype bank
is dropped there rather than carried through reassembly.

Invisible offline and obvious on a GPU: `SinkOutput` never read `masks`, so `mask_area_px`
was `None` in every event this system has published. It reads it now.

`selects_rows = True` was the price: two segment slots that could fill one row are refused at
load, `when: class == …` on one is refused, and `ship_person.yaml` gains `classes: [ship]`.

## 2026-09-04 — The record seam: both planes' `build_records`, byte-compared (P5-A-ALLOC)

The event seam compares the two JSON writers on records a scenario STATES, so
`build_records` -- the translation unit production runs, on both planes -- was covered only
by each side's own unit checks. A record scenario states what the graph LEAVES BEHIND
(detections by class id, the per-object batches with their row indices, the label table, the
field map) and each plane builds its own records: `scenarios/records/` ->
`golden/records/` -> `test_record_parity`, 17 checks, byte-identical on the first comparison.

It also settled a rule that was undocumented and backwards: the candidates are a COVERAGE
union and both planes OVERWROTE, so the last batch to mention a row set the field. The answer
is the one the chain plane already had -- `PoolEmbed._scatter` and `ChainWalk.inbound` raise
on exactly this state -- so `build_records` refuses on both planes, and the contested case is
a scenario with no golden, because what both must do is refuse it.

## 2026-09-04 — The resolved chain plan: the C++ plane stops hard-coding its graph (ADR-020)

`CSRC-TOPOLOGY-Q` answered: no `csrc/topology/` mirror. Python validates the chain and hands
this plane a line-oriented **plan** (#131: `shipinfer plan -t <chain.yaml>`); `plan_stages.cpp`
turns it into the crop sets, the stage names, the label table and the event field map — one
source where four hand-kept lists used to disagree, and one of them was wrong: the ladder said
a ship was class 1 while its crop specs said 8, so every ship left as `unknown`.

The fourth parity seam, and the first with both halves automatic: a Python test holds the
emitter to the committed goldens, `test_plan_parity` (71 checks) re-serialises them and
compares bytes, and `test_plan_stages` (31) gates the decision — which is CUDA-free precisely
so it can be gated, after review found three defects hiding behind an arrangement rather than
a hard question.

## 2026-08-28 — The ingest parity harness: one scenario, two planes, one committed golden (P6 PR-A)

**What.** CLAUDE.md's sync rule says a change to a Python data-plane seam is not finished
until the C++ seam carries it. Nothing checked. `test_ingest.py` and `test_ingest.cpp` assert
the same *properties* in two languages, which catches a property somebody deleted and not a
behaviour that quietly differs. `benchmarks/parity/` is the gate: a line-oriented scenario
drives the **real `IngestManager`** in both planes against a scripted source, each writes a
canonical JSONL trace, and both are held to one golden emitted once by the Python plane and
committed. Per-camera sequences in order; cross-camera interleaving never compared.

**Found on the first run, exactly as the plan expected.** Two real divergences, both in
`CameraHealth`, both registered in `benchmarks/parity/known.py` with citations on both sides
and an open ledger line: `last_error` carries a `"<ExceptionType>: "` prefix on the Python
plane and not on the C++ one; `consecutive_failures` after a fatal open is 0 there and 1
here. A third, the sticky stop fate (`actor.h:139-145`), is documentary — it shows in no
trace field. No golden was regenerated and nothing was `xfail`ed; that is the whole point.

**The seam.** `csrc/tests/` gains three header-only support files, off every link line. The
binary is `test_ingest_parity` so CI's existing `for candidate in csrc/build/test_*` loop
runs it: **zero workflow changes**. Offline in both planes, and no GPU tier by design.

**What internal review broke, and what it cost (round 2).** Three claims the harness made
about itself were falsified by running it, and every fix is a test that fails without it:

- The C++ half of the divergence register could be widened with one line and no entry in
  `known.py` — the mirror test only checked Python → C++. The register is a table now, both
  halves' id sets must be **equal**, and the binary also fails on an id excused past its own
  table.
- `retry.peek_us` could not fail: both planes recomputed `initial · factorⁿ` from the
  scenario's config, so doubling production `ExponentialBackoff.peek()` left the gate green
  while the README called the column coverage. Each recorder now mirrors a **production**
  backoff, stepped in lockstep with the observed failure count; the mutation is red on both
  planes.
- The three entries cited ledger items `P6-D1/D2/D3` that did not exist — the test asserted
  the *shape of a string*. It reads `.claude/TASKS.md` now, and the three items are open.

Plus: a registered divergence that fires in no scenario fails the C++ gate (a fix at the
call site otherwise just stops printing `KNOWN:`); the goldens assert their scenario's
promise rather than a record-count floor; and the emitter's entry point moved to
`scripts/emit_parity_golden.py`, because `require_container.py` denies `-m benchmarks.*`
wholesale and a README documenting a denied command teaches the reader to reach for
`SHIPINFER_ALLOW_HOST_RUN`.

---

## 2026-08-28 — `recognize` as a gallery query: `GalleryRecognize` + the gallery on disk (Phase C7)

---

## 2026-08-28 — The `output` element, event schema v4, and the runnable demo chain (Phase C8b)

**What.** The far end of the chain. A frame that reaches the last element is now one published
`PerceptionEvent`, and `topology/ship_person_cpu.yaml` is the first chain file in this
repository that loads with nothing substituted. The layering half — `core/events/` and
`topology/sinks/` — landed first and on its own; this entry is what sits on top of it.

| Piece | Delivered |
|---|---|
| **schema v4** | `ObjectRecord.global_id`, published as `body_global_id_vec` / `ship_global_id_vec`. Additive: `as_det2mot` untouched, and the test checks a **literal v3 key set** rather than comparing two v4 events to itself |
| `topology/elements/output.py` | `SinkOutput` + `JsonLinesOutput` (`jsonlines`/`jsonl`/`file`) + `NullOutput` (`none`/`null`/`count`); `accepts ("meta@cpu", "bgr@cpu")`, `produces ()` |
| `topology/elements/output_kafka.py` | `KafkaOutput`, registered with `register_lazy`, so a chain that names `jsonlines` never imports a broker client |
| `track_rows` (`elements/track.py`) | one detection row per published track, from `shipvision.mot.association.associate` over IoU — `pipeline/graph/tracking.py::_attribute`'s algorithm in the chain's vocabulary. `params: attribution_iou`, default 0.3 |
| `detections.per_row` | the one rule for reading a per-object `meta` key in its two legal shapes; `track._embeddings` and `output` both call it |
| loader refusals (`chain.py::_check_row_selection`) | `when: class == …` on a `selects_rows` element is refused naming `params: classes:`; a `classes:` label the chain's detector never emits is refused naming both slots |
| `Element` declarations | `selects_rows`, `declared_classes()`, `detection_labels()` — the loader asks the element, never a list of `impl` names |
| `topology/ship_person_cpu.yaml` | the runnable demo: `replay` decode, four `pool` slots, `shipvision` track + mtmc, `jsonlines` output. No `when:`, no `recognize` slot yet |
| **`docs/arch.md` §1 + ADR-017** | the design of record is amended, not just the code — see below |
| tests | `test_output_element.py` (47), `test_chain_to_events.py` (6, end-to-end over the runner), +13 attribution in `test_track_element.py`, +9 `per_row` in `test_detections.py`, +15 in `test_chain.py`, +7 v4 in `tests/core/test_event_schema.py` |

**Decisions.**

- **The event's rows are the detections, and the track→row mapping is the `track` element's
  job.** A track's box is the filtered estimate, so recovering which detection fed it needs
  the frame's detections, the tracker's answer and the association solver at once — and only
  `track` holds all three. An `output` element that redid it would be a second, quieter
  tracker, and would need `shipvision` in a layer that has no other use for it.
  `meta["tracks"]` arriving without `meta["track_rows"]` is **refused**, because the
  alternative is every `track_id` silently `null` on a chain that is tracking.

- **A sink that refuses is counted, never raised.** The runner fails an item's future on any
  exception, so raising would turn a broker outage into a walk that stops over frames that
  were good. `ResultSink.emit`'s `bool` and `drain_delivery_failures()` both land on
  `shipinfer_output_events_dropped_total`, the latter charged to the camera in the tag it came
  with rather than to whichever frame was mid-emit.

- **The element converts nothing itself.** `embedding` goes through
  `core.events.as_embedding` and `bbox` through one `tolist()`, because the per-element
  `float()` generator is a ~30 M-call-a-second Python loop at the documented load, paid even
  with the `null` sink — measured 6.3x slower here at 2048-d. The element keeps only the rule
  that is genuinely its own: a vector nobody filed is `()` and never `None`. A test pins the
  *identity* of the conversion this element and `pipeline/graph/state.py` resolve, because two
  functions that agree today is exactly how this loop came back the second time.

- **`when:` guards frames, `classes:` selects rows — and this amends the design of record.**
  `docs/arch.md` §1's canonical chain snippet had been teaching the spelling the loader now
  refuses, and line 51 explained why to repeat it. The snippet is rewritten to
  `params: {classes: [ship]}` on the three crop slots, §1 gains a paragraph stating the split,
  and **ADR-017 carries an amendment** (2026-08-28, phase C8) recording it with C8a's finding
  and this loader refusal as the evidence. `segment` keeps its `when:` — it submits the whole
  frame and selects no rows — and `when:`'s semantics are otherwise untouched: only the `class`
  field, and only on a row-selecting slot. `topology/ship_person.yaml` still carries the old
  guards, because rewriting them means rewriting the test class that pins skip-and-continue and
  the file stops at `gstreamer-gpu` long before the loader reaches them; its header says so and
  points at the runnable sibling, and phase D will hit the refusal on its first run.

- **The `null` output is registered as `none` with `null` as an alias**, the opposite of the
  sink's own registration: YAML reads a bare `impl: null` as the null *literal*, so leading
  with that name puts a schema error with no diagnosis in front of the one implementation a
  chain reaches for when it has nowhere to publish yet. A test drives both halves — `impl: none`
  loads, `impl: null` is a `ChainSpecError` naming `elements.output.impl`.

**What the evidence does and does not show.** The end-to-end classes in
`test_chain_to_events.py` write real files through a real runner, and the cross-camera merge is
shown with a **double** that assigns one fleet id per object: two cameras, two different
`track_id`s, one `global_id`, asserted on the published bytes. That is deliberate and it is the
honest split — whether the *real* assigner merges a given pair depends on whether their instant
closed and on the clustering threshold, so a `ship_global_id_vec: [0]` out of a live two-camera
run is a **group-of-one assignment, not a cross-camera join**. The deterministic version of the
real merge is pinned one layer down, at the chain item, in
`test_mtmc_element.py::test_near_identical_embeddings_share_one_global_id`. The real-tier class
here asserts what is true of every run instead: an array a row short, or a global id on an
object with no tracklet, both fail.

**Not done, deliberately.** No `recognize:` slot in the demo chain (C7 is another lane; adding
it is one line plus one `after:`). No `mask_area_px` in the event, because `PoolSegment` still
forwards the whole frame and files no areas — a field read from a key nobody writes is worse
than an absent one. `img_fps` is 0: the element is never told a camera's rate, and inventing
one from `params:` would be a number an operator could set wrong with no symptom. No
`shipinfer run` invocation against the demo file: the walk is covered through `InprocessRunner`,
and the CLI composition root is phase D's.

---

## 2026-08-28 — the event value and its transports move to the layers both generations can reach (Phase C8, the layering half)

**What.** The first phase-C element. `recognize: {impl: shipvision}` is a bounded
nearest-neighbour search over a `shipvision.reid` gallery — no model, no repository artefact,
no pool — plus shipinfer's half of that gallery: the on-disk format it is loaded from.

| Piece | Delivered |
|---|---|
| `topology/elements/recognize.py` | `GalleryRecognize`, `@registry_for(ElementKind.RECOGNIZE).register("shipvision")`. Caps copied verbatim from `_PoolElement` (`nv12@gpu, tensor@gpu, bgr@cpu` in, `*@*` out) and it stamps **no** cap on the item it derives. `requires_model_name = needs_model = False` — the divergence C2's split was built for. `_do_open` loads shipvision through `topology/bridge.py`, builds the gallery through `GALLERIES.build`, fills it from disk and refuses a width mismatch; `_do_process` queries once per selected row and files `meta["identities"]` |
| `topology/gallery_store.py` | `<repository>/<entry>/<version>/gallery.npz` — `vectors (N,d) float`, `identities (N,) text`, optional `camera_ids (N,) text` — with `resolve_gallery_path` (newest version wins, Triton's rule) and `load_gallery_file` (shape, dtype, finite and non-zero validated, `allow_pickle=False`). numpy only; no shipvision, no `repository` import |
| `topology/elements/_vectors.py` | `rows_by_index(vectors, detections, who=)` — the rule for reading `meta["vectors"]`, extracted from `recognize` in review and called by `recognize` **only**; `track` still carries its own laxer copy (follow-up TRACK-VECTORS). numpy and `core.errors` only; no element, no gallery, no submodule |
| `topology/elements/mock.py` | `MockRecognize` files the same mapping shape the real element does, so a chain of mocks exercises the *type* the fan-in will merge rather than a stand-in for it |
| `tests/topology/test_recognize_element.py` + `tests/topology/test_vectors_rows.py` | 87 + 45 offline tests, green **with and without** the submodule |

**Why.** `pipeline/graph/graph.py` is the decision of record: there is no `ship_recognizer`
model and there never was one worth training. Identity is a search over the ship embedding,
`shipvision.reid` already carries bounded galleries with the same-camera exclusion protocol,
and the gallery is *state* — so the step belongs in the stateful plane beside tracking, not in
the stateless GPU pool. `PoolRecognize` stays registered for a deployment that really does run
an identity network.

**Decisions.**

- **`meta["identities"]` is a mapping keyed by the detection row**,
  `{index: (identity | None, similarity | None)}`, holding an entry for exactly the rows this
  element queried. Three states, and the third is load-bearing: matched, queried-and-unknown,
  and **absent** — a row nobody embedded, or one `classes:` did not select, is not this
  element's to answer. The runner's fan-in **will** merge branches by unioning their metadata
  once C8a changes `InprocessRunner._inbound`; today that merge is `meta.setdefault`, so it is
  first-writer-wins per key and a second `identities` producer is dropped whole. The shape is
  chosen for the merge that is coming: two recognisers on two branches must claim disjoint
  rows, two positional lists cannot be merged at all, and a `(None, None)` placeholder would
  collide with the other branch's real answer. `ShipvisionTrack` reads `meta["vectors"]` under
  the same convention — which round 2 turned from a convention into one function, below.
- **`exclude_camera` on every published query, with no parameter to turn it off — and the
  default gallery is `flat`, because that is the only shipped implementation that can honour
  it per entry.** A match against the query's own camera measures the tracker and inflates
  every score. `centroid` folds an identity's views into one vector and records only the
  camera of the *most recent* observation, so the same fixture queried through it answers
  `ship-a` at 0.994 — the self-match — with the `exclude_camera=` argument still in place.
  `centroid` remains selectable (its memory scales with the fleet rather than with dwell
  time) and `open()` WARNs when the configured implementation is not in `_EXACT_EXCLUSION`.
- **One deliberate exception, and it is a different question.** With `enrol: true` the
  membership check before an add asks "is this appearance in the gallery **at all**", and that
  one must *not* exclude — a camera can never match what it itself enrolled, so with the
  exclusion it re-enrols the same ship on every frame and the gallery grows per frame. The
  published answer for such a row is still `None`: the identity exists and this camera may not
  claim it.
- **Unknown is `None`, never `0`,** because `0` is a legitimate gallery id. A freshly minted
  enrolment files `(identity, None)`: the identity is real, the similarity behind it does not
  exist.
- **Enrolment writes into a *second* store, so it cannot cost the operator a curated
  identity.** Both shipped galleries evict the least recently *observed* entry and this
  element never re-adds on a match, so a file-loaded identity's clock is frozen at load and it
  is the first thing a full gallery drops: measured at capacity 8 with four curated identities
  and forty stranger frames from one camera — two seconds at 20 fps — the single-store version
  left `curated survivors: []`. That is ADR-005's own failure reproduced inside the gallery.
  The curated store now has no writer after `_fill`; minted entries go into a store built at
  `open` and queried beside it, merged by score. The alternative — refusing to enrol at
  capacity — would need `gallery.capacity`, which is **not** part of `BaseGallery`; two stores
  need only `add`, `query` and `__len__`, which are. The cost is stated honestly in `_enrol`:
  enrolment makes every row two queries and an unmatched row four, not "one extra".
- **Enrolment is opt-in, off, and gated on a *detection* confidence** read from C3's
  `Detections.scores`. A chain that embeds with no decoding detector ahead of it carries no
  per-row quality, so nothing is enrolled — and the element says so once, at WARNING, because
  a switched-on enrolment that silently never fires is the flattering failure this project
  exists to remove. Minted ids are `<prefix>:<camera>:<frame>:<row>` (`auto:cam-03:184102:2`):
  fleet-unique without coordination, greppable, and naming their own provenance.
- **`params: {classes: [ship]}` selects rows by C3's `Detections.labels`**, mirroring
  `ShipvisionTrack`'s key of the same name over the same field. `when:` gates the element per
  *item* and never the rows inside it, so an admitted frame still carries every detection the
  detector found; a ship gallery asked about a person answers *something*. A row the filter
  excludes is not queried, not counted, and not in the mapping. `classes:` with no
  `meta["detections"]` to read is a refusal, not a silent pass-through.
- **The rule for `meta["vectors"]` is written once, and `recognize` is its first caller.** The
  convention has two owners that disagree at the edges: `track` coerces `{"3": v}` and
  `{3.0: v}` where `recognize` refuses them, and `track` refuses a mapping only when *no* key
  names a row where `recognize` refuses when *any* key does not — so one embedder can feed a
  chain whose `track` accepts a frame and whose `recognize` refuses it, over the same metadata
  key, and C8's scatter-back was about to be the third opinion.
  `topology/elements/_vectors.py` is that rule written once: integral keys (`int`/`np.integer`,
  never `bool`/`np.bool_`, never a float or a string), a negative key refused **always** even
  when no detection count is knowable, any out-of-range key refusing the frame, an empty mapping
  legal, and every refusal raised *before* the first pair is yielded, because the callers query
  a gallery per pair and half a frame is not an answer.
  **`track` is deliberately not repointed in this slice**, and the first version of this entry
  claimed otherwise while `track.py` was not in the diff at all. Two reasons it waits: swapping
  the reader makes `track` *stricter*, which is a behaviour change owing its own red tests, and
  `track.py` is the file the queued C8b branch is already editing, so doing it here buys a
  rebase conflict against reviewed work. The gap is executable rather than asserted —
  `TestTheReaderTrackStillDoesNotUse` in `tests/topology/test_vectors_rows.py` pins both edges
  (`{"0": v}` coerced by `track` and refused by the shared reader; `{0: v, 7: v}` on three rows
  half-dropped by `track` and refused by the shared reader) and is the follow-up's acceptance
  criteria in advance. Follow-up: **TRACK-VECTORS**, after C8b merges.
- **The gallery format is shipinfer's (ADR-006), and it is an entry in the model repository.**
  shipvision has no `save`/`load` by design. `.npz` because the payload is an `(N, d)` float32
  matrix; `allow_pickle=False` because a repository is a directory an operator syncs from
  somewhere else — and that is now a test rather than a claim: an object array whose
  `__reduce__` names `os.system` is refused **and does not run**. Flip the flag and the
  archive is still refused, by the shape check, *after* the payload has executed.
- **The dim check's two sources, named.** Declared `params: {dim:}` (the embedder's width,
  which a pure layer cannot discover — `repository` is not importable from `topology`) versus
  the loaded file, including an **empty** one: `np.savez` keeps the shape and the loader
  refuses a zero-width array, so a `(0, 512)` archive states its width as loudly as a full
  one. Both present and disagreeing stops the deploy; only one and it stands; neither and the
  width is not knowable at open, which the warning says. A mismatch that arrives on a frame
  instead (no `dim:` declared) is re-raised as our `ValidationError` rather than escaping as
  `shipvision.errors.DimensionMismatchError`.
- **An empty gallery opens with one WARNING** rather than a refusal: a deployment that has not
  enrolled anyone yet is an ordinary state, and refusing it would mean a chain cannot be
  brought up before its identities exist. What it must not be is quiet — an empty gallery
  answers `None` forever and reads exactly like a recogniser meeting strangers.
- **No lock in the element (the GIL law, V142).** `BaseGallery`'s contract is that
  implementations own their locking, and `FlatGallery.query` deliberately does not hold its
  lock across the gemm because BLAS releases the GIL there. An element-level lock would put
  the convoy back.
- **Every `params:` value is validated at load,** including the switches: `enrol: "off"` is
  truthy and would have turned enrolment *on*, so a non-boolean is refused by name.

**Known gaps, both named in the code.** A `pool` embedder files its response's raw
`{tensor_name: Tensor}` under `meta["vectors"]`; scattering those rows back to the detections
that produced them is C8's, so that shape is **refused loudly** here rather than guessed at.
And the identity published here is a `str` — the gallery's own vocabulary — while
`pipeline/schema.py`'s `ObjectRecord.ship_id` is an `int | None`; there is no integer that
means `auto:cam-A:184102:0`, so the narrowing belongs to whoever fills that record (C8b) and
is named rather than papered over with a cast at this end.

**Evidence.** `pytest -q`: 2914 passed, 1 skipped, 60 deselected (present) / 2740 passed, 175
skipped, 60 deselected (with `shipvision` masked by a meta-path finder), measured after the
rebase onto C6 and the round-2 fixes; collection is 2915/2975 either way, which is the
property that matters — the submodule changes what *runs*, never what is collected. Revert
checks:
filing the identities as a positional list turns **11** red; putting `_DEFAULT_GALLERY` back
to `centroid` turns **6** red, including the cross-camera answer (`assert 'ship-a' ==
'ship-b'`) with `exclude_camera=` untouched; pointing enrolment's `add` back at the curated
store turns **4** red (`curated survivors` becomes `auto:*`); `allow_pickle=True` turns **2**
red on `assert not True` — the marker file the archive's pickle created. On the extracted
reader: dropping the always-refuse-negative rule turns **3** red, loosening the range rule back
to `track`'s "refuse only when *no* key names a row" turns **5**, and putting `int(key)`
coercion back turns **7**.

---

## 2026-08-28 — `PoolEmbed` crops per detection, files the vectors per row, and the fan-in merges them (Phase C8a)

**What.** The embed→track scatter-back, the last missing link before the demo chain runs.
`embed` submitted the *whole payload* — the frame — to a re-identification model whose input
is `3x256x128`, and filed the raw `response.outputs` under `meta["vectors"]`. That is a
`{name: Tensor}` dict, exactly the form `ShipvisionTrack._embeddings` refuses by name, so the
chain could not reach `track` with appearance at all. C4's build report said so: "C8 will need
the embed→track scatter-back before the demo chain runs."

| Piece | Delivered |
|---|---|
| `_PoolCropElement` (`topology/elements/pool.py`) | the fan-out element: `needs_image_ops = True`; `_prepare` reads `meta["detections"]` + `meta["frame_hw"]`, selects rows, cuts them in **one** `ctx.ops.crop_batch(frame, boxes, size, normalize)`; `_submit_crops` chunks at the model's `max_batch_size` and rejoins; `_finish` scatters `{detection index: vector}` |
| `PoolEmbed` | now a `_PoolCropElement`. `meta_key` and caps unchanged: payload untouched, `produces *@*` |
| `params:` | `classes: [ship]` (row filter), `crop: {size, normalize}`, `output: embedding`. Every one of them resolved at `open` against the artefact first |
| `_PoolElement` (lifted) | `_declared`, `_frame_of`, `_max_batch_rows` and `_submit` moved up out of `PoolDetect` — the two elements that read pixels now share one set of refusals |
| `ImageOpsLike.crop_batch` | the second member of the protocol, added the way `topology/base.py` says one should: by the first element that calls it, with the test that needs it |
| `track._embeddings` | the **empty** mapping is exempt from the coverage check |
| `_CropMetrics` | `shipinfer_element_crops_per_frame` (object-count buckets) + `shipinfer_element_crops_total`, both labelled by element, both null-object when the runner offered no registry |
| row selection, shared | `Detections.indices_of_any(classes)` + `Detections.boxes_at(indices)` + `parse_classes(declared, what)` in `topology/elements/detections.py`; `track` and every crop element now call one rule instead of carrying a copy each |
| `RowIndexed` (`topology/base.py`) | **a new vocabulary word on a shared seam.** A thin `dict` subclass that *declares* "this metadata value is keyed by detection row, and is therefore partial by design". Written by `_PoolCropElement._scatter`, read by the fan-in. Carries no behaviour: every `isinstance(..., Mapping)` consumer, `track._embeddings` included, is unchanged |
| **the fan-in merge** (`runners/inprocess.py`) | **a shared seam changed.** `_inbound` merged metadata with `meta.setdefault(key, value)` — first writer wins, per key, *wholesale*. `_merge_meta` now takes the **union** when two branches wrote a `RowIndexed` under one key, and refuses (`InferenceError`) two branches that filed different values for the same detection row. Any other pair of values, mappings included, keeps first-writer-wins |
| tests | `tests/topology/test_pool_embed_crops.py` (60 new), +21 in `test_detections.py` for the shared helpers, +9 in `tests/runners/test_walk.py` for the merge rule, +1 in `test_track_element.py`; four existing tests updated where C8 changed their premise |

**Why.** The chain's cardinality has to change somewhere — arch.md §5's "branch on class →
crop batch → submit crops" — and the embedder is where. Everything here is the proven fan-out
of `pipeline/graph/crop.py` + `objects.py` moved onto the chain, not a second implementation
of it (ponytail principle): one batched crop call, chunking at `max_batch_size`, and the rule
that every row knows which detection it came from.

**Decisions.**

- **The row filter is `params: classes:`, not the chain's `when:`.** A `when:` guard is
  evaluated once per *item* against `item.meta` (`ElementNode.admits`), and an item is a whole
  frame — so `when: class == ship` can only decide whether the element runs on this frame at
  all, and it reads `meta["class"]`, a key nothing in the chain sets today. A frame holds ships
  *and* people; the question is per row. The two mechanisms compose without overlapping:
  `when:` skips frames, `classes:` selects rows.
- **The mapping form, not a NaN-padded per-row array.** The sizing runs two embedders side by
  side (`ship_embedder` on the ship rows, `person_embedder` on the person rows), so partial
  coverage is the *normal* case. The mapping says exactly which rows were covered; a NaN row
  would have to be recognised as absence by every consumer, and the first one that forgot would
  match a track against a vector of NaNs and never say so.
- **The scatter is additive — and so is the rejoin, which is where the shipped chain needs it.**
  Two embedders can meet in two ways and both had to be closed. *In series* — both on one
  branch — the second finds the first's mapping in `item.meta`, and `_scatter` merges into it
  (and refuses a non-mapping already filed under the key rather than overwriting the producer
  that needs fixing). *In parallel* — the wiring `topology/ship_person.yaml` is shaped like
  and the one C8b will give it, `embed_person: after: detect` and
  `track: after: [recognize, embed_person]`; on the shipped file itself both embedders carry
  `when: class == …` against a field nothing in the chain sets, so today neither of them runs
  and the fixture in `tests/topology/test_pool_embed_crops.py` is what declares the shape —
  neither element ever sees the other's item, `_scatter`'s merge branch is never reached, and
  the union has to be taken at the fan-in. It was not: `InprocessRunner._inbound` merged branch
  metadata with `meta.setdefault(...)`, so the *first* branch's whole `vectors` mapping won and
  the second's was dropped. At the sizing that is ~15 000 person crops a second cut, embedded on
  a GPU and discarded, every person reaching the tracker with `embedding=None` — no exception,
  no counter, both elements' per-frame counters reporting the crops they really did make. The
  fix is at the seam and not in the element: `_merge_meta` unions two mappings under one key,
  because a partial coverage is what the mapping form *means*.
- **The shape is declared, not sniffed — and that is what makes the union safe.** The first
  cut of this rule unioned any two `Mapping` values, and a `Mapping` is exactly what it cannot
  decide on: `_PoolElement._finish` files a model's raw `response.outputs` — `{output name:
  Tensor}` — under its `meta_key`, and `PoolSegment` (`masks`) and `PoolRecognize`
  (`identities`) both keep that default. Two rejoining `segment` slots therefore either failed
  *every frame* (engines that name their output the same collided on the output *name*, with a
  message sending the operator to a `params: classes:` that family does not have) or silently
  produced a composite `{'ship_masks': …, 'person_masks': …}` neither engine emitted. So the
  writer declares itself: `_scatter` files a `RowIndexed`, and only two of those union.
  Everything else keeps first-writer-wins, which is the right default for a value whose shape
  nobody declared — and it makes the refusal reachable only from slots that *have* a `classes:`
  to check. The union is itself a `RowIndexed`, because a three-way rejoin merges the third
  contributor into the result of merging the first two.
- **What the merge does not do.** A key one branch wrote as a `RowIndexed` and another as
  something else is not a union — that is two branches disagreeing about what the key *is* — so
  it keeps the existing first-writer-wins rule, as do two scalars (`meta["class"]`), whose
  resolution
  stays a property of the chain file's declaration order rather than of thread timing. Two
  branches filing *different* values for the same detection row is refused with a typed
  `InferenceError` naming the key, the row and every slot that claimed it: there is no answer
  to "which of these two vectors is this object's", and it means both elements cropped it,
  which is the duplicated GPU work `classes:` exists to prevent. Identity is checked before any
  of that, so a mapping written *before* the fork and carried down both branches — a diamond,
  the ordinary case — is not mistaken for a disagreement. Equality is deliberately not
  attempted beyond identity: the values are numpy arrays and `a == b` on two of them is an
  array whose truth value raises.
- **No *general* load-time check behind it, and that is a decision.** CONVENTIONS 2.6 would
  prefer the loader to refuse "two elements on rejoining branches write the same key with
  incompatible shapes" at `from_spec`. To do that it would have to know two things about
  *every* element: which metadata keys it writes, and which of them it writes as a
  `RowIndexed`. The `pool` family declares both — `meta_key` is a `ClassVar` and only
  `_PoolCropElement` scatters — but the family is not the tree: `Element` is an ABC anyone may
  implement, `process` may file any key it likes, and a loader that walked only the classes it
  recognises would pass every pair it cannot see, which reads as coverage and is not. The check
  that *is* sound is narrower, is about `classes:` rather than about shapes, and belongs
  elsewhere: two `pool` crop elements on rejoining branches with the same `meta_key` and
  `classes:` that overlap or are absent is a static fact, and it lands beside the
  `classes:`-against-`class_labels` cross-check that reads the same two fields.
- **The series composition refuses an overlap the same way the fan-in does.** `_scatter` merged
  with `{**existing, **covered}` — silent last-writer-wins per row — so the same `classes:`
  overlap was a typed refusal when the two slots rejoined at `track` and a wrong appearance
  vector when they were declared `after:` one another. "Two elements covering one detection
  means the chain file asked both of them for it" is a property of the chain file, not of the
  wiring, so both compositions now raise `InferenceError` naming the key, the row and the two
  slots. Disjoint rows still union and an empty coverage still hands the item on unchanged.
- **An empty mapping is coverage of no rows, not an off-by-N.** `track._embeddings` refused a
  mapping "whose keys name no row at all". With a crop element in the chain that is the ordinary
  frame — `embed_person` sees three ships and covers none — so the empty mapping is now exempt.
  Keys `{100, 101, 102}` on a three-row frame is still arithmetic that went wrong; zero keys
  index nothing because there was nothing to index.
- **Zero rows means zero requests.** An empty crop batch handed to a model costs a queue slot,
  an instance slot and a round trip to be told nothing; 50 cameras of empty water at 20 fps is a
  thousand of those a second. `_do_process` is overridden for that one line.
- **Chunking at `max_batch_size`, read off the artefact.** A frame holds however many objects
  the detector found (25 was observed) and an engine's plan is built at a fixed batch. Without
  it one crowded frame becomes a single oversized request and *every* crop in it is lost — the
  failure `objects.py::_chunks` exists because of. `max_batch_size: 0` is Triton's "batching
  off", so it means no bound, not batch one.
- **The pixel scale is the slot's own, and it is pinned.** `crop.normalize` is resolved at
  `open` from `params: {crop: {normalize: {mean, std, swap_rb}}}`, defaulting to
  `Normalization()` — not read off the artefact, because a model repository `config.yaml` has
  no normalisation section, and the proven path does the same (`CropStage.__init__`).
  Crops normalised with the wrong mean and std are the right rows, in the right order, at the
  right extent: the engine answers without an error and the only symptom is appearance
  matching that degrades. So `TestThePixelScaleIsTheSlotsOwn` asserts the crop batch equals
  `NumpyImageOps().crop_batch(pixels, boxes, size, <the declared Normalization>)` *and* that
  the default would have produced other pixels — before it existed, replacing the resolved
  normalisation with the default at the `crop_batch` call left the whole offline tier green.
- **One row-selection rule, on the value that owns the labels.** `classes:` parsing and the
  label match were a copy each in `pool.py` and `track.py`, differing only in the kind word
  inside the message. They are now `parse_classes` and `Detections.indices_of_any`, with
  `Detections.boxes_at` doing the contiguous gather the crop path had open-coded (and
  `boxes_of` expressed in terms of it). Two copies of a row filter is two places for "a case
  difference is not a match" to drift, and the drift has no symptom — the element covers no
  rows, silently. C8b adds a third crop element, which is why this moved now.
- **No L2 normalisation here.** Both embedders in the demo repository are already global-pooled
  and L2-normalised (their `config.yaml` says so) and the proven path normalised nothing in
  Python either. Re-normalising would be a silent divide-by-a-tiny-number on the row where the
  engine answered with zeros.
- **`PoolSegment` stays the forwarding element.** Its crop half is one line away — the demo
  repository does feed it 640x640 ship crops — but its `_finish` is a fold over *two* outputs
  (rows × mask prototypes → one area, `pipeline/graph/masks.py::InstanceMaskArea`) that a
  per-row scatter-back cannot express, and filing the raw rows would pin a `(32, 160, 160)`
  prototype tensor per frame alive for the rest of the walk. Half a feature is worse than none.
- **The caps are unchanged and the *frame* is refused instead.** `accepts` keeps
  `nv12@gpu, tensor@gpu, bgr@cpu`: narrowing it would refuse the chain phase D makes work, and
  staying silent would download six megabytes per frame. `_frame_of` refuses a device-resident
  payload by name, with the phase that fixes it — the same refusal `PoolDetect` already made,
  now shared.

**What allocates per frame** (CONVENTIONS 2.5): the crop batch itself, one `Tensor` wrapper,
one `InferenceRequest` per chunk (one, except past `max_batch_size`), and the `{row: vector}`
mapping whose values are numpy *views*. Selecting a subset adds an index tuple and one `(N, 4)`
gather; declaring no `classes:` adds neither — `_selected` returns a `range` and `boxes_at`
recognises it as the whole frame and hands the array through (a *value* test, `indices ==
range(len(self))`, not a length test, so a full-length reordering is permuted rather than
handed back unpermuted). A frame with nothing to crop submits nothing and costs **one**
`derive()`, for the empty mapping that records that this element ran; only where a second
embedder sits *in series* ahead of it is a peer's mapping already there and not even that — on
the shipped parallel wiring the quiet frame costs the one `derive`. The metric handle is bound
once at `open` rather than looked up off `self._metrics` per frame.

**Not done here.** The demo chain (`topology/ship_person.yaml`) still names `gstreamer-gpu` and
`kafka` and does not load; nothing in the chain sets `meta["class"]`, so every `when:` in that
file is currently false for every frame — both are C8's remaining slices, not this one.

**C8b must cross-check `classes:` against the detector's labels.** Now that the empty mapping
is legal at `track`, a `classes:` value naming a label the detector never emits is a permanent,
silent no-op: the element covers no row on any frame, and the only signal is
`shipinfer_element_crops_per_frame` recording zeros. `Topology.from_spec` already sees both
slots, so refusing a crop element whose `classes:` is absent from the upstream detect slot's
`decode.class_labels` is CONVENTIONS 2.6 ("validate at start-up, not at first use") and costs
nothing per frame. Same slice as replacing that file's four `when: class == …` guards with
`params: {classes: [...]}`.


Round 3 (CI): the crop chunk bound is the engine's, not Triton's — `_max_batch_rows` asks
`effective_max_batch_size` and falls back to 1, never None (`max_batch_size: 0` bounds the
assembler at one row; the fake now enforces the same rule, and a real-engine test pins it).
`boxes_at` never aliases the live array; `_scatter` keeps a plain peer's mapping plain. A
crop element's whole-frame response carries chunk 0's `executed_on` — ledger item.

---

## 2026-08-28 — the event value and its transports move to the layers both generations can reach (Phase C8, the layering half)

**What.** No behaviour changes in this slice. One value type and one family of transports move
to homes the layer tables allow, so that the `output` element landing next can reach them
without importing a layer it must not.

| Piece | From | To |
|---|---|---|
| `PerceptionEvent`, `ObjectRecord`, `SCHEMA_VERSION`, `MESSAGE_TYPE` | `pipeline/schema.py` | `core/events/schema.py` (`pipeline/schema.py` is a re-export) |
| `as_embedding` | `pipeline/graph/state.py` | `core/events/convert.py` (re-exported from `state.py`; the DeepStream probe now imports the real home) |
| `ResultSink`, `RESULT_SINKS`, the `jsonlines` / `kafka` / `null` sinks | `pipeline/sinks/` | `topology/sinks/` (`pipeline/sinks/__init__.py` is a re-export) |
| `confluent_kafka` | `FORBIDDEN_EXTERNAL["topology"]` in `check_layers.py` | `TestImportIsCheap`'s subprocess list |
| `tests/pipeline/test_schema.py`, `tests/pipeline/test_sinks.py` | — | `tests/core/test_event_schema.py`, `tests/topology/test_sinks.py` |
| `tests/plugins/mask_shipvision.py` | — | new: the committed `-p tests.plugins.mask_shipvision` that reproduces CI's submodule-less run |

**Why.** `topology` may import `core` and nothing else (ADR-001, `scripts/hooks/check_layers.py`).
Both generations of the pipeline build the *same* event — `pipeline/graph/` from a frame's stage
outputs, the chain's `output` element from a chain item's metadata — so a value shared by the two
has exactly one legal home, and it is not either of them. arch.md §9 says the same thing about
the sinks in its own words: `sinks/{kafka,jsonlines,null}` become `output` element
implementations, and the elements live under `topology/`.

**Decisions.**

- **Re-exports, not renames.** ~30 modules and their tests name `shipinfer.pipeline.schema`, and
  `pipeline/` stays the working application until the chain replaces it (arch.md §9). A shim is
  cheaper and more honest than a rename spread across two generations, and it is a *re-export*
  rather than a second definition: `test_the_shim_re_exports_the_same_class` pins identity, not
  equality, because two definitions under one name is the failure the move exists to prevent.
- **`as_embedding` moves with the value it feeds, and there is exactly one of it.** It converts a
  model row to the JSON array an `ObjectRecord` carries, and the spelling is load-bearing:
  `tuple(float(v) for v in row)` is a per-element Python loop on the emission path — 2048 floats
  per crop at ~15 000 crops/s is ~30 M `float()` calls a second, paid even with the `null` sink —
  and `tolist()` is the same value in one C call, measured ~6x faster on this host at both 512-d
  and 2048-d. The generator has been written and removed twice. It now has three callers on both
  sides of the accelerator seam (the graph, the DeepStream probe, and the `output` element that
  follows), so it lives beside the event in `core` and `tests/core/test_event_convert.py` pins
  the *identity* of the function each of them resolves.
- **`confluent_kafka` left the static row rather than the codebase.** `check_layers.py` walks the
  AST and counts a function-scope import exactly like a module-scope one, so naming the client in
  `FORBIDDEN_EXTERNAL["topology"]` would ban the only legal spelling — the import inside
  `KafkaResultSink.__init__` — and leave none. The ban moved to where it can be enforced:
  `TestImportIsCheap` imports `shipinfer.topology` in a subprocess and fails if a broker client
  came with it. Exactly the precedent `shipvision` set one paragraph above it in the same file.
- **`numpy` in `core` is not a relaxation.** ADR-001 draws the line at torch, tensorrt,
  onnxruntime and the rest; `FORBIDDEN_EXTERNAL["core"]` has never named numpy and `core/types/`
  has used it since the first commit. `core/events/convert.py` touches no device.
- **The move ships alone.** It edits `scripts/hooks/check_layers.py` and
  `tests/test_architecture.py` — the layer tables, the highest-consequence file pair in the
  repository — and a table change deserves to be read on its own rather than as page three of a
  feature.

**Evidence.** `python scripts/hooks/check_layers.py` exits 0. Offline tier green with the
submodule present and under the mask; `pre-commit run --all-files` clean. Numbers in the PR body.

---

## 2026-08-28 — the `mtmc` element: anchored instants across cameras, and a barrier that never takes the last worker (Phase C6)

**What.** The chain gains its cross-camera tier, in two modules and the split is the point.
`topology/barrier.py` is **pure** — it turns a stream of single-camera frames back into
synchronised instants and knows nothing about tracks, so every synchronisation property is
tested with strings and a callback and no submodule. `topology/elements/mtmc.py` is the
element that hands those instants to `shipvision.mtmc` and scatters the answer back.

| Piece | Delivered |
|---|---|
| `InstantBarrier` (`topology/barrier.py`) | **anchored** instants on the *capture* clock: the first arrival opens a window of `sync_window_s`, later frames join while the instant's capture span stays inside it, and a camera already in the bucket reporting its **next** frame seals that instant and opens the following one. Closes on the last live camera, on a seal, or on the window. Whichever worker closes it runs the association under the barrier's lock and publishes to the waiters; late arrivals counted and never retro-fitted; buckets bounded and the one open longest evicted; `close_all(reason=…)` resolves every waiter; every wait bounded by the bucket's own deadline |
| … `WaiterBudget` | the never-starve guard, counted **per process**: at most `workers - 1` waiters across *every* barrier in the runner. The frame that would take the last permit is emitted immediately with `mtmc` in `missing_stages` and counted. `ElementContext.workers is None` collapses to zero permits, i.e. never wait |
| … `camera_added` / `drop_camera` | the live set drives "every camera reported". `drop_camera` also re-checks the **open** buckets and seals any the removal completed — it must not run a tracker on the lifecycle thread |
| … `instant_stats()` / `frame_stats()` | two dictionaries, never one: `shutdown` is a member of both families and a single number would be frames plus instants |
| `ShipvisionMtmc` | `accepts = produces = ("meta@cpu",)`, all three `ClassVar`s `False`. `meta["tracks"]` + `meta["frame_hw"]` in, `meta["global_ids"]` out — a list aligned with this item's tracks, built from a map keyed on `(camera_id, track_id)` |
| … `params:` | `algorithm` (`cluster`), `matrix_builder` (`gated`), `clusterer` (`agglomerative`), `group`, `cameras`, `sync_window_ms` (60), `max_instants` (8), `calibration`, `options` |
| `Element.camera_group()` + `CameraGroup` | a hook on the ABC: "these cameras must be placed together". Default `None`; `ShipvisionMtmc` answers with its parsed roster |
| `ElementContext.waiter_budget` | the process-wide permit pool, built once by `InprocessRunner` with `workers - 1` |
| `bridge.load_errors()` | the fifth loader: an element that has to *catch* a shipvision refusal by name needs the class |
| `runners/fleet.py` | `_camera_groups()` + `_pin_to_group()` — a camera group is an atomic unit of placement, and the launcher learns the membership by asking every node `camera_group()`, with no kind test |
| tests | `tests/topology/test_barrier.py` (57, pure, no submodule), `tests/topology/test_mtmc_element.py` (57), `tests/runners/test_fleet.py` (+9) |

**Why.** `shipvision.mtmc` consumes a `FrameTrackCluster` — every camera of a group at one
instant — and refuses anything less, because handing it one camera at a time turns cross-camera
association into within-camera deduplication. The chain delivers one frame at a time on
whichever worker took it off the fair lane. Something has to bridge that, and everything worth
testing about the bridge is synchronisation rather than geometry — hence a pure module with its
own test file that needs no submodule at all.

**Decisions.**

- **The instant is anchored by its first arrival, never gridded.** The obvious key is
  `floor(capture_s / window)` on an absolute grid, and it is wrong at *every* setting. A window
  wider than one frame period puts two consecutive frames of one camera into the same cell once
  every `window / period` frames: at 20 fps against the 60 ms default that is **one frame in
  six**, each of which is refused an answer, with the cameras perfectly genlocked and nothing
  else wrong. A window narrower than the frame period fixes that and breaks the other half —
  free-running cameras spread their captures over a whole period, so no cell holds the whole
  group. The two constraints are incompatible, which is the proof that no absolute window is
  correct. So the first frame to arrive opens the window, later frames join while the span
  stays inside it, and a camera's *next* frame is the evidence that the instant it was in is
  over: that instant seals and the next one opens with that frame. A camera can no longer
  collide with itself at any window. Measured below.
- **A sealed instant is closed by a waiter, not by the thread that sealed it.** The sealer is
  a frame from a *different* instant; if it ran the association and the callback raised, the
  exception would fail its own future for somebody else's instant. A waiter is a pipeline
  worker holding the callback and owns the answer it is waiting for. Same argument as
  `drop_camera`'s, which is why both use the same `ready` path.
- **The barrier never blocks the last worker, and the budget is per *process*.** The walk is
  synchronous (`arch.md` §5③): a worker inside an element is a worker not draining its lane. If
  every worker parks waiting for cameras whose frames are still *queued*, no instant can close
  on evidence — only on the timeout — and the shard has converted itself into a fixed latency
  with a stalled queue behind it. Counting waiters *per element* did not deliver that: two
  `mtmc` slots in one chain (a supported configuration — the loader takes an explicit `kind:`)
  would have slot A admit `workers - 1` and slot B, which had seen none, admit the last worker.
  So `WaiterBudget` is one object per runner, handed down `ElementContext`, and the rest are
  emitted with an honest gap (`shipinfer_mtmc_would_starve_total`).
- **The scatter is keyed on `(camera_id, track_id)`, never on list position.**
  `FrameTrackCluster` flattens the group into one observation list and the tracker answers in
  that order, so camera B's rows sit at offsets nobody can derive from B's own frame. A
  positional scatter produces a plausible answer rather than an error — ADR-002's tag rule, one
  layer up.
- **The live set is the *announced* set, not the configured roster.** A group's roster names
  every camera in it; a shard runs only the ones placed on it. A barrier that waited for the
  roster would time out on every instant for the life of the process and report it as a healthy
  chain running 60 ms slower. Before the first announcement it falls back to cameras it has seen
  traffic from, so a runner that does not drive the hooks costs one instant of warm-up rather
  than degrading to per-camera MTMC; the fallback latches off for good at the first
  announcement, because a set that came back after `camera_removed` emptied it would resurrect
  the camera that hook exists to forget.
- **`camera_removed` drops the camera from open buckets too.** Dropping it from the live set
  alone leaves every *currently open* instant still counting it, so each one sits out the whole
  window for a camera that will never report again — a permanent per-frame tax. The re-check
  seals rather than closes: it runs under the runner's lifecycle lock, behind which every
  `add_camera`, `remove_camera`, `drain` and `stop` queues.
- **The association runs under the barrier's own lock, and that is the only lock.** The GIL law
  (`arch.md` §7, V142) allows shipinfer one lock around `tracker.track()`;
  `ClusterMTMCTracker` already holds an `RLock` for the whole call, so a second one of ours
  would buy nothing, and the results have to be published to the waiters under this lock anyway.
  The cost is measured, not assumed, and it is **milliseconds**: 0.48 ms at 2 cameras × 2
  tracks, 4.70 ms at 8 × 15 (`CLAUDE.md`'s own people-per-frame figure), 54.32 ms at 50 × 15.
  The growth is quadratic in observations, so fifty cameras in one `mtmc` slot is a frame period
  of serialised association per instant and the answer to that is *more groups*, not more locks.
- **A camera group is an atomic unit of placement, the element declares it and the runner
  enforces it.** `arch.md` §4. The element is told `ElementContext.shard_id` and nothing about
  where any *camera* is, and it opens before a single camera is placed — so the check cannot
  live there. `FleetRunner` owns `{camera_id: shard_id}`, so `add_camera` **pins** a camera to
  its group's home shard and refuses, naming the group, both shards and the recovery, when the
  pin cannot be honoured. What crosses between them is `Element.camera_group()`: the launcher
  asks every node and never asks what kind it is, so there is no `ElementKind.MTMC` test in
  `runners/`, no import of an element implementation module, and no second parse of
  `params: cameras:` — the switch statement ADR-017 §2's registry exists to delete. A second
  kind that needs co-located cameras is a method override, which
  `test_the_launcher_asks_every_element_and_never_what_kind_it_is` pins with a `track` element
  that declares a group.
- **`sync_window_ms = 60` is a proposal, not a measurement.** Nothing in `arch.md` states one
  (plan open question 3). With the anchored instant it is no longer constrained from *below* by
  the frame period — self-collision cannot happen at any window — so it only has to exceed the
  group's arrival spread, and 60 ms is a comfortable margin over the ~1 ms genlock skew of a
  wired group while staying near one frame period at 20 fps, which is the worst-case latency it
  bounds. `test_a_window_narrower_than_the_frame_period_is_also_fine` pins that nothing rests
  on the number.
- **A track with no embedding is a gap, not a dead frame.** `GlobalIdAssigner` refuses to
  identify one, and that is a per-frame data condition — a spilled embedder, a crop that
  produced nothing, a chain with no embedder in front of `track`. Caught, logged once per open
  cycle, emitted with `mtmc` in `missing_stages`. The log line names **both** reasons, because
  the refusal is per *instant*: the one frame whose thread ran the association is counted
  `reason=unassignable` and the rest of the group's, already released by the barrier, are
  counted `reason=failed`.
- **A mis-wired chain is raised on: a missing or zero `meta["frame_hw"]`, and a zero
  `captured_unix_ns`.** `CameraTracks` refuses a zero frame size precisely so the height gate,
  the truncated-box test and the homography's domain cannot all be silently wrong. A zero
  capture clock is the same class of thing — `RequestContext.captured_unix_ns` defaults to `0`,
  and a source that never stamps it would put every frame of every camera into one instant,
  which closes once and leaves the rest of the deployment `late` for good.

**Measured** (offline, no GPU, `process()` entry to return through the real element, one camera
per thread, 60 frames per camera paced at 20 fps of both capture and wall-clock time, 2 tracks
per frame, window 60 ms, `min_hits = 1`). Coverage is the share of frames that came back with
`meta["global_ids"]`:

| Cameras | Workers | min | median | p95 | max | coverage |
|---|---|---|---|---|---|---|
| 2 | 3 | 0.901 ms | 1.957 ms | 2.380 ms | 2.735 ms | **100.0%** |
| 2 | 9 | 0.925 ms | 1.797 ms | 2.346 ms | 2.606 ms | **100.0%** |
| 2 | 32 | 0.914 ms | 1.887 ms | 2.488 ms | 3.419 ms | **100.0%** |
| 8 | 3 | 0.038 ms | 0.119 ms | 4.568 ms | 6.189 ms | 37.5% |
| 8 | 9 | 1.392 ms | 3.237 ms | 4.932 ms | 7.570 ms | **100.0%** |
| 8 | 32 | 1.430 ms | 3.678 ms | 4.591 ms | 6.752 ms | **100.0%** |

Read honestly. **The whole distribution is the association**, and the window never fires: every
run above closed 60 of 60 instants `complete`, so `max` is one association and not one window.
The 8-camera rows are ~3.5 ms because the cluster carries 16 observations rather than 4.

The one row that is not 100% is **8 cameras on 3 workers**, and it is the never-starve guard
working as designed rather than a synchronisation failure: three workers cannot hold seven
cameras' frames while the eighth arrives, so five frames of every instant are emitted
immediately with an honest gap (`would_starve: 300`, and 60/60 instants still closed
`complete`). It is the number that sizes `pipeline.workers` against a group: **a shard needs
at least as many workers as its camera group has cameras** — an instant closes on the *last*
camera's frame and every earlier frame of it is parked until then, so coverage is
`min(1, workers / group_size)` and 8 workers cover 8 cameras exactly. With **two** `mtmc`
slots the requirement is the **sum** of the groups' sizes and not the larger of them, because
the waiter budget is process-wide and its permits are first-come: measured, two 8-camera
groups reach 100% each at 16 workers and 73% / 52% at 9, where whichever group's frames
arrive first takes the permits. The element now warns at `open()` and at the crossing rather
than leaving this to `shipinfer_mtmc_would_starve_total`. Its median is *lower* than the
others precisely because a starved frame does not wait.

The previous entry recorded a p95 of ~51 ms at eight cameras and attributed it to "the window
firing … a camera's frames drift across bucket boundaries". **That diagnosis was wrong**, and
so was the number's cause. The absolute grid made every camera land in a cell it had already
reported once every six frames, so one frame in six was refused an answer and the instants that
did close often closed on the timeout. Coverage on that grid, measured through the same threaded
path with perfectly genlocked cameras and the default window:

| Cameras | Absolute grid | Anchored instant |
|---|---|---|
| 2 | 40/48 = **83.3%** | 48/48 = **100%** |
| 8 | 160/192 = **83.3%** | 192/192 = **100%** |

`TestAGenlockedGridGetsEveryFrameAnswered` is the test that pins it, and restoring the absolute
grid turns it red at exactly those numbers.

**Not done.** No GPU tier (nothing here touches `runtime/`, `backends/` or `native/`). No demo
YAML change — that is C8. The `output` element still does not serialise `global_id`; the event
schema gains it in C8.

---

## 2026-08-28 — the `track` element: one tracker per camera, in-chain, with the camera lifecycle wired (Phase C4 + C5)

**What.** The chain gains its first stateful element. `TrackerShard` / `_CameraShard` move out
of `pipeline/graph/tracking.py` into `topology/elements/track.py`, and `ShipvisionTrack` sits
on top of them: `meta["detections"]` in, `meta["tracks"]` out, `meta@cpu` and no payload.

| Piece | Delivered |
|---|---|
| `topology/elements/track.py` | `TrackerShard` moved unchanged in behaviour — one tracker per camera, one lock per camera, one `frame_id` high-water mark. Its shipvision import is now `bridge.load_mot()`, called inside `__init__` |
| … `update(regression_reset=, on_implicit_reset=)` | a `frame_id` regression past the threshold is a restarted stream nobody announced: forget the tracks and take the frame. Decided **under the camera's lock**, because read-decide-reset-update has to be one step for the same reason check-set-update is |
| … `drop()` / `reset_if_present()` | what a lifecycle hook needs. `drop` takes the table lock and never a camera's; `reset_if_present` never *builds* a tracker for a camera that has none |
| `ShipvisionTrack` | `accepts = ("meta@cpu", "bgr@cpu", "nv12@gpu")`, `produces = ("meta@cpu",)`. Catches `TrackingError`, counts it, emits the item with `track` in `meta["missing_stages"]`. Implements `camera_added` / `camera_removed` — the C2 hooks reaching a stateful element for the first time (C5, folded in) |
| `pipeline/graph/tracking.py` | imports the shard back, so `TrackStage` and `shipinfer.pipeline.TrackerShard` are unchanged. `core/settings/pipeline.py`'s docstring reference repointed |
| `tests/topology/test_track_element.py` | 53 tests, green **with and without** the submodule |

**Why.** `arch.md` §5⑥ puts one stateful tracker per camera on the shard the camera lives on,
and `arch.md` §4's invariant is that the tracker never migrates mid-stream. That object already
existed and was proven; a second copy of an invariant with **no symptom when it breaks** — two
cameras on one tracker report a real identity somewhere nothing happened — is how one of the
two stops being correct. So it moved rather than being rewritten (the ponytail principle
applied to our own code).

**Decisions.**

- **A tracking refusal is caught, counted and emitted — never raised.** The runner fails an
  item's future on anything an element raises and stops the walk, so an out-of-order frame
  raised would cost the frame its whole event: its boxes, its masks and its vectors are all
  still good. It is emitted with `track` in `missing_stages` instead (`arch.md` §5⑤). What
  *is* raised is a mis-wired chain — raw model outputs under `meta["detections"]`, vectors
  that cannot be attributed to rows — because those are faults and a late frame is not.
- **`meta["tracks"]` carries shipvision's own `Track` objects, not a pure record.** They are
  exactly what the cross-camera tier consumes (`mtmc.CameraTracks(tag, tracks, h, w)`), so a
  record here would be a shape `mtmc` converts straight back, losing the embedding, the state
  and the tag on the way. Checked before deciding: `TrackPool` already returns
  `dataclasses.replace` copies and rebinds `track.box` rather than writing into it, so an item
  buffered past the camera's next frame does not change under its reader.
- **A `frame_id` regression has two populations and one threshold, default 64.** A *reorder* is
  bounded by the pipeline worker count (`arch.md` §5③ sizes it at ~32) and must be refused; a
  *restart* is an ingest actor that minted a fresh `FrameCounter` with nobody calling
  `remove_camera` + `add_camera`, and refusing that means refusing every frame of that camera
  for the process's life — the state ADR-018 names remove+add as the recovery for, with
  `shipinfer run` leaving nobody to make the two calls. `regression_reset: 0` restores the old
  behaviour for a deployment that would rather the frames stopped than the identities
  restarted.
- **`camera_removed` takes the table lock and nothing else.** It runs holding the runner's
  `_lifecycle`, so waiting for a worker inside `tracker.update` for that very camera would
  stall every add, remove, drain and the `stop` that would end the wait. The entry is unlinked
  and the in-flight frame finishes against a shard nobody can reach.
- **No `backend:` param.** `TRACKERS.build(name, backend=None)` resolves the fastest backend
  this host can build with a numpy floor. Naming `native` would make a chain that loads on the
  build machine refuse on a machine without the extension, for a step whose cost is tens of
  microseconds against a frame budget of milliseconds. It is also the shape the GIL law (V142)
  wants: shipvision delivers algorithms, and shipinfer holds at most a lock around
  `tracker.update`.
- **Metrics: `shipinfer_track_frames_out_of_order_total{camera}`,
  `shipinfer_track_frames_untracked_total{reason}`, `shipinfer_track_implicit_resets_total{camera}`,
  `shipinfer_track_cameras{element}`.** The plan asked for a `no_shipvision` reason and it is
  **not** there, deliberately: `open()` refuses without the submodule, so no frame can ever be
  processed for that reason and a counter that cannot move reads as evidence that it did not
  happen. The gauge carries `element` because two `track` slots in one chain keep two tables and
  one gauge would report whichever wrote last; it is written only when the count changes, so the
  per-frame cost is an int compare.
- **`tests/topology/test_element_model_declarations.py` was amended.** It opened every
  `needs_model=False` implementation with an empty context and asserted `is_open`.
  `ShipvisionTrack` is the first element with a *runtime* of its own, and
  `elements/__init__.py` says a host lacking one should still list the implementation and fail
  at `open()` naming the fix. The assertion is now "must not refuse for want of a pool", which
  is what the declaration under test actually claims.

**Evidence.** `tests/topology tests/runners tests/pipeline tests/test_architecture.py`: 943
passed with the submodule installed, 859 passed / 84 skipped with it masked out of
`sys.modules` (the state CI runs in). Revert-checks: dropping the `TrackingError` catch reddens
7 tests, dropping `payload=None` reddens 3, and a `camera_removed` that does not drop reddens 3.

---

## 2026-08-28 — decoded detections: `PoolDetect` letterboxes and decodes, and the runner is handed image ops (Phase C3)

**What.** The chain gains a real detector output. `pipeline/graph/detections.py` moves to
`topology/elements/detections.py`; `PoolDetect` letterboxes its frame, submits, and decodes
the rows back into source pixels under `meta["detections"]` + `meta["frame_hw"]`;
`ElementContext.ops` — declared in C2 and `None` everywhere — is filled in by `shipinfer run`
and by the shard.

| Piece | Delivered |
|---|---|
| `topology/elements/detections.py` | moved with `git mv`, unchanged except for a new `Normalization` (the structural twin of `runtime.ops.base.NormalizeParams`, which a pure layer may not import). `pipeline/graph/detections.py` is a one-line re-export so the counting-simulation graph keeps working — a shim, not a copy, because `isinstance` crosses that boundary |
| `topology/elements/pool.py` | `_do_process` splits into `_prepare` / `_finish`; `PoolDetect` replaces both. `letterbox_batch` in, `decode_detections` against the reported scale/pad out. `meta["boxes"]` is **removed** — nothing in `src/` read it |
| `topology/base.py` | `Element.needs_image_ops`, the third of these declarations. `True` on `PoolDetect` only |
| `runners/base.py`, `inprocess.py`, `fleet.py` | `Runner(ops=...)`, `Runner.ops`, and `element_context()` puts it on the context — the same shape `models=` already had |
| `cli/commands/run.py`, `cli/shard.py` | `dependency_is_needed(keyword, runner, chain)` over a `{keyword: element attribute}` table, with `model_pool_is_needed` / `image_ops_are_needed` as its two named rows: a dependency is resolved only when the chain declares it. A mock chain and a `fleet` launcher resolve none. What is handed over is a `ThreadLocalImageOps` — see the first decision below |
| `runtime/ops/thread_local.py` | `ThreadLocalImageOps` + `staging_owner` moved out of `pipeline/graph/ops.py` (re-export shim left behind), plus `get_thread_local_image_ops` — the one call that binds a delegate per worker thread, spreads threads over the visible GPUs and claims a pinned pool each |
| `topology/elements/mock.py` | `MockDetect` files a real `Detections` beside its old `boxes` list, so C4's tracker can be tested offline against the shape a `pool` detector actually produces |

**Why.** `track` cannot consume `meta["boxes"]`, because that key was `response.outputs` filed
under a name: nothing in the chain letterboxed anything — `ChainFrameSink` submits the frame
at whatever size it was decoded — so the rows were in the pixels of an input nobody had
produced. Filling `meta["detections"]` is what unblocks C4/C6/C7, and it is why the decode had
to move into the pure layer first.

**Decisions.**

- **The runner is handed one `ImageOps` per worker thread, not one per process.** `get_image_ops`
  answers for *one* thread and every implementation it can return says so: `NativeImageOps`
  keeps a staging ring inside the extension, `TorchImageOps` binds a device on the constructing
  thread and caches an event and a ping-pong staging pair on the instance. `pipeline.workers`
  threads walk one chain over one shared `PoolDetect`, so a single instance is CONVENTIONS 2.8's
  pinned buffer overwritten mid-DMA — plausible pixels, no error, and invisible to the offline
  tier because `NumpyImageOps` is stateless. The first cut of this slice did exactly that, and
  passed no `device_index` either, so a single-process run on an 8-GPU box pre-processed every
  camera on `cuda:0` — this project's founding bug one layer up. Fixed by *reuse*:
  `ThreadLocalImageOps` already solved it for `PipelineRunner`, so it moved to the layer that
  owns the seam and gained `get_thread_local_image_ops` for the composition roots. Devices come
  off the engine's `DeviceManager`, never from building one here — that costs a CUDA primary
  context per GPU this process never gives back.
- **`PoolDetect` cross-checks the artefact at `open()`, not at the first frame.** `decode.dst_size`
  is the override for a *dynamic*-shape engine; against a declared static `(3, H, W)` it is
  refused, naming both ends, because a static engine catches it loudly and a dynamic one accepts
  it and answers with boxes that are wrong on every camera. An `input` the model does not declare
  is refused too — `_resolve_dst_size` falls back to "the single declared spec" when the name
  misses, so a typo'd `params: {input: pixels}` resolved a perfectly good extent and hid itself
  until the first frame of the deploy. Both checks were in the code this element replaced
  (`pipeline/graph/stage.py::validate`) and were lost in the move; CONVENTIONS 2.6.
- **No `ElementContext.ops` is a refusal at `open()`, not a numpy fallback.** `topology` may
  not import `runtime`, so a fallback would mean a second, unfused letterbox living in the
  pure layer — a reimplementation of the thing the ops seam exists to own (CONVENTIONS 2.1),
  in Python, on the path of a thousand frames a second. A deployment that silently got it
  would read as a successful start-up and measure as a throughput cliff. The model pool is
  handed in for the same reason and refuses the same way, and `needs_image_ops` is what keeps
  the declaration and the requirement from drifting.
- **`meta["boxes"]` is dropped, not kept beside `meta["detections"]`.** Grepped: no reader in
  `src/`. Keeping it would pin the raw output tensor alive for the rest of the walk so that a
  future consumer could redo arithmetic this element has already done correctly. The mocks
  still file their own `boxes` list — that key is theirs, and the chain and runner tests that
  read it are unaffected.
- **The per-frame geometry is returned between the two hooks, never stored on the element.**
  One element instance is shared by every pipeline worker, so scale and pad as attributes
  would be overwritten by whichever frame reached `_prepare` last, and the result is boxes
  computed from the wrong letterbox — no exception, no symptom short of a tracker that swaps
  identities. `tests/topology/test_pool_detect_decode.py` drives two differently-sized frames
  through one element concurrently, held inside `infer` by a barrier, to pin it.
- **Two hooks rather than a flag.** `_prepare` / `_finish` on `_PoolElement`, replaced by the
  one subclass that transforms its payload. An `isinstance` or a `decodes: bool` inside the
  shared method is the switch statement CONVENTIONS 2.3 refuses, and it would also have put
  the geometry back on the element.
- **Geometry is resolved, never guessed.** The model's `config.yaml` is the source of truth
  for the input extent and the output names; this slot's `params: {decode: {...}}` overrides
  it. A model that declares neither a static `(3, H, W)` input nor an unambiguous row output
  stops the deploy. Defaulting to 640×640 is the one answer that cannot be allowed: a
  dynamic-shape engine accepts the wrong extent and every box on every camera is silently
  wrong. The count output is resolved *first*, so `output0` + `num_detections` reads as an
  ordinary detector rather than an ambiguous one.
- **The ops gating reuses `Runner.needs_model_pool` for the runner half.** What that attribute
  declares is "this runner calls `open()` on these elements in this process", which is the
  condition for every dependency an element is handed; the name is the pool's only because the
  pool needed it first, and its docstring now says so. The *element* half is a separate
  declaration, because the two come apart — a chain of `pool` embedders needs a pool and no ops,
  and the first element that crops without running a repository model will need ops and no pool.
  The two predicates are one function over a `{keyword: attribute}` table rather than two copies
  of four lines, because phase D adds a third dependency (the DataPool).
- **`tests/runners/test_pool_element.py` now asserts the shared behaviour through
  `PoolSegment`.** Testing it through the one subclass that overrides both hooks would be
  testing the override.

**Evidence.** Offline tier only, no GPU touched: `2528 passed, 1 skipped, 60 deselected` in
160 s; `check_layers.py` 0; `pre-commit run --all-files` 0 Failed. Revert-checks: deleting
the un-letterbox arithmetic in `decode_detections` turns 6 decode tests red and nothing
else; returning `True` unconditionally from the `ops` row of `dependency_is_needed` turns 3
gating tests red; handing a bare `get_image_ops(...)` back to the two runners turns 5 wiring
tests red across `run` and `shard`; deleting the `_do_open` artefact cross-check turns exactly
the two new start-up refusals red; dropping `extents`, the uint8 check and the mock's class-id
lookup turns 5 more red.

---

## 2026-08-28 — the seam phase C's elements sit on: the shipvision bridge, `needs_model`, `ElementContext`, camera hooks (Phase C2)

**What.** Four seam changes, no new element. ADR-017's `Element` ABC grows two camera hooks
and two model declarations; `ElementContext` grows three fields. ADR-017 §4 is amended in
`DECISIONS.md`, because the rule it states is the one this slice replaced.

| Piece | Delivered |
|---|---|
| `topology/bridge.py` | the ONE site that names `shipvision`: `load_mot/load_mtmc/load_reid/load_types`, each importing its subpackage **inside** the function under `functools.lru_cache`, plus `shipvision_available()`. A missing submodule is a `ConfigurationError` carrying `git submodule update --init 3rdparty/shipvision && pip install -e 3rdparty/shipvision` — `pipeline/graph/tracking.py`'s wording, so an operator meets one sentence and not four. `topology/__init__.py` imports it, so the package-level import-cheapness subprocess covers it |
| `topology/chain.py` + `runners/inprocess.py` | both readers of "does this element run a repository model" now ask the **element**, not the kind — and they ask two *different* declarations, `Element.requires_model_name` and `Element.needs_model`. `MODEL_KINDS` is deleted |
| `topology/base.py` | `ElementContext.metrics` / `.workers` / `.ops`, and `ImageOpsLike` / `LetterboxLike` — a structural stand-in for `runtime.ops.base.ImageOps`, which a pure layer may not import. `InprocessRunner.element_context()` fills the first two; `ops` waits for C3 |
| `topology/base.py` + `runners/inprocess.py` | `Element.camera_added` / `camera_removed`, base no-ops, called best-effort by `add_camera` / `remove_camera` / `drain` inside `_lifecycle` |
| `scripts/hooks/check_layers.py` + `tests/test_architecture.py` | `shipvision` banned in `core`, `scheduling`, `repository` (both tables, asserted equal); `pipeline` may now import `topology`, one-way |

**Why.** C4/C6/C7 move `TrackerShard` in and add three stateful elements. Every one of them
needs the submodule, a per-camera lifecycle and the runner's resolved settings, and none of
those can be added from inside an element module. Doing them first keeps each later slice to
one element.

**Decisions.**

- **The requirement is the implementation's, not the kind's — and it is two declarations,
  not one.** `Element.requires_model_name` is the *chain file's* question ("must this slot
  name a `model:`?"), read by the loader. `Element.needs_model` is *this process's* ("will
  `open()` resolve that name against `ElementContext.models`?"), read by the walk's expiry
  gate and by whatever decides to build an `InferenceServer`. They agree for every
  implementation phase C ships, which is why one attribute looked free, and they come apart at
  the first that runs its model elsewhere: an `nvinfer` detect names a `model:` artefact and
  executes it inside GStreamer — `requires_model_name = True`, `needs_model = False`. Folded
  together, that element must either refuse a correct chain or make a deepstream chain that
  forgot its `model:` load clean and fail at graph-compile time. Consequence, and it is a real
  behaviour change: `detect: {impl: mock}` with no `model:` now **loads**, because a mock
  resolves nothing. The kind-level rule was insisting on a name nobody reads. **ADR-017 §4 is
  amended in place** to say so — leaving the ADR stating the superseded rule is the same drift
  that got `MODEL_KINDS` deleted, one layer up.
- **The refusal names the element and its impl, not the kind.** `element 'detect' (impl 'pool')
  must name a \`model: <repository model name>\``. The old sentence ("`detect` is a detect
  element and needs …") told an operator that every detect element needs a model, which is
  precisely the rule that was removed.
- **A surplus `model:` stays accepted.** `ElementSpec.model` has always been "meaningless for
  the rest" and `describe()` prints it. Refusing it is a separate decision with its own blast
  radius; this slice is about the requirement, not the surplus.
- **`shipvision` is NOT in `FORBIDDEN_EXTERNAL["topology"]`.** The hook walks the AST and
  counts a function-scope import the same as a module-scope one, so that row would ban the
  lazy loaders and leave no legal spelling. The laziness is enforced by the subprocess
  assertions in `tests/test_architecture.py` and `tests/topology/test_bridge.py` instead —
  the same split the file already documents for `runners -> ingest`.
- **`ops` is a `Protocol`, not `Any`.** `topology` may not import `runtime`. `ImageOpsLike`
  names one member — `letterbox_batch`, which is all `pipeline/graph/detect.py` calls — and
  `tests/topology/test_element_context.py` compares its signature against the real `ImageOps`,
  because a stand-in that has quietly stopped matching is worse than none: it type-checks and
  fails on the first call.
- **Hook order.** `camera_added` **after** the ingest actor is placed, so a refused placement
  announces nothing — an element cannot tell a refused camera from a placed one that has yet
  to send a frame, and a duplicate `AddCamera` that reset a live camera's tracker would drop
  every track under it. `camera_removed` **after** the actor is stopped, so nothing can
  rebuild the state the hook just dropped. Both are best-effort: one element raising is logged
  with its name and the loop continues, the shape `_do_stop` uses — a tracker that cannot drop
  a shard must not also keep the camera from being removed, because ADR-018 has no second
  recovery to offer.
- **The hooks are NOT serialised against the walk, and the ABC says so.** They run on the
  caller's thread under the runner's `_lifecycle` lock, which orders the lifecycle operations
  against each other and nothing else; `submit` and the walk deliberately take no lock, so up
  to `ElementContext.workers` threads can be inside the same element's `process()` — for the
  same camera — while `camera_removed` runs. So an implementation guards its per-camera table
  with its own lock (`pipeline/graph/tracking.py`'s `_admit` is the shape), treats a removal
  racing an in-flight frame as ordinary, and returns promptly, because every lifecycle
  operation on the shard queues behind it. `tests/runners/test_camera_lifecycle.py`'s
  `TestTheHooksRunConcurrentlyWithTheWalk` forces the overlap on an `Event` and records it
  from inside the hook, so the contract is pinned rather than merely written down: the
  docstring used to promise "items already admitted are walked after this returns", which
  would have sent the first `TrackerShard.camera_removed` at an intermittent `KeyError`.
- **The announcement resolves the method on the instance.** `hook(node.element, id)` with
  `hook = Element.camera_added` runs the ABC's own no-op past every override: a loop that
  reaches every element and tells none of them anything, with nothing in a log to see. Caught
  by the recording test; the loop uses `getattr(node.element, hook.__name__)`.

**Not done here.** No element. `ElementContext.ops` is `None` in every runner in the tree and
has no producer yet: C3 wires it, in the shape `models=` already has (the CLI or the shard
resolves one implementation from `runtime.ops` and hands it to the runner). The field and its
protocol are declared now so the elements C3 lands are written against them from the first
line, and the docstring says which slice fills it. Shutdown does not announce a removal per
camera — `close()` releases everything.

---

## 2026-08-28 — the engine's lifecycle is a claim: `start()` owns a run, a teardown owns a generation

**What.** `InferenceServer.start()` now performs its entry and its exit as atomic
transitions under the same `_lifecycle_lock` `stop()` has used since #72. `_begin_start`
claims the server (refusing a second start, and waiting out then refusing a teardown in
flight), `_finish_start` publishes only if the claim is still held, `_abandon_start`
releases what a lost start had built. A generation counter rides `stop()` into
`_teardown`/`_release`, and every destructive step checks it. `stats()` reads the trace
state under the same lock and hands out a copy. Non-strict start-up re-raises the
cooperative abort — the run's own, not the flags' — instead of logging "continuing" once per
remaining model.

**Round 4 (the losing start's unwind).** The claim decided who *owns* the server and said
nothing about the thread that lost, which is the one holding models, worker threads and a
sink. Review reproduced it two ways, and the fix binds five things to the run rather than to
the flags: the models' abort predicate (`_start_abort`, generation *or* `_is_stopping` —
`_is_stopping` alone goes false again the moment the next run sets `_starting`), the publish
in `_build_and_start`, the failure path's teardown (`_stop_run(generation)`, so a losing
start does not run a whole-server teardown on the run that replaced it), the service-mesh
assignment, and the release itself — `_abandon_start` pops the table by **identity** and
stops by **ledger** instead of calling `_drain_models`. The trace sink moved with them: it is
built by `start()` but installed only by `_finish_start`, so a losing start closes the one it
built rather than leaving an open file descriptor on a stopped server and wiping the last
run's totals.

**Round 5 (the fifth consumer of "have I lost the server?", and the field that was not
carried).** The second review pass found one hole left in the run binding: `_load`'s
`strict_startup=false` skip still asked `_is_stopping()`. So a non-strict start that lost its
claim did not stop loading — it logged one ERROR with a full traceback per remaining model,
"failed to load … continuing", of a run that is not continuing and about models that did not
fail, and built a whole `Model` for each (a backend, a queue, and on a CUDA host a stream
pool and a graph cache per instance) on devices that belong to the new run. It asks
`abort()` now, the predicate the models poll. That predicate answers the **reason** rather
than a bool, because there are two of them and `Model._check_abort` printed "the server is
stopping" for both — including for the restart that overtook the start, which is the string
an operator greps. The service mesh moved next to the trace sink in `_finish_start`: it had
been published under the generation alone, a weaker question than the claim by exactly "a
`stop()` for this same run has been and gone", and in that window a torn-down server held a
live mesh whose rings a restart then stranded with its peers still writing into them.
`_publish_service_tier` is gone — `start()` carries the joined mesh as a local exactly as it
carries the sink, and `_abandon_start` stops the one a losing start built. Both process-wide
resources now obey one rule: *published by the claim, released by the ledger*. Finally
`_sink_stats` is one guarded `sink.stats()` shared by the teardown and the scrape;
`TRACE_SINKS` is a registry, the teardown had guarded its call from the beginning, and the
unguarded one was the scrape's — the KServe stats route and the metrics exporter, where a
sink that raises turns a monitoring poll into a 500.

**Why.** Review of #72 (rounds 2 and 3) reproduced `is_started == True` with zero models: a
`stop()` concurrent with the *initial* `start()` drained the table and closed the sink under
a start that published itself anyway. A readiness probe reads that as "up and serving
nothing". The trace-totals field #72 added made a second version of it — a late `_release`
overwriting a new run's numbers — and `stats()`'s two-bytecode check-and-act restored the
`{"sink": "none", "recorded": 0}` answer #72 existed to remove.

**Decisions.**

- **A generation, not a flag.** `_starting` cannot tell "still mine" from "cleared by a stop,
  and a later start set it again". Every teardown carries the run it was started for.
- **The grace period's expiry is a refusal, not a downgrade.** `_await_teardown` still
  returns on expiry (`stop()` never raises), but a `start()` on the other side of it is now
  refused with a `ServerStateError` rather than proceeding into a teardown that is still
  running. The generation check is the second line, because the first one is a timed wait.
- **The totals and the sink swap are one transition.** Split, a scrape lands between them
  and reports the null sink's zeros as a run's final sample.
- **A lost start releases what it built, never what it finds.** The first version drained
  the whole table, which is the same destructive act one door to the left: a restart is let
  through as soon as `_torn_down` is set, and that barrier is set by the teardown's
  `finally`, which knows nothing about a start still running. So the losing start stopped the
  *new* run's worker threads and left `is_started` true over an empty table — the readiness
  lie this work exists to remove, re-entered from the other side.
- **`_torn_down` means "the teardown returned", not "nobody is holding this server".** Left
  as it is, and written down in `__init__` instead of strengthened: a barrier that also
  waited for every losing start would be a wait with no bound, because a start is blocked on
  an engine load. What makes it safe is that a losing start can no longer touch a newer run.
- **A release does not publish the between-runs null sink's zeros.** `_traces` holds a
  `NullTraceSink` between runs, so a run that never reached `_finish_start` finds one — and
  publishing its zeros throws away the totals of the run before it. "0 traces recorded" is a
  claim about the workload; "we stopped measuring" is the truth.
- **The lifecycle lock is not held across a sink's `stats()`.** `TRACE_SINKS` is a registry,
  and a third-party sink that touched a file there would hold the one lock `stop()`'s entry
  transition needs — a metrics scrape able to stall a shutdown into a SIGKILL. The pair is
  snapshotted atomically and the call made outside; `is_closed` on the snapshot is what tells
  a scrape it lost the race, and the totals are published before the close, so it can just
  read them.

**Tests.** Twenty-two new offline tests in `tests/engine/test_stop_teardown.py`, every
interleaving forced rather than hammered for: `_WatchedLock` (a lock that records the threads
which had to *wait*) joins `_WatchedEvent`, `_GatedStatsServer` turns `_last_trace_stats` into
a property so a scrape can be parked inside a window two bytecodes wide, `_RecordingSink`
gained a one-shot gate inside `stats()` (the only way into the gap between `_release`'s sink
read and its publish) and a read counter, and `_threads_of` answers "whose worker threads are
these" where `_live_workers` cannot — two runs of one model name their threads identically.
Every fix was revert-checked in a detached copy: F1–F4 in round 3, in round 4 the six run
bindings, the two halves of the sink fix, the two `_release` generation checks the earlier
round left with no coverage, the teardown-owner write and the unlocked `stats()` call, and in
round 5 the non-strict skip's abort, the abort's reason, the deferred mesh install and the
guarded scrape — each red on the test whose name describes it.

**Future work (not this PR).** `InferenceServer` now carries the repository scan, the model
table, the service tier, the stats surface *and* a lifecycle state machine spread over
`_generation`, `_started`, `_starting`, `_torn_down`, `_teardown_owner` and `_lifecycle_lock`.
That state machine is the interesting part and the part with no direct test — every test here
reaches it by monkeypatching a private method. A `_RunState` object owning those six fields
and answering `claim()` / `publish(gen)` / `release_for(gen)` would be testable with no
threads and no repository at all, and round 4's blocker is the kind of bug it would have made
obvious.

---

## 2026-08-27 — a dead shard's cameras are reported lost, never re-placed (Phase B4)

`Fleet.dead_indices()` names exited shards by plan index and `runners.fleet._lost_in()` maps them
to their cameras, which `health()` (`lost`, excluded from the per-shard `placed` lists in the
same snapshot), `stats()` and `StreamInfo.lost` now carry; `remove_camera` on a lost camera
drops the placement and answers `False`, `add_camera` skips dead shards, and `drain()` keeps
an in-flight reservation instead of clearing it out from under an `AddCamera` that was about
to commit. Why loss is reported rather than repaired: ADR-018.

---

## 2026-08-27 — `/streams`: the camera door, over a runner (Phase B3)

**What.** `shipinfer run --topology c.yaml --http` now serves arch.md §2's camera door:
`POST /streams {"url": "rtsp://..."}` starts a camera on the runner, `DELETE /streams/{id}`
stops it, `GET /streams` (alias `GET /cameras`) lists what the runner says it is reading,
`POST /streams/drain?timeout_s=` empties the deployment without tearing it down, and
`GET /health` answers 200 with the state in the body. Five pieces:

| Piece | Delivered |
|---|---|
| `core/errors/launch.py` | `NoShardAvailableError(ServerStateError)`, carrying the camera id and every shard's refusal; `runners/fleet.py` raises it where it raised `ConfigurationError` |
| `api/streams.py` | the five-member `CameraController` protocol + `build_streams_router`, with the status-code mapping and the minting of `cam-<n>` |
| `api/errors.py` | `routes.py`'s `_fail` extracted, so both routers share one table |
| `api/app.py` | `create_app(server=None, *, cameras=None)` mounts whichever routers it was given; `BackgroundHttpServer` runs uvicorn on a thread |
| `cli/commands/run.py` + `cli/__init__.py` | `--http/--host/--port`, and `_wait` supervising with the ingress up — *confirmed* up |

**Why.** arch.md §2 draws two doors into the deployment and only one of them existed. Cameras
could reach a shard over gRPC or `--inputs` at start-up, and a running system could not be
given a fifty-first camera by anything but a restart.

**Decisions.**

- **`run --http`, not `serve --http`.** `run` is the composition root that owns a runner, and
  a runner is the only thing that owns cameras. `serve` builds an engine and no runner; with
  `--runner fleet` there is no engine in the parent at all. So the two commands serve
  different routers, and `create_app` mounts what it was handed rather than assuming both.
- **`api` may import `launch`, and may NOT import `runners`.** The grant is `CameraSpec` and
  `mint_camera_id` — the launcher's vocabulary, which is what `add_camera` takes. What is
  behind the routes arrives as the structural `CameraController` (five members), so an HTTP
  handler can drive the runner it was handed and cannot build one, choose a placement or open
  a chain. Both halves are asserted: the table in `tests/test_architecture.py`, and a
  subprocess that imports `shipinfer.api` and refuses if `shipinfer.runners` came with it.
- **A duplicate id is 400; a fleet with no room is 503.** They were one error, so a control
  plane reading 400 as "my request is malformed" stopped asking about a condition that clears
  as soon as a shard finishes draining. `NoShardAvailableError` is what splits them, and it is
  a `ServerStateError` so the existing rule in `api/errors.py` maps it without a new case.
- **`clean=false` is a body signal, never a 5xx.** `DELETE` removes the camera whatever the
  decoder thread does; a 500 would say the removal failed, and the retry would earn a 404.
- **A runner that manages no cameras is 501.** Its own refusal is a `ServerStateError` (a
  retryable 503), and no amount of retrying gives a `deepstream` runner an ingest plane.
- **`add_camera` runs in a worker thread with a request deadline**, mirroring `routes.py`'s
  `_INFER_TIMEOUT_S`, with `abandon_on_cancel=True` — without that, anyio's cancel scope waits
  for the very thread it is cancelling and the deadline is decorative.
- **A malformed body is refused by the schema, not by a layer below it.** `StreamRequest`
  constrains `url` (non-empty, not whitespace), `fps` (`>= 0`) and `camera_id` (no
  whitespace; `""` still means "mint one for me"), so FastAPI answers 422 naming the field
  before the handler runs. Not a *mirror* of `CameraConfig`'s validators but the same
  predicate: `core/settings/ingest.py::usable_camera_id` was lifted out of the validator so
  the door and the record cannot drift — which they had, `camera_id` being the field the
  argument was written for and the one it was first missing. Without
  them the first thing to inspect those values was `CameraConfig`, whose refusal is a
  *pydantic* `ValidationError` — a `ValueError`, not a `ShipInferError` — which fell past the
  typed mapping: `{"url": ""}` was a **500** in process and, over gRPC, a refusal from every
  shard → `NoShardAvailableError` → a **retryable 503** for a request that can never succeed.
  `add_stream` also maps a leaked `ValueError` to 400 as the net under the other eighteen
  fields of the settings tree.
- **`start()` confirms the bind before the deployment is allowed to look healthy.**
  `uvicorn.Server.startup` answers a taken port by logging the `OSError` and calling
  `sys.exit`, and off the main thread `threading.excepthook` discards the `SystemExit`
  without a word. So `--http --port 8000` against a taken port spawned every shard, placed
  every camera, logged *"serving /streams on ..."*, ran with no ingress at all and exited
  `0` — nothing in the process had a reason to say otherwise. `BackgroundHttpServer.start`
  now polls uvicorn's own `started` flag for `bind_timeout_s` (5 s, a constructor argument)
  and raises `ConfigurationError` naming `host:port`; the INFO line moved below the wait so
  it can no longer assert something untrue. Raised from `_wait` before `supervise()`, it
  travels through `run()`'s existing `finally`, so the runner stops and the command exits
  non-zero.
- **A health report that cannot be fetched is 503 on the write path, 200 on the read paths.**
  `_health` is lenient for every listing — a listing that 500s because one shard is
  unreachable is useless exactly when it is wanted — but `_mint` *acts* on that report, and
  the lenient stand-in carries no `cameras` key, which does not mean "none are running". A
  deployment with fifty cameras up and an unreachable control plane minted `cam-000` and
  answered a **400 naming an id the caller never supplied**: a control-plane fault reported
  as the client's mistake, and terminal, so a well-behaved client stopped retrying. The read
  that feeds the mint now passes `needed=True` and raises `ServerStateError` → 503. A POST
  that supplies its own `camera_id` needs no report and is still placed.
- **`--host`/`--port` without `--http` are refused, not ignored.** Both configure the one
  thing `--http` starts; accepted silently they are a deployment that looks configured and is
  not. Both typer options carry a `None` sentinel rather than their real defaults, so
  `--host 127.0.0.1` typed out in full is still refused and an unmentioned flag is not.
- **The re-mint fires on `DuplicateCameraError` and on nothing wider.** A server-minted id can
  be taken between the report it was read from and the add that uses it, and that one refusal
  is retried under a fresh name. On a bare `ConfigurationError` an unrelated refusal — an
  unregistered source — did the whole add twice before answering the same 400, so the
  duplicate got its own type in `core/errors/config.py` and both raise sites use it.
- **`StreamRequest.loop` reaches `CameraSpec.loop`.** `--inputs` has `--no-loop`; without the
  field a client that posted a finite video over HTTP got it replayed forever. `StreamInfo`
  deliberately does not echo it back — no runner's health carries it, so the answer would be
  `true` for every camera including the one that asked for `false`.
- **A camera's priority band travels on its spec (B5).** New wire vocabulary: `CameraSpec.priority`
  and `shard.proto`'s `CameraPriority` enum (`CAMERA_PRIORITY_UNSPECIFIED` = "the launcher said
  nothing", never a lane), plus a `priority` field on `POST /streams` taking the band by
  **name**, in any case — `Literal["tracking_critical", "high", "normal", "background"]` over a
  before-validator that lower-cases a string, derived from
  `core.request.Priority`, because an `IntEnum` on the wire would publish integers in
  `/openapi.json` that the validator refuses and would let `{"priority": 0}` mean
  `tracking_critical`. A fleet shard's `ingest.cameras` is stripped, so a band an operator
  configured had nowhere to be resolved from once the camera crossed the process boundary;
  `cli/commands/run.py` now reads it where it is still true. On the runner the launcher's band
  and the configured table are two dicts with two lifetimes, so a removed camera's lane does
  not outlive it (`runners/inprocess.py::_priority_for`) — and a placement that is *refused*
  restores the band it recorded, so a 400 cannot re-lane a camera that is already running.
- **The camera ids are minted by one helper.** `launch/control.py::mint_camera_id` is what
  `--inputs` uses and what a `POST` with no `camera_id` uses (lowest free index), because two
  spellings of "the next camera" collide on a deployment that uses both doors.
- **uvicorn goes on a thread and `--host` defaults to loopback.** The main thread is the
  supervising thread (`launch/signals.py`), and `uvicorn.Server` installs its own
  SIGINT/SIGTERM handlers unless it is off the main thread — a Ctrl-C that stopped the web
  server and left fifty decoders reading is exactly what that would buy. Loopback because
  these routes start and stop decoding on a shared GPU box and phase B has no authentication:
  exposing them is a proxy in front, not a different default.

**Evidence.** `tests/api/test_streams.py` (46 cases over a ten-line fake controller — every
status code, and the four a malformed body earns), `tests/api/test_streams_over_a_runner.py` (11 cases: a real `InprocessRunner`
behind a `TestClient`; a posted URL's frames arrive at the sink tagged with the caller's
camera id, `DELETE` stops them), `tests/cli/test_run_http.py` (8 cases: the thread, the
config, `should_exit` on return, and SIGINT still routed to the runner). Offline throughout.

**Known gap.** `StreamInfo.url` reads `""` for any camera this process did not just place: no
runner's health report carries a source URL, and a router that remembered them would answer
from its own memory about cameras added over gRPC. A real `shipinfer run --http` against a
GPU box is container-tier evidence and was not run here.

## 2026-08-27 — the launcher places the fleet; a shard opens only what it was sent (B1 review)

**What.** Round-1 review of PR #71 found a blocking defect in the entry above and this is the
correction. `InprocessRunner._do_start` no longer starts `ingest.cameras` / `ingest.camera_db`;
`cli/commands/run.py::cameras_to_place` derives `CameraSpec`s from the settings tree and places
them — configured first, `--inputs` after — through the same `place_cameras` both already used,
so `add_camera` is the single door on every runner. `launch/supervisor.py::_NOT_INHERITED`
gains `SHIPINFER_INGEST__CAMERAS` and `SHIPINFER_INGEST__CAMERA_DB`. `CameraSpec` gains
`loop: bool = True`, carried on the wire as `optional bool loop = 4`, and `shipinfer run` gains
`--loop/--no-loop` — which supersedes the "Not done here" note in the entry below.

**Why.** A shard *is* an `InprocessRunner` (`cli/shard.py` hard-codes `build_runner("inprocess",
…)`) whose settings come from `build_settings()` with no arguments — env-only, so every child
inherited the operator's whole fleet. `UpdateTopology` → `runner.start()` → the auto-start
branch therefore opened all fifty cameras on all eight shards: 400 RTSP sessions, eight
`FrameCounter`s minting identical `(camera_id, frame_id)` tags for one camera (the ADR-002
misattribution, by construction rather than by race), and a control plane that could then place
nothing because `FleetRunner.add_camera` met "already running" everywhere. The old tests could
not see it: the shard-shaped ones configured no cameras.

**Decisions.**

- **The camera set is a launcher decision, not a property of whichever settings a process
  loaded.** Option (a) of the review, plus the defence from (b): the runner starts nothing, and
  the `IngestManager` is *built* with `cameras=[]`/`camera_db=None` so that even a future
  `start()` on it cannot open a fleet. `_priorities` is still filled from the **full** settings,
  because a band is deployment configuration keyed by camera id — a shard told `cam-7` still
  admits it into the band its config names.
- **The two `SHIPINFER_INGEST__*` names are stripped from a child.** The same argument already
  written for `visible_gpus`: a child is told one thing at `exec`, and an inherited copy of what
  an RPC now carries is worse than absent. Both halves are pinned — the supervisor test asserts
  the child cannot see them, and a shard-shaped `InprocessRunner` handed the whole fleet in its
  settings still reports `cameras == ()` with no `ingest-*` thread until `AddCamera` arrives.
- **`loop` joins `CameraSpec` rather than the help text being corrected.** An `--inputs` camera
  is minted in the CLI and appears in no `ingest.cameras` entry, so the knob the help named was
  unreachable for exactly the cameras that needed it; and now that a configured camera is
  *placed*, a fleet would otherwise have dropped the `loop: false` its operator wrote. Presence
  (`optional`) because the wire default for a bool is false and this field's default is true.
- **A decode root declaring more than one `produces` is refused.** Every decode element hands
  the frame on untouched, so the cap the sink stamps is a claim about a buffer nothing converts;
  with two declarations the loader picks whichever the consumer prefers and stamps *that* on the
  same array. Refused in `_head()` with the reason; a converting decode is phase D. The
  consequence for the test below it is stated rather than hidden: with one `produces` the edge
  and the declaration agree by construction, so "read the edge, not `output_caps[0]`" is now
  pinned by the refusal instead of by a difference.

**Not done here.** `CameraSpec` still carries no `priority`, so a camera placed on a *fleet*
shard whose environment no longer names the fleet is admitted at `NORMAL` unless that shard's
own settings configure it. The bands still work for `inprocess` and for any shard given the
config by other means; carrying the band on the wire is a wire change with a falsy-zero trap in
it (`TRACKING_CRITICAL == 0`) and belongs in its own PR.

---

## 2026-08-27 — the runner owns the cameras: decode elements, `ChainFrameSink`, `--inputs` (Phase B1)

**What.** `shipinfer run --topology c.yaml --inputs a.mp4 b.mp4` now opens the videos, and a
shard's `AddCamera` RPC starts a real camera actor. Five pieces:

| Piece | Delivered |
|---|---|
| `topology/elements/decode.py` | `ReplayDecode` / `GStreamerDecode` / `PyAvDecode` — a `source` ClassVar naming an entry in `SOURCES`, `produces = ("bgr@cpu",)`, and `item.derive()` at walk time |
| `runners/frames.py` | the `TaggedFrame` protocol and `ChainFrameSink`: frame → `ChainItem(context, head caps, Tensor.from_numpy(frame.as_batch()))` → `submit` |
| `runners/inprocess.py` | `manages_cameras = True`; `add_camera`/`remove_camera`/`drain`/`cameras` over an `IngestManager`; `_head()`; `_camera_config`; `_priority_for`; `_do_health["cameras"]`; `_do_stats["ingest"]` |
| `cli/commands/run.py` | `cameras_from_inputs` + `place_cameras`, and the deletion of the `--inputs` refusal |
| `check_layers.py` + `tests/test_architecture.py` | `runners -> ingest`, granted statically and costed dynamically |

**Why.** Phase A2 left a runner that executed a chain nobody could feed: `--inputs` raised
"not wired yet", and `InprocessRunner.add_camera` was the ABC's typed refusal. arch.md §2 has
two doors into the system and neither one worked.

**Decisions.**

- **The runner owns the cameras; the decode element only names a source.** The rejected
  alternative — a decode element that opens its own camera — would drag the camera set and the
  admission door into `topology`, which has to stay pure enough to validate a chain on a
  laptop. So `decode: {impl: replay}` is two declarations (a source name and the chain's head
  cap) and a pass-through, and everything with a thread in it lives in `runners/`.
- **`runners` may import `ingest`, but only inside a method.** `shipinfer.ingest` reaches
  `sources/gstreamer.py` and `shipinfer.runtime` (and, through it, torch on a host where a
  device source is importable — measured on this box, `import shipinfer.ingest` pulls
  `shipinfer.runtime` and not torch), and `import shipinfer.runners` must cost none of them.
  `check_layers.py` grants the edge and cannot see the difference between a module-scope
  import and a function-scope one; `tests/test_architecture.py` adds
  `shipinfer.ingest` to the heavy list it refuses in a subprocess, which is the half that can.
  Both are needed and the hook's comment says so.
- **The head cap comes from `Topology.edges`, never from `root.element.output_caps[0]`.** A cap
  belongs to an edge; an element with two `produces` hands a different one to each consumer.
  A chain whose decode roots disagree — on the cap or on the source — is refused at `start()`,
  because one ingest manager publishes one item and every root sees it.
- **`_do_health()` emits `"cameras"`, and that key is load-bearing.** `ShardService.state()`
  derives `running` from it, so the previous runner would have answered `ready` forever while
  reading fifty cameras. Asserted across both files, over a real `InprocessRunner`.
- **`_do_submit` finally passes a priority.** It was left at the default, so `priority:` on a
  camera applied to nothing and every camera shared one lane — the one customisation ADR-005
  says a generic server cannot express, configured and then ignored. Resolved per camera from
  `IngestManager.configured_cameras()`, with `is not None` and never `or`, because
  `TRACKING_CRITICAL` is `0`.
- **One dropped frame is counted twice, deliberately.** `items_dropped{camera}` at the
  admission door and `ingest_frames_dropped{camera,reason=sink_full}` in the actor answer two
  different operator questions; the pair is documented in `runners/frames.py` and asserted in
  `tests/runners/test_camera_lifecycle.py`.
- **Start opens elements, then workers, then cameras; stop releases cameras first.** Cameras
  are the producers, so joining workers while frames keep arriving is a shutdown racing its own
  input. They get half the shutdown budget against the *same* deadline, so a wedged decoder
  cannot spend the time the workers need. The manager is dropped at the stop rather than
  reused, for the reason the queue and the stop event are rebuilt per cycle.
- **The sink discards the future, and the sink calls `_do_submit`.** An actor cannot wait on a
  future without becoming the chain's pacer; and the manager is started by `_do_start`, which
  runs before `Runner.start` publishes `_running`, so routing through the public `submit` would
  hand a camera a `ServerStateError` the `FrameSink` contract does not name.

- **The three camera methods take the lifecycle lock, and the manager is built lazily.**
  Found in review, and the two are one fix. `add_camera` read `_running` outside
  `Runner._lifecycle` while `_ingest()` builds a manager unconditionally, so an add that
  passed the check just before a `stop()` cleared the flag built a *fresh* `IngestManager` on
  a torn-down runner and started a decoder thread into it — which nothing then stops, because
  `_stop_ingest` has already run and a second `stop()` returns at the idempotence check. B3's
  `POST /streams` calls this from a threadpool, so the race is ordinary rather than exotic.
  `add_camera` / `remove_camera` / `drain` now hold the (re-entrant) lifecycle lock and
  `add_camera` re-checks `_running` under it; `submit`, `health`, `stats` and `cameras`
  deliberately still take nothing.
- **`_do_start` starts the ingest manager only when cameras are configured.** It used to call
  `self._ingest().start()` unconditionally, so every start — including a chain of mock
  elements with no camera in it — imported `shipinfer.ingest` and `shipinfer.runtime`, which
  is the whole cost `_NO_INGEST` and the method-scope import exist to avoid. First use is now
  `_do_start` when `ingest.cameras`/`ingest.camera_db` says so, and `add_camera` otherwise;
  a subprocess test asserts a started, camera-less runner has neither a manager nor the
  modules.

**Not done here.** No `csrc` change: the native ingest halves already exist and are reused
unchanged, and `runners/` has no native mirror. `_camera_config` carries no `loop:` — a
`CameraSpec` has three fields, so a replayed file loops by `CameraConfig`'s default. `--http`
and `POST /streams` are B3.

## 2026-08-27 — `QueueStats` names the camera that paid for each drop (both planes)

**What.** `scheduling.queues.QueueStats` gains four `Mapping[str, int]` fields —
`depth_by_camera`, `rejected_by_camera`, `evicted_by_camera`, `expired_by_camera` — and
`as_dict()` carries them onto the wire. `FairPriorityQueue` and `FifoQueue` count at all three
drop sites (refusal at capacity, `DROP_OLDEST` eviction, expiry at the drain); `Lane.depths()`
is the fair queue's half of the depth walk. `csrc/shipinfer/scheduling/queues/` mirrors it:
`base.h` gains the two maps it was missing, both queues count at the expiry site, and
`Lane::add_depths` fills the breakdown.

**Why.** The totals said a queue refused, evicted or expired work. They could not say *whose*,
and that is precisely the question ADR-005 exists to answer: the inherited failure was observed
per camera — "camera đông người được nhận diện đầy đủ, camera vắng người thỉnh thoảng bị miss",
the crowded cameras recognised in full while the quiet ones occasionally miss — and a per-queue
counter can neither confirm nor refute a per-camera claim. Under `DROP_OLDEST` the fair queue's
victim is the greediest camera *by construction*, so `evicted_by_camera` is now the direct
evidence that the eviction inversion works, rather than a property asserted in a docstring.

**Decisions.**

- **Keyed by `WorkItem.fairness_key`, so a camera-less caller lands in `"-"`.** That is the same
  lane the drain already puts it in; a second, subtly different notion of identity in the stats
  view would make the two readings of one queue disagree.
- **All four maps default to empty.** A queue that cannot attribute an outcome — a third
  implementation, a compiled adapter — constructs unchanged and reports nothing. Reporting a
  zero it never measured would be worse than silence.
- **`close()` feeds none of them.** Shutdown loss is not a per-camera fault and the runner's
  `items_queue_closed` already owns that outcome; charging it here would make an orderly stop
  read like a flood in the one view an operator uses to find floods. Keeping that promise took
  a code change on the `BLOCK` path in both planes: a producer asleep in the make-room wait is
  woken by `close()` as well as by the timeout, and the two exits were indistinguishable, so a
  shutdown charged `rejected_by_camera` and raised `QueueFullError("full (0/1)")`. The closed
  exit is now named before any counter moves — `RequestCancelledError` in Python,
  `PutStatus::Closed` in C++.
- **`depth_by_camera` is computed inside `stats()`, not maintained.** O(cameras x priorities) —
  200 dict entries at the design point — once per stats call, against bookkeeping on a path that
  runs 15 000 times a second. Same trade in both planes.
- **`stats()` and `as_dict()` hand out copies.** `/v2/statistics` serialises this document and a
  health handler nests it into its own; either is free to trim what it was given, and neither
  may be editing a live queue's counters by doing so.
- **`peak` remains a C++-only field.** Noted, not fixed: closing that parity gap is its own
  change and belongs with the parity harness, not with this one.

**Not covered here.** The runner's `_do_stats` per-camera identity (`items["per_camera"]`) is a
follow-up: `runners/inprocess.py` was under concurrent edit, and the queue's attribution is
useful on its own through `/v2/statistics`.

---

## 2026-08-27 — `server/` dissolved: `engine/` + `api/` + `launch/` + `runners/`, and a gRPC control plane (Phase A2, PR-1…PR-6)

**What.** The package `server/` no longer exists. Its parts moved to the seams arch.md §9 names,
in six PRs that each kept the offline tier green:

| PR | Delivered |
|---|---|
| 1 | `engine/` — the model pool moved whole (`pool.py`, `model`, `instance`, `ensemble`, `statistics`, `health`, `cache/`, ADR-015's rings under `engine/spill/`), mirrored in `csrc/shipinfer/engine/` |
| 2 | `api/` — the KServe v2 surface; the one layer that may import fastapi |
| 3 | `runners/` — the `Runner` ABC, `RUNNERS`, `inprocess.py`, and `topology/elements/pool.py` |
| 4 | `launch/` — `Fleet` supervision, moved verbatim, without gRPC |
| 5 | the control-plane contract — `launch/proto/shard.proto` + committed stubs, `ShardClient`, the transport-free `launch/control.py`, `runners/service.py`'s servicer |
| 6 | `runners/fleet.py` over that contract, `cli/shard.py`, `shipinfer run`, and the deletion of the argv mechanism and `server/` itself |

**Why.** Two reasons, and the second is the one that changed behaviour. The tree is meant to be
the architecture — a reader should find every §-heading of `docs/arch.md` as a directory — and
`server/` was four unrelated things in one name: a model pool, an HTTP surface, a process
supervisor, and a set of classes that rendered command lines. And the word "topology" meant
*placement* there while arch.md §1 uses it for the element chain, a collision the operator
called out by name (V129/V132).

The behavioural half is **V140**: *"xóa luôn cách dùng gọi command giữa 2 tiến trình"*. A shard
used to be configured once, at `exec`, by an argv string and a set of environment variables. It
is now spawned with `--shard-id N --control-port P` and told everything else over gRPC —
`UpdateTopology`, `AddCamera`, `RemoveCamera`, `Health`, `Stats`, `Drain`, `Stop`. What that
buys is concrete: a camera can be added to or removed from a *live* shard; health is a typed
answer rather than a scraped log; a shard's state is `ready` vs `running` vs `draining` instead
of an inference from an exit code. vLLM's engine-core split is the pattern reference — processes
talk RPC, nothing meaningful rides argv.

**Decisions.**

- **`CUDA_VISIBLE_DEVICES` stays in the spawn environment, alone.** It has to be set before the
  child imports torch, which is several frames below the first RPC it could answer. That is the
  whole boundary of V140. The four variables that used to ride beside it are now *removed* from
  the child's environment rather than merely unset: an inherited
  `SHIPINFER_DEVICES__VISIBLE_GPUS` naming physical ordinals would fail a child whose devices
  the remap renumbered, with a configuration that is correct for a single-process run.
- **The sharing travels in `UpdateTopology`.** `shared_by`/`share_rank` decide how many
  instances of each model a shard loads (`ModelConfig.placements`), so two shards on one GPU
  each load half. A shard never told loads the full count and the device silently holds twice
  the engines for the same throughput — the assertion `tests/server/test_shard_settings.py`
  made of the environment is now made of the RPC, in `tests/cli/test_shard_entry.py`.
- **The stubs live in `launch/proto/`, not `api/`.** `api` imports `launch` in phase B so
  `POST /streams` can reach the shards; stubs under `api/` would make `launch` import `api` and
  close the cycle. The servicer is `runners/service.py` because it holds a runner and a launcher
  must not — `launch` may not import `runners`, and an architecture test asserts the direction.
- **`cli/shard.py` is the child entry point** for the same reason inverted: it is a composition
  root, building an engine, a topology and a runner, and neither `launch` nor `runners` may
  import all three. `cli` is the layer whose job is that wiring.
- **grpcio and protobuf are an optional extra.** Nothing imports either at module scope; the
  first call on a client raises a `ConfigurationError` naming the extra, the shape `api/app.py`
  uses for FastAPI. `import shipinfer.launch` works on a host that has neither.
- **Two `core/` renames** carried the vocabulary: `core/settings/topology.py` → `runner.py`
  (`TopologySettings.kind` → `RunnerSettings.runner`, section `settings.runner`, env prefix
  `SHIPINFER_RUNNER__`) and `core/errors/topology.py` → `core/errors/launch.py`, which also ends
  its collision with `core/errors/chain.py`'s `TopologyError`.
- **`shipinfer fleet` → `shipinfer run --topology <chain> --runner <name>`.** The old command
  took a model repository and a placement; the new one takes the chain and the runner, and
  names neither in its body — `--shards` is `runner.shards`, `--drain-s` is `runner.drain_s`
  and `--gpus` is `devices.visible_gpus`, so a third runner needs no edit there. Every flag
  the old command had has a home: `--drain` is `--drain-s`, under its settings-tree name.
- **`cli` gained an `ALLOWED_INTERNAL` row.** A *missing* row switches the internal layering
  check off for that package, silently; `cli` had none, so it could have imported anything.
  Three architecture tests now pin it: every package has a row in both tables, no row names a
  package that is not on disk, and nothing below the command line may import it.
- **Supervision is on the `Runner` contract, not probed for.** `request_stop()` records (it is
  a signal handler's whole job), `supervise()` blocks, `describe_plan()` answers `--dry-run`;
  the fleet overrides the last two. `shipinfer run` used to `getattr` for both, which would
  have silently downgraded a renamed fleet method into a runner that never watched its shards.
  `launch/signals.py::forward_signals` is retyped on a one-method `Stoppable` protocol and has
  a production caller for the first time.
- **`start` owns the only unwind.** `FleetRunner._do_start` had its own, so a failed start ran
  two release passes and the second — over an already-emptied client map — *assigned* its zero
  over the count of camera threads the first had abandoned. A fleet with six detached decoders
  reported none, which is the single lie that signal exists to prevent. `_do_stop` is the one
  owner, `_unwind_timeout_s()` is how a subclass says what budget that pass gets (a fleet's
  release is a `Stop` RPC per shard, not a local close), and the counts accumulate.
- **The fleet's lock is never held across an RPC.** A camera is placed by reserving it under
  the lock, asking the shard with the lock released, and committing under it again; `health`
  and `stats` snapshot the two maps under it. `AddCamera` starts a decoder and can sit for
  seconds on an RTSP source that is not answering, and a health probe that waited behind it
  would make the one call an operator reaches for during an incident the one call that hangs.
- **A shard's installs run in parallel.** Each is a `wait_ready` poll plus an `UpdateTopology`
  that deserialises that shard's engines — both waits on another process. Sixteen sequentially
  is an eight-minute deployment turned into two hours with every GPU but one idle. The pool is
  joined before anything is inspected, and the failure re-raised is the first in *plan* order,
  so a fleet fails the same way twice.
- **A retired environment section is refused, not ignored.** `extra="forbid"` does not catch
  `SHIPINFER_TOPOLOGY__SHARDS`: pydantic-settings' environment source only emits keys for
  fields that exist, so the model never sees it and the export is silently unread — an
  operator's pinned process count quietly replaced by the default. `RETIRED_ENV_SECTIONS` is
  a table of old→new names, and the settings tree refuses at start-up with the key named.
- **The wire's zero timeouts read as defaults.** proto3 has no field presence for scalars, so
  an unset `timeout_s` and a deliberate `0.0` are the same bytes; read literally, a client that
  omitted the field asked a shard to detach every camera thread and report a fleet-wide
  lifetime signal for an ordinary shutdown. The servicer clamps, and `shard.proto` says so for
  the other-language clients that read the `.proto` and never this package.
- **A `Drain` that failed does not read as `drained`.** The flag was set in a `finally`, so a
  drain whose runner raised left the shard claiming it had released cameras it was still
  reading — and a launcher acts on that by placing them elsewhere. `drained` now means
  *released*; the reason is in `DrainReply.detail` and `ShardService.drain_detail`.
- **`grpcio-tools` is pinned**, the only pin in `pyproject.toml`: `gen_proto.py --check`
  compares regenerated stubs byte for byte, which is a guard only while every machine runs one
  protoc. It is what already resolved by accident, made deliberate.

**Capabilities temporarily lost, and where they come back.**

- **`deepstream` as a first-class placement.** `DeepStreamTopology` rendered a `shipinfer
  deepstream` command per shard; the command itself remains and is now hand-run over the
  configured cameras. It returns in phase E as a *runner* that compiles the chain into a
  GStreamer graph.
- **The `service` tier's two-process run.** `tests/engine/test_service_multigpu.py` is skipped:
  a shard has no supported way to be told its peers before it starts until phase D's `JoinMesh`
  RPC. The tier itself is unchanged and still covered offline and on one GPU.
- **A fleet of KServe servers.** `shipinfer fleet` spawned `shipinfer serve` children, each
  answering HTTP. A fleet's children run a *chain* now; `shipinfer serve` is still the
  single-process model server, and `/streams` reaches a fleet in phase B.
- **Running the shipped `topology/ship_person.yaml`.** It names `gstreamer-gpu`, `shipvision`
  and `kafka` element implementations that arrive in phases C/E; today the loader refuses it by
  name, which is the refusal working.

**Evidence.** Offline tier green at every step; on the final rebase, 2093 tests collected on
`main` against 2120 on the branch (2059 passed, 1 skipped, 60 deselected) — six deleted
`tests/server/` files against the new `tests/runners/`, `tests/cli/` and `tests/launch/` ones.
`pre-commit run --all-files` clean; `scripts/hooks/check_layers.py` exit 0;
`scripts/gen_proto.py --check` reports the committed stubs current. The GPU tier is not
evidence for this phase — nothing here touches a kernel — but `-m gpu` and a `shipinfer serve`
smoke belong to the release that ships it.

---

## 2026-08-27 — `runners/inprocess.py`: the batch a stale worker must not finish, and a ledger with no caveat

**What.** Three follow-ups to the entry below, from the review of #62. (1) `_work` read the
stop signal only at the top of the outer `while`, so a worker abandoned at a shutdown deadline
finished its **whole** wake-up batch when whatever wedged it let go — up to
`frames_per_wakeup - 1` ghost events emitted through a chain a restart had re-opened, for
futures `_fail_in_flight` had already resolved. The signal is now read in front of every item
and the remainder is left in the slot, which is where the drain has already found it. (2)
`_do_stop` drained the in-flight slots but kept `self._inflight` pointing at the list the
abandoned worker still holds; its `finally` republished the remainder there, so
`stats()["items"]["in_flight"]` came back up after the stop and never came down. The stopped
cycle's list is now *replaced*, not merely emptied. (3) `items_dropped` counted two
populations — an admission refusal (never `accepted`) and a `pool` element's model queue
refusing mid-walk (`accepted`). Split: the new camera-labelled `items_backpressure` takes the
mid-walk half, `items_dropped` stays admission-only.

**Why.** (1) and (2) are the same failure as the abandon/restart bugs below, one level down: a
worker the runner has stopped tracking must be inert on its next turn, and nothing a stopped
cycle owns may still be read. (3) is what let the ledger identity drop its correction term —
an operator who has to subtract `queue["rejected"]` before the numbers add up will not.

**Contract change.** `RunnerMetrics` gained `items_backpressure`
(`shipinfer_runner_items_backpressure_total`) and `totals()` gained `backpressure`.
`stats()["items"]` therefore carries both keys, and the documented identity is now
`accepted == walked + failed + expired + timed_out + backpressure + queue_closed +
queue_evicted + queue_expired + in_flight`, with `dropped` deliberately outside it and two
honest caveats left (an abandoned worker counted twice; the queue's own terms resetting on a
restart). A dashboard that graphed `shipinfer_runner_items_dropped_total` as "all
backpressure" now needs both series.

---

## 2026-08-27 — `runners/inprocess.py`: the failure a submitter sees, and a ledger that adds up

**What.** Three follow-ups to the runner above, from the review of #61. (1) `_walk` re-wrapped
**every** element exception in `InferenceError`, flattening the `ShipInferError` family — a
`QueueFullError` from a saturated `pool` element lost its depth and capacity, a
`RequestTimeoutError` became indistinguishable from a bug, and `pool.py`'s "propagated
untouched" promise was false. One of ours now travels as itself (`_typed`) and is charged to
its own counter: `items_dropped` for backpressure, the new `items_timed_out` for a stage
timeout, `items_failed` for everything else. (2) The per-worker in-flight slot list was
**rebound** on every `_do_start` while workers read it off `self`, so an abandoned worker from
cycle one wrote its "nothing left" into cycle two's list at the same index — abandon, restart,
abandon left a live worker's futures unresolved. The list is built in `_do_start` and passed to
`_work(slot, inflight)`. (3) `stats()["items"]` counted only what the *runner* resolved; items
the queue failed on its own (`close()`, `drop_expired`, `DROP_OLDEST` eviction) had typed
futures and no counter, so `accepted` outran the sum of the outcomes. `items` now carries
`queue_closed` (a camera-labelled runner counter, because the queue keeps no such total),
`queue_evicted`, `queue_expired` and `in_flight`.

**Why.** All three are the same failure in three places: a producer holding a future cannot act
on an outcome the runner refuses to name. Shed-load, add-capacity and open-a-ticket are three
responses, and one `InferenceError` picks none of them.

**Contract change.** `InprocessRunner.stats()["items"]` gained five keys and is documented with
the identity it satisfies — `accepted == walked + failed + expired + timed_out + dropped +
queue_closed + queue_evicted + queue_expired + in_flight` — together with the three ways it
does not hold (`dropped` counting both admission refusals and mid-walk backpressure, an
abandoned worker counted twice, the queue's own terms resetting on a restart). `in_flight` is a
gauge and lags by at most one wake-up batch in either direction; the test helper `settled()`
polls it to zero before asserting the ledger.

---

## 2026-08-27 — `runners/`: the in-process runner, and the chain's client of the model pool (Phase A2, PR-3)

**What.** `src/shipinfer/runners/` — the third of arch.md's three concepts (§1). `Runner` is a
template-method ABC over one validated `Topology` (`start` idempotent and unwinding a partial
start, `stop` on one shared deadline, `submit` refusing before `start`, `health`/`stats`,
context manager) plus `RUNNERS`/`build_runner`. `runners/inprocess.py` is the first
implementation: the fair bounded lane of §5② in front of N workers, each of which walks **one**
item through `topology.nodes` in topological order (§5③), skipping what `node.admits` rejects
and merging a fan-in deterministically. `topology/elements/pool.py` adds the `pool`
implementation of all four model kinds — one request per item, the model resolved once at
`open()` — and `InferenceServer.get(name)` is the one-method `ModelResolver` it reaches it
through.

**Why.** A topology that nothing executes is a data structure. This is also where the two
properties the reset is for become testable offline: admission is the *existing* fair queue
(ADR-005 — there is no second fairness mechanism), and the whole runner runs with mock elements
on a host with no driver, which is why `tests/runners/` is in the offline tier.

**Decisions.** The queue stays typed on `WorkItem`; an item is *wrapped* (a `_ChainWork`
subclass) exactly as `QueueFrameSink` wraps a frame, and the `ChainItem` is taken off it at the
top of `_walk` — the queue's lane, band and expiry are all request fields, so the wrap buys the
per-camera fair lane for free. Fan-in: metadata is the union in `node.inputs` order with
first-writer-wins, payload and caps come from the predecessor whose edge carries the cap the
element prefers, and a skipped `when:` predecessor contributes its own inbound item. An element
that raises costs one item — logged with the tag, counted, its future carrying the typed
failure — never the worker. `runners` may import `core`, `topology`, `scheduling`, `engine` and
`runtime`, but imports none of the last two today: an architecture test asserts
`import shipinfer.runners` loads neither torch nor the engine, and `topology` still imports only
`core` with `pool.py` present.

**Not here, on purpose.** Reassembly (§5⑤) — the walk is synchronous; ingest wiring
(`runners/sink.py`) — `submit()` is the entry until phase B; per-camera priority; and the host
(`bgr@cpu`) path of the pool element, whose `produces: nv12@gpu` is honest only for the device
caps and is named as such in its module docstring. The queue-and-workers shape duplicates
`pipeline/runner.py` deliberately until phase C supersedes `pipeline/graph/`'s hard-coded DAG;
both module docstrings say so.

---

## 2026-08-27 — `topology/`: the element chain as a validated, declarative object (Phase A1)

**What.** `src/shipinfer/topology/` — the first package of the architecture reset (#52,
`docs/arch.md` §1/§8/§9): `Element` template-method ABC with declared caps
(`<format>@<location>`), one `ElementRegistry` per element kind, a pydantic `ChainSpec`
loaded from YAML and a `Topology` that validates the chain at load time — kind inference from
slot names, declaration-order predecessors with `after:` override, Kahn sort, structure
(decode root, output sink, every element reaches an output), and per-edge caps negotiation
that never bridges `gpu` and `cpu`. Ten typed refusals in `core/errors/chain.py`. Mock
elements only; `topology/ship_person.yaml` is the production chain, refused today with the
impl name it is waiting for.

**Why.** The operator's V131 model is "input → topology → output" where every element has
interchangeable OOP implementations; the loader is where a chain that would silently download
a frame to CPU is refused instead (§8). Pure and offline by design — the layering rule gains a
fourth pure layer, enforced by the hook and the architecture tests plus a runtime import guard.

**Decisions.** Registry per kind (impl names repeat across kinds); no implicit converts; default
predecessor = declaration order; `meta@cpu` added to the caps vocabulary; `ElementContext`
inverts the engine dependency. Next: A2 (`runners/inprocess`, `engine/`+`api/` split), A3
(gRPC launch supervisor, argv-command deleted).

**Round-1 review fixes.** A `when:` guards exactly one element — skip-and-continue is the
semantics `admits` fixes — so `topology/ship_person.yaml` (and §1's snippet) now repeat
`when: class == ship` on `embed_ship` and `recognize`; without them the ship embedder and the
ship recogniser ran on every person crop and emitted a ship identity for a person, and two
walk tests (one per class) are what that defect fails. Plus: a root carrying a `when:` is
refused at load (it can never be true, so the chain would ingest nothing); one class
registered under two implementation names is refused instead of having `Element.impl`
rewritten under instances already built; and `Element.__init__` gained a keyword-only
`model:`, so a `pool` element is handed the repository model name the loader validated
instead of having to reach back into the node's spec.

---

## test: a decoded pixel over a real RTSP session, and refusals that name the build (P4-PR2c)

**The evidence #32 and #46 owed.** `csrc/tests/test_ingest.cpp` section P stands up an RTSP
server on 127.0.0.1, opens it through `SOURCES()` as `GStreamerSource`, and asserts on the
bytes that come back: the negotiated size is the served size, the frame carries HWC BGR with
its keepalive, **the pixels vary** (256 distinct byte values, not the blank buffer that would
have passed every other check in the file), consecutive frames differ (so the per-frame copy is
real), the ids are the actor counter's 0.., both clocks are stamped, and it closes cleanly
while the server is still serving. Container: `246 checks, 0 failure(s), 1 skipped`, plus one
deliberate `PIXEL:` line — "246 checks" cannot tell a reviewer whether anything ever looked at
a pixel, which is how two PRs went by. Host (driverless, offline): `219 checks, 0 failure(s),
3 skipped`.

**The server is the one that already existed.** `csrc/tests/rtsp_loopback.h` runs
`scripts/rtsp_serve.py` in a child process — the same fixture the Python ingest tests and
`benchmarks/harness/rtsp.py` use, whose pacing bugs are already argued out in its docstring —
rather than growing a second RTSP server in C++. It could not have grown one anyway:
`shipinfer-gst:jammy` has `libgstrtspserver-1.0.so.0` and the gir binding but **no `-dev`
package** (`pkg-config --modversion gstreamer-rtsp-server-1.0`: not found), so there are no
headers to compile against, and `ffmpeg -f rtsp` cannot serve (in 4.4 `rtsp_flags listen` is a
demuxer option). Ten `testsrc` JPEGs are made on the spot, because the repository ships no
fixture data. Header-only, POSIX-only, no `EXTERNAL` lane: it compiles in the offline tier
everywhere and decides at *runtime* whether this host can serve, with the server's own log in
the skip.

**A refusal now names the build lane.** A lane left out means a unit not compiled, means a
registrar that never ran, means `create_source("gstreamer")` answering "unknown video source"
for a name that is spelled correctly — the third failure mode `ingest/registry.h` warns about,
arriving disguised as the first. `scripts/build_csrc.py` bakes what it left out into every unit
(`-DSHIPINFER_OMITTED_LANES`), `ingest/omitted_lanes.h` maps lane -> the source names that lane
registers (strings, not includes, so the offline-closure invariant holds), and
`SourceRegistry::canonical` checks "not in this build" before "unknown". The wording is neutral
about *why* a lane is absent, because `--offline` omits lanes by design and only a full build's
absence is a missing package. The offline tier asserts this in the branch that used to be a
bare skip, a misspelling still gets the plain message, and `tests/test_build_csrc.py` fails if
the Python lane list and the C++ table ever disagree. The script also re-prints the omitted
lanes after the last `built ...` line (flushing stdout first, or a piped `2>&1` puts the note
back on top of everything).


## feat: the GStreamer camera source crosses to the C++ plane (27 Aug 2026, P4-PR2a+b)

**What it is.** `ingest/sources/gstreamer.py`, ported: the pure half first (#45 —
`gstreamer_pipeline.h`, a header with no GStreamer in it, so the offline tier asserts the
exact `gst-launch-1.0` strings an operator pastes, cross-checked byte-identical against the
Python function over a 12-case matrix), then the gst-linked `GStreamerSource` (this PR):
parse_launch → appsink → PLAYING with the open timeout as a state wait, reads bounded by
`try_pull_sample`, the bus's EOS/ERROR both `FrameDecodeError` (EOS on a live camera is a
fault to reconnect from, never exhaustion), the 4-byte row-stride undone and every frame
copied out of the decoder pool with the vector as `HostFrame.owner`.

**The build grows lanes.** `EXTERNAL` in `scripts/build_csrc.py` declares per-unit
`pkg-config` dependencies; `--with-external gstreamer` opts the lane into an otherwise
offline build — which is how `shipinfer-gst:jammy` (now carrying `libopencv-dev`, extended
by the same run+commit shape that built it) becomes the one place that compiles and RUNS
the gst tests: 234 checks there, 217 plus a counted skip on the driverless host. A full
build leaves an implicitly-missing lane out with a loud warning naming the consequence; a
lane asked for by name that cannot be resolved stays a hard failure.

**Honesty about scope.** No decoded pixel is claimed: `build_pipeline` builds `rtspsrc`
pipelines by construction, so a real frame needs a real RTSP session — PR2c's
`gst-rtsp-server` loopback owns that, and the test section's docstring says so instead of
implying coverage. The two #45 review notes are honoured in code: redaction stays at the
call sites (`SourceOpenError`'s constructor), and the out-of-table codec keeps
byte-faithful parity with Python.


## ingest: the C++ ingest core — contract, registry, actor, manager (27 Aug 2026)

`csrc/shipinfer/ingest/` is now the Python plane's ingest seam, port for port, and it builds
and tests with **g++ alone**: no CUDA, no OpenCV, no GStreamer. Nine units — `frame.h`
(`HostFrame`/`Frame`/`FrameCounter`), `config.h` (a flat, camera-shaped `IngestConfig` P5's
settings tree fills), `sink.h` (the `FrameSink` contract plus `CountingSink`),
`timing/backoff.*`, `timing/pacing.*`, `base.*` (the `open`/`read`/`close` template method),
`registry.*` (`SOURCES()` + `SourceRegistrar`), `camera/health.*`, `camera/actor.*`,
`manager.*` — plus `core/stop_signal.h`, `core/redact.h` and `core/options.*` underneath them.

Four things are the point rather than a side effect:

- **The A1 violation is gone.** `CameraActor` was declared in `sources/replay.h`, which is the
  one ingest unit that reaches `core/platform.h` and OpenCV, so the whole camera plane was
  unbuildable without a driver. It has its own file now, and `ingest/registry.cpp` carries the
  invariant as a comment: **no unit under `ingest/` other than `sources/replay.*` may include
  `sources/replay.h`**, because `scripts/build_csrc.py` follows a header to the `.cpp` beside
  it. `csrc/build/test_ingest` is the fourth CUDA-free binary and its `ldd` names neither
  libcuda nor OpenCV. Visible consequence, stated rather than papered over: an offline binary's
  source registry legitimately contains no *real* source, because replay's registrar is in a
  unit the offline build does not compile. The test asserts on its own fake and skips (counted)
  on replay.
- **`FrameTag` has its second clock.** `captured_ns` is STEADY, `captured_unix_ns` is WALL, and
  `monotonic_ns`/`unix_ns` now live once, in `core/types.h`. The old replay source stamped
  `system_clock` into `captured_ns` while `is_expired` compares `monotonic_ns` — the moment P5
  wires `deadline_ns = captured_ns + budget`, every deadline would land ~54 years out and
  nothing would ever expire. `test_server.cpp` pins both halves of that; `test_pipeline.cpp`
  pins the round trip through `FrameState::capture()`.
- **The reconnect policy is asserted as a sequence, not as "it retried".** The actor's wait is
  injectable, so the offline tier reads back 0.4–0.5 s then 0.8–1.0 s, DEGRADED at one and two
  failures and UNHEALTHY at three (sticky across a retry), a fatal `SourceUnavailableError`
  calling the factory exactly once and surviving `stop()` as UNHEALTHY, the 5-empty-read budget
  with its 5 ms anti-spin sleep, and a `stop()` landing inside a 30 s backoff in under 200 ms.
  `IngestManager::stop` is signal-then-join in two passes: eight cameras parked in
  uninterruptible one-second reads shut down in ~1 s, not 8.
- **`bench.cpp` runs on the registry.** `--source` (default `replay`), an `IngestManager` in
  place of the hand-rolled camera vector, a `QueueSink` that translates refusal into
  `QueueFullError`, and the serial per-camera stop loop replaced by one `manager.stop()`. The
  per-camera drop report reads `manager.health()`. `ReplayLibrary` is now refcounted and shared
  per folder, and every frame carries an `owner` handle, so a reconnect cannot free pages a
  worker is still DMAing out of.

Sized by `csrc/tests/test_ingest.cpp` (131 checks, 1 counted skip) plus 2 checks each in
`test_server.cpp` and `test_pipeline.cpp`. GStreamer is PR2 and needs a toolchain gate. The
fourth binary joins CI's loop in a separate one-line workflows PR, because a PR that edits
`.github/workflows/**` cannot pass the review job.

## feat: topology D, `deepstream` — one NVIDIA graph per shard, the same events out (26 Aug 2026)

**What it is.** The fourth topology — a first-class pipeline implementation, not a competitor benchmark (V108): `@TOPOLOGIES.register("deepstream")`,
whose child is not `shipinfer serve` at all but a DeepStream GStreamer graph — `nvurisrcbin ->
nvstreammux -> nvinfer(detector) -> nvtracker -> nvinfer(embedders) -> fakesink` — with frames
never leaving device memory and only `NvDsBatchMeta` crossing into Python, through one src-pad
probe. Two ends are kept from the rest of the project, and they are what make the comparison a
comparison: the **model repository** generates the nvinfer configs (so the engine paths, dims,
output names, class labels and thresholds have one owner, not two), and the **result sink**
receives the same `PerceptionEvent` every other topology publishes (V108).
`docs/design/topology-deepstream.md` carries the mapping table, the inert-knob table and the
ladder.

**The decisions worth knowing.** *One process per shard, one GPU per shard*, refused at plan
time rather than the reference's one-process-many-branches: `Fleet` sets `CUDA_VISIBLE_DEVICES`
before the child's interpreter starts so the child sees logical 0, per-element physical
`gpu-id` would name devices it cannot see, G contexts are ~300 MiB each, and one plugin
segfault should cost K cameras not fifty. *`nvinfer`, not `nvinferserver`*: the latter needs a
Triton-protocol server we do not run, and `nvinfer` reads the same `model.plan` the `tensorrt`
backend does. *`http_port_base` is refused, not ignored* — a DS shard serves no KServe API.
`server` may not import `pipeline`, so `deepstream_command` names its child by argv only; and
`pipeline` may not import `ingest`, so the GStreamer loader moved verbatim to
`runtime/gstreamer.py` (`load_gst` plus a new `load_pyds`) with the ingest source delegating.

**Honesty about scope (V110).** PR1 is detector + tracker + the two embedder sgies. The
segmenter and the recogniser do not run, and **every event this topology emits names them in
`missing_stages`** rather than passing a partial frame off as a complete one. **No performance
claim is made**: there is no DeepStream image on this box and not one frame has run. What is
verified is everything that can be without one — 93 offline tests (70 pipeline + 23 topology, twenty-one of them review regressions from the seven rounds) —
and the live run is the operator/infra step, recipe in `deploy/deepstream/image.sh` (the
`docker run` + `docker commit`, `--network=host` shape `gst-image.sh` established).

**The parts that would have been silent bugs, pinned offline.** `rect_params` is
`(left, top, w, h)` in *muxed* pixels and `ObjectRecord.bbox` is `(x1, y1, x2, y2)` in *source*
pixels — publishing it unchanged halves every box on a 4K camera and puts extents in a corners
field; `object_id`'s "untracked" is 2^64-1, not 0; `frame_num` restarts at 0 on a reconnect and
`(camera, frame)` is the tag everything keys on (ADR-002), so `FrameNumbering` keeps it
monotonic; NTP 0 means "no capture time", not 1970. The probe runs on a streaming thread, so it
catches everything, counts `build_failures` and always returns `PadProbeReturn.OK`; `_emit` is
`PipelineRunner._emit_resolved`/`_record` copied deliberately, return-value check and delayed
drain included. Config generation refuses eight ways before a GPU is involved — the sharpest
being a single-output detector with no `bbox_parser`, which otherwise runs and reports zero
detections on every frame.
## perf: multi-chunk copies home go through pinned ping-pong staging (26 Aug 2026)

C44's lever 2, converged over three review rounds. The pageable D2H tails were the ops
layer, not TensorRT. The rule that survived review is **structural**: `_to_host` stages a
result only when it spans more than one chunk — one span has no overlap to win and the
staged path would add a full-size serial host memcpy that `.cpu()` never performs. So the
production letterbox frame (1×3×640×640) and design-sizing person-reid batches (~15 crops)
take the plain `.cpu()` path they always had, and the mask-sized batches (every ship its
own span at 640²) stage through a **ping-pong pair with one `torch.cuda.Event` per buffer**
on the worker's own stream — the copy engine runs chunk k+1's DMA while the host drains
chunk k. Budget, re-derivable: 8 MiB per buffer, a pair only for a genuinely multi-chunk
name — at most 16 MiB pinned per worker, released at the runner's stop
(`MemoryPool.release_staging`) and at `close()`; `stats()`/`close()` snapshot the staging
map under the lock. A mid-capture refusal goes pageable for that call only; an allocation
failure degrades once with a warning.

Measurement honesty (recorded because it gates the numbers): this box's inter-invocation
micro noise floor measured 25% on an identical-code control row, wider than every micro
effect attempted — so no per-call speedup is claimed; the claims are the mechanism, the
flat alternating end-to-end A/B, exact-equality tests, and the bounded budget. The
quiet-window pair is the recorded gate for any numeric claim. Deleting the copies entirely
(`letterbox_to_device` through the dispatcher) stays the ADR-007 follow-up.

## perf: crop_batch is one batched pass (26 Aug 2026)

C44's Nsight timeline said the crop stage's ~150 ms/frame was host-side wait — GPUs ~14%
kernel-busy, ~13 k launches/s, and the largest kernel population (`generatedNativePointwise`,
~74 instances/frame) was the per-box loop inside `TorchImageOps.crop_batch`. The loop is now
one batched pass: `_bilinear_axis` builds patch-coordinate `align_corners=False` index/weight
tables on the host (the far neighbour clamped inside the patch — the C45 cross-plane
contract, and the reason `grid_sample`/`roi_align` were rejected, argued in the module
docstring), two gathers + three `torch.lerp`s per chunk, ~10 kernels per crop set constant in
N. The frame crosses as uint8 (no full-frame float32), `swap_rb` rides the transpose gather,
mean/std are cached per (values, device) and shared with `_letterbox`, and the pass is
chunked (8 Mi output elements) so mask-sized batches cannot balloon. Same contract, same
outputs: a frozen copy of the old loop is the test reference (98 offline tests + a CUDA
class; mutation-verified both ways), and `test_ops_parity.py` is byte-identical. The C++
plane already had this shape (C45), so no csrc sync is owed.

## bench: the harness drives the shards (26 Aug 2026)

`--topology fleet|service --shards N --shard-cameras A,B,C` on `run_bench.py`: the parent
plans (the launcher's LPT, or the explicit crowded split), starts one child per shard
through the real `Fleet` with the topology's own environment, each child runs the
single-process measurement on its slice with every guard, and the parent sums throughput,
takes the worst verdict, and adds per-device execution counts where the work ran.
`Topology.adopt(plan)` is the one new seam method (a topology is told a plan someone else
made). `bench.sh` passes `--shm-size=2g` for the tier's rings (ADR-015). Evidence in
PR: B and C sweeps to 72 img/s on GPUs 3–5, C spreading the crowded shard's crop work
(person_embedder 742/377/389 where B had 1582/0/0). Sized by `benchmarks/tests` (21) and
the two adopt tests.

## 2026-08-26 — Topology C, `service`: the crop-stage models served across the fleet's shards

**What it is.** The fleet (topology B, #18) fixed the placement failure the project exists to fix —
every stage of a frame on the GPU that decoded it — and gave up global balance to do it: a crowded
shard's embedder saturates while a quiet shard's idles, which is the uneven-camera case in another
coat. `service` keeps the fleet's shape and adds a cross-process tier for the stateless crop-stage
models (`topology.service.shared_models`: the two embedders by default — crops, never frames,
never the detector; the segmenter's 39 MB batches make sharing it the operator's call). Every shard keeps serving its own GPU's instances and also offers
them to its peers through pinned shared-memory rings: one single-writer ring per (submitter,
owner, model) each way, vLLM's `ShmRingBuffer` discipline (FREE → CLAIMED → WRITTEN → TAKEN),
header = depth / EWMA / heartbeat / closed, read without a lock as the load signal. Three seams,
three PRs: the ring (`runtime/memory/shared_ring.py`), the wire and the proxy
(`server/remote_wire.py`, `server/remote_instance.py`: `RemoteInstance` is a `Placeable` with
`device = cpu`, so `scheduling/` is untouched and a proxy never wins a locality tie;
`RingIngress` and `ResultReader` are the two threads that serve it), and the topology
(`server/topology/service.py`, `server/service_mesh.py`, `Topology.shard_environment`,
`Model.attach_remote`, `InferenceServer` joining the tier on start and leaving it first on stop).

**What changed in behaviour.** Under `service`, a shared model's dispatcher sees its local
instances plus one proxy per peer; `locality_spillover` keeps work home while the local queue is
shallow and borrows a quiet peer when it is not. A full ring is `RingFullError(depth, capacity)`
— a `QueueFullError`, so the dispatcher spills on it like any other (ADR-005). A peer whose
heartbeat stops fails its in-flight requests with `PeerLostError(owner, tags)` and drops out of
the candidate set until it stamps again. The owner's failure crosses the wire as the owner's
error text. `serve` without a shard index and `fleet` build no mesh. Nothing else changed.

**Tested how.** Offline, over real shared memory: 19 ring checks (layout arithmetic, the
protocol, backpressure, the header as load signal, close under a live zero-copy view), 12 wire
checks (round trips, dtypes, the failure form, size accounting), 7 proxy checks (a `Placeable`,
end to end through a real `Dispatcher`, twenty in flight over four slots, the owner's error,
the lost owner with its tags), 7 mesh checks (two shards' meshes in one process, a request that
leaves shard 0's dispatcher and returns from shard 1 as `cuda:1`, stop taking the rings down, a
peer that never appears), 9 topology-contract checks, and 3 engine-level checks that start two
`InferenceServer`s in one process as shards 0 and 1 — the path that found the stray `@property`
which had made `attach_remote` unbound and killed every real shard at start while the fake-model
tests stayed green. Suite: 1132 passed, 43 skipped. **Inside the container** (`-m multigpu`, GPUs
3 and 4): `test_service_multigpu.py` starts two real `serve` processes through the real
`ServiceTopology` and `Fleet`, posts 24 requests to shard 0 over HTTP — 19 executed there, 5 by
shard 1 through the ring, every tag back on its own response, both processes gone after `stop`.

**Measured.** The two-process run above is a proof of the path, not a measurement — the bench-scale
evidence run ("C beats B": per-device retired counts within
10% under `--skew 8`, lower p99 on the busy cameras, no new `frames_failed`) is PR-cut item 4 and
needs `--topology` on the harness with the fleet driving the shards. Until it exists, `service`
is built and tested, not proven.

**Open for the operator** (asked in the topology PR): slot size per model or one size for all;
the detector is never shared — confirm; the pinned budget (ADR-015's derivation, slots per
model and direction: 4 shards × 2 embedders → 48 rings ≈ 1.26 GB of shared memory on the box
existing once, ≈ 0.63 GB registered per process; the 6.36 MB request slots dominate),
acceptable or not; and whether the segmenter joins `shared_models` at 39 MB request slots.

## 2026-08-25 — The port, steps P1a–P1d: the C++ plane takes the Python plane's shape

**What it is.** The operator read `csrc/` against `src/shipinfer/` and saw a different program
(V79); the decision (V80) was to port for real, seam for seam, with the Python plane's tests as
the specification. Four steps, one branch: the queue seam (`FairPriorityQueue`, `Lane`, `FifoQueue`,
`BatchWindow` — `fair.py`, `lanes.py`, `fifo.py`); the five placement policies with their registry
and the `Dispatcher` with its spill; `ModelInstance` (one thread, one bounded queue, bind once,
assemble, execute, scatter) and `Model` over a `Dispatcher`, behind an `Engine` contract that
`TrtInstance` implements through an adapter; and the graph as stages — `Dag`, `DetectStage`,
`CropStage`, `ObjectStage` over `Model::infer`, planned from the frame's state, with the collector
as observer. `cli/bench.cpp` runs that shape; the pool graph (`ModelPool`, `PipelineGraph`) is gone.

**What changed in behaviour.** The C++ queue evicts the greediest camera's *oldest* frame, as
Python does (ADR-014's one recorded divergence, closed); a detector batches in its own instance
queue under a window instead of the drain loop assembling detector-sized batches; a partial batch
is padded to a static plan's batch; a spilled row is peer-copied, as the Python plane copies.

**Tested how.** 62 + 24 + 9 checks named after `tests/scheduling/*.py`, `tests/server/*.py` and
`tests/pipeline/test_graph.py`, over an identity engine in host memory and fake stages — no device;
46 data-plane checks unchanged. The parity harness that drives both planes with one trace (P6) is
the gate the sync rule refers to, and is next.

**Measured.** 50 cameras × 20 fps, GPUs 2–5, 40 s: ~390 img/s at 48 workers, ~470 at 96, balanced
across the four GPUs to 1%, 0 failed, 0 timeouts, against the pool graph's ~470 at 48. A worker is
one frame in flight and waits on each stage in turn — the Python runner's shape — so the lever is
workers and the batch window, and the algo tier's per-stage profile on this shape decides the next
step rather than guesswork.

---

## 2026-08-25 — The topology seam, and the fleet behind it (Phase 7, T1–T2)

**What it is.** The operator's target is topology C — decode shards per GPU and a
cross-process inference tier balanced like Triton's — and there are three deployments that
should share one abstraction for "how the deployment is laid out into processes": the fleet
of shards (B), C itself, and a DeepStream competitor. `server/topology/` is that seam.
`Topology` is a registry-backed contract with four methods — `plan` (cameras + GPUs → a
`ShardPlan`), `command` (the argv one shard runs), `environment` (what every child inherits),
`describe` — and `TOPOLOGIES` is the switch: `SHIPINFER_TOPOLOGY__KIND` / `shipinfer fleet
--topology`. Unknown names fail at configuration time with the known list. The contract is
small on purpose and a test says so (`TestTheContractIsSmallOnPurpose`): a topology decides
*placement of processes*, not scheduling inside one.

**`scheduling/sharding.py` — the plan.** Pure. Longest-processing-time over offered fps
(`fps or 1.0`, because `fps=0` means "whatever the source delivers"), so balance is by load,
not by camera count: four 30 fps cameras and forty 5 fps ones split evenly in *frames*, which
is the failure this project exists to fix seen one level up. Stable across restarts (sorted
input, deterministic ties) so a camera keeps its GPU across a redeploy; GPUs handed out
without leaving one idle; when shards share a GPU the configured per-GPU instances are divided
between them (`instances_for`); an impossible plan (more shards than cameras or than GPUs,
zero of either) fails at plan time. `describe()` is what the launcher prints and what
`--dry-run` shows.

**`server/launcher.py` — one OS process per shard.** `Fleet.start` is all-or-nothing: a shard
that dies during start-up takes the others down before anything is reported running. Each
child gets `CUDA_VISIBLE_DEVICES` for its GPUs alone and `SHIPINFER_SHARD_CAMERAS` for its
cameras; `cli/common.py::_narrow_to_shard` makes `serve` read only its slice, and refuses a
slice naming a camera the configuration does not have. `supervise` turns a dead shard into
`ShardExitedError` for the whole fleet — a fleet that silently keeps running on three of four
GPUs is the imbalance bug wearing a new coat. `stop` drains for `--drain` seconds, then kills,
and leaves nothing behind (tested with real subprocesses on a stand-in command).

**`shipinfer fleet`.** `--shards` (default one per visible GPU), `--gpus`, `--policy`,
`--topology`, `--dry-run`, `--drain`. The dry run prints the plan and exits without spawning.

**What is deliberately not here.** No live multi-GPU run in the PR: the demo repository in
git carries no engines, and the fleet's children are `shipinfer serve`, so the process
semantics are proven with a stand-in command and the plan with a dry run. T3 (`service`, the
cross-process inference tier) and T4 (DeepStream) register against the same contract.

---

## 2026-08-25 — The benchmark's other two tiers, and an RTSP source

**What it is.** R44 asks for three benchmark tiers — system, algo, kernel — and only the
system one existed. R55 makes RTSP mandatory for the benchmark, not only for the tests, and
every measurement so far replayed JPEGs off disk. Both closed.

**`benchmarks/stages.py` — the algo tier.** Where does one frame's time go, stage by stage.
It *reads* rather than instruments: `PipelineStage.run` already stamps `elapsed_us` on every
outcome and `_CollectorObserver` already feeds it into `shipinfer_pipeline_stage_latency_us`,
so a second timing path would be a second implementation that could disagree with the one
operators watch. Reports each stage's exact per-call mean and per-frame cost over the **steady window** — the
histogram's sum over its count, both read at the warm-up boundary and at the end and
differenced, over the frames accepted in the same window — with p50/p95 as bucket-edge colour.
The first version charged stages by p50, which is a bucket's upper edge: two stages in one
bucket rendered a 2.3x cost difference as a tie, and a steady duration was divided by a
whole-run frame count. Review caught both.

`calls_per_frame` is the whole point: a stage costing 8 ms on one frame in three costs 2.7 ms
per frame, and the embedders run once per *object batch*. Assuming one call per frame would
overstate the cheap stages and understate the expensive ones by the same factor.

It runs **below saturation deliberately** and warns loudly when the run did not keep up.
Under saturation a stage's latency includes the time it waited behind other frames, so a
backlog reads as an expensive stage — the same 98% bar `check_offer` holds the system tier to.

**`benchmarks/kernels.py` — the kernel tier.** What one op costs, per implementation, at the
shapes this project runs. Two corrections on the way in, both of the same family — measuring
something adjacent to what production does:

- It called `IMAGE_OPS.create(name)` with no arguments. `TorchImageOps` falls back to the CPU
  without a `device_index` and `PipelineRunner._build_ops` always supplies one, so the first
  run timed torch on the **CPU** and reported it 7–13× slower than numpy. Bound correctly,
  torch on `cuda:0` is 3.27× numpy on letterbox, 1.84× on crop_batch, 2.47× on nms.
- `letterbox_batch` returns numpy by contract, so a device implementation pays a copy home
  that numpy never makes. Timing only that column charges the device implementations for the
  round trip; `letterbox_to_device` is the device-fair case and the one production calls.

Both tiers report what they could **not** measure rather than printing a shorter table — a
missing column with no explanation is how "we never measured it" becomes "it is not faster" —
record the host load, and mark a spread over 20% as noisy. The first kernel run was taken at
load 41 of 48 with spreads to 76%.

**`--source rtsp`.** The bench cameras point at `scripts/rtsp_serve.py` over a real socket,
with `benchmarks/harness/rtsp.py` owning the server's lifetime. It refuses rather than
tolerates: a server that never accepts, or that exits early, raises at start-up with its
output attached — a run whose cameras cannot connect produces a clean-looking zero and this
project has already published one of those. Readiness is a socket poll rather than a sleep;
teardown is terminate-then-kill, because a GLib loop holding the port makes the *next* run
fail with an address already in use, minutes later and nowhere near the cause.

**Replay and RTSP are different experiments, not a fast one and a slow one.** Replay measures
the inference plane with the decode path removed, so a replay number is an upper bound on the
RTSP one. The source is recorded in the run metadata, printed on the console, and explained
in the README, because the failure to avoid is quoting a replay figure as though NVDEC were
in it.

**Tests.** 31 offline tests over the two tiers and the RTSP wiring, pinning the arithmetic and
all four server failure paths. The arithmetic is where a benchmark lies: every defect review
found in `run_bench.py` was a formula producing a plausible number from a run that did not
support it, not a broken measurement loop. 116 tests in `benchmarks/tests`.

**Since then.** Both tiers have run to completion inside the container, and the algo tier has
been re-run after review replaced its cost model (exact steady-window means instead of bucket
edges): 12 cameras × 5 fps on GPUs 2–5, kept up (60.0 of 60 img/s), 1777 frames in the 30 s
steady window, host load 22/48 with another user's 21 GiB job on GPU 0. Per-frame cost: crop
149.6 ms (46%), detect 98.6 ms (30%), ship_segmenter 41.5 ms (13%), ship_embedder 17.7 ms,
person_embedder 17.0 ms; serial 324 ms against wall 16.9 ms. The earlier reading of this run
("p50 16–63 ms") was the histogram's bucket resolution, not the cost — the exact means are two
to three times larger and the p95s reach 0.5–1 s. These are submit-to-result spans (queue,
batch window and work together), so the split between waiting and working is the Nsight
timeline's job (C1a); that `crop` costs 1.5× `detect` is the first thing that timeline has to
explain. The kernel tier, once the fused kernels were reachable, measured native
`letterbox_to_device` at 657 µs against torch's 735 µs — 1.1×, where the inherited figure was
50×. The RTSP path has still not been run under load.

---

## 2026-08-25 — Five of Triton's features, taken (`docs/qa/triton.md` §3)

**What it is.** The five rows of that document's "features Triton has that we should take"
table that were still a plan, implemented and the table rewritten to describe the code rather
than the intention:

1. **`GET /v2/models/{name}/stats`** (and the `/versions/{v}/` spelling) — `server/statistics.py`
   holds `ModelStatistics`, one per model, shared by its instances and by the ensemble path.
2. **Explicit model control** — `model_control: explicit` plus
   `POST /v2/repository/{index, models/{n}/load, models/{n}/unload}`.
3. **A rate limiter** — `scheduling/limits/`, a registry with `off` (default) and
   `concurrency`, configured per model.
4. **Warm-up from declared samples** — Triton's `model_warmup` key, materialised by
   `repository/warmup.py` and run by `ModelBackend.warmup`.
5. **Request tracing** — `core/tracing/`, Triton's seven event names, `none` (default) and
   `jsonlines` sinks, `rate=N` sampling.

**Why each one, in one line.** A histogram has no per-model cumulative count, so an operator
debugging one camera's model had to read the fleet's numbers to find one. A repository that
grows cannot be loaded whole. The queue bounds what is *waiting*, and nothing bounded what was
*running* — eight instances whose windows close together all enter compute at once. A fixed
count of zero-filled batches decides how often a model is warmed but not *what with*, and the
data is what selects the kernels. And six stamps with no sink cannot answer "why was frame
8213 slow".

**Two things the wiring changed that the feature list does not show.**
`DurationStat.observe(ns, count)` now adds `count * ns` rather than `ns`: crediting a batch's
span once instead of once per request divides the reported latency by the batch size, which is
an error in the flattering direction and was caught by the first test written against it. And
`ModelInstance.wait_ready` now returns as soon as the worker has *settled* either way — before
that, a worker that failed on its first line held start-up for the whole 120 s timeout and then
reported "did not become ready", hiding the cause. A typo in `model_warmup` is enough to reach
that path, which is how it was found.

**Where Triton was deliberately not followed**, each recorded in the document: `poll` model
control (a timer can load a half-written config), reload-on-load (it must stop the running copy
first, so a half-failed reload takes a working model down), and the general named-resource rate
limiter (the only resource this pipeline has needed to bound is "an execution").

**Cost.** 90 new offline tests, all class-based; 892 pass with no GPU. Nothing new is on by
default: `off` limiter, `none` trace sink, `none` model control, and no `model_warmup` in any
shipped config, so a deployment that does not opt in pays one virtual call per completed
request and two per batch.

---

## 2026-08-24 — The C++ data plane (`csrc/`)

**What it is.** A standalone binary that owns everything running once per frame or once per
object: ingest and per-camera pacing, the fair bounded queue, letterbox and crop-resize kernels
writing straight into TensorRT bindings, device-affine instance pools, the perception graph,
per-frame reassembly with a timeout, and the occupancy log. Python keeps the control plane.
See ADR-014 for why this is not the optional-extension contract ADR-007 governs.

**Why.** The Python data plane capped at 77 img/s using five cores of forty-eight while every
GPU queue sat empty. Four other candidates were eliminated by measurement first (the GPUs, the
worker pool, the reassembly lock, the load generator), so the remaining explanation was the
pure-Python share of the per-frame path holding the GIL.

**The measurement, and the one design decision that makes it trustworthy.** The binary writes
the *same* `*_buffer_size` occupancy JSONL the Python driver and the baseline binary write, so
`benchmarks/harness/analysis.py` scores all three with one implementation and one set of
guards. A port that exists to look good must not be scored by a friendlier judge than the thing
it is compared against.

    50 cameras x 20 fps on 4x A5000, 70 s, 10 s warmup, scored by the shared analysis
    pipeline: offered 983, growth +592.9/s, sustained 390.5, SATURATED

390.5 against the Python plane's 77.5 — **5.0x**, with 98% of the offered load actually
delivered where the Python driver could never exceed ~87 img/s.

**Four bugs found by running it**, each a design error: static plans refuse any batch but their
own; cross-device execution (an instance on gpu0 executing pixels on gpu1, surfacing as an
illegal access somewhere else entirely); `gpuDeviceSynchronize` after every kernel, which is
device-wide; and a pageable host source for 2 GB/s of copies.

**And a hole in the measurement itself**, which is the part worth remembering. The occupancy
log first carried `busy()` — leases held. This design has no queue in front of a pool (a worker
blocks inside `lease`), so a *fully committed* pool reads as a flat `busy == size` and the
analysis scores it SUSTAINED. `ship_segmenter` sat at exactly 4 with exactly 4 instances:
pegged, and invisible. Logging `waiting()` instead showed 37 of 48 workers blocked on it, and
every bottleneck since has been a model pool rather than the interpreter — which is the
qualitative change the port bought and is worth more than the 5x.

**Review found four blocking defects in the first version**, all real: a per-frame
`gpuMalloc`/`gpuFree` on the dispatch path with the reusable buffer voided by `(void)`; skipped
branches indistinguishable from failed stages (every ship-only frame sealed Incomplete);
reassembly eviction destroying a frame with no event and no per-camera attribution; and no ADR
for a second data plane. Fixing the second took Complete events from a minority to 28656 of
28808. Fixing the first — the one predicted to be the throughput lever — moved 390 to 400,
about 2.5%, which is a reminder that a plausible mechanism is not a measured one.

---

## 2026-08-24 — The benchmark harness: what counts as a measurement

**What it is.** `benchmarks/` drives ShipInfer and `counting-simulation` under one load and
compares them by the baseline's own buffer-growth saturation methodology: a buffer whose
occupancy grows over the steady window is a module that cannot keep up, and
`sustained = offered - growth`. `benchmarks/baseline/` is the upstream repo as a submodule;
its `sim_pipeline_v2.cpp` is compiled unchanged and run as its own binary — nothing here
re-implements it.

**The seam it owns.** `run_bench.system_throughput` is the only place that decides *what
counts as an image, once*. The baseline runs two independent single-model pipelines over
disjoint image streams, so its system rate is `det + seg`. ShipInfer runs one DAG where every
frame enters the pipeline queue once and then fans out into crops, so its rate is the
pipeline queue alone — summing its modules the way the baseline's report does would count
each frame at the queue and again at the detector.

**The taxonomy, which is the part worth remembering.** A run yields one of three things and
conflating them is how a harness publishes a number it did not measure:

- **SATURATED** (not capped) is a **capacity** — the buffer grew, so `offered - growth` is
  exact. This is the whole methodology.
- **SUSTAINED / DRAINING** is a **floor** — nothing grew, so capacity is *at least* the
  offered rate and this run cannot say how much more.
- **UNMEASURED** is nothing — a capped buffer sheds instead of growing, so its slope means
  nothing.

`ratio_of` is the single place a pair combines: capacity/capacity is exact, floor/capacity is
`>= Nx` (can meet a target), capacity/floor is `<= Nx` (can only miss one), floor/floor is
nothing. The first version had this inverted — it refused SATURATED as "a bound" — which made
a speed-up structurally unreachable, because both systems are offered the same load by
construction. Six review rounds to find that.

**The guards, each of which caught a real lie.** Offered is what *entered* (a dropped frame
cannot grow a buffer, so counting it as offered turned a shedding system into a 3.3x
overstatement); a capped module forces UNMEASURED; `check_offer` refuses a run whose
generator delivered under 98% of target; `reconcile` cross-checks the buffer-log rate against
`events_emitted/elapsed`, which a scheduler that *refuses* work cannot fool; every counter is
rated over the same window the fit uses. `--sweep` climbs the offered rate until something
saturates, because one point cannot settle a comparison when both sides get the same load.

**First result.** baseline 868.2 img/s, shipinfer 81.4 img/s, 0.09x against a 5x target. The
binding module is the pipeline queue, not any GPU queue, and it is insensitive to
`--pipeline-workers` over an 8x range — the wall is one Python process. See JOURNAL.

---

## 2026-08-23 — The pipeline plane: the perception DAG, reassembly, and the event contract

**Why.** `src/shipinfer/pipeline/` was empty, so nothing connected the cameras to the models
to anything downstream. This is the application half of PLANE 2 in
`references/bitbucket-subfaceid/docs/new-system-architecture.md`: detect, crop, segment,
embed and recognise, then join a frame's results and publish them on the contract the
tracking tier already consumes.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Frame -> request | `pipeline/sink.py` | `QueueFrameSink`, the production `FrameSink` of ADR-011 |
| Stages | `pipeline/graph/` | subclass `PipelineStage`; `ModelStage`/`ObjectStage` cover a model |
| Reassembly eviction | `pipeline/reassembly/policy.py` | `@EVICTION_POLICIES.register` |
| Result sinks | `pipeline/sinks/` | `@RESULT_SINKS.register` (null / jsonlines / kafka) |
| Pipeline settings | `core/settings/pipeline.py` | one field on `ServerSettings` |

**Decisions.** Reassembly keeps `BodyDataCollector`'s shape (camera -> frame -> results,
complete-or-timeout at its own 1500 ms) and fixes the three things it got wrong: eviction
charges the overflow to the camera holding the most incomplete frames rather than dropping the
globally oldest entry, every internal structure is bounded including the per-camera index, and
a timeout emits a partial event naming the missing stages instead of deleting the frame. The
inherited drop-oldest behaviour ships as a registered policy so the regression test runs the
two side by side. Emission happens when the worker **seals** a frame, not when its
currently-expected stage set is momentarily satisfied — the set grows as branches are decided.
The schema keeps every v1 `Det2MOT` key with its v1 meaning (people only) and adds ships in
the same parallel-array idiom, so a deployed `motservice` needs no rebuild.

**Notable.** Three defects found by running the end-to-end test on a host that has GPUs, all
of which report as something other than what they are: one `ImageOps` shared across worker
threads overwrites a pinned buffer mid-DMA and says `crop_kernel failed: invalid argument`;
preprocessing every frame on `cuda:0` re-creates this project's founding bug one layer up; and
a worker whose current device is 0 holding ops built for `cuda:1` says `invalid resource
handle`. `ThreadLocalImageOps` binds one instance per thread to one device, round-robin over
the visible GPUs (ADR-002). A fourth, caught by the tests: a `RequestQueue` and a `ResultSink`
both define `__len__`, so `self._queue = queue or default` silently discarded an injected empty
one — every default in the runner is now `if x is None`.

**Known cost.** Pre-processing returns to the host before the model stages it to its own
device. A GPU-resident path needs `letterbox_to_device` writing into the chosen instance's
binding buffer, which means knowing the instance — a dispatcher decision, and the "Phase 2
fast path" the architecture document files for when a measurement says the round trip is what
hurts (ADR-007).

**Layering.** `pipeline` has no `ingest` edge in `scripts/hooks/check_layers.py`, so the
adapter describes what it needs from a frame as a four-member `TaggedFrame` protocol and the
runner takes a `FrameProducer` protocol, in the same spirit as `MemoryHandle` in ADR-001. The
rule was left alone rather than widened.

**Evidence.** 113 offline tests, passing identically with GPUs hidden and with eight visible.
Reassembly fairness, at capacity 16 with one camera submitting 100 incomplete frames beside one
submitting 2: `greediest_camera` leaves `{quiet: 2, loud: 14}` with all 86 evictions charged to
`loud`; `oldest_frame` leaves `{loud: 16}` and the quiet camera loses both. End to end, the
`replay` source into the mock backend into the `jsonlines` sink: 6 frames in, 6 events out,
every tag accounted for, none duplicated.

---

## 2026-08-23 — The ingest plane: one stateful actor per camera

**Why.** `src/shipinfer/ingest/` was empty, so the server could not read a camera at all.
PLANE 1 of `references/bitbucket-subfaceid/docs/new-system-architecture.md`: 50 cameras in,
tagged frames out, no inference in the path.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Video sources | `ingest/sources/` | `@SOURCES.register` (gstreamer / pyav / replay) |
| Frame consumers | `ingest/sink.py` | the `FrameSink` protocol — `pipeline` supplies the production one |
| Environment contract | `src/shipinfer/envs.py` | one `EnvVar` per variable, typed, with `describe()` |
| Ingest errors | `core/errors/ingest.py` | four types, one per operator action |

**Decisions recorded.** ADR-011 — ingest depends on a sink protocol it owns, not on the
scheduler.

**Notable.** Two bugs found by the tests, both in code that only runs when something is
already wrong: `ExponentialBackoff.peek()` overflowed a float at ~attempt 1000 (a camera at
the 30 s cap reaches that in under nine hours — a guaranteed actor-thread death on a
long-running deployment), and the `frame_id` counter had to live on the actor rather than the
source, or a reconnect reissues frame 0 and hands a tracker a duplicate `(camera_id,
frame_id)`. Reconnect is exponential + jittered + capped, and a *frame* resets it, not a
successful connect — an RTSP source that opens and delivers nothing is the common real
failure and must not read as healthy.

Two tightenings to `scripts/hooks/check_layers.py` fell out of the work: `from shipinfer
import x` is now checked identically to `import shipinfer.x` (the two spellings had different
rules and the lax one was winning by accident), and `core` may not import the non-layer
top-level modules that every other layer can.

**Evidence.** 163 offline tests, no GPU, no GStreamer, no PyAV, no camera — the `replay`
source over a generated frame directory is what makes that possible and is what the
50-camera stress test will use. Reconnect tests assert the *sequence* of delays
(`[0.1, 0.2, 0.4, 0.8, 0.8, 0.8]`), not that a retry happened. No throughput measurement was
taken; `shipinfer bench` against `CountingSink` is the next step and is not claimed here.

---

## 2026-08-22 — Initial system: scheduler, runtime, backends, server, native kernels

**Why.** The previous generation (`references/bitbucket-subfaceid`) ran every model on GPU 0
and starved quiet cameras through a shared evict-oldest buffer. The requirement is 50
cameras × 20 fps across 16 GPUs with balanced load and bounded tail latency.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Registry primitive | `core/registry.py` | eager + lazy registration |
| Placement policies | `scheduling/policies/` | `@POLICIES.register` |
| Request queues | `scheduling/queues/` | `@QUEUES.register` |
| Batchers | `scheduling/batching/` | `@BATCHERS.register` |
| Backends | `backends/` | `@BACKENDS.register` / `register_lazy` |
| Allocators | `runtime/memory/` | `@ALLOCATORS.register` |
| Graph caches | `runtime/graphs/` | `@GRAPH_CACHES.register` |
| Image ops | `runtime/ops/` | `@IMAGE_OPS.register` |
| Log sinks | `core/logging/sinks/` | `@SINKS.register` |
| Metrics exporters | `core/metrics/exporters/` | `@EXPORTERS.register` |
| Response caches | `server/cache/` | `@RESPONSE_CACHES.register` |
| CUDA providers | `runtime/providers/` | `@PROVIDERS.register` (custom variants only) |

**Decisions recorded.** ADR-001 through ADR-009 — the pure core, one-thread-one-GPU, torch
as substrate, locality-aware spillover, fair queueing, the Triton repository layout, the
Python/C++ split, CUDA-graph buffer lifetime, and the opt-in response cache.

**Evidence.** 149 offline tests (no GPU) + 12 GPU tests; 998 req/s at p99 7.6 ms with
11.7–13.2% per-device share across 8 × A5000; fused letterbox 1.41× faster than torch with
bit-identical output.
