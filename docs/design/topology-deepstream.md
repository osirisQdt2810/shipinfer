# T4 — topology D, `deepstream`: one NVIDIA graph per shard

The fourth topology (V108: a first-class pipeline implementation, not a competitor
benchmark). `fleet` (B) and `service` (C) run *this* project's scheduler over TensorRT
engines; `deepstream` (D) hands the whole per-frame path to NVIDIA's GStreamer graph and keeps
only the two ends — the model repository that describes the models, and the result sink that
publishes the events. It exists so the architecture question ("is our scheduler worth having,
given DeepStream is free and tuned?") can be answered with a measurement instead of an opinion.

**This document describes PR1, which is not the whole DAG.** Detector, tracker and the two
embedders run; the segmenter and the recogniser do not, and every event this topology emits
says so in `missing_stages`. §5 is the ladder for the rest.

**No performance claim is made here.** There is no DeepStream image on this box, so nothing in
this PR has run a frame. What *is* verified is everything that can be verified without one, and
that turns out to be most of the interesting part: 53 offline tests over the config generation,
the metadata walk, the coordinate mapping and the emission discipline. The live run is an
operator/infra step, and §7 is its recipe.

---

## 1. The shape

```
                    one process, one GPU, one shard's cameras
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ nvurisrcbin(cam A) ┐                                                          │
 │ nvurisrcbin(cam B) ┼─► nvstreammux ─► nvinfer(pgie) ─► nvtracker ─►           │
 │ nvurisrcbin(cam C) ┘        batch=K      detector         NvDCF               │
 │                                                                               │
 │   ─► nvinfer(sgie: person_embedder) ─► nvinfer(sgie: ship_embedder) ─► fakesink
 │                                              │                                │
 │                                    src-pad probe: NvDsBatchMeta               │
 └──────────────────────────────────────────────┼────────────────────────────────┘
                                                ▼
                                        PerceptionEvent ──► ResultSink (the same one)
```

Frames never leave the graph. They are decoded into device memory, inferred on there, and
dropped at the `fakesink`; only the metadata crosses into Python, through one pad probe. That
is the property that makes this topology cheap and also the property that makes it opaque —
there is no point in the pipeline where a frame can be inspected, re-queued or spilled to
another GPU, which is exactly what `service` exists to do.

### One process per shard, one GPU per shard

The reference sketch (`mtmc_deepstream.py`) puts two GPU branches in one process and sets
`gpu-id` per element. This topology refuses that, in three places for three reasons:

1. **`CUDA_VISIBLE_DEVICES` is already the mechanism.** `Fleet` sets it before the child's
   interpreter starts, which is the only way to win the race against a module-scope `import
   torch` (see `server/launcher.py`). The child therefore sees exactly one device, numbered 0,
   and a per-element physical `gpu-id` would name a device it cannot see.
2. **Contexts cost.** A process touching G devices holds G CUDA contexts at ~300 MiB each.
   Sixteen GPUs in one process is ~5 GiB of context for nothing.
3. **A blast radius should be a shard.** One plugin segfault takes down the process it is in.
   With one process per shard that is K cameras; with one process for the box it is fifty.

`DeepStreamTopology.adopt` refuses any plan whose shard has more than one GPU, naming
`gpu-id` and the fix (`--shards N >= <gpu count>`). A single-process many-branch variant, if it
is ever wanted, is another file and another decorator — that is what the registry is for.

### `nvinfer`, not `nvinferserver`

The reference uses `nvinferserver`, which speaks Triton's protocol to a Triton server. We do
not run one: this *is* the server, and its API is KServe v2 over HTTP, not Triton's internal
gRPC. `nvinfer` reads `model.plan` directly, which is the same artefact the `tensorrt` backend
loads, so both topologies run the same engine and the comparison is about the serving
architecture rather than about the network. `nvinferserver` becomes interesting the day this
project grows a Triton-gRPC facade (§5), because then a DeepStream graph could dispatch into
*our* scheduler and the two architectures could be composed rather than chosen between.

---

## 2. The mapping: `config.yaml` → nvinfer

Every DeepStream deployment carries hand-written `pgie_config.txt` files. They are a second
description of a model that is already described, and two descriptions of one model drift
silently: an `infer-dims` that no longer matches the engine fails at start-up, a stale
`num-detected-classes` mislabels objects forever. So `pipeline/deepstream/configs.py`
*generates* them from the repository, and the mapping is the interface:

| nvinfer key | comes from | note |
|---|---|---|
| `model-engine-file` | `parameters.engine_file` in the resolved version dir | absolute; nvinfer resolves a relative path against the *config* dir |
| `onnx-file` | `parameters.onnx_file`, when declared | optional, and worth having: nvinfer rebuilds when the TRT version differs |
| `infer-dims` | the single input's `dims` | `c;h;w`, batch excluded, as in Triton |
| `batch-size` (pgie) | `len(cameras)` for this shard | nvstreammux's batch |
| `batch-size` (sgie) | `max_batch_size` | refused at 0 — one inference per object |
| `output-blob-names` | declared `outputs`, in order | |
| `network-mode` | `topology.deepstream.network_mode` | fp32/int8/fp16 → 0/1/2 |
| `num-detected-classes` | `max(pipeline.class_labels) + 1` | the label file's length |
| `labelfile-path` | generated from `pipeline.class_labels` | positional; gaps filled with `unknown` |
| `pre-cluster-threshold` | `pipeline.score_threshold` | |
| `topk` | `pipeline.max_detections` | the fan-out cap, same as the Python DAG's |
| `cluster-mode=4` | fixed | no clustering: yolo26 applied NMS in the engine |
| `net-scale-factor`, `model-color-format=0` | fixed at 1/255 and RGB | matches `NormalizeParams`' defaults — with one stated exception: nvinfer's letterbox pads with 0 where `ImageOps` pads with 114, so the bar pixels differ between planes (#32 r5); everything inside the image is identical |
| `maintain-aspect-ratio=1`, `symmetric-padding=1` | fixed | the letterbox `ImageOps.letterbox` does |
| `operate-on-class-ids` | `operate_on` labels reversed through `class_labels` | labels, not ids: an id is a checkpoint property |
| `process-mode=2`, `network-type=100`, `output-tensor-meta=1` | fixed, sgies | objects not frames; no built-in postprocess; the embedding is reachable |
| `gie-unique-id` | 1 for the pgie, 2..N for the sgies | `operate-on-gie-id=1` on every sgie |

**Nothing is written into the model repository.** The generated files live under
`$TMPDIR/shipinfer-ds-<run>/shard<N>/` (or `topology.deepstream.config_dir`), because
`ModelRepository` refuses a stray Triton config in a model directory and a `.txt` beside a
`config.yaml` is noise in every future diff. A test asserts the repository tree is byte
identical after generation.

### The refusals, all of them offline

Each of these is otherwise a start-up failure inside a GStreamer element on a GPU host, or
worse — a graph that runs and publishes nothing:

* a model whose `platform` is not `tensorrt`;
* more than one input, an input that is not rank 3, an input that is not FP32;
* **a single-output detector with no `bbox_parser`** — nvinfer's built-in parsers expect the
  two-tensor coverage/bbox layout, and against a decoded `(300, 6)` tensor they find zero
  boxes on every frame. The message names the missing key *and* the output shape;
* an `operate_on` label `pipeline.class_labels` does not define (the message lists the ones it
  does);
* two secondaries claiming one label — both would attach a tensor to the same object and the
  probe reads the first it finds, so which embedding survived would depend on GIE order;
* `max_batch_size: 0` on a secondary;
* a camera URI whose scheme is neither `rtsp` nor `file` (a bare path becomes `file://`);
* a missing `model.plan` — except under `--dry-run`, which reports it instead, because a
  control box legitimately has no engine and an engine is host-specific anyway.

---

## 3. What comes back, and what has to be undone

The probe reads `NvDsBatchMeta` and produces the same `PerceptionEvent` every other topology
publishes. Three conversions are load-bearing, and each has been a silent bug in a DeepStream
deployment somewhere:

**Boxes are in the muxed frame's pixels.** `nvstreammux` scaled a 4K camera into 1920x1080
before the detector saw it, so `rect_params` is in *that* space — and it is `(left, top, w, h)`
where `ObjectRecord.bbox` is `(x1, y1, x2, y2)`. Publishing it unchanged halves every box on a
4K camera *and* puts extents into a field every consumer reads as corners. `FrameGeometry`
undoes both: a per-axis scale when `mux_enable_padding` is off, a letterbox undo when it is on.

**`object_id` is unsigned and "no track" is not zero.** It is `UNTRACKED_OBJECT_ID` = 2^64-1,
which read as an integer is a track id nobody will ever see again. It becomes `track_id=None`
exactly once, in `build_event`. A tracked object gets `track_state="tracked"` — coarser than
`TrackerShard`'s `tentative`/`confirmed`, because DeepStream reports no lifecycle and
publishing a state we did not measure would be worse than publishing a coarse one.

**`frame_num` restarts at 0 on a reconnect.** `(camera_id, frame_id)` is the tag this system
keys on (ADR-002), so a reused pair hands downstream a key it has already seen — a wrong join,
not a lost frame. `FrameNumbering` keeps per-camera ids monotonic across a source restart.

Two further distinctions the event keeps:

* **an empty `embedding` is a skip, not a missing stage.** The secondary ran; this object was
  below its minimum size, or of a class it does not operate on. `missing_stages` names a stage
  that never ran at all, and conflating the two would make a healthy frame look broken.
* **NTP 0 means "no capture time", not 1970.** A source before its first RTCP sender report
  stamps 0, and `PerceptionEvent.build` reports latency 0 for it rather than 56 years.

The emission discipline is `PipelineRunner._emit_resolved` and `_record`, copied deliberately:
`emit` returns a `bool` and never raises, so a dropped event is a return-value check;
asynchronous refusals are drained afterwards with the tag they belong to. And the probe runs on
a *streaming thread*, where an exception is swallowed by the C caller and the buffer vanishes —
so it catches everything, counts it against `pipeline_build_failures_total`, and always returns
`PadProbeReturn.OK`.

---

## 4. What is inert under this topology

An operator reading the settings tree should know which knobs stop meaning anything here.

| setting | under `deepstream` |
|---|---|
| `dynamic_batching` (per model) | inert. nvstreammux batches by camera count and `batched-push-timeout`; nvinfer batches objects up to its own `batch-size`. |
| `instance_groups` (per model) | inert. One nvinfer per model per graph, and its streams are NVIDIA's. |
| `scheduler.placement_policy`, the queues | inert. There is no queue to be fair in: the muxer is the only scheduler and it is round-robin over sink pads. |
| `pipeline.workers`, `queue_capacity`, `frame_budget_ms` | inert. No pipeline worker exists. |
| `pipeline.reassembly.*` | inert. A frame's stages complete inside one buffer's trip through the graph; there is nothing to reassemble and nothing to time out. |
| `ingest.*` reconnect/backoff/health | inert. `nvurisrcbin` owns reconnection; only `reconnect_max_ms` is used, as its `rtsp-reconnect-interval`, and `latency_ms` as its jitter buffer. |
| `pipeline.tracking.*` | inert. `nvtracker` is the tracker; `topology.deepstream.tracker_*` configures it. |
| `pipeline.class_labels`, `score_threshold`, `max_detections`, `result_sink*`, `source_id` | **live.** They are what the generated configs and the emitted events are made of. |

**Backpressure is different, and this is not ADR-005 parity.** With `live-source=1` the muxer
does not wait: a camera that misses the `batched-push-timeout` window is simply not in that
batch, and nothing counts it. This project's own fair queue refuses with `QueueFullError` at
the camera actor — a counted loss charged to the camera that caused it, which is the whole
point of ADR-005. DeepStream's drop is uncounted and its fairness is round-robin over pads
rather than over offered load. That is a real difference in the property this project exists to
guarantee, and it should be stated in any comparison rather than discovered in one.

A related gap: a **dark camera is visible only as `frames_emitted == 0`** for that camera.
There is no health plane here — `nvurisrcbin` reconnects on its own and says nothing — so
"camera 7 has been down for an hour" is an absence in a metric rather than an event. Closing
that is PR2 (§5).

---

## 5. The ladder from here

1. **`ship_segmenter` as a ship-only sgie.** `network-type=100`, `output-tensor-meta=1`, and
   the protos reduced to `mask_area_px` in the probe — or in a `shipvision` kernel, because
   the raw tensors are the problem: yolo26n-seg emits `32x160x160` protos plus `300x38`, which
   is ~3.2 MB per *object* through tensor meta. Reducing on the GPU before the meta is
   attached is the only shape of this that scales, and that is a custom lib, not a config key.
2. **`ship_recognizer`.** Gallery matching over the ship embedding, which is CPU work on a
   vector the probe already has; the only question is where the gallery lives.
3. **The health plane.** Per-camera `frames_emitted` is already there; what is missing is the
   "no frames for N seconds" judgement and a `/health` to read it from — the shard exposes no
   HTTP today, deliberately (see §6).
4. **MTMC consumes these events unchanged.** `track_id` plus `embedding`, keyed by camera, is
   exactly what the cross-camera tier needs and it already rides `PerceptionEvent`. No new
   contract, which is the payoff for keeping the sink common.
5. **`nvinferserver` after a Triton-gRPC facade.** Then a DeepStream graph could dispatch into
   this project's scheduler, and B/C/D stop being mutually exclusive.

---

## 6. Live-run blockers (read before promising a number)

* **The bbox parser.** The shipped `ship_detector` is end-to-end, so a `.so` exporting a
  `NvDsInferParseCustomYolo26`-shaped function is required — ~60 lines of C++ against
  `nvdsinfer_custom_impl.h`, which needs the DeepStream SDK to compile. Config generation
  refuses without one, so this cannot be forgotten into a run that reports zero detections.
* **Engine/TensorRT coupling.** `model.plan` is valid only for the TensorRT version and GPU
  architecture that built it, and the DeepStream image's TensorRT is unlikely to be the one
  `scripts/build_engines.py` used. Mitigated by `onnx-file`: declare `onnx_file` in the model's
  `parameters` and nvinfer rebuilds instead of refusing to start.
* **Old vs new `nvstreammux`.** DeepStream 6.1 introduced a second muxer behind
  `USE_NEW_NVSTREAMMUX=1` with no `batch-size`, `width` or `height` — it takes them from its
  own config file, so the generated ones would not describe the running graph. `build_branch`
  asks for the property, refuses with the variable named, and logs the properties the element
  does have.
* **`pyds` is not on PyPI.** It ships with the SDK's `deepstream_python_apps`. Absent, the
  shard fails at start with a typed `SourceUnavailableError` naming the install — never at
  import, which is what keeps this whole package in the offline tier.
* **NTP timestamps: `attach-sys-ts=0` means "the source's NTP stamp if one exists, else
  NOTHING"** — not a fall-back to arrival time (#32 round 4 corrected this inversion). A file
  source never stamps; an RTSP source stamps only once RTCP sender reports flow, and
  DeepStream's own samples additionally call `pyds.configure_source_for_ntp_sync()` on the
  source bin — whether `nvurisrcbin` does that internally is a live-run check. The probe
  covers every absence: an unstamped frame gets the probe's receipt as its capture time,
  labelled `extra.capture_origin="probe"` (a source stamp is `"source"`), so `latency_us`
  is never a silent zero and the bench can tell the two clocks apart.
* **`num-detected-classes` truncates the class space to the labelled ids — a deliberate cross-plane divergence.** The Python plane keeps out-of-map ids and publishes `UNKNOWN_LABEL`; this topology drops them at the parser, its only pre-tracker gate. Same engine, same frame, two object sets on the default COCO checkpoint — stated here so a parity comparison reads the difference as a decision, not a bug. A COCO
  yolo26 engine emits ids up to 79; ids at or above the declared count are dropped by
  nvinfer's contract, which here filters the classes this deployment ignores — a useful
  coincidence, not a decision. What a *custom* parser does with `classId >=
  num-detected-classes` is a live-run check.

---

## 7. The image

This box cannot `docker build` (buildkit gives each `RUN` its own PID namespace and this kernel
refuses to mount `/proc` from an unprivileged userns), and only `--network=host` has outbound
connectivity (the rootless daemon was installed `--skip-iptables`, so the bridge has no NAT).
`deploy/deepstream/image.sh` is therefore `docker run` + `docker commit`, in the same shape as
`deploy/rootless/gst-image.sh`, layering `deepstream_python_apps`' bindings and this project's
wheel onto NVIDIA's `deepstream:*-triton-multiarch` base. It is runnable documentation: on a
box that can pull that base it bakes the image, and everywhere else it is the exact recipe.

Then, per shard:

```bash
shipinfer fleet -r model_repository --topology deepstream --gpus 0,1,2,3
```

and to see what a shard would be given, on any machine, with no GPU and no DeepStream:

```bash
shipinfer deepstream -r model_repository --gpus 3 --dry-run
```
