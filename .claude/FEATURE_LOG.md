# Feature Log

One entry per large feature or seam change. Append-only, newest on top. Skip it for tiny
edits, typo fixes and pure docs.

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
  **name** — `Literal["tracking_critical", "high", "normal", "background"]`, derived from
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
