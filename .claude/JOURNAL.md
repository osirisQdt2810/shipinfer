# Journal

## 2026-09-04 — the V151 handoff, executed: #118–#121, and P6's register emptied

**The handoff worked as written.** "tiep tuc" -> RESUME HERE -> four PRs, three merged.

- **#118 (docs-caps ratchet, V145-W3 + V145-ARM)** — merged after ONE round. The review's
  BLOCKING was real and I checked it against the hook before fixing: `_over_cap` filtered
  stdout on `"(max "`, and `check_docs.py`'s comment-block path short-circuits on a
  REASONLESS `# doc: long` *before* the cap comparison, so its only finding says
  `needs a reason` and carried no `(max N)`. An over-cap comment block was therefore invisible
  to the gate — #89's defect arriving through the gate built on top of the hook. Fixed by
  counting every non-blank stdout line, and PINNED by a probe test rather than a comment.
- **#119 (SourceUnavailableError redaction)** — merged round 1. Found by reading both planes,
  not the ledger. Before opening, two things the branch was missing: a C++ test (the sync rule
  — the Python half had six checks and the C++ half none), and a comment cut to four lines
  because **#118's own ratchet caught it within the hour**.
- **#120 (P6-D1/D2/D3)** — merged round 1. All three went the same way: the C++ plane was
  already right and Python moved, so `csrc/` carries only comment repairs. The register in
  `benchmarks/parity/known.py` is now **empty**, and `test_ingest_parity` reports
  `41 checks, 0 failure(s)` with no `KNOWN:` line at all.
- **#121 (PriorityBands, V149-runners step 4)** — open. The ledger's "may not warrant its own
  module" was right about *placement*; the seam is "which band, and who said so", because one
  lock guards both tables.

**Three traps worth carrying forward.**

1. **A git worktree's editable install resolves `shipinfer` to the PRIMARY checkout.**
   `pyproject.toml`'s `pythonpath = [".", "src"]` protects `pytest`; nothing protected
   `scripts/emit_parity_golden.py`, which was emitting parity goldens from `main`'s plane.
   Caught only because three scenarios came back byte-identical after a deliberate behaviour
   change. Fixed in #120 and pinned by a test that asserts `sys.path[:2]` positionally —
   comparing the imported module would be vacuous on a single-checkout CI runner.
2. **`git checkout -- <file>` after an uncommitted revert-probe eats the work.** Lost four
   edits that way on #118. Commit first, then probe.
3. **Two "already passing" gates disagree.** `Auto-merge` shows SKIPPED (not failed) when a
   review comes back BLOCKING, and the `PR description` check is a separate job whose failure
   also silently skips auto-merge. Neither looks like a verdict.

**Where things stand.** V149 is complete once #121 merges: `runners/`, `topology/`, `cli/`,
`engine/` and `api/` are all trimmed, and `topology/base.py` (3.07) and `api/streams.py`
(2.23) stop high for the structural reason already in the ledger — do not re-open them.
P6's remaining work is PR-B (scheduling-seam parity) and PR-C (csrc runners re-baseline).

**Environment:** `export PATH="/home/dungha15/workspaces/shipinfer/.venv/bin:$PATH"` before
any pytest — the venv is not on PATH after a restart.

## RESUME HERE — 2 Sep 2026, ~02:3x UTC (V151: session handed over) — DISCHARGED 4 Sep, see above

**Typing "tiếp tục" is enough. Do this, in this order:**

1. `gh pr list --state open` — if a shipinfer PR is open, carry it to merged first (one PR at
   a time). On BLOCKING, check each finding against the code before fixing it.
2. If none is open, open the next branch below (they are pushed, rebased and green).
3. Then take the next `[~]`/`[ ]` item in `.claude/TASKS.md`.

**Two branches are pushed and ready to open, in this order:**

| branch | what | state |
|---|---|---|
| `chore/docs-caps-ratchet` | V145-W3 + V145-ARM: the docs-cap and Markdown ratchets | rebased on main, tier green, pre-commit clean |
| `fix/source-unavailable-redaction` | a credential leak in `SourceUnavailableError`, both planes | rebased on main, both tiers green |

Open them with the template (`.github/pull_request_template.md`, every heading, `### Test
Details` included — a missing heading fails the `PR description` check and silently skips
auto-merge), label `automerge`, and write the body **from `git diff origin/main`**.

**`fix/source-unavailable-redaction` is the one to explain carefully.** It is a real defect
found by reading both planes rather than the ledger: `core/errors/ingest.py` states the rule
("the message becomes `CameraHealth.last_error`, which the health API serves") and applies it
to `SourceOpenError` and `FrameDecodeError` — but not to `SourceUnavailableError`, on either
plane, and that is the error the fatal-open path stores. A `rtsp://admin:s3cret@host` that
cannot be opened was served verbatim to every reader of `GET /streams`. Fixed on both planes
per the sync rule; six tests, red-first, parameterised so the two siblings that already
redacted stayed green. It does **not** close P6-D1 — the type-prefix question is untouched and
`last_error_type_prefix` still explains the remaining difference.

**Where V149 got to.** `runners/`, `topology/`, `cli/`, `engine/` and `api/` are all trimmed and
merged (#107, #110–#117). The two files that stop high — `topology/base.py` 3.07 and
`api/streams.py` 2.23 — stop for a structural reason recorded in the ledger: an ABC's remaining
prose is contract text and a router's is the status-code argument, and ~150 lines of code cannot
carry either at a low ratio. Do not re-open them.

**Three checks were added this session and all three are ratchets or zero-tolerance gates**, so
a new PR that regresses prose hygiene now fails in CI rather than merging green:
`TestProseKeepsTheProjectsLineWidth` (96 columns, allowance 48),
`TestNapoleonFieldListsStayIndented` (zero — main had none),
and on the waiting branch `TestDocumentationCapsOnlyGetTighter` + `TestTheProjectsMarkdownKeepsItsCaps`.

**The next real work after those two PRs** is P6-D1/D2/D3, and the reading is already done:

* **P6-D1** (`last_error` spelling) — Python prefixes `"{type}: "`, C++ has only `what()`.
  Converging means dropping the prefix on the Python plane or inventing type names in C++.
  Deleting the `last_error_type_prefix` entry from `benchmarks/parity/known.py` is part of the fix.
* **P6-D2** (`consecutive_failures` after a fatal open: 0 py / 1 cpp) — **Python is internally
  inconsistent**, which decides it: `_record_failure` computes `failures = attempts + 1` for the
  *state* decision but `health()` reports `backoff.attempts`, and the fatal path never calls
  `next_delay()`. So the state says a failure happened and the count says none did. C++ keeps its
  own `consecutive_failures_` and is coherent. Fix: Python counts failures itself.
* **P6-D3** (`stop()` fate stickiness) — documentary. C++ latches `thread_abandoned_` because a
  detach is irreversible; Python re-reads `is_alive()`. Decide whether Python latches too.

Each is a two-plane change, so the C++ side needs `python scripts/build_csrc.py --offline` and
the five `csrc/build/test_*` binaries (395 checks, all green as of this handoff).

**Environment note:** this shell had no venv on PATH after the restart —
`export PATH="/home/dungha15/workspaces/shipinfer/.venv/bin:$PATH"` before any `pytest`.
Work happens in the worktree `/tmp/vw`; the primary checkout stays on `main`.

## 2026-08-27 (evening) — A2 begins: PR-0…PR-2 landed; main's CI had been silent after every auto-merge

- **A2 PR-0 (#54), PR-0b (#55), PR-0c (#56):** workflow-only prerequisites. #55's glob for the
  C++ offline binaries also matched the `.o` files and broke main (`Permission denied`) — my
  error; #56 fixed it after rehearsing the exact step against a fake build dir (memory rule 8).
- **PR-1 (#57, `engine/`):** 74 files, 38 renames, Python + csrc mirror in one PR, silent shim;
  internal review found a vacuous shim test and a stale CLAUDE.md tree before opening; CI
  review APPROVE round 1, and it recorded that a source-first/tests-second split would have
  left the suite red (seven tests import engine submodules the shim does not re-export).
- **Found: a push made with GITHUB_TOKEN creates no workflow run**, so #52, #53 and #57 — every
  PR the pipeline merged itself — landed with no main CI at all (cpp-offline, gst lane, kernels
  never ran). Only hand merges triggered it. **#58** adds `workflow_dispatch` to ci.yml and has
  the Auto-merge job dispatch it; a hand dispatch for `7706b15` was green on all eight jobs and
  ran `test_engine` (27 checks). The self-dispatch half is checked on #59.
- **PR-2 (#59, `api/`):** open. No shim (callers edited); `api` is the only layer allowed to name
  fastapi — every other layer got a row, and a new test asserts every package on disk has one.
- **PR-4 (#60, `launch/`):** merged 16:37, APPROVE round 1 — the supervisor moved byte-for-byte
  with the #33–#41 invariants' tests; its new arch test catches a top-level `import torch` the
  static hook is blind to.
- **PR-3 (#61, `runners/`):** open. Internal review found two real blockers — the `pool` element
  relabelled its payload's cap (`produces` now `*@*`, resolved from the inbound edge) and an
  abandoned worker's in-flight future never resolved (now failed at the stop deadline) — plus
  seven nits (skip-and-continue was untested on a linear chain; expiry re-check and reverse close
  untested; per-camera counters; donor map into the loader; `per:`/`scope:` and the batch ≤ workers
  bound named). Every fix revert-checked.
- **PR-3 merged (#61, 17:33)** after a second CI round: the in-flight registry held one item per
  worker while a worker holds a whole `frames_per_wakeup` batch — fixed to hold the remainder;
  settings now reach the `pool` elements; expiry re-checked per model element; no unnegotiated-cap
  fallback. Round-2 should-fixes (typed errors flattened by the walk; `_inflight` cross-write after
  restart; stats under-reporting queue losses) → PR-3b in build.
- **PR-5 (gRPC contract)** built on #61's tree; internal review found the grpcio floor (1.60) below
  the committed stub's requirement (1.71.2) and a double snapshot in `Health`; fix pass running.
- **PR-3b (#62) merged 18:33** after two rounds: round 1 found the deeper form of the restart bug
  (a stale abandoned worker read the new cycle's stop event and queue) — fixed by handing every piece
  of per-cycle state to the worker thread as arguments. Its approval left four observations → PR-3c
  in build (stale worker breaks out of its batch; `in_flight` after an abandoned stop; a separate
  backpressure counter so the stats identity holds; one `Args:` heading).
- **PR-5 (#63, gRPC control plane)** opened 18:4x on #62's main: 2017 passed; the internal review's
  two blockers (grpcio floor below the stubs' requirement; `Health` double snapshot) fixed first.
- **PR-5 (#63) merged 19:20** after two CI rounds: round 1 found `stats()` importing protobuf
  before the typed path (the venv and CI both carry the extra, the mask fixture masked only `grpc`
  — memory rule 9) and `Stop`'s idempotence check outside the lock; fixed with one guarded door.
- **PR-3c (#64) merged 19:38**, APPROVE round 1 (stale worker breaks out of its batch; `in_flight`
  after an abandoned stop; `items_backpressure`; one `Args:`).
- **PR-6** built (server/ and the argv mechanism gone; child entry `python -m shipinfer.cli.shard
  --shard-id N --control-port P`; `ShardService` with a `RunnerFactory`); internal review BLOCKING
  x3 (stale base reverting #63's fixes; abandonment count zeroed on a failed start; FEATURE_LOG entry
  overwritten) + 11 nits; fix pass running. Split: PR-6a = the two core/ renames (a real behaviour
  change hides in it: the spill gate drops its `kind == "service"` half), PR-6b = the rest.
- **PR-6a (#65)** opened ~20:0x (the two `core/` renames + the deliberately narrowed spill gate);
  CI round 1 found operator-facing error messages still naming the deleted `topology.deepstream.*`
  key and the gate change untested — both fixed (`b740666`), the service-tier design doc frozen
  as the pre-A2 record; round 2 under review.
- **PR-6a (#65) merged 20:35** (round 2). **PR-6b (#66) opened ~20:5x**: `server/` and the argv
  mechanism deleted, the fleet over gRPC, `shipinfer run`, the two-flag shard entry — the last A2
  slice. Phase B planning started on its tree.
- **PR-6b (#66) merged 21:24 — phase A2 complete.** Round 1 caught a real regression: `shipinfer run`
  asked the driver for GPUs before the settings existed and overrode `SHIPINFER_DEVICES__VISIBLE_GPUS`
  at flag priority (8 shards instead of 2); fixed in the deleted command's shape. From #52 (the
  design of record) to #66: `server/` and the argv mechanism are gone, the fleet runs over gRPC, the
  shard child takes exactly `--shard-id --control-port`, `topology/`, `engine/`, `api/`, `launch/`,
  `runners/` are the packages arch.md §9 names. Every PR went through the CI review loop; nine of the
  fifteen needed a second round, each for a real finding.
- **Phase B** planned and started: B1 (runner owns cameras, `--inputs`) and B2 (per-camera queue
  attribution, both planes) in build.
- Main CI dispatch after auto-merge verified on #59–#66.

## 2026-08-27 (late afternoon) — V142 reverses the GIL plan; A1 `topology/` lands as #53; A2 planned as six PRs

- **V142:** the operator revoked V140 (i) the same day — no GIL code in shipvision, ever; it
  delivers algorithms, at most a mutex around `tracker.track()`; slowness accepted. The Phase-0
  coder was stopped before any commit; worktree discarded; shipvision `main` untouched. arch.md
  §7 rewritten to the server-side answer (fewer fatter calls; the hot plane in `csrc/`, V34;
  per-worker streams passed from the server — `NativeImageOps(stream=)` already exists).
- **#53 (A1 `topology/`):** Element ABC + caps + registry per kind + chain loader, offline-only,
  ADR-017. Internal review before opening found three real blockers (fixture refused by its own
  loader; wildcard passthrough laundering a gpu→cpu download; `register_lazy` bypassing the kind
  check), all fixed with revert checks. CI review round 1 found a fourth — under skip-and-continue
  the unconditional `embed_ship`/`recognize` ran on every person crop — fixed the fixture way
  (every element on a class branch guards itself) with person/ship walk tests. Round 2 pushed.
- **A2 planned as six PRs** (engine/ → api/ → runners/+inprocess+pool → launch/ → gRPC contract
  → fleet-over-gRPC + argv deletion), with PR-0 a workflows-only prompt fix. Key traps: the
  `filterwarnings` error rule vs shims/pb2; `shared_by`/`share_rank` must ride the RPC or two
  shards on one GPU double-load instances silently; `CUDA_VISIBLE_DEVICES` is the one thing that
  stays in the spawn env.
- Housekeeping: worktrees of merged PRs removed; only `/tmp/mps` (ledger), `/tmp/a1` (#53),
  `/tmp/pr0`, `/tmp/e1` remain.

## 2026-08-27 (afternoon) — #52 round 2: ADR-016, the committed link probe, C_ctx measured

- **Merged 12:53 UTC after round 3.** Round 3's blocker was mine: the cited probe logs were
  gitignored (`*.log`) and never in the tree; fixed by un-ignoring `benchmarks/link/results/`
  and pointing the runner there. Reviewer's NBs adopted: timed-vs-inferred pairs, PXB as a
  fixed ~49 ms per-copy penalty, C_ctx annotated as whole-context at K = 1.

- Review round 1 of #52 (docs/arch.md) had three real blockers: the DataPool reversed
  ADR-002/ADR-015 without naming them (and left ADR-015's per-peer-context-cost and
  handle-lifecycle objections unanswered); V127–V140 were not in the repo's user.md; the
  P2P table had no traceable artefact. All three were correct.
- Wrote **ADR-016** (supersedes ADR-015's payload transport, amends ADR-002's payload
  clauses, threading core intact): K-neighbourhood (K = 3 default) bounds the contexts;
  the cost is a measured input; the budget is enforced at start-up; lifecycle = open at
  join / close at drain after quiesce / generation-stamped tickets / death-notice
  invalidation / graph-I/O carves pinned (ADR-008).
- f6's probe scripts existed in the shared scratchpad but no raw log did, so the probe was
  **re-run in the pytorch container** and committed as `benchmarks/link/` with logs. Same
  numbers to the µs. New cell: **one foreign CUDA context costs 208 MiB on the OWNER's
  device and 0 on the opener's** (own context 243 MiB) — so a device is charged by its K
  openers, `K × 208 MiB`, not by how many peers it opens. That inverted the budget formula
  in both documents (`slabs + own_ctx + K × C_ctx + engines ≤ device − reserve`).
- Two-session state: shipinfer-f6 vanished (not in ListAgents); shipinfer-32 is an
  unrelated fresh session with no lane. f6's lanes (shipvision GIL+streams Phase 0,
  C-lanes, /tmp/ci, /tmp/t4) revert to this session. Recorded in the QUEUE block.
- Lesson (memory rule 6 again, positive case): the probe artefacts were written to the
  branch in the SAME command as the run, before any prose cited them.


Newest on top. One entry per working session: what changed, what it cost, what is next.

---

## 27 Aug (early) — #31 merged; #32 (deepstream topology) through six review rounds

- **#31 (`perf/pinned-staging`) merged 26 Aug 23:43 UTC after five rounds** — pinned
  ping-pong staging for the ops layer's D2H copies, with the structural single-chunk rule;
  the 25% identical-code control row exposed the box's noise floor, so every micro claim
  was withdrawn; the quiet-window rerun fired 04:55 and the table is POSTED to #31 (staged wins 1.4–1.7× with a 0.06–4% A/B/A control; crop stays staged).
- **#32 (`feat/deepstream-topology`) opened and driven through rounds 1–6, all findings
  real**: the lazy sink (r1), the unfiltered secondary (r1), the lost-in-transit sink wiring
  + missed round (r2–3, both process defects mine and told plainly), attach-sys-ts semantics
  inverted → capture_origin distinguishability (r4), the parser/lib pair validator + start()
  inside the try (r5), and pad-added matching names instead of caps + zero builder coverage
  (r6: FakeBuilderGst now drives chain order, video-links/audio-does-not, missing element,
  USE_NEW_NVSTREAMMUX, refused link). Tier 1675/93 suite at `2ea6c09`; awaiting round 7.
- **V124 answered with a fact-check** (gitlink WAS latest main; the checkout was on a stale
  branch — synced; drift into torch_ops acknowledged, lanes V124a/V124b opened) and **V125
  honored + indexed** (shipvision checkout stays on main).
- **P4-PR1 pre-flighted**: already based on #31's merge commit; pushes the moment #32 lands.

## 27 Aug (midday) — two workers, the C1 answers, and the architecture pause

- **The session forked**: a process restart left two live workers sharing one lineage
  (shipinfer-23 = pre-restart, shipinfer-f6 = continuation). A live /tmp/p4 write collision
  was caught (f6's resumed coder vs 23's session), the coder stopped, and a QUEUE protocol
  went to the top of TASKS.md: one open PR, announce before opening, partitioned lanes
  (23: P4/P-lanes; f6: C-lanes/model_repository/bench).
- **P4-PR2 completed by 23**: #48 (the RTSP loopback pixel — the oldest owed evidence),
  then #49/#50/#51 CI fixes; the kernels job passed for the first time in its existence;
  main fully green, 8/8 acceptance.
- **f6's C-lane answers**: C1c — fleet floor 95.9 img/s/3 GPUs; segmenter's single
  instance was the binding stage; count:2 → 143.8 (+50%), shipped as #47 with the
  measurement in the comment (and the expand()-divides caveat from its review). C1a —
  closed with all three tiers run; algo tier: crop 52.3 ms/frame > detect 45.8, pools
  buying 8.4×. C1b — the 8.4× ROOT-CAUSED as a GIL convoy (shipvision bindings never
  release the GIL; Little's law pins ρ≈1.0) and CONFIRMED by a workers=1 discriminator
  (crop 52.1→8.5 ms, serial 140.5→51.6 ms); the fix collides with V70 and awaits the
  operator; true serial DAG cost 51.6 ms/frame.
- **V129 pause**: the operator questioned the whole architecture (server/ tangled,
  topology-vs-command, their 4-point model). 23 restated as-built (converging
  independently with f6's three facts), and the discussion ran V130–V136 live. V132
  DECIDED: `topology` = the declared element chain, `runner` = the execution mode
  (inprocess/fleet/deepstream); track/MTMC are chain elements yet shardable; KServe stays
  as the engine's second face. All new PRs held until the discussion resolves.

## 27 Aug (late morning) — the review loop as a grinder: #39–#44, fourteen merges on the day

- **#39 (P4-NB3)**: four rounds. The reviewer reverted my fix under ASan to validate it,
  then proved my regression couldn't tell fix from no-fix in the plain build (their
  weak_ptr witness applied; flip-proven both ways); round 2 blocked ONLY on a ledger claim
  that wasn't in the diff — the rule's letter, enforced; round 3 found the guard's own
  docstring arguing for its defeat (measured, not asserted). Merged 06:03.
- **#40 (ring hygiene R1+R2)**: magic lands last (the fourth birth window unobservable);
  the mesh names absent vs stuck-mid-birth. Merged 06:17 round 1.
- **#41 (ingest parity)**: Python re-check (deadly order driven by a monkeypatched start),
  remove_camera returns the answer, self-stop reads the atomic fate (flip 191/1). Merged
  06:34 round 1. The P4 review-debt ledger emptied.
- **#42→#43→#44 (the DetectNet chain)**: each round's findings real — the EfficientNMS
  quartet slipping the arity gate; my body claiming a pin that didn't exist (reviewer
  measured !=2→<2 green); then MY tightened gate false-accepting swapped-role and
  cov_bbox pairs (their table reproduced as 2 test failures); then V86 re-sourcing that
  CHANGED the answer — NVIDIA's own DetectNet_v2 sample uses NMS+iou 0.5, not DBSCAN.
  Merged 06:42 / 06:53 / 07:19. Chain closed; the take-or-leave trio ledgered (T4-NB1c).
- Process rules that had to be relearned got memory entries: body-before-push (review
  reads the body at trigger time — bit twice), commit-before-flip-proof (a checkout
  restore ate an uncommitted fix once), grep the sentence's FAMILY across code+tests+body.
- P4-PR2 (the GStreamer C++ source) exploration launched.

## 27 Aug (morning) — five merges, the CI-health sweep, V124a phase 1 landed

- **#32 MERGED** (r7 APPROVE, `e72955f`): DeepStream is the fourth topology. **#33 MERGED**
  (3 rounds: shared_ptr lifecycle, the redact fail-open + the ported redaction suite,
  fleet-wide stop deadline): the C++ ingest core. **#34 MERGED** (V109): CI runs
  test_ingest. **#35 MERGED** (3 rounds: the abandonment debt paid at the re-check with a
  flip-proven weak_ptr witness; Python stop synced; the ledger-claim-must-be-in-the-diff
  lesson). **#36 MERGED** (V109): the kernels job can finally clone its submodule (born
  broken in #28). **#37 MERGED** (r1): the mesh flake — three hits, two spellings, one
  birth race — is dead at the root: both mid-birth windows are retryable "unborn" now.
- **shipvision #12 MERGED** (V124a phase 1: swap_rb + nms max_output; the float-cap
  finding); parent gitlink bumped to `90b0c41` on main, operator checkout synced (V125).
- **The quiet-window table posted to #31**: staged beats pageable 1.4–1.7× with a 0.06–4%
  A/B/A control (box never quiet, load ~25 — the control is what licensed the medians);
  **crop stays on staging**, the merge-time commitment discharged.
- Reviews escalated P4-NB3 (concurrent CameraActor::stop = latent std::terminate in the
  hammer test → crashing-CI class; NEXT ingest fix) and found a fourth, narrower ring
  window (R2: magic lands first in create's header write). Ledgered: R1/R2, P4-NB2-py,
  P4-NB4.
- Process: two evidence blocks caught written-before-run (both replaced with real output;
  memory extended — REPLACE_* is absolute); the sv12 fix once posted a reply before its
  commit existed (fresh clone had no git identity) — commit landed minutes later.


## 26 Aug (evening → night) — T3 shipped end to end; C44 answered; B4/B5 closed

- **#30 (`perf/batched-torch-crop`) merged 19:43 UTC, APPROVE round 1** — C44's lever 1:
  `crop_batch` one batched pass, constant in the crowd; 98 offline tests on a frozen-loop
  reference, parity byte-identical; the honest three-layer evidence (degenerate-biased
  fixture, valid-box wash per call, system −11% host CUDA-API / upsample 4083→322). The
  C44 `generatedNativePointwise` attribution corrected (it is TensorRT's, invariant).

- **#26 (`service` tier) merged 17:39 UTC after five review rounds** — every finding real:
  round 1 the owner threading (ONE sweeper, workers never touch a ring); round 2 `_DeviceSpan`
  retention + saturation held with the slot (not failed as text); round 3 probe-not-spin and
  the `Model` admission seam (`admit_local` / `try_dispatch_local` / `count_local_rejection`);
  round 4 `WireRefusedError` spills, per-lane stamping, typed status codes, D2H-before-claim,
  in-flight depth, reader snapshot, `pending_timeout_ms`; round 5 APPROVE.
- **#27 (bench/topology) merged 18:00 UTC, APPROVE round 1**: both sweeps re-run on the merged
  tree — every rung SUSTAINED to 72 img/s; B put every person crop on the crowded GPU
  (1572/0/0), C spread them (847/345/382) with ship_embedder shared back. T3 closed; T3b
  (crowd fan-out) opened as the honest residual.
- **#28 (ci: C++ offline tier + prompt rename) merged 18:06 UTC by hand under V109** after
  py3.10/py3.12 tests passed and the review job failed with exactly the documented
  workflow-validation error. B4 and B5 closed.
- **C44 answered by the Nsight timeline** (12×5, GPUs 3–5, 30 s, merged tree): GPUs ~14%
  kernel-busy; host threads blocked in `cudaMemcpyAsync` (22.9 s, max 116 ms) and ~400 k
  kernel launches (~13 k/s); the per-crop torch pointwise storm is 74 launches/frame; D2H
  tails hit 39 ms (pageable, segmenter-sized). Levers for C1: fused `crop_batch`, pinned
  D2H staging for big outputs. Evidence: `c44-nsys-stats.txt` (scratchpad).
- Operator: **V108** (T4 is a topology, not a competitor benchmark — four topologies under
  one backbone), **V109** (standing self-merge grant), **V110** (the tier must hold for the
  full DAG), **V111** (the stale repo copy of user.md — answered by the snapshot PR).

## 2026-08-26 — Topology C is built, and the first real runs find what the fakes could not

**What merged.** #21 (the docs snapshot) after three review rounds, each of which found something
real: the request log carried two assignments for V41–V51 (renumbered once, every citation shifted,
verified mechanically — Section 1 runs V1–V90 plus V91, the appendix V41–V51, no duplicate, no gap),
the design doc's pinned budget had three values for one quantity (derived once: 72 rings ≈ 0.9 GiB
on the box existing once, ≈ 0.44 GiB registered per process for 4 shards × 3 models), and its error
block declared `RingFullError` a sibling of the type the spill path catches (it is a
`QueueFullError`, as built). #22 (the submodule at the per-algorithm tree; the tracking stage imports
`shipvision.mot`). Open: #23, the RTSP path, with the baseline gate fix (C48) folded in.

**T3, steps 2–4, on three stacked branches.** The wire and the proxy (`RemoteInstance` is a
`Placeable` with `device = cpu`; `ResultReader`, `RingIngress`), the `service` topology
(`ServiceSettings`, `Topology.shard_environment`, `ServiceMesh`, `Model.attach_remote`,
`InferenceServer` joining the tier after the models load and leaving it first on stop), and the
harness driving the shards (`benchmarks/harness/shards.py`: one child per shard through the real
`Fleet`, an explicit `shard_cameras` split to model a crowd that moved after planning). ADR-015 and
the feature-log entry are on the topology branch. Also on the way: every fleet shard ran `serve`
with no HTTP, so none was addressable (`serve_command(http_port_base=)`); `Model.infer_local` so
work that crossed once is never re-routed; `slot_bytes` = 1.5 MiB + 64 KiB because the heads travel
in the slot; a ring `open` no longer registers with the resource tracker (bpo-38119).

**What the real runs found.** 37 fake-model tests were green while every real shard died at start:
the edit that inserted `attach_remote` had landed between `total_depth`'s `@property` and its `def`,
making `attach_remote` a property and `total_depth` a method. The two-process multigpu test (two
`serve` processes through the real `ServiceTopology`, GPUs 3 and 4) found the first; the harness's
occupancy probe — reading `total_depth` as main declares it and handing a bound method to
`json.dumps` — found the second, as three shards with a metadata line and no samples. Both fixed,
and `test_service_engine.py` (two `InferenceServer`s in one process as shards 0 and 1) now covers
the engine's own path. The rule this confirms: a seam is proven by the first run through the real
object on both sides, and the fakes are for the protocol, not for the wiring.

**Evidence so far.** The two-process test: 24 requests to shard 0, 19 executed there, 5 by shard 1
through the ring. The fleet under a crowded split (6/3/3 cameras on GPUs 3–5, swept from 12 × 2 fps
to ×3): every rung SUSTAINED up to 72 img/s — 36 on the crowded GPU — and the per-device table shows
the fleet's locality exactly (every person crop on GPU 3, every ship crop on 4 and 5). `person_2K`
yields about one person crop per frame, so the crop stage is nowhere near binding at these rungs
and C has nothing to balance there; the informative rungs are ×4–×5, where the first run failed on
the crowded shard's embedder time-outs. The `service` sweep then needed three more fixes the fakes
could not see (the wire refused the pipeline's device-resident tensors; `abandon` existed only in
the tests' imagination; no real batch fit a 1.5 MiB slot — slots are per model and direction now),
and with them, plus the bounded replies and the ring's closed-handle rule (#25 rounds 1–2), **C's
confirmation sweep is clean: x1–x3 all SUSTAINED to 72 img/s — B's own ceiling — zero errors, the
borrow visible at every rung (x3: person_embedder 742/377/389 across GPUs 3/4/5 against B's
1582/0/0)**. With ~1 crop per frame the tier has nothing to *win* here — C matches B and spreads
the crop work; its win case needs the crowd fan-out the sizing assumes, which this dataset cannot
produce.

## 2026-08-25/26 — The split lands piece by piece, and the review keeps finding the unwired guard

**What merged.** shipinfer #12–#17 in order — pipeline, the bench's other two tiers, the offline
tier's hidden devices, the C++ data plane (six review rounds), the infra/docs slice with the
`Stop` hook, and the three reasons the fused kernels were unreachable. shipvision #6–#10: reid,
mot, mtmc, eval/tune, and S1 — the `tracking` → `mot` rename with one package per algorithm.
Open at the end of the stretch: shipinfer #18 (the topology seam and the fleet, review round 1
answered) and shipvision #11 (S3: the native tree, bindings and build in the per-algorithm
layout — `342 passed` on the native tier inside the container, `2025 passed` for the whole tree).

**S2 does not exist.** The ledger promised "StrongSORT and BoostTrack and the Python tweaks" as
the second slice of the V79 restructure. `git ls-tree` over every remote ref finds no such file,
and the remaining Python diff between the source branch (08:39) and main is main being *newer*
(#3, #5, #9 landed 14:27–19:13). Struck with the evidence, not deferred — a ledger line that
describes work nobody can do is the kind that gets "continued" forever.

**The port is real now (P1a–d).** `FairPriorityQueue<T>`, the five policies behind a registrar,
the dispatcher, `ModelInstance` as thread + queue + window, `Model::infer` → `std::future`, the
graph as stages over it, and a bench that runs the Python plane's shape: ~390 img/s at 48
workers, ~470 at 96, balanced across GPUs 2–5 to 1%, 0 failed, 0 timeouts. Rebased onto main's
squash of #15 with an identical base tree (`git diff` empty where it matters); the PR is queued
behind #18.

**What the review found, and the shape it keeps having.** The `Stop` hook parsed only below a
`## Now` heading the ledger never had — a mechanism that reported nine open items as "clear",
worse than the reminder it replaced. The same hook would have fired in the CI review job, where
none of its escapes are reachable. `scripts/build_native.py` asserted the phantom `is_available`
that the whole PR was about, so a good build failed the container recipe. `Fleet.supervise()`
consulted only `dead()` over a fleet `stop()` had just emptied, so Ctrl-C spun until SIGKILL.
`ShardPlan.instances_for` — the guard whose docstring promised half the instances per shard on a
shared GPU — had no caller. Round two of the same PR added three more of the same shape: the plan balanced *shards* while two
of four GPUs carried twice the load of the other two — the founding bug rebuilt one level up, hidden
by a metric that measured processes rather than the resource; a signal handler that called a blocking
`stop()` under a lock and deadlocked on the second Ctrl-C; floor division that silently dropped a third
of a device's configured instances. And one of my own making: an append to the append-only feature
log that duplicated the file end to end. Every one of these was **a guard written and not wired**, a
name fixed in one place and not the other, or a metric measuring the wrong thing. The tests that now pin them were written to fail on the
previous commit first, and did.

**The RTSP path, run for the first time.** The ledger's C4 said "wired and tested offline; not yet
run", and six container runs later that sentence reads as a warning. Each run died one layer
deeper, each on a defect no offline test could see: NVDEC's decoder opening a GL display the
headless container does not have; fifty camera threads racing `Gst.init` so half of them probed
an empty plugin registry and reported "no decoder found" on an image with three; PyGObject's
lazy attribute loading racing the same way (`'GLib' object has no attribute 'Idle'`); GIO's
default proxy resolver, libproxy, throwing a C++ exception through the C boundary and
terminating the process (`Unable to read configuration`) with fifty cameras connected and zero
frames read; and the appsink's methods missing because nobody had loaded the GstApp typelib.
Five commits on `fix/rtsp-headless-decode`, each with a test that fails without it. The sixth
run decoded fifty streams through NVDEC and pushed frames through the whole graph — and the
harness refused the number, correctly: one interpreter generated 2% of the offered load. The
RTSP measurement is a 12 × 5 run; the 50 × 20 one is the fleet's to make.

**What it cost.** The org's monthly spend limit cut off both background agents mid-task (the
S3 coder had already built the extension in the container; the T3 planner had read nothing).
The rest of S3 was done in the main session; the T3 plan is still owed and will be written here
rather than delegated until the limit resets. One host-run refusal from the advisory hook, for
naming `tests/runtime/test_native.py` in a host `pytest` command — the offline tier is exempt,
but that file imports torch and the hook is right to stop and ask.

**Next.** The queue: #18 → `port/p1-scheduling` → the docs snapshot (this journal, the ledger,
V78–V90, the two CLAUDE.md rules) → C47's CUDA-free test binaries (every C++ test binary still
links four accelerator libraries, so the fairness invariants cannot run on a machine with no
driver). Then P4 ingest, P5 config-in/events-out, P6 the parity harness; T3 (`service`) and T4
(DeepStream) against the topology seam; after shipvision #11, the parent's
`pipeline/graph/tracking.py` switches to `shipvision.mot` with the submodule bump in its own
commit.

---

**The request log, numbered once.** Review of the docs snapshot (#21) found the request log
carrying two assignments for V41–V51: the verbatim appendix's C++-convention requests, cited
by ~18 rules-index rows and the ledger, and the 24-Aug shipvision requests added to Section 1
under the same numbers. Repaired in one commit: V40 is the `csrc` mirrors `src` request the
index already cited, the duplicate of V49 folded into V49 with its time, the 24–26 Aug entries
renumbered V52–V90 (+9) with every citation in TASKS, CLAUDE.md, this journal and the design
doc shifted the same way, and main's misplaced `V43 — 14:40` (the trackers-the-services-used
note) is V91. Verified mechanically: no duplicate, no gap, no citation that resolves to nothing.
The rule-value paraphrases (V86, V88–V89) moved from the verbatim appendix into the rules table.

## 2026-08-23/24 — The benchmark answers, and the answer is no

**The number exists now.** Baseline **868.2 img/s** measured capacity at 50x20; ShipInfer
**81.4 img/s** measured capacity, found by `--sweep` saturating at the x2 rung (120 img/s
offered from 12 cameras). Ratio **0.09x** against a 5x target — missed by ~53x.

**Where the wall is.** Not the GPUs. At 120 img/s offered, every model queue sustained its
full offered rate with a CI straddling zero — the detector took 120 and retired 120. The
queue that grew, at +38.7/s, is the *pipeline* queue: in front of the Python worker pool and
behind nothing. Not the worker count either, which was swept rather than assumed:

    workers=24    87.6 img/s
    workers=96    81.4 img/s
    workers=192   85.0 img/s

An 8x range for under 8% of movement, non-monotonic. What is left is the interpreter — 12
decode threads plus up to 192 worker threads doing per-frame Python work in one process.
Section 9 of `new-system-architecture.md` already puts decode in separate **processes**, and
this driver does not. **The next PR is the process split, not a kernel.**

**Six review rounds on PR #3, 8 blocking items in the last one.** The one that mattered:
the harness was structurally incapable of producing a speed-up. `is_rate` refused SATURATED
as "a bound", which has the buffer-growth methodology backwards — a saturated run is the
*only* regime in which `offered - growth` is exact. Both systems are offered the same load
by construction, so either neither saturated (each reported the offer back, 1.00x) or one
did (comparison refused). A number is now a CAPACITY, a FLOOR or NOTHING, `ratio_of` is the
single place a pair combines, and floor-over-capacity reads `>= Nx` while
capacity-over-floor reads `<= Nx`.

Three of the eight were holes in guards I had added in *earlier rounds of the same review*.
The pattern: the idea right, the detail wrong, always in the flattering direction.

**Other real bugs found by review this round.** `redact()` failed *open* on a `/` or `@`
inside a password — `urlsplit` follows RFC 3986 where `/` ends the authority, so
`parts.password` was None and the URI was echoed whole, into `SourceOpenError` ->
`_last_error` -> the ingest health endpoint, on every retry. The fair queue's drain skipped
`notify_all` after its row-budget exits became `return`, so a blocked producer slept the
whole 500 ms timeout instead of waking at 50. `pipeline_sink_failures_total` could never
increment, because `emit` is documented "never raises" and the runner counted failures from
an `except` around it — so a dead broker lost every event with a green dashboard.

**The constraint that shaped what could be measured.** The baseline binary only survives
while *saturated*: static-batch plans, `setInputShape` with whatever batch it assembled, so
a partial batch throws inside a worker and aborts the process. It dies at 60 img/s. The two
capacities therefore come from different offers, which is fine for a capacity comparison but
has to be said out loud.

**Still absent.** Plane 3 (MOT/MTMC) does not exist — `shipinfer` imports only
`shipvision.detection.engine_build`. The `std::memcpy` audit is deferred by the operator
until the system is complete.

---

## 2026-08-22 — Project reset: from Sale Hunter scaffolding to ShipInfer

**What this session did.** Replaced the inherited scaffolding (a Shopee desktop-app
template) with the real project: a Triton-shaped multi-GPU inference server for the ship /
person perception pipeline described in `references/bitbucket-subfaceid/docs/`.

**Built.**

- `core/` as packages, not modules — `types`, `request`, `errors`, `settings`, `logging`
  (with a bounded async sink), `metrics` (with pluggable exporters), plus the `Registry`
  primitive every extension point uses.
- `scheduling/` — the part this project exists to own. Fair per-camera queueing with
  priority lanes, dynamic batching windows, five placement policies, and a dispatcher that
  spills to the next-shortest queue rather than dropping a frame.
- `runtime/` — the accelerator seam, on torch. Device manager, streams, CUDA graph capture,
  a pinned staging pool, and three image-op implementations (numpy / torch / native).
- `backends/` — mock, TensorRT (engine + persistent bindings + graph replay), ONNX Runtime,
  TorchScript.
- `server/` — instances, models, a validated ensemble DAG, response cache, health, KServe v2.
- `native/` — C++17 + CUDA/HIP behind pybind11: fused letterbox, batched crop, device NMS.
- A demo `model_repository/` carrying the real DAG on the mock backend, so the whole thing
  is runnable and testable anywhere.

**The correction that shaped the result.** The first version of `runtime/` hand-rolled a
CUDA layer against `cuda-python` — its own caching allocator, pinned-memory management,
stream wrappers, raw graph capture. It worked and it was slower than torch, missing exactly
the parts that are hard. Rewritten onto torch (ADR-003). The hand-written code was kept as
registry-selectable `custom_*` variants: runnable documentation for what the library does,
and the other half of the parity tests.

**Evidence.**

- offline suite: 149 passed, 12 deselected, ~7 s, no GPU
- GPU tier: 12 passed on 8 × RTX A5000
- `shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 3 --skew 8`:
  3000/3000 completed at 998 req/s, p99 7.6 ms, 0 rejected; per-device share 11.7–13.2%
  across all 8 GPUs; every quiet camera served despite 8× skew
- fused letterbox, 8 × 1080p → 640², writing into a torch CUDA tensor: **9.7 ms (822 img/s)
  vs torch's 13.7 ms (585 img/s)**, output bit-identical

**Notable bugs found and fixed on the way.**

- `np.clip(..., out=fancy_index)` in the numpy crop path neither wrote back nor cast — it
  raised, and only for some dtypes. Caught by the GPU parity test; host-tier tests added so
  it would have been caught without a GPU.
- The ensemble DAG validated tensor *names* but not shapes, so a 512×512 ship crop fed a
  256×128 embedder and failed on the first frame instead of at start-up. Validation now
  type-checks every edge.
- The bench discarded futures completed during intermediate waits and reported 496 of 3000.

**Next.**

- `pipeline/` and `ingest/` are declared in the layout but not yet implemented: NVDEC
  decode, per-camera actors with backpressure to the decoder, and the `(cam, frame)`
  reassembler.
- Kafka result publishing, to hand tracklets to `motservice`.
- A real TensorRT engine in the repository, so the GPU tier exercises the production path
  rather than the mock.
