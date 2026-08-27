# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**ShipInfer** (`shipinfer`) is a **Triton-shaped, hackable multi-GPU inference server** for
maritime perception: detect ships and people across ~50 cameras, segment ships, embed both,
recognise ship identity, and hand tracklets downstream. It is the next generation of the
`subfaceid` → `motservice` → `mtmcservice` trio under `references/`.

Sizing that drives every decision (from `references/bitbucket-subfaceid/docs/`):

| Parameter | Value |
|---|---|
| Cameras | 50 |
| FPS per camera | 20 → **1000 frames/s** |
| People per frame | 10–20 → **~15 000 crops/s** |
| GPUs | 16 (8 × RTX A5000 on the current dev box) |

At that sizing the box is **GPU-rich**: 1000 fps of detection over 16 GPUs is ~63 fps each.
So the bottleneck is not raw throughput. It is **(a) load balance** when cameras are uneven
and **(b) end-to-end latency**. Everything in this codebase is arranged around those two.

The failure being fixed is documented and specific. The previous system funnelled every
camera into one shared 1000-slot buffer that evicted the *oldest* entry when full, so a
crowded camera silently starved a quiet one, and all inference ran on GPU 0 because nothing
ever called `cudaSetDevice`.

## Stack & runtime reality (read this before anything else)

- **Language:** Python **3.10+** for the control plane, **C++17/CUDA** for the data plane.
- **Torch is a hard dependency, not an extra.** It is the runtime substrate: caching device
  allocator, caching pinned-host allocator, streams, events, CUDA graph capture, and one
  API that covers CUDA *and* ROCm. See ADR-003 — this is the single most important
  convention in the project.
- **This is a server, not a notebook and not a desktop app.** Linux + NVIDIA (or ROCm).
  Long-running, 24/7, observable.
- **Concurrency is threads, not asyncio.** One worker thread per model instance, bound to
  one GPU for its whole life. The GIL is not the problem because every thread spends its
  time inside TensorRT or a CUDA memcpy, both of which release it.
- **The fused kernels are optional and live in their own repository.**
  `3rdparty/shipvision` is a submodule; every native component has a Python
  counterpart, so a machine with no build still runs — and CI deliberately does not check
  the submodule out, which is how that promise stays true.
- **Distribution:** a wheel plus a container. There is no installer and no GUI.

### Where commands run (RULE — enforced, not advisory)
**Anything that touches an accelerator runs inside a container.** That means the GPU test
tiers (`-m gpu`, `-m multigpu`), every benchmark, `shipinfer bench|serve`, and any engine
build. `deploy/` holds the recipes: `deploy/rootless/test.sh` for the suite,
`deploy/rootless/bench.sh` for the benchmark, `deploy/rootless/prove.sh` for the
container+GPU attestation, `make shell` for an interactive shell.

**The offline tier is exempt, deliberately.** `pytest` with no marker must pass on a machine
with no driver — that is ADR-001, it is what CI does on a plain runner, and it is the promise
that makes the pure layers verifiable anywhere. The rule is about *measurements*, and the
offline tier produces none.

Enforced in two places, and the split matters:

- **`src/shipinfer/runtime/containment.py`** is the gate. It runs inside the process that
  would do the work — `tests/conftest.py` for the device tiers, the `serve` and `bench`
  commands for the CLI — so there is no spelling that avoids it. A container is established
  by *agreement* between the marker file, pid 1's cgroup and an overlay root: `/.dockerenv`
  alone is a file anyone can `touch`, and resting on it let a host run self-certify.
- **`scripts/hooks/require_container.py`** is a `PreToolUse` hook on `Bash`, and is now an
  advisory fast path rather than the enforcement point. It catches the common case before a
  command runs at all, which is useful. It is *not* sound: a deny-list over command text
  cannot be, and review found nine ordinary spellings past it — `( pytest -m gpu )`,
  `eval "..."`, `coverage run -m pytest`, `echo pytest | sh`. Fix bypasses when you find
  them, but do not add a rule that only the hook enforces.

Per-command override `SHIPINFER_ALLOW_HOST_RUN=1`, only when the operator has said so, and
say in the report that you used it. There is no session-wide switch: "I turned it off an hour
ago and forgot" is how the rule was lost the first time.

Two consequences worth stating plainly:
- **A host number is not a production number.** Host `nvcc` here is 11.5 against a 12.6
  driver, so `sm_89` will not build and host timings are not the deployment's timings.
- **The offline tier being CPU-only by design is not the same as "the GPU path is covered".**
  `pytest` with no accelerator proves the pure-logic tier; only `-m gpu` inside the container
  says anything about the data plane.

## Architecture (layered; one-way imports)

```
src/shipinfer/
├── __main__.py            # python -m shipinfer
├── cli/                   # typer commands, one per file
│
├── core/                  # PURE. no torch, no tensorrt, no network, no model files.
│   ├── types/             # DataType, Device, MemoryKind, TensorSpec, Tensor
│   ├── request/           # InferenceRequest/Response, RequestContext, Priority, Timings
│   ├── errors/            # the typed failure vocabulary, split by domain
│   ├── settings/          # pydantic settings tree, one section per file
│   ├── logging/           # logger factory + pluggable sinks (incl. async)
│   ├── metrics/           # counters/gauges/histograms + exporters
│   └── registry.py        # the Registry primitive every extension point uses
│
├── repository/            # the on-disk model repository (Triton's layout, config.yaml)
│
├── scheduling/            # PURE. the part this project exists to own.
│   ├── work.py            # WorkItem
│   ├── queues/            # fair (default) / fifo, bounded, priority lanes
│   ├── batching/          # assemble N requests -> one batch, scatter back
│   └── policies/          # round_robin / jsq / power_of_two / locality_spillover /
│                          #   sequence_affinity            <- @POLICIES.register
│
├── runtime/               # THE ACCELERATOR SEAM. torch underneath.
│   ├── platform.py        # CUDA vs ROCm vs CPU detection
│   ├── device.py          # DeviceManager, thread binding
│   ├── stream.py          # Stream / StreamPool over torch.cuda.Stream
│   ├── tensor.py          # core.Tensor <-> torch.Tensor bridge
│   ├── graphs/            # torch (default) / custom (raw-driver reference)
│   ├── memory/            # staging pool, allocators (torch_* default, custom_* reference)
│   ├── ops/               # numpy / torch / native fused kernels
│   ├── providers/         # raw driver access — only for the `custom` variants
│   └── native.py          # loads shipinfer._C
│
├── backends/              # one execution runtime per module   <- @BACKENDS.register
│   ├── mock.py            # deterministic, hardware-free — the default in tests
│   ├── tensorrt/          # the production path (engine, bindings, logger, backend)
│   ├── onnx.py            # portable fallback
│   └── torch_backend.py   # TorchScript, for prototyping and numeric parity
│
├── topology/              # PURE. the element chain (ADR-017): Element ABC, caps, registry
│   ├── base.py            #   per kind, YAML chain loader — validated at load time
│   └── elements/          #   one impl per file; mock today, real impls in phase C/E
│
├── engine/                # the model pool (arch.md §6) — what `server/` used to be
│   ├── pool.py            # InferenceServer: repository, devices, memory pool, loaded models
│   ├── instance.py        # 1 backend copy + 1 queue + 1 worker thread, pinned to 1 GPU
│   ├── model.py           # instances + dispatcher + batcher + cache
│   ├── ensemble.py        # the KServe-visible model DAG, validated at load time
│   ├── cache/             # response cache (off by default)
│   └── spill/             # the ADR-015 ring tier (kept as the control channel, ADR-016)
│
├── api/                   # KServe v2 over FastAPI — the engine's side-door, arch.md §6
│                          #   the ONE layer that may import fastapi/uvicorn, and lazily
│
├── runners/               # §1 HOW a topology executes   <- @RUNNERS.register
│   ├── inprocess.py       #   the whole chain on a thread pool here
│   ├── fleet.py           #   one shard process per GPU, driven over gRPC — the default
│   └── service.py         #   the shard's half of the control plane (holds a runner)
│
├── launch/                # §2 spawn + supervise shards; the parent's half of the RPCs
│   ├── supervisor.py      #   Fleet: one process per shard, all-or-nothing start, one drain
│   ├── client.py          #   ShardClient: Ready/UpdateTopology/AddCamera/Health/Stop
│   ├── control.py         #   the vocabulary, with no transport in it
│   ├── proto/             #   shard.proto + the committed generated stubs
│   └── signals.py         #   Ctrl-C/SIGTERM -> the fleet. Never imports torch: it sets
│                          #   CUDA_VISIBLE_DEVICES before the child's interpreter starts
│
├── cli/shard.py           # §2 the child: `--shard-id N --control-port P` and nothing else.
│                          #   Binds, answers `starting`, builds its runner when told what
│                          #   to run. A composition root, which is why it is under cli/
│
└── pipeline/, ingest/     # the ship+person application on top of the engine

3rdparty/                  # first-party libraries with their own repos, as submodules
  shipvision/              #   algorithms + C++17/CUDA/HIP fused kernels -> shipvision._C
model_repository/          # the demo repository: the real DAG, on real TensorRT engines
benchmarks/                # the head-to-head against the counting-simulation architecture
deploy/                    # Dockerfile + compose; everything runs in a container
tests/                     # offline tier (default) + `-m gpu` tier
```

### The five shared seams (this is the design — protect it)

1. **The registry** (`core/registry.py`). Every pluggable family is a folder of
   one-class-per-file modules plus a `Registry`. Adding an implementation is a new file and
   a decorator, never an edit to a switch statement.
2. **The scheduler** (`scheduling/`). Queues, batching and placement. Pure Python, no
   hardware, fully testable on a laptop — which is the point, because this is the part most
   worth having tests for.
3. **The accelerator seam** (`runtime/`). The only package that knows a GPU exists.
   Everything above it works unchanged with no driver installed.
4. **The backend contract** (`backends/base.py`). A backend receives an assembled batch and
   returns one. It does not decide what to batch, where to run, or when.
5. **The kernel boundary** (`3rdparty/shipvision` ↔ `runtime/ops/`). Fused kernels
   only, numpy in and device-pointer out. Nothing torch already does well lives there, and
   the parent depends on it the way it depends on any library — a pinned commit (ADR-010).

### Coupling rule (enforced in `scripts/hooks/check_layers.py` *and* `tests/test_architecture.py`)
`core/` imports nothing from the project but `core/`. `core`, `scheduling` and `repository`
import **no torch, no tensorrt, no onnxruntime, no fastapi**. `runtime` may not import
`server`. `import shipinfer` must not pull in a backend.

## Adding things

### A placement policy
1. `src/shipinfer/scheduling/policies/<name>.py`, subclass `PlacementPolicy`.
2. Decorate with `@POLICIES.register("<name>", "<alias>")`; import it in the package
   `__init__.py` so registration happens.
3. Add a test in `tests/scheduling/test_policies.py` using the `FakeInstance` double —
   a policy is given four attributes, so a four-field dataclass tests it completely.

### A backend
1. `src/shipinfer/backends/<name>.py` (or a package if it has several concerns, as
   TensorRT does), subclass `ModelBackend`.
2. Implement `_do_initialize` / `execute`; override `input_specs` if the artefact knows the
   truth better than the config does (TensorRT does — use it, and fail at load on a mismatch).
3. Register **lazily** in `backends/registry.py` if importing it is expensive.
4. Raise the typed errors: `BackendUnavailableError` when the runtime is missing,
   `BackendLoadError` when the model is wrong. Never return empty outputs to mean failure.

### A model
1. `model_repository/<name>/config.yaml` + `<name>/1/<artefact>`.
2. `shipinfer repo show <name>` to check the resolved config, then `shipinfer repo ls` to
   see how many instances it expands to on this host.
3. Put it in an ensemble only after the DAG validates — mismatched shapes fail at start-up.

### A fused kernel
It belongs in the **`shipvision` repository**, not here — see its own `CLAUDE.md`.
Then, in this repository:
1. Extend the `ImageOps` ABC in `runtime/ops/base.py` and both implementations.
2. Add it to `tests/runtime/test_ops_parity.py`. **A fused kernel is only trustworthy if a
   readable implementation agrees with it**, and the readable one lives here.
3. Bump the submodule pointer in its own commit, so a kernel change and a server change are
   never entangled in one revert.

## GPU hygiene (RULE)
**Free the GPU the moment the task no longer needs it.** This box is shared; a
finished-but-alive process keeps its CUDA context (~220-480 MiB per device) and starves the
next person. Bound every GPU run with `timeout`, prefer `docker run --rm`, and drop tensors
plus `torch.cuda.empty_cache()` in a `finally` — a crash must not be what frees the device.
Then **verify instead of assuming**:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # want: empty
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader # want: ~15-20 MiB idle
```

Before killing anything, tell a leak from live work: a background agent's probe inside its
own `timeout` is working, not leaking — check elapsed time and parent first. The operator
watches VRAM continuously via `~/workspaces/tools/vram_log.sh`, so a leak is visible whether
or not it gets mentioned.

## Testing: two tiers, both mandatory
- **Offline (default, must stay green):** `pytest` — pure logic, config validation,
  scheduling invariants, architecture rules, ops on numpy. **Needs no GPU**, by design.
- **GPU (opt-in, run before every release and after touching `runtime/`, `backends/` or the
  kernels submodule):** `pytest -m gpu`, and `-m multigpu` for the balancing evidence.

Plus one operator command that produces the evidence a PR needs — the benchmark harness,
inside the container, for at least the analysis's 10 s warm-up plus a steady window. It
needs the host-built baseline binary first (`benchmarks/build/sim_pipeline_v2`, built with
`python -c 'from benchmarks.harness import baseline; baseline.build_binary()'`): the script
refuses to start without it, today even when `--systems shipinfer` names no baseline at all
(ledger C48 relaxes that gate):

```bash
deploy/rootless/bench.sh --systems shipinfer --seconds 40      # the system tier, per-device table
SHIPINFER_BENCH_SCRIPT=benchmarks/stages.py  deploy/rootless/bench.sh <run.json>   # algo tier
SHIPINFER_BENCH_SCRIPT=benchmarks/kernels.py deploy/rootless/bench.sh --op letterbox  # kernel tier
```

The harness takes `--cameras --fps --gpus --seconds --warmup --source replay|rtsp --systems
--sweep`; it has no `--skew`. `shipinfer bench <model> --cameras 50 --fps 20 --skew 8` is the
in-process scheduler demonstration (one model, synthetic load, skewed cameras), not the
system measurement. The design load — 50 cameras × 20 fps — is more than one interpreter can
generate or serve: it needs the multi-process generator, which is the benchmark harness's own
sharded mode (`benchmarks/harness/shards.py`, one process per shard) driving each shard's
cameras. `shipinfer fleet` was that entry point until A2 PR-6; `shipinfer run --runner fleet`
is the deployment command now, and it runs a chain rather than a repository.

"The offline suite is green" is **not** evidence that the server balances load. A bench run
with a per-device breakdown is.

## Branching & PRs (branch + PR for everything non-trivial)

1. Never commit non-trivial work directly to `main`. `git checkout -b feat/<kebab-topic>`
   (`fix/…`, `chore/…`, `docs/…`), implement there.
2. Push and surface the PR URL; **do not merge into main yourself** unless told to — or
   unless the PR carries the `automerge` label and CI's own gate merges it.
3. Small standalone edits (a `.gitignore` line, a `.claude/*` tweak) may go straight on main.
4. **Co-author trailer:** add `Co-Authored-By: Claude …` only to large feature commits.
   Small incidental commits do not carry it.
5. **All remotes are SSH** (`git@github.com:…`), never HTTPS.

### Three rules from one afternoon of review rounds (V80 follow-through)

- **A push is `&&`-chained to the check that gates it.** Twice in one hour a branch was pushed
  after its own check had just failed, because the push was on the next line. `test && commit
  && push` — a red check cannot be followed by a push.
- **Before opening or pushing a PR, grep every test name and every claim in the body against
  `git diff origin/main`.** Three bodies in one day described tests that were not in the diff:
  written from the plan, or from a sibling branch where the tests actually lived. The body is
  written *after* the diff, from the diff, and each `Test*` class it names must appear in
  `git diff --name-only`/`git diff` output.
- **A hook that rewrites files "passes" on its second run.** shipvision's PR pipeline runs
  `pre-commit run --show-diff-on-failure` (black, isort, ruff, pinned). Black *modifies* the
  files and reports Failed; a second run reports Passed on the rewritten tree — and a push at
  that point ships a commit that does not match the working tree. After `pre-commit`, run
  `git status`; if it is dirty, amend first.

- **Format only the files in the diff, with the formatter CI runs.** shipvision's PR pipeline
  runs `pre-commit run --from-ref <base> --to-ref HEAD` with pinned **black**; a whole-tree
  `ruff format` on a scoped branch dragged two unrelated files into #4 and failed lint on a PR
  that was already APPROVEd. Before pushing a shipvision branch, run exactly that command.
  Old split branches compared against a *newer* main show the merged packages as "changed";
  that is not contamination — rebuild each branch from the current main at its turn.

### PR description format (MANDATORY)
Every PR body follows `.github/pull_request_template.md`: **Context → Content / Changes →
Test Plan** (with `### Test Details` and `### Test Output / Feature Demonstration`), then
the checklist. `gh pr create --body` bypasses GitHub's template, so fill it explicitly:

```bash
sed -e 's/^<!--.*-->$//' .github/pull_request_template.md > /tmp/pr-body.md   # then fill it
gh pr create --base main --title "…" --body-file /tmp/pr-body.md --label automerge
```
- **Paraphrase the request in Context — never paste the chat message.** Requests often
  arrive in Vietnamese; the PR states the problem in the author's own English words.
- **Never delete a heading.** A non-applicable section gets `N/A — <reason>`.
- **Test Output is evidence, not a claim.** Paste the real `pytest` tail and the real
  `shipinfer bench` table. "Tests pass" is not a test plan.

### CI: auto-review and auto-merge
`.github/workflows/` holds `ci.yml` (offline suite + lint on main), `pr-pipeline.yml`
(tests → Claude review → gated auto-merge), and `claude.yml` (`@claude` in comments).
A PR auto-merges only when tests are green **and** the review verdict is APPROVE **and**
the `automerge` label is present **and** the reviewed commit is still HEAD.
Known permanent exception: a PR that edits `.github/workflows/**` cannot pass the review
job, so those need a manual merge.

**The review is a loop, not a handoff** (see `.claude/WORKFLOW.md` for the diagram). Push,
then go do other work — the review takes 10-20 minutes. On APPROVE with the label on, it
merges itself and there is nothing to say. On BLOCKING, **check each finding against the code
before fixing it**: fix and push if it is real, comment with the evidence if it is wrong, and
re-trigger either way — a push fires `synchronize`, and toggling the `automerge` label fires
`labeled`, which re-runs the review without an empty commit. Loop until merged.

**Keep a PR small — this is a hard limit, not a preference (V80).**

> **At most ~15 commits and ~25 files changed.** Measure before opening:
> `git diff <base> --stat | tail -1` and `git log --oneline <base>..HEAD | wc -l`.
> Over either number, **split and push the pieces one at a time** — do not open it.

PR #3 reached ~100 commits and 20k+ lines, past GitHub's diff API limit, so the reviewer had
to check the branch out rather than read a diff, and six review rounds followed. shipvision
PR #2 then repeated it at 45 commits / 290 files.

**Writing "this PR is too big" in the description is not splitting it.** That happened on
shipvision #2 and the operator had to ask again. The paragraph explaining why a split would be
expensive is the paragraph that should have been the split.

**How to split when the commits interleave.** Cherry-picking is usually not needed: packages
are file-disjoint, so build each branch by taking *paths* from the big branch onto `main`
(`git checkout <big-branch> -- <paths>`) rather than by replaying commits. Order the pieces by
dependency, and merge each before opening the next.

**Open them one at a time, in order.** When work splits into several PRs, push the first and
carry it to merged before opening the second. Two reasons, and the second is the one that
bites: a review that comes back BLOCKING has to be fixed, and if PR #2 is already stacked on
#1's branch the fix has to be threaded through both — while a reviewer looking at four open
PRs from the same session reviews none of them well.

## Documentation Language
All documentation (README, `docs/`, any `.md`, docstrings, comments, commit messages, PR
bodies, ADRs) **must be in English**, regardless of the conversation language. Exception:
only when the user explicitly asks for Vietnamese in a specific file.

## Two planes, one architecture (RULE — V88, V89)
`csrc/` is a **port** of the Python data plane, not a second design. Every per-frame seam
exists twice and must be the *same* seam: one thread per instance with its own bounded queue,
a dispatcher and a placement policy chosen by name, the batch window with `max_queue_delay`,
the fair queue with the same eviction order, the perception graph, reassembly with the same
event schema, and real ingest. The control plane — settings, the model repository, registries,
the CLI, the HTTP surface — stays Python (ADR-014) and hands the C++ plane a resolved config.

**Sync rule:** a change to a Python data-plane seam is not finished until the C++ seam carries
the same change and the cross-plane parity harness agrees (same inputs → same events, same
eviction counts per camera, same batches). A PR that changes one plane and not the other says
so in its body and opens the ledger item for the other; it does not merge as "done".

## Reference implementations first (RULE — V86)
When the way to build something is not clear, **read how Triton Inference Server or vLLM does
it before inventing.** Their shape is the default: Triton for the model repository, instance
groups, the per-model queue that instances on several GPUs pull from, model control and
metrics; vLLM for process-per-GPU data parallelism, the coordinator that publishes per-engine
load, the router that picks the least-loaded engine, and the CUDA-graph / allocator plumbing.
A departure from their shape is fine when it is stated with its reason in the ADR or the PR;
an unexplained departure is a reinvention. Reference checkouts live outside the repository
(the scratchpad), never as a dependency.

## Coding Conventions
Standards live in **`.claude/CONVENTIONS.md`** — read it in full before non-trivial work.
Part 1 = universal Python. Part 2 = project-specific (the layering rule, the *ponytail
principle*, registries, threading, the two test tiers, native code). Part 3 = Agent Working
Principles.

## Finishing (RULE — mechanised, because the promise failed)

**A stopping point is exactly three things:** every item in `.claude/TASKS.md` is `[x]`, `[!]`
or `[-]`; an action needs the operator's confirmation before it is safe; or the operator
interrupted. Nothing else.

**Opening a PR is not a stopping point. Pushing is not. Writing a summary is not.** Those are
milestones, and a milestone is something to report *while continuing*. On 24 Aug this failed
twice in the same shape: a PR was opened, a summary was written with the words "I'll continue"
in it, and the turn ended. Ending a turn *is* stopping — there is no "and then I keep going".

The rule was in front of me both times, so a stronger reminder was not the fix. What was
missing is that nothing *checked*: "is there work left?" was answered by a feeling at the end
of a long stretch of tool calls rather than by a file. So:

- **`.claude/TASKS.md`** is the ledger. One line per outstanding item. Mark `[~]` while working
  and `[x]` with the evidence when done. `[!]` means genuinely blocked on the operator, with
  the question on the line.
- **`scripts/hooks/unfinished_work.py`** is a `Stop` hook that reads it and returns
  `decision: block` while anything is open, handing the list back. Escapes, so it cannot trap a
  session: an `[!]` line, `SHIPINFER_ALLOW_STOP=1` for one command, and a cap of twelve
  consecutive blocks against an unchanged ledger — past which the loop is not converging and
  saying so is more useful than spinning.

Keep the ledger current *as you go*, not at the end. It is the file that decides whether the
session may end, so a stale ledger is either a session that stops early or one that will not
stop at all.

## Project State Files (DYNAMIC — read at session start)
- **`docs/qa/user.md`** — every request the operator has made, verbatim. **RULE: append each
  new request here as it arrives**, exactly as written, before acting on it. A rule stated
  once in passing is otherwise lost to the next compaction; Section 3 of that file is the
  standing-rules index and is the fastest way to reload the constraints.
- **`.claude/memory/MEMORY.md`** — durable facts about how this project is worked on.
- **`.claude/JOURNAL.md`** — daily work log; newest on top. Read first for recent context.
- **`.claude/FEATURE_LOG.md`** — one entry per large feature (append-only, newest on top).
- **`.claude/DECISIONS.md`** — ADRs; read before changing an architectural pattern.
- **`.claude/WORKFLOW.md`** — startup, the build/verify loop, GPU testing, pre-commit.

**Feature-log rule:** after any large feature/change (a new backend, a new policy, a change
to a shared seam, a native kernel), append an entry to `.claude/FEATURE_LOG.md`. Skip it for
tiny edits, typo fixes, and pure docs.

## Sub-agent Workflow

> **STATUS: ENABLED** for non-trivial feature/refactor work. Small tasks (<~20 lines) or
> obvious fixes: do them directly in the main session.

| Phase  | Agent                | Tools                   | Purpose |
|--------|----------------------|-------------------------|---------|
| Search | `Explore` (built-in) | read-only               | Locate code/patterns before planning. |
| Plan   | `planner`            | Read, Grep, Glob        | Produce the implementation plan; writes no code. |
| Build  | `coder`              | Bash, Read, Write, Edit | Implement the plan + tests; validate with the tooling. |
| Verify | `reviewer`           | Read, Grep, Bash        | Senior **solution-architecture** review: abstraction, reuse, cohesion, coupling — plus correctness, conventions, tests. |
| Fix    | `debugger`           | Read, Grep, Bash, Edit  | Root-cause a failing test/error; minimal fix. |

Typical order: **Explore → planner → coder → reviewer**, invoking `debugger` only when a
test or run fails.

## Slash Commands
- `/gpu-check` — run the GPU tier plus a bench, and report the evidence
- `/parallel-tasks` — decompose a task list into file-disjoint lanes, build them in parallel,
  then one integration review over the combined diff

JOURNAL.md, DECISIONS.md and FEATURE_LOG.md are still maintained — by editing them directly
when there is something worth recording, not through a command.
