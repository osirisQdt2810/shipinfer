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
  `3rdparty/shipinfer-imgproc` is a submodule; every native component has a Python
  counterpart, so a machine with no build still runs — and CI deliberately does not check
  the submodule out, which is how that promise stays true.
- **Distribution:** a wheel plus a container. There is no installer and no GUI.

### Where commands run
**In a container.** `deploy/` holds the recipes: `make shell` drops you into the dev image
with the repository bind-mounted, so you edit on the host and every command runs inside.
A host `.venv` still works for the offline tier and is what CI uses, which is exactly why
that tier must never need a GPU.

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
├── server/                # instances, models, ensembles, the engine, health, HTTP
│   ├── instance.py        # 1 backend copy + 1 queue + 1 worker thread, pinned to 1 GPU
│   ├── model.py           # instances + dispatcher + batcher + cache
│   ├── ensemble.py        # the DAG, validated at load time
│   ├── cache/             # response cache (off by default)
│   └── api/               # KServe v2 over FastAPI
│
└── pipeline/, ingest/     # the ship+person application on top of the server

3rdparty/                  # first-party libraries with their own repos, as submodules
  shipinfer-imgproc/       #   C++17 + CUDA/HIP fused kernels -> shipinfer_imgproc._C
model_repository/          # the demo repository: the real DAG, on the mock backend
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
5. **The kernel boundary** (`3rdparty/shipinfer-imgproc` ↔ `runtime/ops/`). Fused kernels
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
It belongs in the **`shipinfer-imgproc` repository**, not here — see its own `CLAUDE.md`.
Then, in this repository:
1. Extend the `ImageOps` ABC in `runtime/ops/base.py` and both implementations.
2. Add it to `tests/runtime/test_ops_parity.py`. **A fused kernel is only trustworthy if a
   readable implementation agrees with it**, and the readable one lives here.
3. Bump the submodule pointer in its own commit, so a kernel change and a server change are
   never entangled in one revert.

## Testing: two tiers, both mandatory
- **Offline (default, must stay green):** `pytest` — pure logic, config validation,
  scheduling invariants, architecture rules, ops on numpy. **Needs no GPU**, by design.
- **GPU (opt-in, run before every release and after touching `runtime/`, `backends/` or the
  kernels submodule):** `pytest -m gpu`, and `-m multigpu` for the balancing evidence.

Plus one operator command that produces the evidence a PR needs:

```bash
shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 5 --skew 8
```

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

## Documentation Language
All documentation (README, `docs/`, any `.md`, docstrings, comments, commit messages, PR
bodies, ADRs) **must be in English**, regardless of the conversation language. Exception:
only when the user explicitly asks for Vietnamese in a specific file.

## Coding Conventions
Standards live in **`.claude/CONVENTIONS.md`** — read it in full before non-trivial work.
Part 1 = universal Python. Part 2 = project-specific (the layering rule, the *ponytail
principle*, registries, threading, the two test tiers, native code). Part 3 = Agent Working
Principles.

## Project State Files (DYNAMIC — read at session start)
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
- `/resume` — read JOURNAL.md and brief on where work left off
- `/daily-wrap` — update JOURNAL.md with today's session summary
- `/adr` — create a new Architecture Decision Record in DECISIONS.md
- `/gpu-check` — run the GPU tier plus a bench, and report the evidence
