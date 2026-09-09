# Workflow

## Session start

1. Read `.claude/memory/MEMORY.md` — durable facts about how this project is worked on.
2. Read the top of `.claude/JOURNAL.md` — where the last session left off.
3. `git status && git log --oneline -5` — what is actually on disk.
4. If the change touches an architectural pattern, read the matching ADR first.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cli]"

# torch must match the DRIVER, not the newest release. Check with `nvidia-smi`:
#   driver 12.6 -> cu126 wheels. Installing a cu130 build on a 12.6 driver gives
#   `torch.cuda.is_available() == False` and an unhelpful "driver is too old".
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

pre-commit install
shipinfer doctor          # confirm devices, provider and native status
```

Optional extras: `.[tensorrt]` for the production backend, `.[server]` for the HTTP API,
`.[onnx]`, `.[video]`, `.[kafka]`.

## The build/verify loop

```bash
bash scripts/run_tests.sh               # offline tier, exactly as CI runs it  <- use this
pytest                                  # the same selection, but WITH your GPUs visible
pytest tests/scheduling -q              # one area
pytest -q -m gpu                        # real devices
pytest -q -m multigpu                   # the balancing evidence (needs >= 2 GPUs)
python -m pytest -p tests.plugins.mask_shipvision # the CI-shaped run: `3rdparty/shipvision` hidden

ruff check src tests scripts
black --check src tests scripts
isort --check src tests scripts
mypy src/shipinfer                      # strict; not a commit gate yet
pre-commit run --all-files
```

**Run the masked selection too, whenever you touched `topology/` or `pipeline/`.** CI does
not check the submodule out, so a green run on this box says nothing about the run that gates
the PR: `-p tests.plugins.mask_shipvision` makes `shipvision` unimportable exactly as a
checkout without it is (`ModuleNotFoundError`, so `pytest.importorskip` skips rather than
fails), and prints a header line saying so — paste that line with the numbers, because a
masked run and an unmasked one differ by ~130 skips and by nothing else visible.

**Run `scripts/run_tests.sh`, not bare `pytest`, before pushing.** It exports
`CUDA_VISIBLE_DEVICES=""`, so the offline tier runs the way CI runs it: with no accelerator
at all. A bare `pytest` on this box has eight A5000s in view, so a test that quietly takes
a CUDA path still passes — and then fails on the runner. `torch.empty(pin_memory=True)` is
the worked example: it succeeds here and raises on a GPU-less host.

## Fused kernels (the `shipvision` submodule)

```bash
git submodule update --init 3rdparty/shipvision
pip install -e 3rdparty/shipvision      # the Python surface; no toolkit needed
python scripts/build_native.py --arch 86       # delegates to the submodule's build.py
python scripts/build_native.py --hip           # ROCm
```

Both steps are needed and they fail differently: without the editable install the server
cannot import the package at all, and without the build it imports but reports
`cuda_available() == False` and falls back to torch. `shipinfer doctor` says which.

Three failure modes worth recognising, all of which the script now handles or explains:

- **`parameter packs not expanded`** deep inside `std_function.h` — nvcc is older than the
  system GCC. The script picks a compatible `g++-N` automatically.
- **`bytecode stream ... LTO version`** at the final link — the CUDA host compiler and the
  C++ compiler are different GCC majors. The script sets both to the same one.
- **`undefined symbol: fatbinData`** on import — separable compilation plus pybind11's LTO.
  It is disabled in `native/CMakeLists.txt`; do not turn it back on.

After a build, `shipinfer doctor` should report the extension.

## Operator evidence

The number that matters is not "tests pass", it is the per-device breakdown:

```bash
shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 5 --skew 8
shipinfer bench ship_detector --policy round_robin --seconds 5     # the baseline
shipinfer repo ls
shipinfer repo show ship_detector
shipinfer serve --http --port 8000       # then curl /v2/health, /v2/statistics, /metrics
```

`--skew 8` reproduces the inherited failure: camera 0 submits eight times the traffic of
the others. Fair queueing keeps `per-camera served min` proportional to submission; without
it, quiet cameras trend to zero.

## Comparing two systems (RULE — measured 9 Sep, and it is not a preference)

**One run against one run decides nothing on this box.** Four runs of the same arm at
*identical* settings spread 26 669 to 39 375 events — 36.7% of the mean; dropping the
earliest still leaves 14.9%. So a single-run A/B cannot resolve anything under ~15%, and
three conclusions in `TASKS.md` sit inside that (the 4x-queue arm's +1.8%, the -12%
`queue_rejected`, and `replay`'s +16.6% over `nvdec`).

**Interleave the arms and quote the pairwise ratios, not the means.** A worker-count sweep
said 72 beat the default 115 by 6% on its first pass and lost on both repeats; the means hid
the reversal. Three pairs minimum, alternating, so a drift in the box's load cannot favour
one side:

```bash
for pass in a b c; do
  deploy/rootless/bench.sh --systems baseline --cameras 50 --fps 20 --seconds 40 --gpus 1,3,4,5,6
  SHIPINFER_BENCH_SOURCE=nvdec SHIPINFER_BENCH_GPUS=1,3,4,5,6 \
    SHIPINFER_BENCH_SECONDS=40 scripts/run_cpp_bench.sh "ratio-$pass"
done
```

**And it is not a choice: a simultaneous pair aborts.** `--systems baseline,shipinfer` starves
our in-process generator below the offer gate, and at a load small enough to avoid that the
baseline logs too few samples to bound a growth rate. `--topology fleet` at 50x20 fails every
shard the same way. The harness says it in its own words — "run one at a time to keep the GPUs
uncontended".

**Both arms print their host CPU**, so the like-for-like denominator is a grep:

```
baseline  host cpu: 510.8 CPU-s over 44.3 s = 11.53 cores, 12.12 ms CPU/image
shipinfer host cpu: 126.3 CPU-s over 38.2 s = 3.31 cores,  75.14 ms CPU/image
```

Divide by the *rows* `cli/bench` prints rather than by events: a baseline image passes through
one model, one of our frames through four. Watch the box's own load line in the header — the
baseline's throughput is asserted from its configuration, so CPU starvation lowers its
CPU-seconds and not its images, which flatters it.

## Pre-commit

Runs on staged files at every `git commit`:

- hygiene (whitespace, EOF, YAML/TOML/JSON parse, large files, private keys)
- `isort` → `black` → `ruff --fix`
- `clang-format` on `native/`
- **`check_layers.py`** — the one-way import rule (ADR-001)
- **`check_model_configs.py`** — every staged `config.yaml` parses

`references/` and `3rdparty/` are excluded: they are read-only upstream checkouts.

## Branch and PR

```bash
git checkout -b feat/<kebab-topic>
# ... work, with tests ...
pytest && pytest -m gpu && ruff check src tests scripts

sed -e 's/^<!--.*-->$//' .github/pull_request_template.md > /tmp/pr-body.md   # fill it
gh pr create --base main --title "…" --body-file /tmp/pr-body.md --label automerge
```

### Keep it small (RULE)

**Few commits, few files.** PR #3 reached ~100 commits and 20 000+ lines, past GitHub's diff
API limit, so the reviewer had to check out the branch instead of reading a diff — and six
review rounds were partly a consequence of the size. Split by seam: one PR for the harness,
one for the fixes it found, one for the docs. If a branch is growing past ~15 commits, that
is the signal to open the next PR rather than to keep going.

### The review loop (RULE — this is a loop, not a handoff)

The pipeline fires on `opened, synchronize, reopened, ready_for_review, labeled`, so **a push
re-runs it and so does adding a label**. Auto-merge needs three things at once: tests green,
the review verdict `APPROVE`, and the `automerge` label present at merge time.

```
push
  └─> review runs (slow — 10-20 min; go do other work, do not idle-poll)
        ├─ APPROVE + automerge label  ->  it merges itself. Nothing to say, nothing to do.
        └─ BLOCKING
              ├─ check each finding against the code before trusting it
              ├─ real      -> fix, push. The push re-triggers the review.
              └─ wrong     -> comment on the PR with the evidence, then re-trigger:
                              gh pr edit N --remove-label automerge
                              gh pr edit N --add-label automerge     # `labeled` fires
        loop until merged
```

**Check the finding before you fix it.** A review can be wrong — round 6 claimed stale
`shipinfer-imgproc` references that `git show HEAD:.claude/CLAUDE.md` proved absent. Fixing
what is not broken is worse than arguing: it makes the next reviewer's map of the code wrong.
Say which it is, with the command that shows it.

**Edit the title and body BEFORE the push, not after (RULE — cost two blocking rounds on
9 Sep).** The review snapshots them when the job starts, so a body edited afterwards is
reviewed against the *previous* revision's argument. #188 round 3 blocked entirely on four
claims that were already deleted. And the title matters most: with `automerge` on it becomes
`main`'s subject line, so one left describing an approach you abandoned inverts the diagnosis
for whoever bisects that code next. When a revision changes what the PR does — a reversed
approach, a new mechanism — the order is: rewrite body, retitle, then `git push`.

**Keep the label on.** Removing it turns the loop into a handoff that waits for a human. The
only reason to remove it is a PR that edits `.github/workflows/**`, which cannot pass the
review job at all (a GitHub App restriction, not a bug) and needs a manual merge.

Commit messages: imperative mood, a body explaining *why*. Add the
`Co-Authored-By: Claude …` trailer only to large feature commits.

## Releasing

1. `pytest && pytest -m gpu && pytest -m multigpu`
2. `shipinfer bench` on the target hardware; paste the table into the release notes.
3. Rebuild the native extension for every architecture in the fleet (not `--arch <local>`).
4. Bump `version` in `pyproject.toml`; append to `.claude/FEATURE_LOG.md`.
