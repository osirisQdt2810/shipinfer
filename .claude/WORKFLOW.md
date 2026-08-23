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
pytest                                  # offline tier — must stay green, no GPU needed
pytest tests/scheduling -q              # one area
pytest -q -m gpu                        # real devices
pytest -q -m multigpu                   # the balancing evidence (needs >= 2 GPUs)

ruff check src tests scripts
black --check src tests scripts
isort --check src tests scripts
mypy src/shipinfer                      # strict; not a commit gate yet
pre-commit run --all-files
```

## Fused kernels (the `shipinfer-imgproc` submodule)

```bash
git submodule update --init 3rdparty/shipinfer-imgproc
pip install -e 3rdparty/shipinfer-imgproc      # the Python surface; no toolkit needed
python scripts/build_native.py --arch 86       # delegates to the submodule's build.py
python scripts/build_native.py --hip           # ROCm
```

Both steps are needed and they fail differently: without the editable install the server
cannot import the package at all, and without the build it imports but reports
`is_available() == False` and falls back to torch. `shipinfer doctor` says which.

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
shipinfer repo show ship_pipeline
shipinfer serve --http --port 8000       # then curl /v2/health, /v2/statistics, /metrics
```

`--skew 8` reproduces the inherited failure: camera 0 submits eight times the traffic of
the others. Fair queueing keeps `per-camera served min` proportional to submission; without
it, quiet cameras trend to zero.

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

Then watch the pipeline: `gh pr checks --watch`. If the Claude review returns BLOCKING,
fix, push, and it re-runs. A PR that edits `.github/workflows/**` cannot pass the review
job and needs a manual merge — that is a GitHub App restriction, not a bug.

Commit messages: imperative mood, a body explaining *why*. Add the
`Co-Authored-By: Claude …` trailer only to large feature commits.

## Releasing

1. `pytest && pytest -m gpu && pytest -m multigpu`
2. `shipinfer bench` on the target hardware; paste the table into the release notes.
3. Rebuild the native extension for every architecture in the fleet (not `--arch <local>`).
4. Bump `version` in `pyproject.toml`; append to `.claude/FEATURE_LOG.md`.
