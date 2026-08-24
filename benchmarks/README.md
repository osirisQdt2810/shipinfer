# Benchmarks

ShipInfer against [`counting-simulation`](baseline/), head to head, under one load.

## What is compared

Both systems are offered the same thing: **50 cameras x 20 fps = 1000 images/s**, from the
same JPEG frames, on the same four GPUs, against engines built from the same ONNX. Both
write a once-a-second buffer-occupancy log, and both are analysed by the baseline's own
methodology — **if a module's buffer grows, that module is not keeping up**, and its
sustained throughput is `offered - growth`.

Nothing here re-implements the baseline. `harness/baseline.py` compiles the submodule's own
`sim_pipeline_v2.cpp` unchanged and runs the resulting binary. The submodule is read-only,
and the linters are configured to leave it alone: a baseline we reformatted is not a
baseline.

## The one asymmetry, and why it is not hidden

The systems are shaped differently and it matters for the arithmetic:

| | baseline | ShipInfer |
|---|---|---|
| topology | two independent single-model pipelines | one DAG |
| where an image enters | `det` queue **or** `seg` queue | the pipeline queue |
| per image | one model | detect -> conditional segment -> embed |

So an image enters the baseline at one of two disjoint queues, and its system throughput is
`det_sustained + seg_sustained`. An image enters ShipInfer once, at the pipeline queue, and
then fans out into *crops* — the segmenter and embedders see work derived from a frame
already counted. Summing ShipInfer's modules the way the baseline's report legitimately sums
its own would report **roughly twice** the real throughput.

That decision lives in exactly one place, `run_bench.system_throughput`, and
`benchmarks/tests/test_comparison_metric.py` asserts the trap directly — including that
`RunAnalysis.total_sustained` really does return 2000 for a 1000 img/s ShipInfer run, so a
future refactor reaching for it would be caught.

The comparison is therefore **conservative in our disfavour**: equal images-per-second means
strictly more work on our side.

## Running it

Everything runs in a container on real GPUs — a host number is not a production number
(`.claude/CLAUDE.md`, "Where commands run"). The hook on `Bash` refuses a host run.

```bash
# both systems, the full 50x20 load, 70 s with a 10 s warmup
deploy/rootless/test.sh --bench

# or, inside a container
python benchmarks/run_bench.py --cameras 50 --fps 20 --seconds 70 --gpus 2,3,4,5

# one system at a time, to keep the GPUs uncontended
python benchmarks/run_bench.py --systems baseline
python benchmarks/run_bench.py --systems shipinfer
```

Output lands in `.artifacts/bench/run-<label>/`: each system's occupancy JSONL, its console
capture, and `summary.json` carrying the config, the per-module fits and the verdict.

## Reading the verdict

Three outcomes, and conflating them is how a harness publishes a number it did not measure.

- **SATURATED** (and not capped) is a **capacity**. A buffer grew linearly, so
  `offered - growth` is the rate the module actually retired. This is the whole of the
  buffer-growth methodology and the one regime in which the number is exact.
- **SUSTAINED** or **DRAINING** is a **floor**. Nothing grew, so the system kept up with what
  it was given and its capacity is *at least* the offered rate — how much more, this run
  cannot say.
- **UNMEASURED** is nothing. A capped buffer sheds instead of growing, so its slope stops
  meaning anything.

The ratio inherits its meaning from the pair, and `ratio_of` says which it has:

| baseline | shipinfer | the ratio is |
|---|---|---|
| capacity | capacity | an exact speed-up |
| capacity | floor | a **floor** on the speed-up, `>= Nx` — enough to *meet* a target |
| floor | capacity | a **ceiling**, `<= Nx` — enough to miss one, never to meet one |
| floor | floor | nothing. Both kept up; raise the offered rate. |
| either | UNMEASURED | nothing |

> An earlier version of this harness had the first two rows inverted: it refused SATURATED as
> "an upper bound, not a rate". Since both systems are offered the same load by construction,
> that made a speed-up structurally unreachable — either neither side saturated and each
> reported the offer back at 1.00x, or one did and the comparison was declared unavailable.
> Six review rounds to find it. If you are tempted to re-tighten one of these refusals, check
> first that a measurable run still exists on the other side of it.

Which is why `--sweep` exists: one offered rate can leave both sides on the floor, and the
answer is to climb until somebody saturates.

## Two things every number here is qualified by

**The box is shared, and the two systems are not equally exposed to that.** ShipInfer's
pipeline plane is CPU-bound Python; the baseline is a GPU-bound C++ binary. A busy neighbour
therefore depresses *our* number much more than theirs, so a ratio taken under load is
indicative rather than defensible. Every run records `load_average` and `cpu_count` in its
metadata and prints them, with a `BUSY` warning past half the CPU count. Check that line
before quoting a ratio.

**The baseline's offered rate is asserted, not measured.** Ours comes from ingest's own
counters and `check_offer` refuses a run that delivered under 98% of target. The baseline's is
read off its own configuration, because its JSONL carries buffer depths and no arrival
counter — and it is run *unchanged* on purpose, so adding one means patching the submodule.
If its CPU-decoding source threads under-deliver, `offered - growth` over-reports each of its
queues. That errs in the **baseline's** favour, so it works against a conclusion in ours
rather than for it; it is still a real limit on how many digits of the ratio mean anything.

One more constraint worth knowing before you pick an offered rate: **the baseline binary only
survives while saturated.** Its plans are static-batch and `TrtRunner::infer` calls
`setInputShape` with whatever batch it assembled, so a partial batch throws inside a worker
and aborts the process — `terminate called ... what(): setInputShape failed`, reproducibly at
60 img/s. It cannot be run at the low rates our own driver can currently deliver.

## Offline tests

```bash
deploy/rootless/test.sh benchmarks/tests -q
```

Pure logic over synthetic occupancy logs — no GPU, no engines, no baseline binary. They run
as part of the default `pytest`.
