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

- **SUSTAINED** — no buffer grew; the number is a throughput.
- **SATURATED** — at least one buffer grew; the number is an **upper bound**, not a rate.
  `run_bench` refuses to divide two bounds, because a ratio of bounds is not a speed-up.
- **DRAINING** — every buffer shrank; the system had headroom at this offered rate.

A speed-up is only printed when both sides produced a real rate. If a side saturated, lower
the offered rate to find its sustainable point rather than reporting the bound as a result.

## Offline tests

```bash
deploy/rootless/test.sh benchmarks/tests -q
```

Pure logic over synthetic occupancy logs — no GPU, no engines, no baseline binary. They run
as part of the default `pytest`.
