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

## Where the frames come from (R55)

```bash
python benchmarks/run_bench.py --systems shipinfer                 # replay (default)
python benchmarks/run_bench.py --systems shipinfer --source rtsp   # over a real socket
```

**These are two different experiments, not a fast one and a slow one.** `replay` reads JPEGs
off disk, which measures the inference plane with the decode path *removed*. `rtsp` pulls
H.264 from `scripts/rtsp_serve.py` over a real socket, so NVDEC, the jitter buffer, reconnects
and the NV12 conversion are all included — and the deployment reads fifty RTSP cameras, so
those are part of its real cost. A replay number is an **upper bound** on the RTSP one.

`config.as_dict()` records which was used and the console names it, because the failure to
avoid is quoting a replay figure as though the decode path were in it.

The server is started and stopped around the run by `benchmarks/harness/rtsp.py`. It refuses
rather than tolerates: a server that never accepts a connection, or that exits early, raises
at start-up with the reason attached. A run whose cameras cannot connect produces a
clean-looking zero, and this project has already published one of those.

## The three tiers (R44)

`system → algo → kernel`. Each answers a different question, and reading one as another is the
mistake they exist to prevent.

| tier | file | question | when to reach for it |
|---|---|---|---|
| **system** | `run_bench.py` | how many images a second does the whole thing retire? | the number a claim is made on |
| **algo** | `stages.py` | where does one frame's time go, stage by stage? | before optimising anything |
| **kernel** | `kernels.py` | what does one op cost, per implementation? | deciding whether a fused kernel earns its build |

All three are measurements, so all three run in the container. `deploy/rootless/bench.sh`
selects the tier with `SHIPINFER_BENCH_SCRIPT` (default: the system tier) and only demands the
baseline binary for the tier that uses it:

```bash
deploy/rootless/bench.sh --systems shipinfer                                  # system
SHIPINFER_BENCH_SCRIPT=benchmarks/stages.py  deploy/rootless/bench.sh --cameras 12 --fps 5  # algo
SHIPINFER_BENCH_SCRIPT=benchmarks/kernels.py deploy/rootless/bench.sh --op letterbox         # kernel
```

**A kernel speed-up is not a system speed-up.** An op that is 2% of the frame budget caps out
at 2% however fast it gets. That is why the algo tier sits between them: it is the one that
turns "this op takes 400 µs" into "this op is 14% of a frame".

### The algo tier reads, it does not instrument

`PipelineStage.run` already stamps `elapsed_us` on every outcome and `_CollectorObserver`
already feeds it into `shipinfer_pipeline_stage_latency_us`. `stages.py` drives a run and
renders those histograms. A second timing path would be a second implementation that could
disagree with the one operators actually watch.

It runs **below saturation on purpose**. Under saturation a stage's latency includes the time
it waited behind other frames, so a queueing artefact reads as an expensive stage; the report
warns loudly when the run did not keep up with 98% of its offered load — the same bar
`check_offer` holds the system tier to.

### The kernel tier binds to a device the way production does

`TorchImageOps` falls back to the CPU unless it is given a `device_index`, and
`PipelineRunner._build_ops` always gives it one. The first version of `kernels.py` called
`create(name)` with no arguments and therefore timed torch on the *CPU*: it came out 7–13×
slower than numpy, which is a true fact about a configuration nobody runs and a false one
about this project. Bound correctly, on this box:

```
op                   impl         per call   spread   vs numpy   where
letterbox            torch         5390.7 us   35.0%      3.27x   torch kernels on cuda:0
letterbox            numpy        17613.5 us   54.5%      1.00x   numpy (host)
crop_batch           torch         4416.9 us   32.1%      1.84x   torch kernels on cuda:0
nms                  torch         3365.2 us    5.9%      2.47x   torch kernels on cuda:0
```

Those spreads are the point: taken at load 31–41 of 48 cores, they are **not reproducible**
and the tool says so rather than printing three significant figures of noise. `native` is
absent because the fused kernels are not built on this box — reported as a skip with the
remedy, not dropped from the table.

`letterbox` returns numpy by contract, so a device implementation pays a copy home that numpy
never makes; `letterbox_to_device` is the device-fair column and the one production calls.

## Offline tests

```bash
deploy/rootless/test.sh benchmarks/tests -q
```

Pure logic over synthetic occupancy logs — no GPU, no engines, no baseline binary. They run
as part of the default `pytest`.
