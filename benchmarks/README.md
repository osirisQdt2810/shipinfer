# Benchmarks

## `compare_baseline.py` — ShipInfer against the `counting-simulation` architecture

### What is being compared, and what is not

`references/counting-simulation` cannot be run on this host: it needs TensorRT, OpenCV and
two built `.engine` files, and the repository ships none of them. So the *serving
architecture* is re-implemented faithfully from `sim_pipeline_v2.py` and both systems are
driven against an identical synthetic backend.

That is not a workaround, it is the better experiment. In production both call the same
TensorRT engine, so the engine is a constant and the serving layer is the variable.
Comparing two separately-built engines would have measured the wrong thing.

The baseline's five decisions, taken from its source:

- one shared bounded `queue.Queue` per model, holding bare frames with no identity
- a global `pop_lock` held while a worker dequeues its whole batch
- workers statically bound to GPUs by `gpu_ids[i % len(gpu_ids)]`
- a fixed batch size
- a producer that blocks and retries forever on a full queue rather than shedding

None of those is a strawman. Each is a reasonable first thing to write, and the comparison
is about what they cost at fifty cameras.

### Results, on 8 × RTX A5000

**Preprocessing — measured, not simulated.** 8 × 1080p → 640×640:

| | ms | img/s |
|---|---:|---:|
| `cv2.resize` + `cvtColor` per image (their path) | 438–548 | 15–18 |
| fused CUDA kernel into a torch tensor (ours) | 16–18 | 448–499 |

**~28×.** The whole batch in one pass over the pixels instead of four passes per image on
the CPU. This is the single largest difference between the two systems and the one with no
caveats.

**Serving, at the design point** — 50 cameras × 20 fps = 1000 req/s over 8 GPUs, camera 0
at 8× traffic, a launch-bound cost model (1.2 ms/batch + 0.03 ms/row):

| | baseline | ShipInfer |
|---|---:|---:|
| achieved | 935 req/s | 997 req/s |
| p50 | 17.3 ms | 4.6 ms |
| p99 | 127.8 ms | 13.1 ms |
| device share | 11.7–12.8% | 12.0–12.8% |

**~10× better p99.** The cause is the fixed batch: the baseline makes every request wait
for 31 companions even when the GPU is idle, while dynamic batching takes whatever is
actually queued. Load balance is a **tie** — the baseline's shared queue with pull-based
workers self-balances perfectly well, and it would be dishonest to claim otherwise.

**At saturation** — 2 GPUs, a segmentation-weight model (8 ms/batch + 1.5 ms/row):

| | baseline | ShipInfer |
|---|---:|---:|
| achieved | 935 req/s | 982 req/s |
| p50 | 72 ms | 80 ms |
| p99 | 87 ms | 256 ms |
| rejected | 0 (invisible) | 73 (reported) |

**The baseline wins on the latency number, and the reason matters.** Its producer *blocks*
on a full queue, so it never admits more than it can serve — the offered rate is silently
not met (935 of 1000) and everything that does get in waits about the same. ShipInfer meets
the rate, refuses what it cannot serve, and pays with a longer tail.

Neither is better without saying what you value. What the number does show is the standing
cost of queue depth: at `max_queue_size: 64` with `max_batch_size: 32`, two batches are
always queued, and at 56 ms per batch that is ~112 ms of latency that exists by
configuration. Shallower queues trade it away directly:

| total capacity | ShipInfer p50 | ShipInfer p99 | rejected |
|---:|---:|---:|---:|
| 512 | 81 ms | 462 ms | 0 |
| 128 | 80 ms | 292 ms | 42 |
| 32 | 55 ms | 238 ms | 538 |
| 16 | 37 ms | 179 ms | 1095 |
| 8 | 30 ms | 149 ms | 1692 |

That is the knob, and the bench is how a deployment picks it.

### A negative result worth recording

The saturation numbers suggested the dynamic-batching delay window was pure loss once the
queue is deep: waiting 3 ms to grow a batch from 29 to 32 costs 1.84 ms/row against
1.78 ms/row for running immediately. An adaptive version that skips the window after a
full batch was implemented and A/B'd — and changed nothing measurable (p99 196 ms against
209 ms, inside run-to-run noise; identical mean batch size). It was reverted. The server's
own metrics explain why: batch size was already at p50 = p99 = 32, so the window was rarely
firing in the first place, and the latency was queue depth rather than batching delay.

### Running it

```bash
python benchmarks/compare_baseline.py --seconds 5 --cameras 50 --fps 20 --skew 8
python benchmarks/compare_baseline.py --gpus 0,1 --fixed-ms 8 --per-item-ms 1.5 --capacity 32
```

`--fixed-ms` / `--per-item-ms` are the cost model. Set them from a real engine's measured
batch-1 and batch-N times; the ratio between them is what decides whether batching helps at
all.
