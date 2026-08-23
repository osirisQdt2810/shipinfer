# ShipInfer

A **Triton-shaped, hackable multi-GPU inference server** for maritime perception: detect
ships and people across ~50 cameras, segment ships, embed both, recognise ship identity,
and feed tracklets downstream.

It exists because the previous generation of this pipeline ran every model on GPU 0 and
silently starved quiet cameras through a shared evict-oldest buffer. This one puts the
scheduler — placement, batching, fairness, backpressure — at the centre, and builds
everything else on libraries that are already fast.

```bash
pip install -e ".[dev,cli]"
# torch must match your DRIVER, not the newest release (see `nvidia-smi`):
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

shipinfer doctor                      # what this host can run
shipinfer repo ls                     # what the repository holds, and where it would place it
shipinfer serve --http --port 8000    # KServe v2 on http://localhost:8000/docs
shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 5 --skew 8
```

## What it does

| Concern | How |
|---|---|
| **Model repository** | Triton's layout and config vocabulary, in YAML — `max_batch_size`, `instance_group`, `dynamic_batching`, `version_policy`, ensembles |
| **Multi-GPU placement** | Five policies behind a registry; the default keeps a frame on the GPU that holds it and spills only when that queue backs up |
| **Fairness** | Per-camera round-robin inside priority lanes, so one crowded camera cannot fill a batch |
| **Backpressure** | Bounded per-instance queues that *say* they are full, carrying depth and capacity |
| **Batching** | Dynamic batching with a delay window and preferred sizes, plus zero-copy scatter back to per-request responses |
| **Execution** | TensorRT (with CUDA graph replay), ONNX Runtime, TorchScript, or a deterministic mock |
| **Ensembles** | A declared DAG with conditional steps, type-checked against the loaded models at start-up |
| **Fused kernels** | `3rdparty/shipinfer-imgproc`: resize + colour convert + normalise + NHWC→NCHW in one pass, batched crop, device-side NMS — CUDA and HIP from one source |
| **Observability** | Prometheus/JSONL metrics, structured logging with a non-blocking sink, `/v2/health` and `/v2/statistics` |

## Measured

8 × RTX A5000, `person_embedder`, 50 cameras × 20 fps, camera 0 sending 8× the traffic:

```
submitted 3000  completed 3000  rejected 0  failed 0  in 3.01s -> 998 req/s
latency ms  p50=5.68  p95=6.48  p99=7.59  max=12.94

 device   requests   share
 cuda:0        370   12.3%
 cuda:1        388   12.9%
 cuda:2        352   11.7%
 cuda:3        381   12.7%
 cuda:4        395   13.2%
 cuda:5        369   12.3%
 cuda:6        361   12.0%
 cuda:7        384   12.8%

per-camera served  min=40  median=52  max=430  (submission skew was 8x)
```

Every GPU inside 11.7–13.2%, and the quietest camera served every frame it submitted.

The fused preprocessing kernel, 8 × 1080p → 640², writing straight into a torch CUDA
tensor: **9.7 ms (822 img/s)** against torch's **13.7 ms (585 img/s)**, output bit-identical.

## Design in one paragraph

Python owns the control plane; C++/CUDA owns the data plane. **Torch is the runtime
substrate** — its caching allocators, streams, events and CUDA graph capture are better than
anything worth hand-writing, and on ROCm the same API covers HIP for free. What this project
writes itself is the layer above (queues, batching, placement) and the fused kernels torch
has no equivalent for. A GPU is the unit of a worker: one thread, one context, one device,
for that thread's whole life; work moves between GPUs by being *queued* elsewhere, never by
reaching across. `core`, `scheduling` and `repository` import no accelerator runtime at all,
which is why the scheduler's fairness and balancing behaviour is tested on a laptop.

See [`.claude/DECISIONS.md`](.claude/DECISIONS.md) for the nine ADRs behind that.

## Layout

```
src/shipinfer/
  core/        pure types, errors, settings, logging, metrics, the Registry primitive
  repository/  the on-disk model repository
  scheduling/  queues / batching / policies        <- the part this project exists to own
  runtime/     devices, streams, memory, graphs, image ops   <- the accelerator seam
  backends/    tensorrt / onnx / torchscript / mock
  server/      instances, models, ensembles, cache, health, KServe v2
3rdparty/shipinfer-imgproc/   fused CUDA/HIP kernels — its own repository, as a submodule
model_repository/  the real DAG, runnable anywhere on the mock backend
benchmarks/        the head-to-head against the counting-simulation architecture
deploy/            Dockerfile and compose; everything runs in a container
```

Every extensible family is a package plus a registry: adding a placement policy, a queue, a
backend, a log sink or a metrics exporter is a new file and a decorator.

## Testing

```bash
pytest              # offline tier — no GPU needed, and that is enforced
pytest -m gpu       # real devices
pytest -m multigpu  # the balancing evidence
```

The offline tier covers the scheduler's invariants precisely because tests that need
sixteen GPUs get written once and then never run.

## Fused kernels

They live in [shipinfer-imgproc](https://github.com/osirisQdt2810/shipinfer-imgproc),
vendored here as a submodule:

```bash
git submodule update --init 3rdparty/shipinfer-imgproc
pip install -e 3rdparty/shipinfer-imgproc
python scripts/build_native.py --arch 86     # picks a compatible toolkit and compiler
```

Optional. Every native component has a Python counterpart, and `execution.provider`
(`auto` / `native` / `python`) chooses — `native` refuses to start without it, so a
production deploy cannot silently regress to the slow path. CI deliberately does not check
the submodule out, which is how that promise stays honest.

## Status

Working: the repository, the scheduler, all four backends, ensembles, the HTTP API, the
native kernels, and the metrics.
Not yet: `ingest/` (NVDEC decode and per-camera actors) and `pipeline/` (the reassembler and
Kafka publishing) are declared in the layout and still to be written.
