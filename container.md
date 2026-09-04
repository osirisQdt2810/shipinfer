# Where this actually runs

Written because the question was asked and the answer was not obvious from the repository —
which is itself the problem. Everything here is measured on this machine, not intended.

## The short answer

**Nothing has run in a container. Not once.**

Every test, every benchmark number, every build of the native extension has happened in a
**host virtualenv**:

```
/.dockerenv          absent
/proc/1/cgroup       no container markers
PID 1                systemd
hostname             optix                     <- the host itself
interpreter          <repo>/.venv/bin/python   (CPython 3.10.12, host)
```

This contradicts the project's own requirement — *"mọi thứ đều phải chạy trong docker …
không được chạy nền trên host luôn"*. `deploy/` was written and has never been exercised.
The reason is one permission:

```
$ docker info
permission denied while trying to connect to the Docker daemon socket

$ id -nG | tr ' ' '\n' | grep -x docker
(nothing — the user is not in the docker group)
```

Docker itself is installed (29.4.0) and so is the NVIDIA container toolkit
(`nvidia-container-cli` 1.18.2), so the host is *ready*; only the group membership is
missing. One command unblocks it:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

Until then `deploy/` is a design, not a deployment, and this file exists so nobody mistakes
one for the other.

## CPU versus GPU — what is deliberate and what is a gap

These are two different things and they were being conflated.

### Deliberate: the offline test tier is CPU-only

`ADR-001` and `scripts/run_tests.sh` make the default suite run with
`CUDA_VISIBLE_DEVICES=""`. That is not timidity, it is the property being protected: `core/`,
`scheduling/` and `repository/` must import no torch, no TensorRT, no CUDA, so that

* the scheduling invariants — the part this project exists to own — are testable on a laptop
  in seconds, which is the only way they get tested at all;
* CI needs no GPU runner;
* a machine with no driver still starts the server.

Hiding the devices rather than merely deselecting a marker is the honest version of that
check: an unmarked test can take a CUDA path by accident, pass on a box with eight A5000s and
fail on a runner. That already happened once with `torch.empty(pin_memory=True)`.

So: **a green offline suite is not evidence that anything works on a GPU, and was never
claimed to be.**

### The gap: the GPU path is barely exercised

| Layer | State on this machine |
|---|---|
| Host driver / devices | **8 × RTX A5000**, driver CUDA 12.6, `torch.cuda.is_available()` True, 8 devices |
| torch | 2.7.1+cu126 — real CUDA build |
| `shipvision._C` fused kernels | **built and run on GPU.** Parity against the numpy oracle at 1.788e-07 (3 ULP) over 14 shape/target combinations |
| Fused letterbox, device-to-device | measured: 10.10 ms vs torch 15.12 ms per batch (1.50x), bit-identical output |
| Preprocessing vs OpenCV | measured: 8.8–9.3 ms vs 469–704 ms (~50x), median of 5, `cv2.setNumThreads` pinned |
| TensorRT backend | **never executed an engine.** `import tensorrt` fails — not installed in this venv |
| ONNX backend | **never executed.** `onnxruntime` not installed |
| End-to-end camera → GPU → output | **never run.** `ingest/` and `pipeline/` are new and have no GPU path yet |
| `nvcc` | 11.5 on the host, against a 12.6 driver — which is why the native build needs `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10` and cannot target `sm_89` |

The fused kernels are the only part with real GPU evidence. The **production inference path
has not run on a GPU at all**, because the only backend that has ever executed is `mock`,
which is CPU by construction and exists so the scheduler can be tested without hardware.

That is the honest state, and it is a gap rather than a decision.

## Why the container matters more than convenience here

The `nvcc` 11.5 / driver 12.6 mismatch above is exactly what the container fixes. `deploy/`
pins:

```
base image     nvidia/cuda:12.6.3-cudnn-{runtime,devel}-ubuntu22.04
torch          2.7.1 + torchvision 0.22.1 from the cu126 index
CUDA arches    70;75;80;86;89
stages         base -> builder -> dev -> runtime
```

Inside it, `nvcc` matches the driver, `sm_89` is buildable, and TensorRT can be installed
against a known CUDA. On the host none of that is true, so *the container is the only place
the production backend can be exercised honestly* — it is not a packaging nicety.

GPU access is already declared in `deploy/compose/docker-compose.yml` via
`deploy.resources.reservations.devices` with `capabilities: [gpu]` and `count: all`, using
the modern form rather than the deprecated `runtime: nvidia`.

## What to do, in order

1. `sudo usermod -aG docker $USER && newgrp docker`
2. **SUPERSEDED.** This said `cd deploy && make shell`, and there is no Makefile: this kernel
   refuses `docker build` from an unprivileged user namespace (`deploy/rootless/setup.sh`,
   KERNEL LIMIT), so the compose path was replaced by the scripts in `deploy/rootless/`. The
   door for one command inside the tier's container is `deploy/rootless/run.sh <cmd>`.
3. Re-run the offline suite **inside** the container and confirm the same counts.
4. Install TensorRT in the image, build one engine, and run `pytest -m gpu`. That is the
   first moment the production path has GPU evidence.
5. `shipinfer bench` with a per-device breakdown, from inside the container.

Until step 4, any claim about production inference performance in this repository is about
the fused kernels or about the mock backend, and should say which.

## How to check this file has not gone stale

```bash
[ -f /.dockerenv ] && echo container || echo host
docker info >/dev/null 2>&1 && echo "daemon reachable" || echo "daemon blocked"
python -c "import tensorrt" 2>/dev/null && echo "trt present" || echo "trt absent"
```
