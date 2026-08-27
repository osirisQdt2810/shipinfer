# `csrc/` — the C++ data plane

## Why this exists

The Python implementation was measured and the wall was located. Four candidates were
eliminated by measurement, not by argument:

| candidate | evidence it is not the wall |
|---|---|
| the GPUs | at 120 img/s offered the detector retired 119.9 with a CI straddling zero; every model queue SUSTAINED while only the pipeline queue grew |
| the worker pool | 24 / 96 / 192 workers → 87.6 / 81.4 / 85.0 img/s: an 8× range for under 8%, non-monotonic |
| the reassembly lock | 98.9% of its hold removed (770 → 8.7 µs/frame), no measurable change |
| the load generator | it delivered 100% of the offered 120 img/s while the pipeline retired 77 |

What is left is one interpreter. `docker stats` during a saturated run reads **390–534% CPU**
— five cores of forty-eight. Not one core, so the C extensions that release the GIL (OpenCV,
numpy, torch, TensorRT) really do run in parallel; but five of forty-eight while every GPU
queue sits empty is the signature of the *pure-Python* share of the per-frame path holding the
GIL and capping the whole process at about five cores' worth of frames.

`CLAUDE.md` has said from the beginning: **Python for the control plane, C++17/CUDA for the
data plane.** The control plane is here in Python and correct. The data plane was also in
Python, and that is the discrepancy this directory closes.

## What is in scope, and what is deliberately not

**In scope — everything that runs once per frame or once per object:**

- ingest: replay/RTSP source, per-camera pacing, frame handoff
- preprocess: letterbox and normalise on the GPU
- scheduling: the fair per-camera queue, the dynamic batcher
- execution: TensorRT engine load, bindings, `enqueueV3`
- the perception graph: detect → crop → segment/embed
- reassembly: per-frame collection with a timeout, and the eviction policy
- the sink and the occupancy log

**Not in scope — everything that runs once, at start-up:**

the settings tree, the model repository reader, the CLI, the KServe HTTP surface. Porting
those buys nothing measurable and would lose the pydantic validation that makes a bad
`config.yaml` fail at load with a readable message. The Python layer keeps owning them; it
hands this plane a resolved configuration and gets out of the way.

The registries for the families that *do* live here are the exception: `POLICIES()` for
placement and `SOURCES()` for video sources. They are twenty lines apiece and exist for the
same reason the Python ones do — adding a policy or a source must be a new file and a
registrar, never an edit to a switch statement (seam 1). They resolve a *name* the Python
layer already validated; they do not read configuration.

That split is not a way of doing less. It is where the 1000-frames-a-second boundary actually
falls, and it is what `CLAUDE.md` specified before any of this was written.

## The measurement is deliberately shared

`cli/bench.cpp` writes **the same buffer-occupancy JSONL** the Python driver and
the baseline binary write. So `benchmarks/harness/analysis.py` judges all three with one
implementation and one set of guards — no new measurement code, and no way for the C++ side to
be scored by a friendlier judge than the thing it is being compared against.

## Building

Built on the **host** and run in the **container**, which is exactly what
`benchmarks/harness/baseline.py` already does for `sim_pipeline_v2` and for the same reason:
the container image has a toolchain but not OpenCV, the host has both, and the binary's
shared-library closure is staged into a directory the container mounts. See
`stage_runtime_libs`.

```bash
python scripts/build_csrc.py            # host build -> csrc/build/bench and the test binaries; run them with deploy/rootless/cpp.sh (SHIPINFER_CPP_BINARY=test_dataplane for the checks)
deploy/rootless/bench.sh --systems cpp  # run it in the container, on real GPUs
```

Host `nvcc` is 11.5 against a 12.6 driver. That is fine for this box — the A5000s are sm_86,
which 11.5 supports, and a cudart-linked binary is forward-compatible with a newer driver —
but it is why `sm_89` cannot be built here, and why a production build belongs in a container
with a matching toolkit.

## Where it runs

`deploy/rootless/cpp.sh` runs the binaries in the container, which is where every measurement
belongs (`.claude/CLAUDE.md`, "Where commands run"). Nothing stops `csrc/build/bench` being
invoked directly on the host — `runtime/containment.py` is a Python gate and does not see this
binary — so a number from a host run is **not gated** and is not a production number. Ledger
C47 tracks making the binary consult the gate itself.
