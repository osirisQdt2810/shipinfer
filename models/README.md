# Model artefacts

Weights, ONNX exports and TensorRT engines. **Gitignored** — these are large binaries with
their own licences, and an engine is only valid for the GPU architecture and TensorRT version
that built it, so committing one would ship something that fails on any other machine.

Fetch and build them with:

```bash
# 1. Checkpoints and ONNX. The .pt files come from the ultralytics assets release named
#    in the table below; the ONNX is exported from them:
python -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt').export(format='onnx')"

# 2. Engines for THIS machine's GPU and TensorRT version. Runs in the container, because
#    a plan built anywhere else will not load here:
deploy/rootless/bench.sh --help    # the image this needs
python scripts/build_engines.py --check   # what is present
python scripts/build_engines.py           # build what is not
```

There is no `scripts/fetch_models.py`. This file used to name one, and three other places
pointed at `scripts/build_engines.py` before it existed — so anyone reproducing the
benchmark hit `No such file or directory`. The build half now exists; the fetch half is two
lines of ultralytics and is written out above rather than wrapped in a script that would
need a network the container does not have.

| file | what | provenance |
|---|---|---|
| `yolo26n.pt` | detection weights | ultralytics assets v8.4.0 |
| `yolo26n-seg.pt` | segmentation weights | ultralytics assets v8.4.0 |
| `yolo26n.onnx` | ONNX export, batch 8, 640x640, static | exported here |
| `yolo26n_fp32.engine` | TensorRT plan | built here, this GPU only |
| `yolo26n_fp16.engine` | TensorRT plan, half precision | built here, this GPU only |
| `timing.cache` | TensorRT tactic timings | reused across builds to cut build time |

`yolo26n` is what `references/counting-simulation` benchmarks against, so both systems run
the same engine and the comparison is about the serving architecture rather than the model.
