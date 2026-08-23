# Model artefacts

Weights, ONNX exports and TensorRT engines. **Gitignored** — these are large binaries with
their own licences, and an engine is only valid for the GPU architecture and TensorRT version
that built it, so committing one would ship something that fails on any other machine.

Fetch and build them with:

```bash
scripts/fetch_models.py            # downloads the weights, exports ONNX
scripts/build_engines.py           # builds TensorRT engines for THIS machine
```

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
