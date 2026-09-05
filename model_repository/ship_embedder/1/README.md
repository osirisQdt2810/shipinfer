Version directory for `ship_embedder`.

Drop the built artefact here (`model.plan` for TensorRT, `model.onnx` for ONNX Runtime)
and switch `platform` in `../config.yaml`. Engines are host-specific — they are built on
the target node, never committed.

To build this one, on the node that will run it and inside the container:

    deploy/rootless/run.sh python scripts/build_engines.py --only reid
    cp models/reid_r50_fp32.engine model_repository/ship_embedder/1/model.plan

Two steps, because the `reid` target builds ONE engine that this model and its sibling
both use, and installs it into no version directory — `--only ship_embedder` exits 2 with
"unknown model(s)".
