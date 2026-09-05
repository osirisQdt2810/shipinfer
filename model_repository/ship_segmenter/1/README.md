Version directory for `ship_segmenter`.

Drop the built artefact here (`model.plan` for TensorRT, `model.onnx` for ONNX Runtime)
and switch `platform` in `../config.yaml`. Engines are host-specific — they are built on
the target node, never committed.

To build this one, on the node that will run it and inside the container:

    deploy/rootless/run.sh python scripts/build_engines.py --only ship_segmenter

That target installs the plan here. `--fp16` builds the half-precision engine instead.
