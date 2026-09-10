Version directory for `ship_embedder`.

Drop the built artefact here (`model.plan` for TensorRT, `model.onnx` for ONNX Runtime)
and switch `platform` in `../config.yaml`. Engines are host-specific — they are built on
the target node, never committed.

To build this one, on the node that will run it and inside the container:

    deploy/rootless/run.sh python scripts/build_engines.py --only reid

ONE step now. The `reid` target builds ONE engine that this model and its sibling both use,
and installs it into BOTH version directories — so `--force` is a remedy that works. It used
to install into neither, which left the benchmark's byte-identity guard refusing every run
over a plan nothing had put here while naming that command as the fix.

The target is still called `reid`: `--only ship_embedder` exits 2 with "unknown model(s)",
because the name is the build target's and one target feeds two models.
