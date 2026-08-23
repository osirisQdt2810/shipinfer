Version directory for `person_embedder`.

Drop the built artefact here (`model.plan` for TensorRT, `model.onnx` for ONNX Runtime)
and switch `platform` in `../config.yaml`. Engines are host-specific — they are built on
the target node, never committed.
