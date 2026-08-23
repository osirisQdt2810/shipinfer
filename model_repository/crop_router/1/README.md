Version directory for `crop_router`.

In a real deployment this is a TensorRT or Python backend wrapping
`shipinfer.runtime.ops.ImageOps.crop_batch` — the fused device kernel that extracts and
resizes detected boxes without ever moving the full frame off the GPU.
