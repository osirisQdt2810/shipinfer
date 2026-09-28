#!/usr/bin/env python3
"""Export the dynamic-batch ONNX that `build_engines.py --dynamic` builds from. CPU only.

The shipped ONNX fix the batch (8, 8, 16), and the YOLO graphs bake that 8 into their attention
Reshapes, so re-dimming the input is not enough: they are re-exported from the `.pt` with the
ORIGINAL recipe, read back from their own metadata, changing only `dynamic`. The ReID graph has
no batch constant (a `Flatten(axis=1)`), so its batch axis is made symbolic in place. Each
result is checked: a symbolic batch on every input and output, and no batch literal left.

    python scripts/export_onnx.py        # -> models/{yolo26n,yolo26n-seg,reid_r50}_dyn.onnx
"""

from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Before anything imports torch: a CPU export must never open a CUDA context on the host.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("YOLO_OFFLINE", "1")
# The vendored ultralytics is a read-only submodule whose repository tracks its `.pyc` files.
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
#: ultralytics 8.4.34, the version whose name is in the shipped ONNX's metadata.
VENDORED = REPO / "benchmarks" / "baseline"
YOLO_STEMS = ("yolo26n", "yolo26n-seg")
REID_STEM = "reid_r50"


def batch_literals(graph: Any, batch: int) -> int:
    """Shape-bearing constants still holding `batch` -- what a re-dim alone would leave behind."""
    from onnx import numpy_helper

    values = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attribute.t)
    found = 0
    for node in graph.node:
        if node.op_type not in ("Reshape", "Expand", "Tile"):
            continue
        for name in node.input[1:]:
            array = values.get(name)
            if array is not None and array.dtype.kind in "iu" and array.size <= 8:
                found += batch in array.flatten().tolist()
    return found


def check(path: Path, batch: int) -> None:
    """Refuse an export whose batch is fixed anywhere, or survives as a shape constant."""
    import onnx

    graph = onnx.load(str(path)).graph
    for value in [*graph.input, *graph.output]:
        first = value.type.tensor_type.shape.dim[0]
        if not first.dim_param:
            raise SystemExit(
                f"{path.name}: {value.name} keeps a fixed batch of {first.dim_value}"
            )
    left = batch_literals(graph, batch)
    if left:
        raise SystemExit(f"{path.name}: {left} shape constant(s) still hold the batch {batch}")


def export_yolo(stem: str) -> Path:
    """Re-export from the checkpoint, with the recipe the original's metadata records."""
    import onnx

    original = onnx.load(str(MODELS / f"{stem}.onnx"), load_external_data=False)
    meta = {prop.key: prop.value for prop in original.metadata_props}
    sys.path.insert(0, str(VENDORED))
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != meta["version"]:
        raise SystemExit(
            f"{stem}: exported by ultralytics {meta['version']}, but {VENDORED} holds "
            f"{ultralytics.__version__}; a different exporter is a different graph"
        )
    recipe = dict(ast.literal_eval(meta["args"]), dynamic=True)
    target = MODELS / f"{stem}_dyn.onnx"
    with tempfile.TemporaryDirectory() as scratch:
        # Copied first: the exporter writes its ONNX beside the weights, which in `models/`
        # would overwrite the static export the baseline's engines are built from.
        weights = Path(shutil.copy(MODELS / f"{stem}.pt", scratch))
        exported = YOLO(str(weights)).export(
            format="onnx", device="cpu", imgsz=ast.literal_eval(meta["imgsz"]), **recipe
        )
        shutil.move(str(exported), target)
    check(target, int(meta["batch"]))
    return target


def redim_batch(source: Path, target: Path) -> Path:
    """The batch axis of every input and output made symbolic; nothing else touched."""
    import onnx

    model = onnx.load(str(source))
    batch = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_param = "batch"  # a oneof: clears dim_value
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    check(target, batch)
    return target


def main() -> int:
    reid = redim_batch(MODELS / f"{REID_STEM}.onnx", MODELS / f"{REID_STEM}_dyn.onnx")
    for path in [*(export_yolo(stem) for stem in YOLO_STEMS), reid]:
        print(f"{path.relative_to(REPO)}  {path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
