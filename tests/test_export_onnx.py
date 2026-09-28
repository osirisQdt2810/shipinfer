"""`scripts/export_onnx.py` refuses an export whose batch is still fixed anywhere.

The shipped YOLO ONNX baked batch 8 into their attention Reshapes, so a re-dim of the input alone
would build a plan that accepts batch 3 and then reshapes it as 8. `check` is what stops that
reaching `build_engines.py --dynamic`, so it is tested on graphs built here: one with a fixed
batch, one symbolic but with the batch left in a Reshape's shape, one clean.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

onnx = pytest.importorskip("onnx")
import numpy as np  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_onnx.py"


@pytest.fixture(scope="module")
def export_onnx() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scripts.export_onnx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model(path: Path, batch, reshape_to: list[int] | None) -> Path:
    """`images [batch, 3, 4, 4]` -> Reshape or Flatten -> `out [batch, 48]`."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [batch, 3, 4, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [batch, 48])
    if reshape_to is None:
        nodes, inits = [helper.make_node("Flatten", ["images"], ["out"], axis=1)], []
    else:
        shape = numpy_helper.from_array(np.array(reshape_to, dtype=np.int64), "shape")
        nodes, inits = [helper.make_node("Reshape", ["images", "shape"], ["out"])], [shape]
    graph = helper.make_graph(nodes, "g", [images], [out], initializer=inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


def test_a_fixed_batch_is_refused(export_onnx: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="fixed batch of 8"):
        export_onnx.check(_model(tmp_path / "m.onnx", 8, None), 8)


def test_a_batch_left_in_a_reshape_is_refused(export_onnx: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="still hold the batch 8"):
        export_onnx.check(_model(tmp_path / "m.onnx", "batch", [8, 48]), 8)


def test_a_clean_dynamic_graph_passes(export_onnx: ModuleType, tmp_path: Path) -> None:
    export_onnx.check(_model(tmp_path / "m.onnx", "batch", [-1, 48]), 8)


def test_redim_makes_the_batch_symbolic_and_nothing_else(
    export_onnx: ModuleType, tmp_path: Path
) -> None:
    """The ReID route: a graph with no batch constant, re-dimmed in place."""
    out = export_onnx.redim_batch(_model(tmp_path / "s.onnx", 16, None), tmp_path / "d.onnx")

    graph = onnx.load(str(out)).graph
    dims = [
        [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*graph.input, *graph.output]
    ]
    assert dims == [["batch", 3, 4, 4], ["batch", 48]]
