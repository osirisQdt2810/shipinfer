"""A dynamic-batch plan's rows are its static twin's rows -- the GPU half of `--dynamic`.

`build_engines.py --dynamic` builds `models/*_fp16_dyn.engine` from `export_onnx.py`'s
re-export, and the static `models/*_fp16.engine` stays for the baseline. The dynamic plan is a
serving change ONLY if a row it runs at batch 3 is the row the static plan computes at batch 8
padded, so both run the same real frames, through the pipeline's own preprocessing, and are
compared by what the chain reads, and a dynamic plan's batch-N rows must equal its batch-1 rows.
Across the two builds, tolerances: an fp32 control (re-export against original) agrees to 2e-4
in score, so the fp16 gap -- up to 0.041, measured -- is tactic choice. Skips where a plan is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tests.support.devices import a_test_device

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SCRIPT = ROOT / "scripts" / "build_engines.py"
#: What the chain's `decode` keeps (`topology/ship_person_cpu.yaml`), and the slack either side.
#: SCORE_DELTA covers two fp16 builds of one network: 0.041 measured, fp32 control 2e-4.
KEEP, SLACK, SCORE_DELTA, MATCH_IOU = 0.25, 0.08, 0.06, 0.9


@pytest.fixture(scope="module")
def build_engines() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scripts.build_engines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plans(stem: str) -> tuple[Path, Path]:
    static, dynamic = MODELS / f"{stem}_fp16.engine", MODELS / f"{stem}_fp16_dyn.engine"
    if not (static.is_file() and dynamic.is_file()):
        pytest.skip(f"{static.name} and {dynamic.name} are built per machine; build both first")
    return static, dynamic


def _frames(build_engines: ModuleType, target_name: str, rows: int) -> np.ndarray:
    """`rows` real frames, preprocessed exactly as the served path preprocesses them."""
    target = next(t for t in build_engines.TARGETS if t.name == target_name)
    files = build_engines._calibration_files()
    if not files:
        pytest.skip("no replay frames: `benchmarks/baseline` is not checked out")
    stacked = np.concatenate(
        [np.asarray(b) for b in build_engines._calibration_batches(target, files)]
    )
    while len(stacked) < rows:
        stacked = np.concatenate([stacked, stacked])
    return np.ascontiguousarray(stacked[:rows], dtype=np.float32)


def _run(path: Path, batch: np.ndarray) -> dict[str, np.ndarray]:
    """Every output for `batch`; a static plan padded the way `adapter.cpp` pads it."""
    import tensorrt as trt
    import torch

    torch.cuda.set_device(a_test_device())
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
        path.read_bytes()
    )
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    (source,) = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    planned, rows = engine.get_tensor_shape(source)[0], len(batch)
    if planned > 0 and rows < planned:  # repeat the last real row; read back only real ones
        batch = np.concatenate([batch, np.repeat(batch[-1:], planned - rows, axis=0)])
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    images = torch.from_numpy(batch).cuda()
    if planned < 0:
        context.set_input_shape(source, tuple(images.shape))
    context.set_tensor_address(source, images.data_ptr())
    outputs = {}
    for name in names:
        if name != source:
            shape = tuple(context.get_tensor_shape(name))
            outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: value[:rows].cpu().numpy() for name, value in outputs.items()}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / union) if union > 0 else 0.0


def _unmatched(ours: np.ndarray, theirs: np.ndarray) -> list[tuple[int, int]]:
    """Rows of `ours` kept by the chain that `theirs` has no twin for: (image, row)."""
    missing = []
    for image, (mine, other) in enumerate(zip(ours, theirs, strict=True)):
        pool = other[other[:, 4] >= KEEP - SLACK]
        for index, row in enumerate(mine[mine[:, 4] >= KEEP]):
            twins = pool[(pool[:, 5] == row[5]) & (np.abs(pool[:, 4] - row[4]) <= SCORE_DELTA)]
            if not any(_iou(row[:4], twin[:4]) >= MATCH_IOU for twin in twins):
                missing.append((image, index))
    return missing


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize(
    "target, stem", [("ship_detector", "yolo26n"), ("ship_segmenter", "yolo26n-seg")]
)
def test_a_yolo_plan_keeps_the_same_detections(build_engines, target, stem, rows) -> None:
    static, dynamic = _plans(stem)
    frames = _frames(build_engines, target, rows)
    padded, exact = _run(static, frames), _run(dynamic, frames)
    kept = int((padded["output0"][..., 4] >= KEEP).sum())
    assert kept > 0, "no detection above the chain's threshold: the comparison would be vacuous"
    assert _unmatched(padded["output0"], exact["output0"]) == []
    assert _unmatched(exact["output0"], padded["output0"]) == []
    if "output1" in padded:  # the segmenter's prototype bank, which the mask fold reduces
        error = np.linalg.norm(padded["output1"] - exact["output1"]) / np.linalg.norm(
            padded["output1"]
        )
        assert error < 0.02, f"prototypes differ by {error:.4f} relative L2"


@pytest.mark.parametrize(
    "target, stem, rows",
    [
        ("ship_detector", "yolo26n", 8),
        ("ship_segmenter", "yolo26n-seg", 8),
        ("reid", "reid_r50", 16),
    ],
)
def test_a_dynamic_plan_computes_a_row_the_same_at_any_batch(build_engines, target, stem, rows):
    """The property batching must not break: a row's answer does not depend on its batchmates."""
    _, dynamic = _plans(stem)
    frames = _frames(build_engines, target, rows)
    together = _run(dynamic, frames)
    for index in (0, rows // 2, rows - 1):
        alone = _run(dynamic, frames[index : index + 1])
        for name, value in alone.items():
            assert np.abs(value[0] - together[name][index]).max() < 1e-3, (name, index)


@pytest.mark.parametrize("rows", [1, 11, 16])
def test_the_reid_plan_embeds_the_same(build_engines, rows) -> None:
    static, dynamic = _plans("reid_r50")
    crops = _frames(build_engines, "reid", rows)
    a, b = _run(static, crops)["embedding"], _run(dynamic, crops)["embedding"]
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert cosine.min() >= 0.999, f"lowest cosine {cosine.min():.5f}"
