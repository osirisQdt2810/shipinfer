"""Real TorchScript fixtures — what the offline tier runs instead of a fake backend.

`backends/mock.py` gave tests shaped output, a latency knob and injected failure. The first
two a real module gives honestly: a scripted `Linear` is a real function of its input, and a
loop of real matmuls costs real time *and blocks the thread with the GIL released*, which is
what a worker waiting on an accelerator actually does — closer to the truth than `sleep`.

The cost is calibrated once per session (`unit_cost_ms`) and turned into an iteration count,
because a machine's speed is not a constant anyone should hard-code.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

__all__ = [
    "HIDDEN",
    "build_model",
    "iterations_for",
    "materialise",
    "unit_cost_ms",
    "write_model",
]

#: Width of the square matmul that is one unit of work. Big enough that one step is
#: measurable, small enough that a 5 ms fixture is not a memory event.
HIDDEN = 96


class _Fixture(torch.nn.Module):
    """One real projection per declared output, over every declared input.

    Arity matters: a model that declares two outputs must return two tensors, or the backend
    refuses it — which is the contract a real engine is held to as well.
    """

    def __init__(
        self, in_features: list[int], out_features: list[int], work: int, seed: int = 20260828
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.projections = torch.nn.ModuleList(
            [torch.nn.Linear(sum(in_features), width) for width in out_features]
        )
        self.register_buffer("hidden", torch.eye(HIDDEN) * 0.5 + 0.001)
        self.work = work

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        flat = torch.cat([x.flatten(1) if x.dim() > 2 else x for x in inputs], dim=1)
        if self.work > 0:
            spin = self.hidden
            for _ in range(self.work):
                spin = spin @ self.hidden
            flat = flat + spin[0, 0] * 0.0
        return [projection(flat) for projection in self.projections]


class _Wrapper(torch.nn.Module):
    """The positional call the backend makes, over the fixture underneath."""

    def __init__(self, fixture: _Fixture, single_output: bool) -> None:
        super().__init__()
        self.fixture = fixture
        self.single_output = single_output

    def forward(self, *inputs: torch.Tensor):
        outputs = self.fixture(list(inputs))
        return outputs[0] if self.single_output else tuple(outputs)


def build_model(
    in_features: Sequence[int],
    out_features: Sequence[int],
    work: int,
    seed: int = 20260828,
    in_shapes: Sequence[Sequence[int]] | None = None,
) -> torch.jit.ScriptModule:
    """A traced module with the arity the config declares.

    Traced rather than scripted: the arity comes from the config, so the signature is not
    known when this file is written, and a class built at runtime has no source for
    ``torch.jit.script`` to read. Tracing records the real operations either way.
    """
    fixture = _Fixture(list(in_features), list(out_features), work, seed)
    wrapper = _Wrapper(fixture, single_output=len(out_features) == 1).eval()
    # Traced with the shape the config declares, not a flattened stand-in: tracing freezes
    # the path it saw, so a 2-D example would bake out the flatten a 4-D input needs.
    shapes = list(in_shapes) if in_shapes is not None else [[width] for width in in_features]
    example = tuple(torch.zeros(1, *shape) for shape in shapes)
    with torch.no_grad():
        return torch.jit.trace(wrapper, example, check_trace=False)


@functools.lru_cache(maxsize=1)
def unit_cost_ms() -> float:
    """Milliseconds for one unit of work on this machine. Measured once per process."""
    module = build_model([4], [4], 32)
    sample = torch.zeros(1, 4)
    for _ in range(3):
        module(sample)
    start = time.perf_counter()
    module(sample)
    return max((time.perf_counter() - start) * 1000.0 / 32, 1e-4)


def iterations_for(latency_ms: float, unit_ms: float) -> int:
    return max(round(latency_ms / unit_ms), 0) if latency_ms > 0 else 0


def write_model(
    directory: Path,
    *,
    in_features: Sequence[int] | int,
    out_features: Sequence[int] | int,
    latency_ms: float = 0.0,
    unit_ms: float | None = None,
    seed: int = 20260828,
    broken: bool = False,
    in_shapes: Sequence[Sequence[int]] | None = None,
) -> Path:
    """Save a scripted fixture whose one call costs roughly ``latency_ms``.

    The calibration is measured once per process, so a caller that has no opinion passes
    nothing: replacing a fake backend should not mean threading a number through every test.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # An int is the ordinary case (one tensor in, one out); a sequence is the general one.
    in_features = [in_features] if isinstance(in_features, int) else list(in_features)
    out_features = [out_features] if isinstance(out_features, int) else list(out_features)
    path = directory / "model.pt"
    unit = unit_cost_ms() if unit_ms is None else unit_ms
    shapes = list(in_shapes) if in_shapes is not None else None
    if broken:
        # A module that disagrees with its own config: it loads, and every execution raises
        # for a real reason. That is a deployment's actual failure mode -- someone rebuilt
        # the artefact and not the config -- and the honest replacement for an injected fault.
        in_features = [width + 1 for width in in_features]
        shapes = None
    module = build_model(
        in_features, out_features, iterations_for(latency_ms, unit), seed, in_shapes=shapes
    )
    torch.jit.save(module, str(path))
    return path


def materialise(root: Path, *, latency_ms: float | Mapping[str, float] = 0.0) -> list[Path]:
    """Give every model under ``root`` a real ``model.pt`` matching its own config.

    A repository already declares each model's contract — input and output shapes, and the
    version directory. Reading it and building to it keeps the fixture honest: the test says
    what the model is, once, in the config, and the file on disk is a real module of exactly
    that shape rather than a second copy of the same numbers.
    """
    written: list[Path] = []
    for config in sorted(root.glob("*/config.yaml")):
        spec = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if spec.get("platform") != "pytorch":
            continue
        name = config.parent.name
        params = spec.get("parameters") or {}
        # The config already says how expensive this model is and which weights it wants;
        # reading it here is what keeps `latency_ms:` meaningful after the fake backend that
        # invented it is gone -- the cost is now real compute, declared in the same place.
        cost = latency_ms.get(name, 0.0) if isinstance(latency_ms, Mapping) else latency_ms
        cost = float(params.get("latency_ms", cost))
        seed = int(params.get("seed", 20260828))
        broken = bool(params.get("disagrees_with_its_config", False))
        for version in sorted(p for p in config.parent.iterdir() if p.is_dir()):
            written.append(
                write_model(
                    version,
                    in_features=_features(spec.get("inputs")),
                    out_features=_features(spec.get("outputs")),
                    in_shapes=_shapes(spec.get("inputs")),
                    latency_ms=cost,
                    seed=seed,
                    broken=broken,
                )
            )
    return written


def _features(specs: Any) -> list[int]:
    """One flattened width per declared tensor, in declaration order."""
    return [
        int(math.prod(int(d) for d in (entry.get("dims") or [1]))) for entry in (specs or [{}])
    ]


def expected_output(
    *,
    in_features: Sequence[int],
    out_features: Sequence[int],
    inputs: Sequence[Any],
    seed: int = 20260828,
) -> list[Any]:
    """What the fixture with these weights answers — the honest replacement for a fake's RNG.

    A test that has to predict a model's output now predicts it by *building the same model*
    and asking, rather than by re-implementing a fake backend's random draw.
    """
    module = build_model(in_features, out_features, work=0, seed=seed)
    with torch.no_grad():
        result = module(*[torch.as_tensor(value, dtype=torch.float32) for value in inputs])
    values = result if isinstance(result, (tuple, list)) else (result,)
    return [value.detach().cpu().numpy() for value in values]


def _shapes(specs: Any) -> list[list[int]]:
    """Each declared tensor's dims, as the config wrote them."""
    return [[int(d) for d in (entry.get("dims") or [1])] for entry in (specs or [{}])]
