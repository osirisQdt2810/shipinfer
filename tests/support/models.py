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
import tempfile
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

#: Calibration: enough iterations that fixed call overhead is a small share of the sample,
#: averaged over several runs so one scheduling hiccup cannot set every fixture's cost.
_CALIBRATION_WORK, _CALIBRATION_RUNS = 128, 5


class _Fixture(torch.nn.Module):
    """One real projection per declared output, over every declared input.

    Arity matters: a model that declares two outputs must return two tensors, or the backend
    refuses it — which is the contract a real engine is held to as well.
    """

    @staticmethod
    def _rng(seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def __init__(
        self,
        in_features: list[int],
        out_features: list[int],
        work: int,
        seed: int = 20260828,
        flag_offset: float = 0.0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.projections = torch.nn.ModuleList(
            [torch.nn.Linear(sum(in_features), width) for width in out_features]
        )
        # ORTHOGONAL, and that is the whole trick: `spin @ hidden` repeated thousands of
        # times must neither blow up nor decay. A contraction (the old `eye * 0.5`) drives
        # every entry subnormal within ~100 iterations, and subnormal matmul is an order of
        # magnitude slower on CPU -- so the per-iteration cost grew tenfold with the count
        # and no linear calibration could exist. A norm-preserving matrix keeps every
        # iteration the same price, which is what makes `latency_ms` mean something.
        orthogonal, _ = torch.linalg.qr(torch.randn(HIDDEN, HIDDEN, generator=self._rng(seed)))
        self.register_buffer("hidden", orthogonal.contiguous())
        #: Small enough that the spin cannot move an output past its own rounding, large
        #: enough that it is not zero: the cost has to be real without the arithmetic under
        #: test changing. `expected_output()` builds the same module, so predictions stay exact.
        self.register_buffer("epsilon", torch.tensor(1e-12))
        self.work = work
        #: Added to the LAST output's bias. An ensemble condition reads a flag output and
        #: runs a branch when it clears 1; without this the flag is the untrained bias,
        #: |b| < 0.5, which truncates to 0 on the INT32 cast every single run -- so no
        #: branch-taken path was exercised anywhere in the ensemble tests.
        if flag_offset:
            with torch.no_grad():
                self.projections[-1].bias.add_(float(flag_offset))

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        flat = torch.cat([x.flatten(1) if x.dim() > 2 else x for x in inputs], dim=1)
        if self.work > 0:
            # Seeded from the INPUT and folded back into the output, both deliberately.
            # `TorchScriptBackend` runs `torch.jit.optimize_for_inference`, which freezes the
            # module: work that starts from a buffer is constant, and a result multiplied by
            # zero is dead, so the earlier version of this loop was folded away entirely and
            # a `latency_ms: 60` model executed in 0.05 ms. Data in, data out, no folding.
            spin = self.hidden + flat.sum() * self.epsilon
            for _ in range(self.work):
                spin = spin @ self.hidden
            flat = flat + spin[0, 0] * self.epsilon
        return [projection(flat) for projection in self.projections]


class _Wrapper(torch.nn.Module):
    """The positional call the backend makes, over the fixture underneath."""

    def __init__(self, fixture: torch.jit.ScriptModule, single_output: bool) -> None:
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
    flag_offset: float = 0.0,
) -> torch.jit.ScriptModule:
    """A traced module with the arity the config declares.

    Traced rather than scripted: the arity comes from the config, so the signature is not
    known when this file is written, and a class built at runtime has no source for
    ``torch.jit.script`` to read. Tracing records the real operations either way.
    """
    fixture = _Fixture(list(in_features), list(out_features), work, seed, flag_offset)
    # The fixture is SCRIPTED, then wrapped and traced. Tracing alone unrolls the work loop
    # into one node per iteration, so a 60 ms model became a 4000-node graph that took longer
    # to freeze than to run and whose per-iteration cost grew tenfold with the count -- the
    # calibration could not be linear because the thing being calibrated was not. Scripting
    # keeps the loop a loop; the wrapper is still traced, because its arity comes from the
    # config and is not known when this file is written.
    wrapper = _Wrapper(torch.jit.script(fixture), single_output=len(out_features) == 1).eval()
    # Traced with the shape the config declares, not a flattened stand-in: tracing freezes
    # the path it saw, so a 2-D example would bake out the flatten a 4-D input needs.
    shapes = list(in_shapes) if in_shapes is not None else [[width] for width in in_features]
    example = tuple(torch.zeros(1, *shape) for shape in shapes)
    with torch.no_grad():
        return torch.jit.trace(wrapper, example, check_trace=False)


@functools.lru_cache(maxsize=1)
def unit_cost_ms() -> float:
    """Milliseconds for one unit of work on this machine. Measured once per process.

    Timed through ``optimize_for_inference``, which is what ``TorchScriptBackend`` runs
    (``backends/torch_backend.py``) -- calibrating on the raw traced module measures a
    module the server never executes, and the two differ by several times over.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.pt"
        build_model([4], [4], _CALIBRATION_WORK).save(str(path))
        module = torch.jit.optimize_for_inference(torch.jit.load(str(path)).eval())
        sample = torch.zeros(1, 4)
        for _ in range(3):
            module(sample)
        start = time.perf_counter()
        for _ in range(_CALIBRATION_RUNS):
            module(sample)
        elapsed = (time.perf_counter() - start) * 1000.0 / _CALIBRATION_RUNS
    return max(elapsed / _CALIBRATION_WORK, 1e-4)


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
    flag_offset: float = 0.0,
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
        in_features,
        out_features,
        iterations_for(latency_ms, unit),
        seed,
        in_shapes=shapes,
        flag_offset=flag_offset,
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
        # `always: 1` raises the last output above the threshold an ensemble condition tests,
        # so a branch-taken path is actually exercised. Read here rather than ignored: three
        # configs already declared it and nothing honoured it.
        flag_offset = float(params.get("always", 0.0)) * 2.0
        for version in sorted(p for p in config.parent.iterdir() if p.is_dir()):
            written.append(
                write_model(
                    version,
                    in_features=_features(spec.get("inputs")),
                    out_features=_features(spec.get("outputs")),
                    in_shapes=_shapes(spec.get("inputs")),
                    latency_ms=cost,
                    seed=seed,
                    flag_offset=flag_offset,
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
    flag_offset: float = 0.0,
) -> list[Any]:
    """What the fixture with these weights answers — the honest replacement for a fake's RNG.

    A test that has to predict a model's output now predicts it by *building the same model*
    and asking, rather than by re-implementing a fake backend's random draw.
    """
    module = build_model(in_features, out_features, work=0, seed=seed, flag_offset=flag_offset)
    with torch.no_grad():
        result = module(*[torch.as_tensor(value, dtype=torch.float32) for value in inputs])
    values = result if isinstance(result, (tuple, list)) else (result,)
    return [value.detach().cpu().numpy() for value in values]


def _shapes(specs: Any) -> list[list[int]]:
    """Each declared tensor's dims, as the config wrote them."""
    return [[int(d) for d in (entry.get("dims") or [1])] for entry in (specs or [{}])]
