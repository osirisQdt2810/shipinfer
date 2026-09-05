"""``shipinfer plan`` — resolve a chain file into the plan the C++ data plane reads.

The composition ADR-014 describes: the Python control plane validates the chain (ADR-017's
one door) and reads the model repository, then hands the other plane a resolved
configuration. This is that hand-over, spelled as a command so it is reviewable and
diffable rather than happening invisibly inside a launcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

from shipinfer.repository import ModelRepository
from shipinfer.repository.resolved import model_extents, model_runtimes
from shipinfer.topology import Topology, load_topology
from shipinfer.topology.plan import plan_text, resolve_plan

__all__ = ["plan"]

#: The models `scripts/build_engines.py` has a target for. The two embedders are absent from
#: it, so a note that sent an operator there for them cost a container start and changed
#: nothing -- `build_engines.py --only ship_embedder` exits 2 with "unknown model(s)".
_BUILDABLE = frozenset({"ship_detector", "ship_segmenter"})


def plan(topology: Path, repository: Path, out: Path | None = None) -> int:
    """Write the resolved plan for ``topology`` to ``out``, or to stdout."""
    chain = load_topology(topology)
    models = ModelRepository.load(repository)
    runtimes = model_runtimes(models)
    text = plan_text(resolve_plan(chain, dims=model_extents(models), runtimes=runtimes))
    if out is None:
        print(text, end="")
    else:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text.splitlines())} lines) for chain {chain.name!r}")
    _report_missing(chain, models)
    return 0


def _report_missing(chain: Topology, models: ModelRepository) -> None:
    """Say which artefacts the plan names and the repository does not hold.

    Reported and NOT refused: writing a plan where the engine is absent is the documented
    workflow. ADR-014 lets the control plane run on a driverless box and
    `model_repository/*/1/README.md` says engines are built on the node that runs them, so a
    fresh checkout legitimately has a `config.yaml` and no artefact. What this removes is the
    SILENCE -- otherwise the operator learns at bench start-up, inside a container, while the
    machine that knew is the one that wrote the plan.
    """
    named = {node.spec.model for node in chain.nodes if node.spec.model}
    missing = [
        (name, wanted)
        for name in sorted(named)
        for wanted in [_wanted_file(models, name)]
        if wanted is not None
    ]
    if not missing:
        return
    buildable = sorted(name for name, _ in missing if name in _BUILDABLE)
    rest = sorted(f"{name}/{wanted}" for name, wanted in missing if name not in _BUILDABLE)
    lines = [f"note: {len(missing)} artefact(s) this plan names are not in {models.root}:"]
    if buildable:
        lines.append(
            f"  {', '.join(buildable)} — `python scripts/build_engines.py "
            f"--only {' '.join(buildable)}` on the node that runs them, inside the container"
        )
    if rest:
        # NOT the build script: it has no target for these, so sending an operator there
        # costs them a container start and leaves the artefact exactly as absent.
        lines.append(
            f"  {', '.join(rest)} — see `model_repository/<name>/1/README.md`; "
            f"`scripts/build_engines.py` has no target for these"
        )
    print("\n".join(lines), file=sys.stderr)


def _wanted_file(models: ModelRepository, name: str) -> str | None:
    """The artefact THIS model's backend opens, if the repository does not hold it yet.

    ``None`` when it is there, when the platform has no file of its own (`ensemble`), or when
    TensorRT will BUILD it: `resolve_engine` compiles a sibling `.onnx` at load, so a version
    directory holding one -- which the README tells an operator to drop there -- is complete
    and a note about the plan would be a false alarm on every run.
    """
    entry = models.entry(name)
    wanted = entry.config.artefact_file
    if wanted is None:
        return None
    directory = entry.root / str(entry.latest)
    if (directory / wanted).is_file():
        return None
    if entry.config.platform == "tensorrt" and any(directory.glob("*.onnx")):
        return None
    return wanted
