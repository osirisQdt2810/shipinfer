"""``shipinfer plan`` — resolve a chain file into the plan the C++ data plane reads.

The composition ADR-014 describes: the Python control plane validates the chain (ADR-017's
one door) and reads the model repository, then hands the other plane a resolved
configuration. This is that hand-over, spelled as a command so it is reviewable and
diffable rather than happening invisibly inside a launcher.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

from shipinfer.repository import ModelEntry, ModelRepository
from shipinfer.repository.resolved import ModelRuntime, model_extents, model_runtimes
from shipinfer.topology import Topology, load_topology
from shipinfer.topology.plan import plan_text, resolve_plan

__all__ = ["plan"]


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
    _report_missing(chain, models, runtimes)
    return 0


def _report_missing(
    chain: Topology, models: ModelRepository, runtimes: Mapping[str, ModelRuntime]
) -> None:
    """Say which artefacts THE PLAN NAMES that the repository does not hold.

    Reported and NOT refused: writing a plan where the engine is absent is the documented
    workflow. ADR-014 lets the control plane run on a driverless box and
    `model_repository/*/1/README.md` says engines are built on the node that runs them, so a
    fresh checkout legitimately has a `config.yaml` and no artefact. What this removes is the
    SILENCE -- otherwise the operator learns at bench start-up, inside a container, while the
    machine that knew is the one that wrote the plan.
    """
    # A model the repository does not index is SKIPPED, not looked up: `resolve_plan` is
    # deliberately tolerant of one (a slot declaring its own extent needs no config), so
    # `entry()` here turned a written plan into a `ModelNotFoundError` after the file was on
    # disk -- a diagnostic refusing, which is what this exists not to do.
    named = {node.spec.model for node in chain.nodes if node.spec.model}
    missing: list[tuple[ModelEntry, str]] = []
    for name in sorted(named & set(models.names())):
        wanted = _wanted_artefact(models.root, runtimes[name].artefact)
        if wanted is not None:
            missing.append((models.entry(name), wanted))
    if not missing:
        return
    lines = [f"note: {len(missing)} artefact(s) this plan names are not in {models.root}:"]
    lines += [
        f"  {path} — {remedy}" for path, remedy in sorted(_remedies(models.root, missing))
    ]
    print("\n".join(lines), file=sys.stderr)


def _wanted_artefact(root: Path, artefact: str) -> str | None:
    """The artefact the plan names for this model, if the repository does not hold it.

    ``engine_file`` and not what this model's own backend opens: a plan's `artefact` line is
    `engine_file` whatever `platform:` says, because the plane reading a plan is TensorRT-only.
    So a `platform: pytorch` repository genuinely lacks it -- what would be wrong is
    prescribing a TensorRT build, which is the REMEDY's job. ``None`` ONLY where the file is
    there: a sibling `.onnx` is not an exemption, because `resolve_engine` is a PYTHON-plane
    mechanism while the plan's reader opens the path verbatim, and would not write this name
    anyway. The ONNX belongs in the remedy, which says which plane can use it.
    """
    # THE PLAN'S OWN string, threaded in rather than rebuilt here: this note's whole claim
    # is "the artefact the plan names", and two constructions of `<name>/<version>/<file>`
    # agree by coincidence until one of them moves.
    return None if (root / artefact).is_file() else artefact


def _is_tensorrt(platform: str) -> bool:
    """Through the registry, so a repository spelling it `trt` is not told to build one.

    `BACKENDS.register_lazy("tensorrt", ..., "trt")` makes that alias a valid TensorRT
    repository, and a literal comparison sent it down the "this is not TensorRT" branch and
    withheld the remedy that works.
    """
    from shipinfer.backends.registry import BACKENDS

    return BACKENDS.canonical(platform) == "tensorrt"


def _remedies(root: Path, missing: list[tuple[ModelEntry, str]]) -> list[tuple[str, str]]:
    """One `(path, remedy)` per missing artefact — and NO build command.

    Five review rounds went into a remedy naming `scripts/build_engines.py`, and each fix
    opened the next hole; the lesson is the shape rather than the last bug. A command is only
    ever right for ONE repository and this runs against any of them, so what is said is what
    is true everywhere: which artefact is absent, which plane could use what is beside it, and
    where the repository's own instruction lives -- `<name>/<version>/README.md`, which is the
    only thing that knows how its engines are built.
    """
    out: list[tuple[str, str]] = []
    for entry, path in missing:
        onnx = sorted(p.name for p in (entry.root / str(entry.latest)).glob("*.onnx"))
        readme = f"{root / entry.name / str(entry.latest) / 'README.md'}"
        if not _is_tensorrt(entry.config.platform):
            out.append(
                (
                    path,
                    f"this repository is `platform: {entry.config.platform}` and a plan is "
                    f"read by a TensorRT-only plane, so it names an artefact nothing here "
                    f"builds",
                )
            )
        elif len(onnx) == 1:
            # The one distinction the data supports and the operator cannot see: the same
            # directory is complete for one plane and not for the other.
            out.append(
                (
                    path,
                    f"`{onnx[0]}` is beside it, which the PYTHON plane builds at load "
                    f"(`resolve_engine`) and the plan's reader does not — build the plan "
                    f"itself on the node that runs it",
                )
            )
        else:
            out.append((path, f"build it on the node that runs it; see {readme}"))
    return out
