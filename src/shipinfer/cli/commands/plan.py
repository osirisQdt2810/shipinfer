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
from shipinfer.repository.build_targets import (
    BUILT_BY_REID_TARGET,
    INSTALLED_BY_BUILD_ENGINES,
    REID_ENGINE,
)
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
    indexed = set(models.names())
    missing = [
        (entry, wanted)
        # A model the repository does not index is skipped, not looked up: `resolve_plan` is
        # deliberately tolerant of one (a slot declaring its own extent needs no config), so
        # `entry()` here turned a written plan into a `ModelNotFoundError` AFTER the file was
        # on disk -- a diagnostic refusing, which is what this exists not to do.
        for name in sorted({n.spec.model for n in chain.nodes if n.spec.model} & indexed)
        for entry in [models.entry(name)]
        for wanted in [_wanted_artefact(models.root, runtimes[name].artefact)]
        if wanted is not None
    ]
    if not missing:
        return
    lines = [f"note: {len(missing)} artefact(s) this plan names are not in {models.root}:"]
    lines += [f"  {path} — {remedy}" for path, remedy in sorted(_remedies(missing))]
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


def _remedies(missing: list[tuple[ModelEntry, str]]) -> list[tuple[str, str]]:
    """One `(path, remedy)` per missing artefact, with the remedy that works for IT.

    Four, because one command covers none of the others: a repository that is not TensorRT, a
    directory holding the `.onnx` only the PYTHON plane builds from, a model
    `build_engines.py` installs, and the two it builds an engine for and installs nowhere.
    """
    out: list[tuple[str, str]] = []
    for entry, path in missing:
        onnx = sorted(p.name for p in (entry.root / str(entry.latest)).glob("*.onnx"))
        if not _is_tensorrt(entry.config.platform):
            out.append(
                (
                    path,
                    f"this repository is `platform: {entry.config.platform}`, and a plan is "
                    f"read by a TensorRT-only plane; build a TensorRT repository for it",
                )
            )
        elif len(onnx) == 1:
            # The Python plane builds this at load; the plane that reads a plan does not.
            # Saying so is the point: the two behave differently for one directory, and the
            # operator is about to run the one that cannot.
            out.append(
                (
                    path,
                    f"`{onnx[0]}` is beside it, which the PYTHON plane builds at load "
                    f"(`resolve_engine`) and the plan's reader does not — build the plan "
                    f"itself on the node that runs it",
                )
            )
        elif entry.name in INSTALLED_BY_BUILD_ENGINES:
            out.append(
                (
                    path,
                    f"`python scripts/build_engines.py --only {entry.name}` on the node that "
                    f"runs it, inside the container",
                )
            )
        elif entry.name in BUILT_BY_REID_TARGET:
            # That target builds one engine both of them use and installs it into no version
            # directory, so `--only ship_embedder` exits 2. Two commands, stated, because the
            # README it would otherwise point at says only "drop the built artefact here".
            out.append(
                (
                    path,
                    "`python scripts/build_engines.py --only reid` then copy "
                    f"`{REID_ENGINE}` to it — that target builds an engine and installs it "
                    "nowhere",
                )
            )
        else:
            # A model this repository's build script has never heard of, which is EVERY
            # repository but this one's demo. Naming a command here would be confidently
            # wrong -- worse than the silence this exists to remove.
            out.append(
                (
                    path,
                    "build a TensorRT engine for this model on the node that runs it, inside "
                    "the container; `scripts/build_engines.py` has no target for it",
                )
            )
    return out
