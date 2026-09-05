"""``shipinfer plan`` — resolve a chain file into the plan the C++ data plane reads.

The composition ADR-014 describes: the Python control plane validates the chain (ADR-017's
one door) and reads the model repository, then hands the other plane a resolved
configuration. This is that hand-over, spelled as a command so it is reviewable and
diffable rather than happening invisibly inside a launcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

from shipinfer.repository import ModelEntry, ModelRepository
from shipinfer.repository.resolved import model_extents, model_runtimes
from shipinfer.topology import Topology, load_topology
from shipinfer.topology.plan import plan_text, resolve_plan

__all__ = ["plan"]


def plan(topology: Path, repository: Path, out: Path | None = None) -> int:
    """Write the resolved plan for ``topology`` to ``out``, or to stdout."""
    chain = load_topology(topology)
    models = ModelRepository.load(repository)
    text = plan_text(
        resolve_plan(chain, dims=model_extents(models), runtimes=model_runtimes(models))
    )
    if out is None:
        print(text, end="")
    else:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text.splitlines())} lines) for chain {chain.name!r}")
    _report_missing(chain, models)
    return 0


def _report_missing(chain: Topology, models: ModelRepository) -> None:
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
        for wanted in [_wanted_artefact(entry)]
        if wanted is not None
    ]
    if not missing:
        return
    lines = [f"note: {len(missing)} artefact(s) this plan names are not in {models.root}:"]
    lines += [f"  {path} — {remedy}" for path, remedy in sorted(_remedies(missing))]
    print("\n".join(lines), file=sys.stderr)


def _wanted_artefact(entry: ModelEntry) -> str | None:
    """The artefact the plan names for this model, if the repository does not hold it.

    ``engine_file`` and not what this model's own backend opens: a plan's `artefact` line is
    `engine_file` whatever `platform:` says, because the plane reading a plan is TensorRT-only.
    So a `platform: pytorch` repository genuinely lacks it -- what would be wrong is
    prescribing a TensorRT build, which is the REMEDY's job. ``None`` where the file is there,
    or where `resolve_engine` will BUILD it from the one `.onnx` `1/README.md` says to drop
    there; two of them is `_find_onnx`'s own refusal, not a missing artefact.
    """
    directory = entry.root / str(entry.latest)
    wanted = entry.config.engine_file
    if (directory / wanted).is_file():
        return None
    if entry.config.platform == "tensorrt" and len(list(directory.glob("*.onnx"))) == 1:
        return None
    return f"{entry.name}/{entry.latest}/{wanted}"


def _remedies(missing: list[tuple[ModelEntry, str]]) -> list[tuple[str, str]]:
    """One `(path, remedy)` per missing artefact, with the remedy that actually works.

    Three, because one command does not cover them: `build_engines.py` has targets that
    install into a version directory, a `reid` target that builds an engine and installs it
    nowhere, and nothing at all for a repository that is not TensorRT.
    """
    out: list[tuple[str, str]] = []
    for entry, path in missing:
        if entry.config.platform != "tensorrt":
            out.append(
                (
                    path,
                    f"this repository is `platform: {entry.config.platform}`, and a plan is "
                    f"read by a TensorRT-only plane; build a TensorRT repository for it",
                )
            )
        elif entry.name in _installed_by_the_script():
            out.append(
                (
                    path,
                    f"`python scripts/build_engines.py --only {entry.name}` on the node that "
                    f"runs it, inside the container",
                )
            )
        else:
            # The `reid` target builds `models/reid_r50_fp32.engine` and installs it into no
            # version directory, so `--only ship_embedder` exits 2. Two commands, stated,
            # because the README it would otherwise point at says only "drop it here".
            out.append(
                (
                    path,
                    "`python scripts/build_engines.py --only reid` then copy "
                    "`models/reid_r50_fp32.engine` to it — that target builds an engine and "
                    "installs it nowhere",
                )
            )
    return out


def _installed_by_the_script() -> frozenset[str]:
    """Which models `scripts/build_engines.py` writes into a version directory.

    Read from the script rather than restated: a hard-coded copy makes the note say "no target
    for these" the day a target is added, which is the class of stale instruction this whole
    diagnostic exists to remove.
    """
    from scripts.build_engines import TARGETS

    return frozenset(target.name for target in TARGETS if target.version_dir is not None)
