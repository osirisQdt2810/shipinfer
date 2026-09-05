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
    _report_missing(chain, models.root, runtimes)
    return 0


def _report_missing(chain: Topology, root: Path, runtimes: dict[str, ModelRuntime]) -> None:
    """Say which artefacts the plan names and the repository does not hold.

    Reported and NOT refused: writing a plan where the engine is absent is the documented
    workflow. ADR-014 lets the control plane run on a driverless box and
    `model_repository/*/1/README.md` says engines are built on the node that runs them, so a
    fresh checkout legitimately has a `config.yaml` and no `model.plan`. What this removes is
    the SILENCE -- otherwise the operator learns at bench start-up, inside a container, while
    the machine that knew is the one that wrote the plan.
    """
    named = {node.spec.model for node in chain.nodes if node.spec.model}
    missing = sorted(
        runtime.artefact
        for name, runtime in runtimes.items()
        if name in named and not (root / runtime.artefact).is_file()
    )
    if not missing:
        return
    print(
        f"note: {len(missing)} artefact(s) this plan names are not in {root}: "
        f"{', '.join(missing)}. An engine is built on the node that runs it — "
        f"`python scripts/build_engines.py` there, inside the container.",
        file=sys.stderr,
    )
