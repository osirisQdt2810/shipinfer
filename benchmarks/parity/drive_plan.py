"""The plan seam: a chain file resolved by the Python plane, byte-compared by the C++ one.

The fourth seam after ingest, the queue and events, and the cheapest of the four: the
artefact that crosses IS the golden, so there is no trace to render and nothing to run. What
is being held to agreement is that both planes read one text identically —
`topology/plan.py` and `csrc/shipinfer/pipeline/graph/plan.cpp`.

The dims come from the committed demo repository, so a golden is reproducible on a machine
with no driver — which is where CI emits and checks it.
"""

from __future__ import annotations

from pathlib import Path

from shipinfer.repository import ModelRepository
from shipinfer.repository.resolved import model_extents, model_runtimes
from shipinfer.topology import load_topology
from shipinfer.topology.plan import plan_text, resolve_plan

ROOT = Path(__file__).resolve().parents[2]
#: Chain files, not `.scn` scripts: the input this seam takes is a chain, and inventing a
#: grammar for one would be a third format nobody reads.
SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "plans"
GOLDEN = Path(__file__).resolve().parent / "golden" / "plans"
REPOSITORY = ROOT / "model_repository"

__all__ = ["GOLDEN", "REPOSITORY", "SCENARIOS", "load_plan_scenario", "render_plan"]


def load_plan_scenario(name: str) -> Path:
    """A scenario name (or path) to the chain file it means."""
    named = Path(name)
    return named if named.suffix in (".yaml", ".yml") else SCENARIOS / f"{name}.yaml"


def render_plan(path: Path, *, repository: Path | None = None) -> str:
    """The plan text for one chain file — what the golden holds and the C++ gate re-writes.

    Resolved WITH the runtimes, so the goldens carry the `instances`/`queue_delay_us`/
    `artefact` lines and the cross-plane gate covers them. A plan resolved without them is
    the shape of the chain alone, which is what a caller wanting no repository gets.
    """
    models = ModelRepository.load(repository or REPOSITORY)
    return plan_text(
        resolve_plan(
            load_topology(path), dims=model_extents(models), runtimes=model_runtimes(models)
        )
    )
