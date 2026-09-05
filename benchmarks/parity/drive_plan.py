"""The plan seam: a chain file resolved by the Python plane, byte-compared by the C++ one.

The fourth seam after ingest, the queue and events, and the cheapest of the four: the
artefact that crosses IS the golden, so there is no trace to render and nothing to run. What
is being held to agreement is that both planes read one text identically —
`topology/plan.py` and `csrc/shipinfer/pipeline/graph/plan.cpp`.

The dims come from the committed demo repository, so a golden is reproducible on a machine
with no driver — which is where CI emits and checks it.
"""

from __future__ import annotations

from dataclasses import dataclass
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

__all__ = ["GOLDEN", "REPOSITORY", "SCENARIOS", "SETTINGS", "load_plan_scenario", "render_plan"]


@dataclass(frozen=True)
class _Pipeline:
    workers: int
    queue_capacity: int
    stage_timeout_ms: int
    reassembly: object


@dataclass(frozen=True)
class _Reassembly:
    capacity: int
    timeout_ms: int
    sweep_interval_ms: int


@dataclass(frozen=True)
class _Scheduler:
    max_queue_size: int
    enqueue_block_timeout_ms: int
    placement_policy: str
    placement_policy_options: dict


@dataclass(frozen=True)
class _Settings:
    pipeline: _Pipeline
    scheduler: _Scheduler


# doc: long the two properties a settings double for a byte-compare golden must have
#: A FIXED double, whose every value is deliberately NOT the tree's default.
#:
#: FIXED, because `ServerSettings()` reads `SHIPINFER_*`: a golden emitted from it encodes
#: whichever box emitted it and re-checks red on the next one.
#:
#: NOT-THE-DEFAULT, because that is what lets the golden fail. A plane that ignores a
#: `setting` line and substitutes its own default reproduces the file exactly when the two
#: agree. `tests/topology/test_plan.py` pins the field names against the real tree.
SETTINGS = _Settings(
    pipeline=_Pipeline(
        workers=7,
        queue_capacity=257,
        stage_timeout_ms=5001,
        reassembly=_Reassembly(capacity=1025, timeout_ms=1501, sweep_interval_ms=101),
    ),
    scheduler=_Scheduler(
        max_queue_size=65,
        enqueue_block_timeout_ms=51,
        # Not the default (`locality_spillover`), for the same reason every number here is not:
        # a plane that ignored the `policy` line and used its own would reproduce the golden.
        placement_policy="power_of_two",
        placement_policy_options={"probes": "3"},
    ),
)


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
            load_topology(path),
            dims=model_extents(models),
            runtimes=model_runtimes(models),
            settings=SETTINGS,
        )
    )
