"""``shipinfer plan``: the hand-over ADR-014 describes, spelled as one command.

Offline. What is asserted is the composition -- the chain is validated by the loader and the
geometry comes from the repository -- and that a chain the loader refuses produces no plan at
all, because a plan is only trustworthy if the door it came through was the validating one.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from shipinfer.cli.commands.plan import plan
from shipinfer.core.errors import ChainStructureError
from shipinfer.topology.plan import parse_plan

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT / "model_repository"

CHAIN = textwrap.dedent("""
    name: cli_chain
    elements:
      decode: {impl: replay}
      detect:
        impl: pool
        model: ship_detector
        params: {decode: {class_labels: {0: person, 8: ship}}}
      embed_ship: {impl: pool, model: ship_embedder, params: {classes: [ship]}}
      output: {impl: jsonlines}
    """)


@pytest.fixture()
def chain_file(tmp_path: Path) -> Path:
    path = tmp_path / "cli_chain.yaml"
    path.write_text(CHAIN)
    return path


class TestThePlanCommand:
    def test_it_writes_a_plan_the_reader_accepts(
        self, chain_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "chain.plan"

        assert plan(chain_file, REPOSITORY, out) == 0
        resolved = parse_plan(out.read_text(), source=str(out))

        assert resolved.name == "cli_chain"
        assert [node.slot for node in resolved.nodes] == [
            "decode",
            "detect",
            "embed_ship",
            "output",
        ]

    def test_the_geometry_comes_from_the_repository(
        self, chain_file: Path, tmp_path: Path
    ) -> None:
        """The chain names no extent, so the plan's numbers are the models' own."""
        out = tmp_path / "chain.plan"
        plan(chain_file, REPOSITORY, out)
        resolved = parse_plan(out.read_text())

        assert resolved.node("embed_ship").crop == (256, 128)
        assert resolved.node("detect").letterbox == (640, 640)

    def test_it_prints_to_stdout_with_no_out(
        self, chain_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert plan(chain_file, REPOSITORY, None) == 0
        printed = capsys.readouterr().out

        assert printed.startswith("# A RESOLVED chain")
        assert parse_plan(printed).name == "cli_chain"

    def test_a_chain_the_loader_refuses_produces_no_plan(self, tmp_path: Path) -> None:
        """`from_spec` is the one door (ADR-017); this command is not a second one."""
        broken = tmp_path / "broken.yaml"
        broken.write_text("elements:\n  decode: {impl: replay}\n")
        out = tmp_path / "broken.plan"

        with pytest.raises(ChainStructureError):
            plan(broken, REPOSITORY, out)
        assert not out.exists(), "a refused chain must not leave a half-written plan"
