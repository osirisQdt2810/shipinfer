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


class TestItSaysWhichArtefactsAreNotThereYet:
    """Reported, not refused -- and the distinction is the documented workflow.

    ADR-014 lets the control plane run on a box with no driver, and
    `model_repository/*/1/README.md` says engines are host-specific and built on the node that
    runs them. So a fresh checkout legitimately holds a `config.yaml` and no `model.plan`, and
    refusing would break the case the design is built for. What this removes is the SILENCE:
    without it the operator learns at bench start-up, inside a container, from a loader that
    could not open a path -- while the machine that knew was the one that wrote the plan.
    """

    @pytest.fixture()
    def bare(self, tmp_path: Path) -> Path:
        """A repository with the configs and no artefacts, which is a fresh checkout."""
        root = tmp_path / "repo"
        for name in ("ship_detector", "ship_embedder"):
            (root / name / "1").mkdir(parents=True)
            (root / name / "config.yaml").write_text(
                (REPOSITORY / name / "config.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        return root

    def test_a_missing_artefact_is_named_with_the_build_command(
        self, chain_file: Path, bare: Path, tmp_path: Path, capsys
    ) -> None:
        assert plan(chain_file, bare, tmp_path / "chain.plan") == 0
        note = capsys.readouterr().err

        assert "ship_detector/1/model.plan" in note
        assert "ship_embedder/1/model.plan" in note
        assert "build_engines.py" in note, "and how to fix it, on the node that runs it"

    def test_the_plan_is_still_written(
        self, chain_file: Path, bare: Path, tmp_path: Path
    ) -> None:
        """A note and not a refusal: the plan is the artefact this command exists to produce."""
        out = tmp_path / "chain.plan"

        assert plan(chain_file, bare, out) == 0
        assert parse_plan(out.read_text(), source=str(out)).name == "cli_chain"

    def test_nothing_is_said_when_every_artefact_is_there(
        self, chain_file: Path, bare: Path, tmp_path: Path, capsys
    ) -> None:
        """Non-vacuity: the note is about the files, not about the command."""
        for name in ("ship_detector", "ship_embedder"):
            (bare / name / "1" / "model.plan").write_bytes(b"not a real plan, but present")

        assert plan(chain_file, bare, tmp_path / "chain.plan") == 0

        assert capsys.readouterr().err == ""

    def test_a_model_the_chain_does_not_name_is_not_reported(
        self, chain_file: Path, bare: Path, tmp_path: Path, capsys
    ) -> None:
        """The repository may hold models this chain does not run, and an operator told to
        build one the plan never names would be sent to do nothing."""
        (bare / "unused" / "1").mkdir(parents=True)
        (bare / "unused" / "config.yaml").write_text(
            (REPOSITORY / "ship_segmenter" / "config.yaml")
            .read_text(encoding="utf-8")
            .replace("name: ship_segmenter", "name: unused"),
            encoding="utf-8",
        )

        assert plan(chain_file, bare, tmp_path / "chain.plan") == 0

        assert "unused" not in capsys.readouterr().err


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
