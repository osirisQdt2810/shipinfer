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
from shipinfer.repository.build_targets import INSTALLED_BY_BUILD_ENGINES
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

    def test_each_artefact_gets_the_remedy_that_works_for_IT(
        self, chain_file: Path, bare: Path, tmp_path: Path, capsys
    ) -> None:
        """Per artefact, because one command does not cover them. `build_engines.py` installs
        `ship_detector` into its version directory; its `reid` target builds an engine and
        installs it NOWHERE, so `--only ship_embedder` exits 2 with "unknown model(s)" -- a
        container start spent to change nothing.
        """
        assert plan(chain_file, bare, tmp_path / "chain.plan") == 0
        note = capsys.readouterr().err

        assert "2 artefact(s)" in note
        assert "ship_detector/1/model.plan" in note, "the path the PLAN names, copyable"
        assert "ship_embedder/1/model.plan" in note
        assert "build_engines.py --only ship_detector" in note, "the one it installs"
        assert "--only reid" in note, "and the two-step that works for the one it does not"
        assert "build_engines.py --only ship_embedder" not in note, "which would exit 2"

    def test_the_shipped_set_and_the_script_agree(self) -> None:
        """Pinned HERE, where both are importable, and not by `src/` importing the script.

        `scripts/` is in neither the wheel (`packages.find` is `src` only) nor the runtime
        image, so `from scripts.build_engines import ...` inside `src/shipinfer/` crashes
        exactly where the note is meant to run -- and `pythonpath = [".", "src"]` hides that
        from the offline tier. The fact lives in the shipped package, the script imports it,
        and this is what keeps the two from drifting.
        """
        from scripts.build_engines import TARGETS

        assert {
            target.name for target in TARGETS if target.version_dir is not None
        } == INSTALLED_BY_BUILD_ENGINES
        assert "ship_embedder" not in INSTALLED_BY_BUILD_ENGINES, "the case that bit"

    def test_a_model_the_repository_does_not_index_is_skipped(
        self, chain_file: Path, tmp_path: Path, capsys
    ) -> None:
        """`resolve_plan` is deliberately tolerant of one -- a slot declaring its own extent
        needs no config -- so looking it up here turned a written plan into a
        `ModelNotFoundError` AFTER the file was on disk: a diagnostic refusing."""
        root = tmp_path / "partial"
        (root / "ship_detector" / "1").mkdir(parents=True)
        (root / "ship_detector" / "config.yaml").write_text(
            (REPOSITORY / "ship_detector" / "config.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        chain = tmp_path / "declared.yaml"
        chain.write_text(
            CHAIN.replace(
                "embed_ship: {impl: pool, model: ship_embedder, params: {classes: [ship]}}",
                "embed_ship: {impl: pool, model: ship_embedder, "
                "params: {classes: [ship], crop: {size: [256, 128]}}}",
            )
        )
        out = tmp_path / "chain.plan"

        assert plan(chain, root, out) == 0
        assert parse_plan(out.read_text(), source=str(out)).name == "cli_chain"
        assert "ship_embedder" not in capsys.readouterr().err

    def test_a_non_tensorrt_repository_gets_a_remedy_that_is_not_a_tensorrt_build(
        self, chain_file: Path, tmp_path: Path, capsys
    ) -> None:
        """A plan's `artefact` line is `engine_file` whatever `platform:` says, because the
        plane that reads a plan is TensorRT-only. So a `platform: pytorch` repository DOES
        lack the artefact the plan names -- saying otherwise would restore the silence. What
        would be wrong is prescribing `build_engines.py` for a chain running TorchScript.
        """
        root = tmp_path / "torch_repo"
        for name in ("ship_detector", "ship_embedder"):
            (root / name / "1").mkdir(parents=True)
            (root / name / "config.yaml").write_text(
                (REPOSITORY / name / "config.yaml")
                .read_text(encoding="utf-8")
                .replace("platform: tensorrt", "platform: pytorch"),
                encoding="utf-8",
            )
            (root / name / "1" / "model.pt").write_bytes(b"a scripted module")

        assert plan(chain_file, root, tmp_path / "chain.plan") == 0
        note = capsys.readouterr().err

        assert "platform: pytorch" in note, "which is why the plan cannot be run as written"
        assert "build_engines.py" not in note, "and not a TensorRT build for a torch chain"

    def test_an_onnx_beside_it_is_reported_with_the_difference_between_the_planes(
        self, chain_file: Path, bare: Path, tmp_path: Path, capsys
    ) -> None:
        """`resolve_engine` is a PYTHON-plane mechanism. The plan's reader is `csrc/`, which
        opens the path verbatim -- and autobuild would not write `model.plan` anyway, since
        `cache_name` produces `<stem>.<digest>.trt<v>.sm<cc>.<precision>.plan`. Staying quiet
        here was the silence coming back one layer down.
        """
        for name in ("ship_detector", "ship_embedder"):
            (bare / name / "1" / "yolo26n.onnx").write_bytes(b"an onnx graph")

        assert plan(chain_file, bare, tmp_path / "chain.plan") == 0
        note = capsys.readouterr().err

        assert "2 artefact(s)" in note, "absent for the plan's reader, so reported"
        assert "yolo26n.onnx` is beside it" in note, "and the remedy names what IS there"
        assert "the plan's reader does not" in note

    def test_two_onnx_files_do_not_get_the_one_onnx_remedy(
        self, chain_file: Path, bare: Path, tmp_path: Path, capsys
    ) -> None:
        """A remedy naming "the" ONNX would have to pick one, and `_find_onnx` refuses that
        pair without `parameters.onnx_file` -- so the artefact is reported with the build
        remedy instead of a file the loader will not accept."""
        for name in ("ship_detector", "ship_embedder"):
            for stem in ("a", "b"):
                (bare / name / "1" / f"{stem}.onnx").write_bytes(b"an onnx graph")

        assert plan(chain_file, bare, tmp_path / "chain.plan") == 0
        note = capsys.readouterr().err

        assert "2 artefact(s)" in note
        assert "is beside it" not in note, "no single ONNX to name"
        assert "build_engines.py --only ship_detector" in note

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
