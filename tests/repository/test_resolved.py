"""What a resolved plan takes from the repository, and why each answer is the one it is.

The numbers here were restated by hand on the C++ bench's command line until P5-B, and they
disagreed with the files: `--seg-instances 3 --emb-instances 3 --ship-emb-instances 3` against
a repository saying 2, 2 and **1**, no `--det-instances` at all so the binary's own default of
2 applied, and one global `--batch-delay-us 2000` against four different windows. So the two
planes were measured at a configuration neither file described, and the head-to-head was not
like for like. These tests are what makes the one remaining source true.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shipinfer.repository import ModelRepository
from shipinfer.repository.resolved import ModelRuntime, model_extents, model_runtimes

REPOSITORY = Path(__file__).resolve().parents[2] / "model_repository"

#: One model that declares everything this module reads, written out rather than reused from
#: the demo repository: a test that moved when a production config did would be a test of that
#: file rather than of this reader.
DECLARED = """
name: {name}
platform: tensorrt
max_batch_size: 8
inputs: [{{name: images, data_type: FP32, dims: [3, 320, 192]}}]
outputs: [{{name: embedding, data_type: FP32, dims: [512]}}]
instance_groups: [{{kind: KIND_AUTO, count: 3, streams: 2}}]
dynamic_batching: {{enabled: true, max_queue_delay_us: 4000}}
parameters: {{engine_file: model.plan}}
"""


def repository(tmp_path: Path, name: str = "m", **edits: object) -> ModelRepository:
    """A one-model repository, with `edits` merged into its config."""
    config = yaml.safe_load(DECLARED.format(name=name))
    config.update(edits)
    root = tmp_path / "repo"
    (root / name / "1").mkdir(parents=True)
    (root / name / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return ModelRepository.load(root)


@pytest.fixture(scope="module")
def runtimes() -> dict[str, ModelRuntime]:
    """The demo repository's four models, read once."""
    return model_runtimes(ModelRepository.load(REPOSITORY))


class TestWhatTheDemoRepositoryResolvesTo:
    """The four shipped models, which is the table the bench used to restate."""

    def test_the_instance_counts_are_the_repositorys_own(
        self, runtimes: dict[str, ModelRuntime]
    ) -> None:
        """`ship_embedder` is the one that matters: the bench passed 3 where the file says 1,
        so its instances were triple the deployment's for every measurement."""
        counts = {name: runtime.instances for name, runtime in runtimes.items()}

        assert counts == {
            "ship_detector": 2,
            "ship_segmenter": 2,
            "ship_embedder": 1,
            "person_embedder": 2,
        }

    def test_the_batch_window_is_per_model_and_not_one_number(
        self, runtimes: dict[str, ModelRuntime]
    ) -> None:
        """The bench had one global `--batch-delay-us`; the repository has four windows."""
        windows = {name: runtime.queue_delay_us for name, runtime in runtimes.items()}

        assert windows == {
            "ship_detector": 5000,
            "ship_segmenter": 8000,
            "ship_embedder": 8000,
            "person_embedder": 3000,
        }
        assert len(set(windows.values())) > 1, "one number could not have carried these"

    def test_each_artefact_is_repository_relative(
        self, runtimes: dict[str, ModelRuntime]
    ) -> None:
        """`<name>/<version>/<engine_file>`, so the other plane joins it to a root it was
        given rather than being handed an absolute path from this machine."""
        assert runtimes["ship_detector"].artefact == "ship_detector/1/model.plan"
        assert all(
            not runtime.artefact.startswith("/") for runtime in runtimes.values()
        ), "an absolute path here would be this box's filesystem in a portable artefact"


class TestHowEachAnswerIsReached:
    def test_the_version_in_the_path_is_the_latest_one(self, tmp_path: Path) -> None:
        models = repository(tmp_path)
        (models.entry("m").root / "4").mkdir()
        reloaded = ModelRepository.load(models.entry("m").root.parent)

        assert model_runtimes(reloaded)["m"].artefact == "m/4/model.plan"

    def test_a_named_engine_file_is_carried(self, tmp_path: Path) -> None:
        models = repository(tmp_path, parameters={"engine_file": "fp16.plan"})

        assert model_runtimes(models)["m"].artefact == "m/1/fp16.plan"

    def test_batching_switched_off_is_no_window_at_all(self, tmp_path: Path) -> None:
        """`0` and not the declared delay: a window nothing waits is not a window, and
        carrying 4000 with `enabled: false` would make the other plane wait what this one
        does not."""
        models = repository(tmp_path, dynamic_batching={"enabled": False})

        assert model_runtimes(models)["m"].queue_delay_us == 0

    def test_two_device_groups_are_added_and_a_cpu_group_is_not(self, tmp_path: Path) -> None:
        """`expand`'s own arithmetic: each device group places `count` on every device it
        targets, so two of them is their sum -- while a `KIND_CPU` group places its instances
        on the host, and adding it would run a GPU instance for a rule that asked for none."""
        models = repository(
            tmp_path,
            instance_groups=[
                {"kind": "KIND_GPU", "count": 2},
                {"kind": "KIND_AUTO", "count": 3},
                {"kind": "KIND_CPU", "count": 5},
            ],
        )

        assert model_runtimes(models)["m"].instances == 5, "2 + 3, and not 10"

    def test_a_cpu_only_model_carries_no_instance_count(self, tmp_path: Path) -> None:
        """`None` and not `0`: the plan then says nothing and the consumer refuses by name,
        where a zero would read as a slot that loads its engine and runs nothing."""
        models = repository(tmp_path, instance_groups=[{"kind": "KIND_CPU", "count": 2}])

        assert model_runtimes(models)["m"].instances is None

    def test_a_model_with_no_instance_groups_runs_one(self, tmp_path: Path) -> None:
        models = repository(tmp_path, instance_groups=[])

        assert model_runtimes(models)["m"].instances == 1

    def test_the_extent_is_the_last_two_dims_of_a_chw_input(self, tmp_path: Path) -> None:
        assert model_runtimes(repository(tmp_path))["m"].extent == (320, 192)

    def test_a_dynamic_input_has_no_extent(self, tmp_path: Path) -> None:
        """Refused by name later rather than guessed at here."""
        models = repository(
            tmp_path, inputs=[{"name": "images", "data_type": "FP32", "dims": [3, -1, -1]}]
        )

        assert model_runtimes(models)["m"].extent is None
        assert "m" not in model_extents(models), "and it is absent from the extent view"
