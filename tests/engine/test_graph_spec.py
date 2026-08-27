"""The CUDA-graph capture set is derived from the batcher, not stated independently.

``execution.cuda_graph_batch_sizes`` used to be the whole answer, and its default is
``[1, 2, 4, 8, 16, 32]``. A model with ``max_batch_size: 8`` therefore asked for graphs at
16 and 32 — sizes its batcher can never emit, so those captures are paid for at every
start-up and never replayed — while a model with ``preferred_batch_sizes: [6]`` got no
graph for the one size the batcher actually stops at. Neither is an error and neither is
visible: capture failure falls back to the ordinary launch path, so the only symptom is a
replay counter stuck at zero.

These tests live in ``tests/engine`` rather than ``tests/runtime`` because the derivation
only means anything as an answer the *model* gives: it reads the same batching window the
instance's queue batches against, and that pairing is the property worth protecting.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import ServerSettings
from shipinfer.engine import InferenceServer
from shipinfer.runtime.graphs import derive_graph_batch_sizes, resolve_graph_spec


def _write_model(
    root: Path,
    name: str = "embedder",
    *,
    max_batch_size: int = 8,
    preferred: Sequence[int] = (),
    parameters: dict[str, Any] | None = None,
) -> Path:
    config: dict[str, Any] = {
        "platform": "mock",
        "max_batch_size": max_batch_size,
        "inputs": [{"name": "images", "data_type": "FP32", "dims": [4]}],
        "outputs": [{"name": "embedding", "data_type": "FP32", "dims": [3]}],
        "instance_groups": [{"kind": "KIND_CPU", "count": 1}],
        "dynamic_batching": (
            {
                "enabled": True,
                "max_queue_delay_us": 0,
                "preferred_batch_sizes": list(preferred),
            }
            if max_batch_size
            else {"enabled": False}
        ),
        "parameters": {"latency_ms": 0.0, **(parameters or {})},
    }
    directory = root / name
    (directory / "1").mkdir(parents=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    return root


def _server(root: Path, **execution: Any) -> InferenceServer:
    return InferenceServer(
        ServerSettings(
            model_repository=root,
            devices={"visible_gpus": []},
            execution={"warmup_iterations": 0, **execution},
        )
    )


def _capture_set(root: Path, **execution: Any) -> dict[str, Any]:
    with _server(root, **execution) as server:
        graphs = server.model("embedder").stats()["cuda_graphs"]
    return dict(graphs)


class TestDerivedFromTheBatchingWindow:

    def test_the_capture_set_never_leaves_the_models_batching_window(
        self, tmp_path: Path
    ) -> None:
        """The regression test. A graph for batch 32 on a max_batch_size 8 model is dead
        weight: captured at start-up, replayable never."""
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=8))

        assert graphs["batch_sizes"], "a model with a batcher must capture something"
        assert max(graphs["batch_sizes"]) <= 8
        assert min(graphs["batch_sizes"]) >= 1

    def test_the_capture_set_and_the_window_are_one_answer(self, tmp_path: Path) -> None:
        """The invariant the whole change exists for: the sizes the server captured are a
        function of the window it publishes, so a config edit cannot move one without the
        other. Asserted against both facts as the server reports them, not against a
        literal, because a literal would still pass if they drifted together."""
        with _server(_write_model(tmp_path / "repo", max_batch_size=8, preferred=[4])) as s:
            stats = s.model("embedder").stats()

        window = stats["window"]
        assert stats["cuda_graphs"]["batch_sizes"] == list(
            derive_graph_batch_sizes(window["max_batch_size"], window["preferred"])
        )

    def test_preferred_sizes_are_the_capture_set(self, tmp_path: Path) -> None:
        """Triton's rule: 1, the preferred sizes, and max_batch_size. A batcher that stops
        early at 6 will hand the backend a 6, and nothing else between 1 and 8."""
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=8, preferred=[6]))

        assert graphs["batch_sizes"] == [1, 6, 8]
        assert graphs["source"] == "derived"

    def test_without_preferred_sizes_the_ladder_is_clamped_to_the_window(
        self, tmp_path: Path
    ) -> None:
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=8))

        assert graphs["batch_sizes"] == [1, 2, 3, 4, 6, 8]

    def test_a_model_that_does_not_batch_captures_only_batch_one(self, tmp_path: Path) -> None:
        """``max_batch_size: 0`` means the model owns its own batch dimension, so the only
        shape the server ever hands it is one row."""
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=0))

        assert graphs["batch_sizes"] == [1]


class TestExplicitOverrides:

    def test_a_model_can_state_its_own_graph_spec(self, tmp_path: Path) -> None:
        """Triton's ``graph_spec`` is per model, and so is this: whether a shape is worth
        capturing is a property of the engine, not of the deployment."""
        root = _write_model(
            tmp_path / "repo", max_batch_size=8, parameters={"graph_spec": [2, 4]}
        )
        graphs = _capture_set(root)

        assert graphs["batch_sizes"] == [2, 4]
        assert "parameters.graph_spec" in graphs["source"]

    def test_an_explicitly_set_settings_list_wins_over_the_derivation(
        self, tmp_path: Path
    ) -> None:
        """An operator who typed the key is running an experiment; the derivation must not
        silently overrule them."""
        root = _write_model(tmp_path / "repo", max_batch_size=8)
        graphs = _capture_set(root, cuda_graph_batch_sizes=[1, 8])

        assert graphs["batch_sizes"] == [1, 8]
        assert graphs["source"] == "settings execution.cuda_graph_batch_sizes"

    def test_the_untouched_settings_default_does_not_win(self, tmp_path: Path) -> None:
        """The other half of the same rule: a default nobody typed is this file's opinion,
        and the model's own batching window beats an opinion."""
        graphs = _capture_set(
            _write_model(tmp_path / "repo", max_batch_size=4, preferred=[2, 4])
        )

        assert graphs["batch_sizes"] == [1, 2, 4]
        assert graphs["source"] == "derived"

    def test_a_model_graph_spec_beats_the_settings_list(self, tmp_path: Path) -> None:
        root = _write_model(tmp_path / "repo", max_batch_size=8, parameters={"graph_spec": [8]})
        graphs = _capture_set(root, cuda_graph_batch_sizes=[1, 2])

        assert graphs["batch_sizes"] == [8]

    def test_the_settings_override_survives_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "Explicitly set" has to mean the same thing through `SHIPINFER_EXECUTION__*` as
        through a settings file, because that env var is how an operator on a running box
        sets it — and a rule that only worked for one of the two spellings would be a trap
        rather than an escape."""
        monkeypatch.setenv("SHIPINFER_EXECUTION__CUDA_GRAPH_BATCH_SIZES", "[1,2]")
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=8))

        assert graphs["batch_sizes"] == [1, 2]
        assert graphs["source"] == "settings execution.cuda_graph_batch_sizes"

    def test_an_override_outside_the_window_stops_the_deploy(self, tmp_path: Path) -> None:
        """Clamping would leave the config saying something the server ignores, which is
        the failure this whole change exists to remove."""
        root = _write_model(
            tmp_path / "repo", max_batch_size=8, parameters={"graph_spec": [8, 32]}
        )

        with pytest.raises(
            ConfigurationError, match=r"model .*: .*\[32\].*outside the batching window"
        ):
            _server(root).start()


class TestObservability:

    def test_stats_says_which_sizes_and_why(self, tmp_path: Path) -> None:
        """A replay counter of zero is the symptom; this is where the cause is readable."""
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=8, preferred=[4]))

        assert set(graphs) == {"enabled", "batch_sizes", "source", "reason"}
        assert "max_batch_size=8" in graphs["reason"]
        assert "[4]" in graphs["reason"]

    def test_deriving_a_capture_set_does_not_turn_capture_on(self, tmp_path: Path) -> None:
        """ADR-013: ``execution.cuda_graphs`` is off by default, and answering "which sizes
        would be captured" must not be what turns it on."""
        graphs = _capture_set(_write_model(tmp_path / "repo", max_batch_size=8))

        assert graphs["enabled"] is False


class TestDerivationArithmetic:
    """The rule itself, without a server around it — ``instance_state.cc:3683-3700``."""

    @pytest.mark.parametrize(
        ("max_batch_size", "preferred", "expected"),
        [
            (1, (), (1,)),
            (4, (), (1, 2, 3, 4)),
            (16, (), (1, 2, 3, 4, 6, 8, 12, 16)),
            (32, (), (1, 2, 3, 4, 6, 8, 12, 16, 32)),
            (8, (4, 8), (1, 4, 8)),
            (8, (8, 4, 4), (1, 4, 8)),
        ],
    )
    def test_the_ladder(
        self, max_batch_size: int, preferred: tuple[int, ...], expected: tuple[int, ...]
    ) -> None:
        assert derive_graph_batch_sizes(max_batch_size, preferred) == expected

    def test_an_empty_override_is_refused(self) -> None:
        """An empty list is ambiguous — "capture nothing" and "derive it" look identical,
        and only one of them is what anybody means."""
        with pytest.raises(ConfigurationError, match="at least one batch size"):
            resolve_graph_spec(max_batch_size=8, override=[])

    def test_a_window_with_no_batch_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            derive_graph_batch_sizes(0)
