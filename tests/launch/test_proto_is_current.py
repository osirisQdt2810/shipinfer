"""The committed stubs are what the ``.proto`` compiles to.

The stubs under ``src/shipinfer/launch/proto/`` are generated **and committed**, because
``grpcio-tools`` is a build-time dependency and a wheel that compiled its own stubs at install
time would need a protobuf compiler on every deployment host. The price of committing them is
drift: an edit to ``shard.proto`` that nobody regenerated leaves a wire contract that says one
thing and a client that sends another, and both sides keep working — against each other's old
field numbers — until a field silently arrives as its default.

So: regenerate into a temporary directory and diff. Byte-for-byte, because protoc's output is
deterministic for a given input and version, and a "close enough" comparison would not catch a
renumbered field.

``grpcio-tools`` is in the ``dev`` extra precisely so this runs in CI. If it is absent this
test skips — and the drift it guards against is then unguarded, which is the argument for
keeping it in ``dev`` rather than only in ``grpc``.
"""

from __future__ import annotations

import filecmp
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("grpc_tools", reason="grpcio-tools is not installed (the dev extra)")

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "src" / "shipinfer" / "launch" / "proto"


def _gen_proto():
    """Load ``scripts/gen_proto.py`` as a module. It is a script, not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "gen_proto", ROOT / "scripts" / "gen_proto.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_proto"] = module
    spec.loader.exec_module(module)
    return module


class TestRegeneratingIsANoOp:
    def test_the_committed_stubs_match_the_proto(self, tmp_path: Path) -> None:
        gen_proto = _gen_proto()

        fresh = gen_proto.generate(tmp_path)

        stale = [
            path.name
            for path in fresh
            if not filecmp.cmp(path, COMMITTED / path.name, shallow=False)
        ]
        assert not stale, (
            "these generated stubs no longer match shard.proto: "
            + ", ".join(stale)
            + " — run: python scripts/gen_proto.py"
        )

    def test_all_three_artefacts_are_produced_and_committed(self, tmp_path: Path) -> None:
        """The ``.pyi`` matters as much as the ``.py``: it is what mypy reads."""
        gen_proto = _gen_proto()

        names = {path.name for path in gen_proto.generate(tmp_path)}

        assert names == {"shard_pb2.py", "shard_pb2.pyi", "shard_pb2_grpc.py"}
        for name in names:
            assert (COMMITTED / name).is_file(), f"{name} is generated but not committed"

    def test_the_check_mode_agrees(self) -> None:
        """`python scripts/gen_proto.py --check` is what a human runs; it must say the same."""
        gen_proto = _gen_proto()

        assert gen_proto.main(["--check"]) == 0


class TestTheGeneratedFilesSayWhereTheyCameFrom:
    @pytest.mark.parametrize("name", ["shard_pb2.py", "shard_pb2.pyi", "shard_pb2_grpc.py"])
    def test_the_header_names_the_generator(self, name: str) -> None:
        """protoc writes "DO NOT EDIT"; it does not write *what to run instead*."""
        head = (COMMITTED / name).read_text(encoding="utf-8").splitlines()[:2]

        assert "scripts/gen_proto.py" in head[0]
        assert "python scripts/gen_proto.py" in head[1]

    def test_the_grpc_stub_imports_its_messages_by_package_path(self) -> None:
        """A bare `import shard_pb2` resolves only when the package directory is on the path.

        protoc emits the cross-import from the proto's path relative to the include root, so
        this is what says `-I src` was used rather than `-I` the package directory — a
        difference that shows up as an ImportError on a deployment host and nowhere else.
        """
        text = (COMMITTED / "shard_pb2_grpc.py").read_text(encoding="utf-8")

        assert "from shipinfer.launch.proto import shard_pb2 as" in text
        assert "\nimport shard_pb2" not in text
