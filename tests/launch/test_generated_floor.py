"""The dependency floors in ``pyproject.toml`` are the ones the committed stubs enforce.

Both generated modules under ``src/shipinfer/launch/proto/`` check their runtime **at import
time** and refuse to load below a version baked in when they were generated:

* ``shard_pb2_grpc.py`` compares ``grpc.__version__`` against ``GRPC_GENERATED_VERSION`` and
  raises a bare ``RuntimeError``;
* ``shard_pb2.py`` calls ``ValidateProtobufRuntimeVersion`` with the gencode version.

So the floors declared for the ``grpc`` and ``dev`` extras are not style: a floor below the
baked-in number makes ``pip install "shipinfer[grpc]"`` resolve a *supported* combination that
cannot import. That was real - the stubs said 1.71.2 while ``pyproject`` said ``grpcio>=1.60``
- and it is exactly the kind of drift a regeneration on a newer machine reintroduces silently,
because nothing else in the suite reads either number.

This test reads both numbers out of the committed stubs rather than restating them, so the
next regeneration either keeps the floors honest or fails here with the two versions named.
``pyproject.toml`` is parsed with a TOML reader, not grepped: a floor that moved into a
different extra should be found, not silently pass because the string is still somewhere in
the file.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

try:  # `tomllib` is the standard library from 3.11; `tomli` is the same parser before it.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - taken on 3.10, which CI also builds
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = ROOT / "src" / "shipinfer" / "launch" / "proto"

#: The line protoc writes into the grpc stub: ``GRPC_GENERATED_VERSION = '1.71.2'``.
_GRPC_GENERATED = re.compile(r"^GRPC_GENERATED_VERSION = '([^']+)'", re.MULTILINE)
#: protobuf's gencode version, as the positional arguments of the validation call the pb2
#: module makes at import: ``ValidateProtobufRuntimeVersion(Domain.PUBLIC, 5, 29, 0, ...)``.
_PROTOBUF_GENCODE = re.compile(
    r"ValidateProtobufRuntimeVersion\(\s*[\w.]+,\s*(\d+),\s*(\d+),\s*(\d+),", re.DOTALL
)
#: ``grpcio>=1.71.2,<2`` -> ``("grpcio", "1.71.2")``. ``==`` counts as a floor too: a pin is
#: the tightest floor there is, and ``grpcio-tools`` is pinned so that the byte-for-byte
#: regeneration guard compares against one protoc and not whichever one resolved today.
_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)\s*[=>]=\s*([0-9][0-9A-Za-z.]*)")

#: Which distribution's floor is set by which generated file. ``grpcio-tools`` is what
#: *produces* the stub, so a floor below the stub's own version means a regeneration on a
#: developer's machine cannot reproduce what is committed.
FLOOR_SOURCE = {"grpcio": "grpc", "grpcio-tools": "grpc", "protobuf": "protobuf"}


def version_key(text: str) -> tuple[int, int, int, int]:
    """``"1.71.2"`` -> ``(1, 71, 2, 0)``.

    Numeric, so ``1.9`` does not sort above ``1.71``, and zero-padded to a fixed width, so
    ``5.29`` and ``5.29.0`` are the same floor rather than the shorter one comparing lower.
    """
    parts = [int(part) for part in re.findall(r"\d+", text)][:4]
    return tuple(parts + [0] * (4 - len(parts)))  # type: ignore[return-value]


def generated_floors() -> dict[str, str]:
    """What the committed stubs demand of their runtimes, as the stubs themselves say it."""
    grpc_match = _GRPC_GENERATED.search(
        (PROTO_DIR / "shard_pb2_grpc.py").read_text(encoding="utf-8")
    )
    assert grpc_match is not None, "shard_pb2_grpc.py no longer declares GRPC_GENERATED_VERSION"
    pb_match = _PROTOBUF_GENCODE.search(
        (PROTO_DIR / "shard_pb2.py").read_text(encoding="utf-8")
    )
    assert pb_match is not None, "shard_pb2.py no longer validates a protobuf runtime version"
    return {"grpc": grpc_match.group(1), "protobuf": ".".join(pb_match.groups())}


def declared_specs() -> dict[str, list[str]]:
    """Every optional dependency's requirement string, verbatim, keyed by extra."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return dict(pyproject["project"]["optional-dependencies"])


def declared_floors() -> dict[str, dict[str, str]]:
    """The ``>=``/``==`` floors of every optional dependency, keyed by extra then dist."""
    extras = declared_specs()
    return {
        extra: {
            match.group(1).lower(): match.group(2)
            for match in (_REQUIREMENT.match(spec) for spec in specs)
            if match is not None
        }
        for extra, specs in extras.items()
    }


class TestTheDeclaredFloorsMatchTheCommittedStubs:
    def test_every_extra_that_ships_the_stubs_declares_a_floor_they_can_import_at(
        self,
    ) -> None:
        """A floor below the baked-in version is an install that resolves and then raises.

        ``grpc`` is what a deployment installs and ``dev`` is what CI installs; both pull the
        generated modules in, so both have to clear the same bar.
        """
        generated = generated_floors()
        declared = declared_floors()

        for extra in ("grpc", "dev"):
            for distribution, source in FLOOR_SOURCE.items():
                floor = declared[extra].get(distribution)
                if floor is None:
                    continue
                needed = generated[source]
                assert version_key(floor) >= version_key(needed), (
                    f"the {extra} extra declares {distribution}>={floor} but the committed "
                    f"stubs need {needed} - raise the floor in pyproject.toml, or regenerate "
                    "the stubs against the older one"
                )

    def test_both_extras_name_grpcio_and_protobuf_at_all(self) -> None:
        """The loop above skips what is absent; this is what says nothing may be absent."""
        declared = declared_floors()

        assert {"grpcio", "protobuf"} <= set(declared["grpc"])
        assert {"grpcio", "grpcio-tools", "protobuf"} <= set(declared["dev"])

    def test_the_two_extras_agree_with_each_other(self) -> None:
        """CI installs ``dev`` and a deployment installs ``grpc``; a split floor means the
        combination CI proves is not the combination that ships."""
        declared = declared_floors()

        for distribution in ("grpcio", "protobuf"):
            assert version_key(declared["dev"][distribution]) == version_key(
                declared["grpc"][distribution]
            )

    def test_grpcio_tools_can_reproduce_the_committed_stub(self) -> None:
        """The generator's floor is the stub's own version: below it, `gen_proto.py` writes a
        different GRPC_GENERATED_VERSION and `test_proto_is_current.py` fails for a reason
        that has nothing to do with the .proto."""
        declared = declared_floors()

        assert version_key(declared["dev"]["grpcio-tools"]) >= version_key(
            generated_floors()["grpc"]
        )

    def test_grpcio_tools_is_pinned_exactly_so_the_byte_compare_means_something(self) -> None:
        """`gen_proto.py --check` regenerates and compares BYTES, so it is only a guard while
        every machine runs one protoc: grpcio-tools 1.83 emits a different `shard_pb2_grpc.py`
        for the same .proto and would fail the check on a tree nobody touched. Today the pin
        also happens to be what resolves anyway - `protobuf<6` drags grpcio-tools back - and
        an accident is not a guarantee. This is what makes it one."""
        pins = [spec for spec in declared_specs()["dev"] if spec.startswith("grpcio-tools")]

        assert pins == [f"grpcio-tools=={generated_floors()['grpc']}"], (
            "grpcio-tools must be pinned to the version the committed stubs were generated "
            f"with; found {pins}"
        )


class TestAFloorThatDriftsAnywayIsStillTyped:
    """Belt and braces: if the declaration above is ever wrong, the refusal is still typed.

    protoc's output raises a bare ``RuntimeError`` when the installed grpcio is older than
    the stub was generated against, and a ``RuntimeError`` out of a lazy import is not what
    either half of the control plane promises its caller - both promise a
    ``ConfigurationError`` naming the extra to install, the same way ``api/app.py`` does for
    FastAPI. So both catch it, and this is what says so.

    The grpcio installed here is stood in for by a module that has a version and nothing
    else, which is what makes the *real* generated guard run and raise: no old grpcio has to
    be installed to test the case it produces.
    """

    @pytest.fixture()
    def grpcio_too_old(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("google.protobuf", reason="the grpc extra is not installed")
        stand_in = types.ModuleType("grpc")
        stand_in.__version__ = "1.60.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "grpc", stand_in)
        # Dropped so the guard at the top of the stub runs again on the next import; the
        # monkeypatch puts the real module back afterwards.
        monkeypatch.delitem(sys.modules, "shipinfer.launch.proto.shard_pb2_grpc", raising=False)

    def test_the_stub_does_raise_a_runtime_error_below_its_floor(
        self, grpcio_too_old: None
    ) -> None:
        """The premise of the two tests below, asserted rather than assumed."""
        with pytest.raises(RuntimeError, match=r"GRPC_GENERATED_VERSION|depends on"):
            import shipinfer.launch.proto.shard_pb2_grpc  # noqa: F401

    def test_the_client_names_the_extra_rather_than_raising_it(
        self, grpcio_too_old: None
    ) -> None:
        from shipinfer.core.errors import ConfigurationError
        from shipinfer.launch import ShardClient

        with pytest.raises(ConfigurationError) as excinfo:
            ShardClient(50199).health()

        assert 'pip install "shipinfer[grpc]"' in str(excinfo.value)
        assert "RuntimeError" in str(excinfo.value)

    def test_serve_shard_names_the_extra_rather_than_raising_it(
        self, grpcio_too_old: None
    ) -> None:
        """And it refuses *before* binding anything, so nothing has to be unwound."""
        from shipinfer.core.errors import ConfigurationError
        from shipinfer.runners.service import serve_shard

        with pytest.raises(ConfigurationError) as excinfo:
            serve_shard(runner=None, shard_id=1, control_port=0)  # type: ignore[arg-type]

        assert 'pip install "shipinfer[grpc]"' in str(excinfo.value)
        assert "RuntimeError" in str(excinfo.value)
