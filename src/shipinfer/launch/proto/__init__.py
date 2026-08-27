"""The generated gRPC stubs for the shard control plane (``shard.proto``), behind a guard.

**This package imports nothing at module scope.** ``shard_pb2_grpc`` imports ``grpc`` at its
first line and ``shard_pb2`` imports ``google.protobuf``; both are the optional ``grpc``
extra (``pip install "shipinfer[grpc]"``), and re-exporting either module from here would
make ``import shipinfer.launch`` - which every launcher does, on the one CPU-only process in
the deployment - fail on a host that never installed it.

What it does export is the **one door** every caller goes through:
:func:`load_pb`, :func:`load_grpc` and :func:`load_json_format` import lazily and turn the
extra being absent, or present and too old, into a
:class:`~shipinfer.core.errors.ConfigurationError` naming what to install. They exist so that
no call site has to remember the guard, and so no refusal rests on statement order - a direct
``from google.protobuf import json_format`` in a method that happens to run after a guarded
call is correct only until someone reorders two lines, and that is exactly how the offline
tier grew a ``ModuleNotFoundError`` once already.

Both halves of the control plane use them: :mod:`shipinfer.launch.client` and
:mod:`shipinfer.launch.control` on the launcher's side, :mod:`shipinfer.runners.service` on
the shard's. That is also what makes :data:`MISSING_GRPCIO` "one string" true rather than
aspirational - the sentence an operator reads is defined once, here.

The stubs are generated and committed; ``python scripts/gen_proto.py`` puts them back and
``tests/launch/test_proto_is_current.py`` fails if they drift from the ``.proto``.
"""

from __future__ import annotations

from typing import Any

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "MISSING_GRPCIO",
    "UNUSABLE_GRPCIO",
    "load_grpc",
    "load_json_format",
    "load_pb",
]

#: What to tell an operator who has not installed the extra. One string, shared by the
#: launcher's client, the shard's servicer and the tests that assert on it, so the message in
#: a refusal and the message a test looks for cannot drift.
MISSING_GRPCIO = 'the gRPC control plane needs grpcio: pip install "shipinfer[grpc]"'

#: What a lazy import of the generated stubs can fail with. ``ImportError`` is the extra
#: being absent; ``RuntimeError`` is it being present and too old - protoc's output compares
#: ``grpc.__version__`` against its own ``GRPC_GENERATED_VERSION`` at import and raises a
#: bare ``RuntimeError`` below it, and ``shard_pb2`` does the same for the protobuf runtime.
#: The floors in ``pyproject.toml`` are set so a supported install never sees the second
#: (``tests/launch/test_generated_floor.py``); catching it anyway means a floor that drifts
#: reaches an operator as a typed refusal naming the extra rather than as a raw traceback.
UNUSABLE_GRPCIO = (ImportError, RuntimeError)


def _refuse(exc: BaseException) -> ConfigurationError:
    """The one refusal, with the underlying failure kept for whoever has to diagnose it."""
    return ConfigurationError(f"{MISSING_GRPCIO} ({type(exc).__name__}: {exc})")


def load_pb() -> Any:
    """The generated messages (``shard_pb2``).

    Raises:
        ConfigurationError: protobuf is missing, or older than the committed stubs were
            generated against (:data:`UNUSABLE_GRPCIO`).
    """
    try:
        from shipinfer.launch.proto import shard_pb2
    except UNUSABLE_GRPCIO as exc:
        raise _refuse(exc) from exc
    return shard_pb2


def load_grpc() -> tuple[Any, Any]:
    """``grpc`` and the generated service stubs, as one pair.

    Together because neither is useful alone and importing them separately would give two
    call sites two chances to forget the guard.

    Returns:
        ``(grpc, shard_pb2_grpc)``.

    Raises:
        ConfigurationError: grpcio is missing or unusable (:data:`UNUSABLE_GRPCIO`).
    """
    try:
        import grpc

        from shipinfer.launch.proto import shard_pb2_grpc
    except UNUSABLE_GRPCIO as exc:
        raise _refuse(exc) from exc
    return grpc, shard_pb2_grpc


def load_json_format() -> Any:
    """``google.protobuf.json_format``, for decoding a ``Struct`` into a plain dict.

    protobuf is installed by the same extra as grpcio but is a *separate* distribution, so a
    host can have one without the other; this is guarded for its own sake rather than relying
    on some earlier call having already refused.

    Raises:
        ConfigurationError: protobuf is missing or unusable (:data:`UNUSABLE_GRPCIO`).
    """
    try:
        from google.protobuf import json_format
    except UNUSABLE_GRPCIO as exc:
        raise _refuse(exc) from exc
    return json_format
