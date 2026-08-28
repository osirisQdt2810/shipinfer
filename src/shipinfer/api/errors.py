"""One place where a typed failure becomes a status code.

Two routers now answer over the same vocabulary — the KServe side-door in
:mod:`shipinfer.api.routes` and the stream control plane in :mod:`shipinfer.api.streams` —
and a mapping copied into both is a mapping that drifts. The drift is not cosmetic: the
difference between 400 and 503 is whether a caller retries, so a
:class:`~shipinfer.core.errors.NoShardAvailableError` answered as 400 on one router and 503
on the other means a control plane that gives up on one path and backs off on the other for
the same condition.

FastAPI is imported *inside* the function, as everywhere else in this package: the ``server``
extra may not be installed, and this module has to be importable anyway
(``tests/api/test_optional_dependency.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shipinfer.core.errors import (
    ConfigurationError,
    ModelControlError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    QueueFullError,
    ServerStateError,
    ShipInferError,
    ValidationError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; fastapi is imported where it is used
    from fastapi import HTTPException

__all__ = ["http_error"]


def http_error(exc: ShipInferError) -> HTTPException:
    """Map a domain error onto the status code that tells the client what to do.

    The distinction matters operationally: 503 on a saturated pool is retryable and a load
    balancer will back off; 400 on a malformed tensor is not, and retrying it forever is how
    a client turns its own bug into an outage.

    Returns the exception rather than raising it, so a caller writes
    ``raise http_error(exc) from exc`` and keeps the original in ``__cause__`` — the
    traceback an operator reads is the domain failure, not this function.
    """
    from fastapi import HTTPException

    if isinstance(exc, (ModelNotFoundError, ModelVersionNotFoundError)):
        return HTTPException(404, str(exc))
    # A refused load/unload is the caller asking for something this server is not
    # configured to do. 400, never 503: it will not start working on a retry, and a
    # control-plane script that retries a 503 forever is how one bug becomes a load.
    # `ConfigurationError` alongside them: a `config.yaml` the caller asked us to load
    # and that does not parse is the caller's mistake, and it will parse no better on a
    # retry. It fell through to 500, which is what a control-plane script retries.
    # A duplicate camera id on `POST /streams` is the same shape of mistake and lands here
    # for the same reason; a fleet with no room is NOT, which is why it is a
    # `NoShardAvailableError` (a `ServerStateError`) and reaches the rule below.
    if isinstance(exc, (ValidationError, ModelControlError, ConfigurationError)):
        return HTTPException(400, str(exc))
    if isinstance(exc, (QueueFullError, ServerStateError)):
        return HTTPException(503, str(exc))
    return HTTPException(500, str(exc))
