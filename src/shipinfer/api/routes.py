"""KServe v2 routes."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from shipinfer.api.schemas import (
    InferenceRequestBody,
    InferenceResponseBody,
    ModelMetadata,
    RequestTag,
    ServerMetadata,
    TensorMetadata,
    tensor_from_wire,
    tensor_to_wire,
)
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
from shipinfer.core.logging import get_logger
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.engine.health import check_health
from shipinfer.engine.pool import InferenceServer

__all__ = ["build_router"]

# The logger name stays "server.api" on purpose: an operator's log filter is behaviour,
# and this move promises none changed. It is retargeted when server/ is deleted (A2 PR-6).
_LOG = get_logger("server.api")

#: Ceiling on how long one HTTP request may hold a worker. Generous enough that a cold
#: model finishing its first batch is not cut off, short enough that a wedged backend
#: cannot accumulate stuck workers until the server stops answering its own health check.
_INFER_TIMEOUT_S = 120.0


def build_router(server: InferenceServer) -> Any:
    """Build the router for one server instance.

    A factory rather than a module-level ``APIRouter`` with a global: the server is an
    argument, so two of them can be mounted in one process and a test can spin one up
    without touching module state.
    """
    from fastapi import APIRouter, HTTPException, Response

    router = APIRouter()

    def _fail(exc: ShipInferError) -> HTTPException:
        """Map a domain error onto the status code that tells the client what to do.

        The distinction matters operationally: 503 on a saturated pool is retryable and a
        load balancer will back off; 400 on a malformed tensor is not, and retrying it
        forever is how a client turns its own bug into an outage.
        """
        if isinstance(exc, (ModelNotFoundError, ModelVersionNotFoundError)):
            return HTTPException(404, str(exc))
        # A refused load/unload is the caller asking for something this server is not
        # configured to do. 400, never 503: it will not start working on a retry, and a
        # control-plane script that retries a 503 forever is how one bug becomes a load.
        # `ConfigurationError` alongside them: a `config.yaml` the caller asked us to load
        # and that does not parse is the caller's mistake, and it will parse no better on a
        # retry. It fell through to 500, which is what a control-plane script retries.
        if isinstance(exc, (ValidationError, ModelControlError, ConfigurationError)):
            return HTTPException(400, str(exc))
        if isinstance(exc, (QueueFullError, ServerStateError)):
            return HTTPException(503, str(exc))
        return HTTPException(500, str(exc))

    # -- health -------------------------------------------------------------------------

    @router.get("/v2/health/live")
    def live() -> Response:
        return Response(status_code=200 if check_health(server).live else 503)

    @router.get("/v2/health/ready")
    def ready() -> Response:
        return Response(status_code=200 if check_health(server).ready else 503)

    @router.get("/v2/health")
    def health() -> dict[str, object]:
        return check_health(server).as_dict()

    # -- metadata ------------------------------------------------------------------------

    @router.get("/v2", response_model=ServerMetadata)
    def server_metadata() -> ServerMetadata:
        from shipinfer import __version__

        return ServerMetadata(
            version=__version__,
            extensions=["health", "statistics", "model_repository", "metrics"],
        )

    @router.get("/v2/models/{name}", response_model=ModelMetadata)
    def model_metadata(name: str) -> ModelMetadata:
        try:
            entry = server.repository.entry(name)
        except ShipInferError as exc:
            raise _fail(exc) from exc
        return ModelMetadata(
            name=entry.name,
            versions=[str(v) for v in entry.versions],
            platform=entry.config.platform,
            inputs=[
                TensorMetadata(name=s.name, datatype=s.dtype, shape=list(s.shape))
                for s in entry.config.input_specs
            ],
            outputs=[
                TensorMetadata(name=s.name, datatype=s.dtype, shape=list(s.shape))
                for s in entry.config.output_specs
            ],
        )

    # -- inference -----------------------------------------------------------------------

    @router.post("/v2/models/{name}/infer", response_model=InferenceResponseBody)
    def infer(name: str, body: InferenceRequestBody) -> InferenceResponseBody:
        """Run one inference.

        The ``(camera_id, frame_id)`` tag travels in KServe's request-level ``parameters``
        and comes back the same way. Without it every HTTP request shares one fairness
        lane, the queue degenerates to FIFO, and two responses are indistinguishable —
        which is the defect ADR-005 exists to prevent, reintroduced at the ingress.
        """
        try:
            tag = RequestTag.from_parameters(body.parameters)
            inputs = {spec.name: tensor_from_wire(spec) for spec in body.inputs}
            request = InferenceRequest(
                model_name=name,
                inputs=inputs,
                parameters=body.parameters,
                context=RequestContext(
                    camera_id=tag.camera_id,
                    frame_id=tag.frame_id,
                    trace_id=body.id or "",
                ),
            )
            # Bounded, because an unbounded .result() hands a wedged backend the power to
            # hold an HTTP worker forever; enough of those and the server stops answering
            # its own health check.
            response = server.infer(request).result(timeout=_INFER_TIMEOUT_S)
        except ShipInferError as exc:
            raise _fail(exc) from exc
        except FuturesTimeout as exc:
            raise HTTPException(
                504, f"{name} did not respond within {_INFER_TIMEOUT_S:.0f}s"
            ) from exc
        except Exception as exc:  # a backend failure surfaced through the future
            _LOG.exception("inference failed for %s", name)
            raise HTTPException(500, str(exc)) from exc

        return InferenceResponseBody(
            id=body.id,
            model_name=response.model_name,
            model_version=str(response.model_version),
            outputs=[tensor_to_wire(k, v) for k, v in response.outputs.items()],
            parameters={
                "camera_id": response.context.camera_id,
                "frame_id": response.context.frame_id,
                "executed_on": str(response.executed_on),
                "queue_us": round(response.timings.queue_us, 1),
                "compute_us": round(response.timings.compute_us, 1),
            },
        )

    # -- statistics ----------------------------------------------------------------------

    @router.get("/v2/statistics")
    def statistics() -> dict[str, object]:
        return server.stats()

    @router.get("/v2/models/{name}/stats")
    def model_statistics(name: str) -> dict[str, object]:
        """Triton's per-model statistics, for one model.

        `/v2/statistics` returns the whole server, which is the wrong shape for the question
        an operator actually has at 3am: one camera is slow, its model is `person_embedder`,
        what has *that* model done. Reading it out of a fleet-wide document, or off a
        histogram that has no per-model cumulative count at all, is how that question goes
        unanswered.

        The body is Triton's ``model_stats`` array with one entry, so a Triton client parses
        it unchanged.
        """
        try:
            model = server.model(name)
        except ShipInferError as exc:
            raise _fail(exc) from exc
        return {"model_stats": [model.model_stats()]}

    @router.get("/v2/models/{name}/versions/{version}/stats")
    def model_version_statistics(name: str, version: int) -> dict[str, object]:
        """The same, addressed by version — the spelling a Triton client generates.

        A version that is not the loaded one is a 404 rather than the loaded one's numbers:
        answering with a different version's statistics under the requested version's URL is
        how a rollout gets declared healthy on the old build's data.
        """
        try:
            model = server.model(name)
            if model.version != version:
                raise ModelVersionNotFoundError(name, version, [model.version])
        except ShipInferError as exc:
            raise _fail(exc) from exc
        return {"model_stats": [model.model_stats()]}

    # -- model control --------------------------------------------------------------------
    #
    # Triton's model-repository extension, same paths and same verbs, so an existing control
    # plane works against this server unchanged. They are POST even where they read, because
    # that is what the protocol says.

    @router.post("/v2/repository/index")
    def repository_index() -> list[dict[str, str]]:
        try:
            return server.index()
        except ShipInferError as exc:
            raise _fail(exc) from exc

    @router.post("/v2/repository/models/{name}/load")
    def load_model(name: str) -> dict[str, object]:
        """Load a model into the running server.

        Synchronous, and deliberately so: the response arrives when the model is serving or
        when it has failed, so a deploy script needs no polling loop and cannot mistake
        "accepted" for "ready".
        """
        try:
            model = server.load_model(name)
        except ShipInferError as exc:
            raise _fail(exc) from exc
        except Exception as exc:  # a backend or engine failure during load
            _LOG.exception("loading model %s failed", name)
            raise HTTPException(500, str(exc)) from exc
        return {"name": model.name, "version": str(model.version), "state": "READY"}

    @router.post("/v2/repository/models/{name}/unload")
    def unload_model(name: str) -> dict[str, object]:
        try:
            server.unload_model(name)
        except ShipInferError as exc:
            raise _fail(exc) from exc
        return {"name": name, "state": "UNAVAILABLE"}

    @router.get(server.settings.http.metrics_path)
    def metrics() -> Response:
        from shipinfer.core.metrics import EXPORTERS

        exporter = EXPORTERS.create(server.settings.observability.metrics_exporter)
        return Response(
            exporter.render(server.metrics.registry), media_type=exporter.content_type
        )

    return router
