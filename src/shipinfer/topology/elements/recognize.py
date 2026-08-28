"""The ``recognize`` element that is a **gallery query**, not a model.

Identity here is a nearest-neighbour search over the ship embedding, not a second network.
``pipeline/graph/graph.py`` is the decision of record and the reasoning is worth repeating,
because the shape of this file follows from it: ``shipvision.reid`` already carries bounded
galleries with the same-camera exclusion protocol and CMC/mAP evaluation, and running an
identity *model* would mean training one to answer a question a search over the embedder's
output already answers. Recognition against a gallery is also **stateful** — the gallery is
the state — so it belongs with tracking in the stateful plane rather than in the stateless
GPU pool.

That is why this element declares ``needs_model = False`` and ``requires_model_name =
False`` while :class:`~shipinfer.topology.elements.pool.PoolRecognize`, one registry entry
away, declares both ``True``. The two are the same *kind* wearing two shapes, and the split
those two declarations exist for (ADR-017, amended in C2) is what lets ``recognize: {impl:
shipvision}`` load with no ``model:`` at all. ``PoolRecognize`` stays registered and is not
deprecated: a deployment that really does run an identity network writes ``impl: pool,
model: ship_recognizer`` and gets it.

**The caps are copied from ``_PoolElement``, deliberately verbatim.** It accepts
``nv12@gpu`` first because the device path is the default end to end (arch.md §8), then
``tensor@gpu``, then ``bgr@cpu``; it produces ``*@*``. The wildcard is the precise claim
that this element **hands the payload on untouched** and only adds a metadata key, so its
outbound cap *is* its negotiated inbound cap and the loader resolves it as such. The
corollary, which :meth:`_do_process` obeys and a reviewer should check first: it must
**not** stamp a cap on the item it derives. A concrete ``produces`` here would relabel a
``bgr@cpu`` frame as device memory and make the download arch.md §8 exists to refuse
invisible one element further down (``elements/pool.py`` argues this at length).

**Four rules this element exists to keep, and each one is a way re-identification lies.**

1. ``exclude_camera`` is passed on **every** query, always, and it is not decoration. The
   standard protocol excludes gallery entries from the query's own camera because a match
   there measures the *tracker*, not the recogniser; a live system that re-identifies a ship
   against its own camera's last frame has learned nothing and will report near-perfect
   accuracy while doing it (``shipvision.reid.gallery.base``). There is no parameter to turn
   it off, because the only reason to want one is a better-looking number.
2. **Unknown is ``None``, never ``0``.** ``0`` is a legitimate gallery id
   (``pipeline/schema.py``), so a stranger recorded as ``0`` is indistinguishable from ship
   zero. A row with no accepted match is filed as ``(None, None)``.
3. **Enrolment is opt-in and off.** A gallery that enrols whatever it does not recognise
   converges on one identity per crop — and the operator sees a growing identity count that
   looks like coverage. With ``enrol: false`` (the default) an unmatched row changes nothing.
4. **A dim mismatch stops the deploy**, at :meth:`_do_open`, not on the first frame. Two
   models feeding one gallery is the failure ``BaseGallery.dim`` names, and its only other
   symptom is a similarity matrix that cannot be formed — or one that can, because a
   broadcast succeeded.

**Locking: none here, on purpose (the GIL law, V142).** Several pipeline workers walk this
element at once and they all share one gallery, which is exactly the case
``BaseGallery``'s contract covers: *"Implementations own their own locking. The server calls
a gallery from several worker threads, and 'the caller should lock it' is how two threads
end up appending to the same row."* ``FlatGallery.query`` takes its lock twice and briefly
and deliberately **not** across the gemm, because BLAS releases the GIL there and a lock
held over it makes eight threads perform like one. An element-level lock around ``query``
would put that back and would be the one thing this file could do to make the library slower.

**Two gaps, both named, neither hidden.** They are the honest state of phase C at this slice:

* the vectors this element reads have to arrive **one per detection row**, and a ``pool``
  embedder files its response's raw ``{tensor_name: Tensor}`` mapping under
  ``meta["vectors"]``. Scattering a model's output rows back to the detections that produced
  them is the embed element's job and does not exist yet (slice C8), so that shape is
  refused loudly here rather than guessed at;
* the row **count** comes from the vectors themselves. When C3's decoded ``Detections``
  lands under ``meta["detections"]`` this element cross-checks against it and reads the
  per-row confidence enrolment needs — both are one attribute lookup, both are optional
  today, and neither invents a number when the key is absent.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.logging import get_logger
from shipinfer.core.metrics import Counter, Histogram, MetricsRegistry
from shipinfer.topology import bridge
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.gallery_store import (
    GalleryFile,
    load_gallery_file,
    resolve_gallery_path,
)
from shipinfer.topology.registry import registry_for

__all__ = ["GalleryRecognize"]

_LOG = get_logger("topology.recognize")

#: The gallery implementation used when the chain does not name one. ``centroid`` folds every
#: view of an identity into one vector, so memory depends on how many ships exist rather than
#: on how long each one stays in view — the right default for a 24/7 server with a fixed
#: fleet of known vessels. A deployment whose views vary enough that one centroid cannot
#: represent them (a ship's bow and stern average to a vector resembling neither) writes
#: ``gallery: flat`` and gets the exact search instead.
_DEFAULT_GALLERY = "centroid"

#: Minimum cosine similarity for a match. ``shipvision``'s own documented example value, and
#: a real number rather than ``None`` on purpose: ``threshold=None`` accepts the best match
#: unconditionally, which is how a re-identification system assigns every stranger an
#: identity (``shipvision.reid.types.QueryResult``). Writing ``threshold: null`` is still
#: allowed — it is the right setting when every query is known to be someone in the gallery —
#: and it opens with a warning.
_DEFAULT_THRESHOLD = 0.55

#: How good a detection must be before ``enrol: true`` will mint an identity from it. High,
#: because the cost of the two mistakes is not symmetric: refusing to enrol a real new ship
#: loses one crop, and enrolling a blurred half-occluded one puts a bad prototype in the
#: gallery for every future query to match against.
_DEFAULT_ENROL_CONFIDENCE = 0.9

#: Prefix on a minted identity. See :meth:`GalleryRecognize._mint` for the scheme.
_DEFAULT_ENROL_PREFIX = "auto"

#: Every ``params:`` key this element reads. Anything else is a typo and is refused at load:
#: a silently ignored config key means the element ran with a default nobody chose and the
#: run looked successful, which is the argument ``shipvision.registry.Registry.build`` makes
#: for not swallowing unknown keyword arguments.
_KNOWN_PARAMS = frozenset(
    {
        "gallery",
        "capacity",
        "dim",
        "threshold",
        "top_k",
        "enrol",
        "enrol_min_confidence",
        "enrol_prefix",
        "gallery_dir",
        "gallery_name",
        "gallery_version",
        "gallery_file",
    }
)


@dataclass(slots=True)
class _RecognizeMetrics:
    """The element's metric handles, resolved once at ``open``.

    Same shape and the same reason as :class:`~shipinfer.runners.metrics.RunnerMetrics`: at
    1000 frames a second with 10-20 objects in each, a handle looked up by string per row is
    a hash and a dict probe nobody needs to pay for.

    Labelled by ``camera`` **and** ``element``: by camera because the number that matters in
    this system is never the shard total (ADR-005), and by element because a chain may hold
    two ``recognize`` slots — ships and, one day, something else — and two slots sharing a
    counter is a counter that answers neither.

    The identity to read them by: ``queries == matches + unknowns``, and ``enrolments`` is a
    subset of ``unknowns`` — an enrolled row matched nothing, which is *why* it was enrolled.

    The histogram is the one that does **not** count rows. It times *every* call into the
    gallery, including the membership check ``enrol: true`` adds on an unmatched row, because
    what an operator reads it for is "how long is the gallery taking", and a latency number
    that ignored a third of the queries would be the wrong answer to that question.
    """

    queries: Counter
    matches: Counter
    unknowns: Counter
    enrolments: Counter
    latency_us: Histogram

    @classmethod
    def build(cls, registry: MetricsRegistry) -> _RecognizeMetrics:
        return cls(
            queries=registry.counter(
                "shipinfer_recognize_queries_total",
                "Gallery queries made, per camera and element. One per embedded row.",
            ),
            matches=registry.counter(
                "shipinfer_recognize_matches_total",
                "Queries that cleared the threshold and were filed with an identity.",
            ),
            unknowns=registry.counter(
                "shipinfer_recognize_unknown_total",
                # The number an operator watches after a model change: unknowns climbing
                # while queries hold steady is an embedder whose output no longer resembles
                # the enrolled vectors, and it is invisible in `queries` alone.
                "Queries that matched nobody and were filed as `None`, per camera and "
                "element. Never filed as 0: 0 is a legitimate gallery id.",
            ),
            enrolments=registry.counter(
                "shipinfer_recognize_enrolments_total",
                "Unmatched rows added to the gallery under a minted identity, per camera "
                "and element. Zero unless `enrol: true`; a subset of `unknown_total`.",
            ),
            latency_us=registry.histogram(
                "shipinfer_recognize_query_latency_us",
                # Deliberately unlabelled by camera: one gallery serves every camera and the
                # cost of a query is the gemm's, which scales with how many entries the
                # gallery holds, not with which camera asked. A per-camera histogram would be
                # fifty copies of one distribution.
                "Wall time of one gallery query, in microseconds.",
            ),
        )


@registry_for(ElementKind.RECOGNIZE).register("shipvision")
class GalleryRecognize(Element):
    """Query a bounded ``shipvision`` gallery once per embedded row; file the answers.

    Args:
        name: the chain slot. Also the default gallery entry name — a slot called
            ``recognize`` looks for ``<gallery_dir>/recognize/<version>/gallery.npz`` — so a
            deployment whose entry has its own name says so with ``gallery_name:``.
        params: see below. Every key is validated here, at construction, which is where the
            chain loader builds the element, which is what makes a bad value stop the deploy
            instead of the first frame.
        model: accepted and ignored, as a surplus ``model:`` always has been. There is
            nothing to run: see the module docstring.

    ``params:`` keys, all optional:

    * ``gallery`` — which :data:`shipvision.reid.GALLERIES` implementation
      (``centroid`` by default, ``flat`` for an exact search over several views each).
    * ``capacity`` — passed to that implementation. Left unset, its own bound applies; both
      shipped galleries are bounded by construction, so there is no unbounded state either
      way. What ``capacity`` counts differs between them (identities for ``centroid``, rows
      for ``flat``), which is why this element does not restate a default for it.
    * ``dim`` — the embedder's output width, declared. The one number this element cannot
      discover for itself: the model repository knows it and a pure layer may not import
      ``repository``. Declaring it is what turns a two-model mix-up into a start-up refusal.
    * ``threshold`` — minimum cosine similarity for a match (:data:`_DEFAULT_THRESHOLD`).
      ``null`` accepts the best match unconditionally and warns.
    * ``top_k`` — how wide a ranking the gallery computes. Only the accepted match is filed.
    * ``enrol`` / ``enrol_min_confidence`` / ``enrol_prefix`` — see :meth:`_enrol`.
    * ``gallery_file`` — an explicit path to a ``gallery.npz``, which wins over the three
      below. For a deployment that keeps its identities somewhere that is not the model
      repository, and for a test.
    * ``gallery_dir`` / ``gallery_name`` / ``gallery_version`` — the model repository, the
      entry in it and the version directory (newest by default). See
      :mod:`shipinfer.topology.gallery_store` for the layout and why it is ADR-006's.

    With none of the four path keys the element opens on an **empty** gallery and says so
    once, at ``WARNING``. That is a legitimate state — a deployment that enrols later, a
    test — and refusing it would mean a chain could not be brought up before its identities
    existed. What it must not be is silent: an empty gallery answers ``None`` to every query,
    forever, and that reads exactly like a recogniser that is working and finding strangers.
    """

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE

    #: Verbatim from :class:`~shipinfer.topology.elements.pool._PoolElement`, and the
    #: module docstring says why each half is what it is.
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "tensor@gpu", "bgr@cpu")
    produces: ClassVar[tuple[str, ...]] = ("*@*",)

    #: Both ``False``, and this is the implementation the C2 split was built for: the chain
    #: names no ``model:`` (there is no artefact) and ``open()`` reaches into no model pool
    #: (there is nothing to submit). ``shipinfer run`` therefore builds no ``InferenceServer``
    #: for a chain of these, and the walk charges this element no expiry re-check — it is
    #: local work with no wait in it.
    requires_model_name: ClassVar[bool] = False
    needs_model: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        unknown = sorted(set(self.params) - _KNOWN_PARAMS)
        if unknown:
            raise ConfigurationError(
                f"recognize element {name!r}: unknown params {unknown} "
                f"(known: {sorted(_KNOWN_PARAMS)})"
            )
        self._gallery_impl = str(self.params.get("gallery", _DEFAULT_GALLERY))
        self._capacity = self._positive_int("capacity", None)
        self._dim = self._positive_int("dim", None)
        self._top_k = self._positive_int("top_k", 1)
        self._threshold = self._optional_float("threshold", _DEFAULT_THRESHOLD)
        self._enrol_enabled = bool(self.params.get("enrol", False))
        self._enrol_floor = float(
            self.params.get("enrol_min_confidence", _DEFAULT_ENROL_CONFIDENCE)
        )
        self._enrol_prefix = str(self.params.get("enrol_prefix", _DEFAULT_ENROL_PREFIX))
        self._gallery_file = self.params.get("gallery_file")
        self._gallery_dir = self.params.get("gallery_dir")
        self._gallery_name = str(self.params.get("gallery_name", name))
        self._gallery_version = self._positive_int("gallery_version", None)

        #: The gallery, and therefore this element's whole state, between open and close.
        self._gallery: Any = None
        #: ``shipvision.types``, for :class:`~shipvision.types.Embedding` on the enrol path.
        self._types: Any = None
        self._metrics: _RecognizeMetrics | None = None
        #: One warning, not one per frame, when enrolment is on and nothing tells this
        #: element how good a row is. See :meth:`_enrol`.
        self._warned_unconfident = False

    # -- construction-time validation ---------------------------------------------------

    def _positive_int(self, key: str, default: int | None) -> int | None:
        """A count from ``params:``, refused at load if it is not one.

        At load rather than at open because the loader builds every element to read its caps,
        so ``top_k: 0`` stops the deploy before a camera is opened. The refusal names the slot
        because a chain has several elements and a pydantic-style report that names only the
        key leaves an operator grepping for which one.
        """
        if key not in self.params:
            return default
        value = self.params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(
                f"recognize element {self.name!r}: {key} must be a positive integer, "
                f"got {value!r}"
            )
        return value

    def _optional_float(self, key: str, default: float | None) -> float | None:
        """A number or an explicit ``null`` — the two are different settings, not one."""
        if key not in self.params:
            return default
        value = self.params[key]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(
                f"recognize element {self.name!r}: {key} must be a number or null, "
                f"got {value!r}"
            )
        return float(value)

    # -- lifecycle ----------------------------------------------------------------------

    def _do_open(self, context: ElementContext) -> None:
        """Load shipvision, build the gallery, fill it from disk, and check the width.

        In that order, because each step's refusal is only useful once the one before it has
        succeeded: there is no point checking a dim against a gallery that could not be built
        on a host that has no ``shipvision`` at all.

        Raises:
            ConfigurationError: the submodule is missing (the message carries the command
                that fixes it), the gallery implementation is unknown, the file is not a
                gallery, or its width disagrees with the declared ``dim``. Every one of them
                is a deploy-stopping configuration problem, which is what ``open()`` is for.
        """
        reid = bridge.load_reid()
        self._types = bridge.load_types()

        loaded = self._load_file()
        self._check_dim(loaded)

        options: dict[str, Any] = {}
        if self._capacity is not None:
            options["capacity"] = self._capacity
        if self._dim is not None:
            options["dim"] = self._dim
        try:
            self._gallery = reid.GALLERIES.build(self._gallery_impl, **options)
        except Exception as exc:
            # `GALLERIES.build` raises *shipvision's* ConfigurationError, a different class
            # from ours, and it would otherwise leave `open()` as a foreign exception type.
            # Re-raised with the names that exist, which is the half an operator who
            # misspelled `centroid` actually needs.
            raise ConfigurationError(
                f"recognize element {self.name!r}: cannot build gallery "
                f"{self._gallery_impl!r} ({exc}); available: {reid.GALLERIES.names()}"
            ) from exc

        if loaded is not None:
            self._fill(loaded)
        if context.metrics is not None:
            self._metrics = _RecognizeMetrics.build(context.metrics)
        self._warn_about_what_will_be_quiet(loaded)

    def _do_close(self) -> None:
        """Drop the gallery. The state goes with it, and that includes enrolments.

        Nothing in this element writes back to disk: ``enrol: true`` grows the gallery *in
        memory*, and a restart re-reads the file. That is stated here and in :meth:`_enrol`
        rather than being a surprise, because an operator who turns enrolment on and expects
        the identities to survive a restart has a data-loss expectation this element does not
        meet. Persisting them is a writer with a caller — a ``shipinfer gallery`` command —
        and neither exists yet.
        """
        self._gallery = None
        self._types = None
        self._metrics = None

    # -- open-time helpers ---------------------------------------------------------------

    def _load_file(self) -> GalleryFile | None:
        """The identities on disk, or ``None`` when the chain configured no path.

        ``gallery_file:`` wins over the repository triple because it is the more specific
        statement; naming both is not an error, it is an operator being explicit about a file
        that lives outside the repository.
        """
        if self._gallery_file is not None:
            return load_gallery_file(Path(str(self._gallery_file)))
        if self._gallery_dir is not None:
            return load_gallery_file(
                resolve_gallery_path(
                    Path(str(self._gallery_dir)),
                    self._gallery_name,
                    version=self._gallery_version,
                )
            )
        return None

    def _check_dim(self, loaded: GalleryFile | None) -> None:
        """Refuse a two-model mix-up now, while it is still a configuration problem.

        There are exactly two sources for the width at open, and which one is in play is
        worth being explicit about because they are not equally strong:

        * the **declared** ``params: {dim: …}`` — the embedder's output width, which this
          element cannot discover (``topology`` may not import ``repository``);
        * the **loaded file** — authoritative about what is in the gallery and silent about
          what the embedder will send.

        With both, a disagreement is refused here and the message names both numbers. With
        one, there is nothing to compare it against and the single value simply stands. With
        **neither** — an empty gallery and no declaration — the width is not knowable at open
        at all, and the first vector to arrive fixes it inside the gallery; that is what the
        warning below says, and it is the case a deployment removes by declaring ``dim``.
        """
        if self._dim is None or loaded is None or len(loaded) == 0:
            return
        if loaded.dim != self._dim:
            raise ConfigurationError(
                f"recognize element {self.name!r}: {loaded.path} holds {loaded.dim}-d "
                f"vectors and `dim: {self._dim}` was declared. Two models are feeding one "
                "gallery; re-enrol the gallery with the embedder this chain runs"
            )

    def _fill(self, loaded: GalleryFile) -> None:
        """Add every row of the file, then say what capacity dropped.

        ``add_many`` and not a loop: it is the bulk path the gallery contract offers, and a
        thousand-row gallery loading one ``add`` at a time is a Python cost paid at every
        shard start-up.

        The dropped-identity check works for both shipped galleries without knowing which
        one it has, which is why it is phrased over *identities* rather than rows: ``flat``'s
        capacity counts rows and ``centroid``'s counts identities, so comparing ``len()``
        against the file's row count would be a false alarm on one of them every time.
        """
        embedding = self._types.Embedding
        try:
            self._gallery.add_many(
                embedding(vector=loaded.vectors[row], identity=identity, camera_id=camera)
                for row, (identity, camera) in enumerate(
                    zip(loaded.identities, loaded.camera_ids, strict=True)
                )
            )
        except Exception as exc:
            # The store validated the arrays, so anything raising here is the gallery's own
            # refusal on data that looked well-formed. Re-raised with the file's name, which
            # is the thing the operator has to go and fix.
            raise ConfigurationError(
                f"recognize element {self.name!r}: {loaded.path} was refused by the "
                f"{self._gallery_impl!r} gallery ({exc})"
            ) from exc

        wanted = len(set(loaded.identities))
        got = len(self._gallery.identities)
        if got < wanted:
            _LOG.warning(
                "recognize %r: %d of %d identities in %s did not fit the gallery's capacity "
                "and were evicted at load; raise `capacity:`",
                self.name,
                wanted - got,
                wanted,
                loaded.path,
            )

    def _warn_about_what_will_be_quiet(self, loaded: GalleryFile | None) -> None:
        """The three settings whose consequence is silence, said once, at open.

        Each of these produces a server that runs, answers, and is wrong in a way that reads
        like ordinary operation. A log line at start-up is the cheapest place to make them
        visible; none of them is refused, because every one is a legitimate state of a
        deployment that is still being brought up.
        """
        if len(self._gallery) == 0:
            source = "no gallery file was configured" if loaded is None else f"{loaded.path}"
            _LOG.warning(
                "recognize %r opened on an EMPTY gallery (%s): every query will answer "
                "unknown until something is enrolled. This looks exactly like a recogniser "
                "that is working and meeting only strangers",
                self.name,
                source,
            )
        elif self._dim is None:
            _LOG.info(
                "recognize %r: gallery width %s taken from %s; declare `dim:` to have a "
                "mismatch with the embedder refused at start-up",
                self.name,
                self._gallery.dim,
                loaded.path if loaded is not None else "the first vector",
            )
        if self._threshold is None:
            _LOG.warning(
                "recognize %r has `threshold: null`: the best match is accepted whatever its "
                "score, so every stranger is assigned the identity it happens to resemble",
                self.name,
            )
        if loaded is not None and len(loaded) and not any(loaded.camera_ids):
            _LOG.warning(
                "recognize %r: no row of %s carries a camera_id, so `exclude_camera` can "
                "exclude nothing and a ship may be matched against its own camera's "
                "enrolment — which measures the tracker, not the recogniser",
                self.name,
                loaded.path,
            )

    # -- the walk ------------------------------------------------------------------------

    def _do_process(self, item: ChainItem) -> ChainItem:
        """One query per embedded row; file ``(identity, similarity)`` aligned with the rows.

        Returns:
            The successor item, payload untouched, with ``meta["identities"]`` added: a list
            as long as the rows the item carried vectors for, each entry either
            ``(identity, similarity)`` or ``(None, None)``. A list, aligned by position,
            because that is what ``MockRecognize`` and every other metadata key in this
            vocabulary already are — a dict keyed by row index would make every consumer
            handle two shapes.

            **No ``caps=``.** The payload is handed on unchanged, so the cap it carries is
            the cap it arrived with; see the module docstring for what stamping one here
            would relabel.

        An **empty** ``vectors`` is not an error: it is a frame with no ships in it, and it
        files an empty list. A **missing** ``vectors`` is a different thing entirely — this
        element was placed where nothing embeds — and raises, because the alternative is a
        chain that recognises nobody and reports no failure.

        Raises:
            ValidationError: ``meta["vectors"]`` is absent, or is a shape this element cannot
                align with detection rows. Both are wiring, both name the fix.
            ServerStateError: called before :meth:`Element.open` — the base class's refusal.
        """
        if "vectors" not in item.meta:
            raise ValidationError(
                f"recognize element {self.name!r} found no `vectors` in the metadata for "
                f"{item.key}. It queries a gallery with the embeddings an `embed` element "
                "produced, so it must sit after one on every path that reaches it; an empty "
                "list would mean 'no ships in this frame', which is not the same thing"
            )
        rows = self._rows(item.meta["vectors"], item.meta.get("detections"), item)
        camera = item.context.camera_id
        confidences = _confidences(item.meta.get("detections"))

        observe = None if self._metrics is None else self._metrics.latency_us.observe
        identities: list[tuple[str | None, float | None]] = []
        matched = 0
        enrolled = 0
        for index, vector in rows:
            # `exclude_camera` on every identity query, with no way to turn it off. Rule 1.
            result = self._query(
                vector, exclude_camera=camera, top_k=self._top_k, observe=observe
            )
            accepted = result.accepted
            if accepted is not None:
                matched += 1
                identities.append((accepted.identity, float(accepted.score)))
                continue
            minted = self._enrol(vector, index, item, confidences, observe)
            if minted is not None:
                enrolled += 1
            # A minted identity IS this row's identity — it was just declared — and the
            # similarity behind it is `None` because it matched nothing. Rule 2 holds either
            # way: what is never filed for an unknown row is `0`.
            identities.append((minted, None))

        self._record(camera, len(rows), matched, enrolled)
        return item.derive(identities=identities)

    def _rows(
        self, vectors: Any, detections: Any, item: ChainItem
    ) -> tuple[tuple[int, np.ndarray], ...]:
        """``(row_index, vector)`` pairs, from either shape an embedder may file.

        Two shapes are accepted, and the second is not a convenience:

        * a **per-row array** — ``(N, d)`` — whose row *i* is detection *i*. The shape an
          embedder that embedded the whole frame produces.
        * an **index → vector mapping** with integer keys. The shape a *branch* embedder
          produces: ``embed_ship`` embeds the ship rows of a frame that also holds people, so
          its output has fewer rows than the frame has detections and only the original index
          says which is which. Losing that mapping is how an embedding ends up attached to
          the wrong object — a failure with no visible symptom until a tracker starts
          swapping identities (``pipeline/graph/detections.py``).

        The count is cross-checked against ``meta["detections"]`` when the item carries one.
        That key is C3's and is absent on this branch, so the check is written as "if it is
        there, it must agree" rather than as a requirement — the dependency is one ``len()``
        and it fails loudly rather than drifting.

        Raises:
            ValidationError: any other shape, and in particular the raw
                ``{tensor_name: Tensor}`` a ``pool`` embedder files straight from its
                response. That one is refused with the scatter-back gap named, because
                guessing which of its rows belongs to which detection is exactly the mistake
                above.
        """
        count = _detection_count(detections)
        if isinstance(vectors, Mapping):
            keys = list(vectors)
            if not all(
                isinstance(key, (int, np.integer)) and not isinstance(key, bool) for key in keys
            ):
                raise ValidationError(
                    f"recognize element {self.name!r} was handed `vectors` as a mapping "
                    f"keyed by {sorted(str(key) for key in keys)} for {item.key}. That is a "
                    "model response filed verbatim by a `pool` embedder, and scattering its "
                    "output rows back to the detections that produced them is the embed "
                    "element's job (phase C8). This element needs one vector per detection "
                    "row: an (N, d) array, or a mapping keyed by row index"
                )
            pairs = tuple(
                (int(key), _as_vector(vectors[key], self.name, int(key)))
                for key in sorted(keys)
            )
            if count is not None:
                for index, _ in pairs:
                    if not 0 <= index < count:
                        raise ValidationError(
                            f"recognize element {self.name!r}: `vectors` names row {index} "
                            f"and {item.key} has {count} detection(s)"
                        )
            return pairs

        if isinstance(vectors, np.ndarray):
            if vectors.ndim != 2:
                raise ValidationError(
                    f"recognize element {self.name!r} was handed a {vectors.ndim}-d "
                    f"`vectors` array for {item.key} and needs (N, d), one row per detection"
                )
            rows = tuple((index, vectors[index]) for index in range(vectors.shape[0]))
        elif isinstance(vectors, Sequence) and not isinstance(vectors, (str, bytes)):
            rows = tuple(
                (index, _as_vector(value, self.name, index))
                for index, value in enumerate(vectors)
            )
        else:
            raise ValidationError(
                f"recognize element {self.name!r} was handed `vectors` of type "
                f"{type(vectors).__name__} for {item.key}; it needs an (N, d) array, a "
                "sequence of vectors, or a mapping from row index to vector"
            )

        if count is not None and len(rows) != count:
            raise ValidationError(
                f"recognize element {self.name!r}: {len(rows)} vector(s) for {count} "
                f"detection(s) on {item.key}. A positional array must be one row per "
                "detection; an embedder that ran on a *subset* of the rows — a `when:` "
                "branch — must file a mapping from row index to vector, or the identities "
                "land on the wrong objects"
            )
        return rows

    def _query(
        self,
        vector: np.ndarray,
        *,
        exclude_camera: str | None,
        top_k: int,
        observe: Any,
    ) -> Any:
        """One gallery query, timed. The only place this element calls the library.

        Args:
            exclude_camera: the camera whose rows are dropped before ranking. Always the
                item's own for an identity query; see :meth:`_enrol` for the one other
                caller and why its answer to this is different.
            top_k: how wide a ranking to compute.
            observe: the histogram's bound ``observe``, or ``None`` when the runner offered
                no metrics registry. Bound by the caller before its loop so that the
                per-row cost of "are we counting?" is one predictable branch.
        """
        started = time.perf_counter()
        result = self._gallery.query(
            vector,
            top_k=top_k,
            threshold=self._threshold,
            exclude_camera=exclude_camera,
        )
        if observe is not None:
            observe((time.perf_counter() - started) * 1e6)
        return result

    def _enrol(
        self,
        vector: np.ndarray,
        index: int,
        item: ChainItem,
        confidences: Sequence[float] | None,
        observe: Any,
    ) -> str | None:
        """Add an unmatched row under a minted identity, if the deployment asked for it.

        Returns:
            The minted identity, or ``None`` when enrolment is off, when the row is not good
            enough, or when nothing in the item says how good it is.

        Off by default, and the default is the whole point. A gallery that enrols every
        stranger it meets grows one identity per crop of the same ship — and the operator
        watching the identity count sees it rise and reads it as coverage.

        **The floor is a detection confidence, and it comes from C3's ``Detections``.** Until
        that key exists an item carries no per-row confidence at all, and a row whose quality
        is unknown is **not** enrolled: the safe answer is the one that changes nothing. That
        would be silent, so the first time it happens the element says so, once.

        The minted identity is ``<prefix>:<camera>:<frame>:<row>`` — for example
        ``auto:cam-03:184102:2``. Three properties, each of which is a thing an operator
        needs: it is **unique across the fleet** without coordination (camera ids are, frame
        ids are per-camera monotonic, the row index separates two ships in one frame), it is
        **greppable**, so an id that a human enrolled is distinguishable at a glance from one
        the server minted at 3 a.m., and it **names its own provenance**, so the crop behind a
        bad prototype can be found in the recording.

        It costs one extra gallery query per unmatched row — the membership check below —
        and that is the whole cost of the opt-in.

        The entry lives in memory only; see :meth:`_do_close`.
        """
        if not self._enrol_enabled:
            return None
        confidence = (
            confidences[index] if confidences is not None and index < len(confidences) else None
        )
        if confidence is None:
            if not self._warned_unconfident:
                self._warned_unconfident = True
                _LOG.warning(
                    "recognize %r has `enrol: true` but the items reaching it carry no "
                    "per-row confidence (no `detections` in the metadata), so nothing will "
                    "be enrolled. Put a detect element that decodes its boxes ahead of it",
                    self.name,
                )
            return None
        if confidence < self._enrol_floor:
            return None
        if (
            self._query(vector, exclude_camera=None, top_k=1, observe=observe).accepted
            is not None
        ):
            # Already in the gallery — from this row's own camera, which is precisely why the
            # identity query above did not find it. Asking a second time WITHOUT the
            # exclusion is deliberate and is not a hole in rule 1: the two questions are
            # different. "Which identity may I publish for this row?" must exclude the row's
            # own camera, because a same-camera match measures the tracker. "Is this
            # appearance in the gallery at all?" must not, because a camera can never match
            # what it itself enrolled — so with the exclusion it would enrol the same ship
            # again on every frame it saw it, and a gallery that grows per frame is the
            # memory leak with a plausible name that `BaseGallery` warns about. The published
            # answer for this row stays `None`: the identity exists, and this camera is not
            # allowed to claim it.
            return None

        identity = self._mint(item, index)
        self._gallery.add(
            self._types.Embedding(
                vector=vector,
                identity=identity,
                camera_id=item.context.camera_id,
                frame_id=item.context.frame_id,
            )
        )
        return identity

    def _mint(self, item: ChainItem, index: int) -> str:
        """The minted identity for one row. Scheme documented in :meth:`_enrol`."""
        camera, frame = item.key
        return f"{self._enrol_prefix}:{camera}:{frame}:{index}"

    def _record(self, camera: str, queries: int, matched: int, enrolled: int) -> None:
        """Charge one frame's counters, once, after the loop.

        After rather than inside: the per-row cost of a counter is a hash and a dict probe,
        and this loop runs ten to twenty times per frame at a thousand frames a second. The
        histogram is the one thing observed per row, because a distribution summarised per
        frame is not a distribution.

        A ``None`` registry means the runner offered none, and then this element counts
        **nothing** — minting a private registry no exporter scrapes would be worse than an
        absent metric, because it reads as evidence
        (:class:`~shipinfer.topology.base.ElementContext`).
        """
        metrics = self._metrics
        if metrics is None:
            return
        labels = {"camera": camera, "element": self.name}
        metrics.queries.inc(queries, **labels)
        metrics.matches.inc(matched, **labels)
        metrics.unknowns.inc(queries - matched, **labels)
        if enrolled:
            metrics.enrolments.inc(enrolled, **labels)


def _detection_count(detections: Any) -> int | None:
    """How many rows the frame's detections hold, or ``None`` when the item carries none.

    ``len()`` and nothing else, so the only thing this element assumes about C3's
    ``Detections`` is that it is sized — which every sequence-shaped value is. A value that
    is not sized is ignored rather than refused: it is not this element's key to validate.
    """
    if detections is None:
        return None
    try:
        return len(detections)
    except TypeError:
        return None


def _confidences(detections: Any) -> Sequence[float] | None:
    """Per-row detection scores, when the item carries them, else ``None``.

    Reads ``detections.scores`` — the attribute C3's ``Detections`` declares — and is indexed
    by *row index*, the same index ``_rows`` yields. On a subset embedder that index is the
    detection's own, which is exactly the alignment the mapping shape exists to preserve, so
    the two line up without this having to know which shape the vectors arrived in.

    Anything that is not a sequence of numbers is treated as absent rather than refused: this
    is not this element's key to validate, and the consequence of getting it wrong here is
    only that enrolment declines a row.
    """
    scores = getattr(detections, "scores", None)
    if scores is None:
        return None
    try:
        return [float(score) for score in scores]
    except (TypeError, ValueError):
        return None


def _as_vector(value: Any, element: str, index: int) -> np.ndarray:
    """One row as a 1-D float32 array, or a refusal naming the row.

    ``np.asarray`` and not a copy where the caller already handed one over: the gallery
    normalises into its own buffer, so a second copy here would be per-row work on the hot
    path for nothing.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValidationError(
            f"recognize element {element!r}: vector for row {index} has shape "
            f"{array.shape} and an embedding is a non-empty (d,)"
        )
    return array
