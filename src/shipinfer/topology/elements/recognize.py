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

   **How completely it can be honoured is the gallery's decision, not this element's, and
   that is why the default is ``flat``.** ``flat`` keeps every enrolled view as its own row
   carrying its own camera, so ``exclude_camera`` drops exactly the rows that camera
   produced and nothing of them survives anywhere else. ``centroid`` folds an identity's
   views into one vector and can record only the camera of the *most recent* observation,
   so a ship enrolled from ``cam-A`` and later refreshed from ``cam-B`` stops being excluded
   from a ``cam-A`` query — the library states this plainly and is not at fault. Measured on
   this element's own fixture, ``ship-a``'s ``cam-A`` enrolment queried from ``cam-A``
   answers ``ship-b`` (0.900) under ``flat`` and ``ship-a`` (0.994) under ``centroid``: the
   self-match rule 1 exists to prevent. Choosing ``centroid`` is legitimate — its memory
   scales with the fleet rather than with dwell time — but choosing it *silently* is how a
   deployment reports near-perfect accuracy while measuring its tracker, so
   :data:`_EXACT_EXCLUSION` names the implementations whose exclusion is per-entry and
   ``open()`` warns for anything else.
2. **Unknown is ``None``, never ``0``.** ``0`` (and ``"0"``) is a legitimate gallery id, so
   a stranger recorded as ``0`` is indistinguishable from ship zero. A row that was queried
   and matched nobody is filed as ``(None, None)``; a row that was **not queried** — it
   carried no vector, or ``params: classes:`` did not select it — is absent from the mapping
   altogether, which is a third state and not a synonym for either of the first two.
3. **Enrolment is opt-in, off, and cannot evict what an operator curated.** A gallery that
   enrols whatever it does not recognise converges on one identity per crop — and the
   operator sees a growing identity count that looks like coverage. With ``enrol: false``
   (the default) an unmatched row changes nothing. With ``enrol: true`` the minted entries
   go into a **second** store, so the bound they run into evicts another minted entry and
   never a row that came off disk; see :meth:`GalleryRecognize._enrol`.
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

**What it files, and why it is a mapping.** ``meta["identities"]`` is
``{detection row index: (identity, similarity)}``, holding an entry for exactly the rows
this element queried. Keyed rather than positional because two ``recognize`` slots can sit
on two branches of one chain — a ship recogniser and, one day, something else — and the
runner's fan-in merges their metadata into one item at the rejoin. Two lists aligned by
position cannot be merged: they would have to agree on a length nobody owns, and the first
writer would win the whole key. Two mappings over *disjoint* detection rows merge by union,
which is why an unqueried row is absent rather than ``(None, None)`` — an unqueried row is
the other branch's to answer, and filing a placeholder for it would collide with the answer.
``ShipvisionTrack`` reads ``meta["vectors"]`` under the same convention (a mapping of
detection index to value), so the two elements of this phase agree about what a row index
means.

**Two gaps, both named, neither hidden.** They are the honest state of phase C at this slice:

* the vectors this element reads have to arrive **one per detection row**, and a ``pool``
  embedder files its response's raw ``{tensor_name: Tensor}`` mapping under
  ``meta["vectors"]``. Scattering a model's output rows back to the detections that produced
  them is the embed element's job and does not exist yet (slice C8), so that shape is
  refused loudly here rather than guessed at;
* the identity published here is a **``str``** — the gallery's own vocabulary (``"ship-b"``,
  ``"auto:cam-A:184102:0"``), and :mod:`shipinfer.topology.gallery_store` refuses a numeric
  identity column precisely so a label crosses the wire unchanged. The record this
  eventually lands in, ``pipeline/schema.py``'s ``ObjectRecord.ship_id``, is an
  ``int | None``. Whoever fills that record therefore has a narrowing to do, and it is named
  here rather than papered over with a cast at this end: there is no integer that means
  ``"auto:cam-A:184102:0"``, and inventing one would be a second identity vocabulary nobody
  could map back. Slice C8b owns that record and owns the choice.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError, ShipInferError, ValidationError
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

#: The gallery implementation used when the chain does not name one. ``flat`` keeps every
#: enrolled view as its own row with its own camera, which is what makes rule 1 — the
#: same-camera exclusion — *exact* rather than best-effort, and rule 1 is the reason this
#: element exists in the shape it does. It is bounded twice (``capacity`` overall and
#: ``per_identity`` per ship), so "keep every view" is still a fixed amount of memory: a
#: thousand ships at sixteen views of 512 floats is 32 MB, and the search is one gemm.
#:
#: ``centroid`` is the other shipped choice and it is not wrong — it folds an identity's
#: views into one vector, so memory depends on how many ships exist rather than on how long
#: each stays in view. What it cannot do is exclude per view: it records the camera of the
#: most recent observation only. That trade belongs to a deployment that has measured it,
#: not to a default, which is why the default moved here and why ``open()`` warns when the
#: configured implementation is not in :data:`_EXACT_EXCLUSION`.
_DEFAULT_GALLERY = "flat"

#: The gallery implementations whose ``exclude_camera`` is exact — one stored entry, one
#: camera, so excluding a camera removes every trace of what it saw. ``shipvision``'s
#: ``flat`` (registered alias ``exact``) is the only one today.
#:
#: An allow-list rather than a deny-list, and that direction is the point: an implementation
#: this element has never heard of is warned about until somebody reads its exclusion
#: semantics and adds it here. Being told about a gallery that does exclude exactly costs an
#: operator one log line; not being told about one that does not costs a re-identification
#: number that is measuring the tracker.
_EXACT_EXCLUSION = frozenset({"flat", "exact"})

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
        "per_identity",
        "classes",
        "dim",
        "threshold",
        "top_k",
        "enrol",
        "enrol_capacity",
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
                "Gallery queries made, per camera and element. One per row this element "
                "selected and was handed a vector for — not one per detection.",
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

    * ``gallery`` — which :data:`shipvision.reid.GALLERIES` implementation (``flat`` by
      default, for the exact same-camera exclusion rule 1 depends on; ``centroid`` for one
      folded vector per identity, which warns at open because its exclusion is best-effort).
    * ``capacity`` / ``per_identity`` — passed to that implementation for the **curated**
      store. Left unset, its own bound applies; both shipped galleries are bounded by
      construction, so there is no unbounded state either way. What ``capacity`` counts
      differs between them (rows for ``flat``, identities for ``centroid``), which is why
      this element does not restate a default for it, and ``per_identity`` is ``flat``'s
      second bound — how many views one ship may keep — so naming it with ``centroid``
      selected is refused at open by the gallery itself.
    * ``classes`` — the detection labels to query, by name, read off C3's ``Detections``
      (:class:`~shipinfer.topology.elements.detections.Detections`). Default: every row an
      embedder handed over. A row this list excludes is not queried and does not appear in
      ``meta["identities"]`` at all. Names and not class ids, exactly as
      :class:`~shipinfer.topology.elements.track.ShipvisionTrack` takes them, because
      ``Detections`` resolved the id table once and a second copy of it here is a second
      thing to get wrong.
    * ``dim`` — the embedder's output width, declared. The one number this element cannot
      discover for itself: the model repository knows it and a pure layer may not import
      ``repository``. Declaring it is what turns a two-model mix-up into a start-up refusal.
    * ``threshold`` — minimum cosine similarity for a match (:data:`_DEFAULT_THRESHOLD`).
      ``null`` accepts the best match unconditionally and warns.
    * ``top_k`` — how wide a ranking the gallery computes. Only the accepted match is filed.
    * ``enrol`` / ``enrol_capacity`` / ``enrol_min_confidence`` / ``enrol_prefix`` — see
      :meth:`_enrol`. ``enrol_capacity`` bounds the *second*, minted-only store and defaults
      to that implementation's own bound.
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
        self._per_identity = self._positive_int("per_identity", None)
        self._dim = self._positive_int("dim", None)
        self._top_k = self._positive_int("top_k", 1)
        self._threshold = self._optional_float("threshold", _DEFAULT_THRESHOLD)
        self._classes = self._parse_classes(self.params.get("classes"))
        self._enrol_enabled = self._flag("enrol", False)
        self._enrol_capacity = self._positive_int("enrol_capacity", None)
        self._enrol_floor = float(
            self.params.get("enrol_min_confidence", _DEFAULT_ENROL_CONFIDENCE)
        )
        self._enrol_prefix = str(self.params.get("enrol_prefix", _DEFAULT_ENROL_PREFIX))
        self._gallery_file = self.params.get("gallery_file")
        self._gallery_dir = self.params.get("gallery_dir")
        self._gallery_name = str(self.params.get("gallery_name", name))
        self._gallery_version = self._positive_int("gallery_version", None)

        #: The curated gallery — what :meth:`_load_file` read off disk. Nothing writes to
        #: it after ``open``, which is what makes it impossible for enrolment to evict it.
        self._gallery: Any = None
        #: The enrolled gallery: minted identities only, built only when ``enrol: true`` and
        #: ``None`` otherwise, so the default path holds one store and asks one query.
        self._enrolled: Any = None
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

    def _flag(self, key: str, default: bool) -> bool:
        """A boolean from ``params:``, refused at load if it is not one.

        ``bool(value)`` would accept anything, and ``enrol: "off"`` is truthy — an operator
        who wrote it would get exactly the behaviour they were trying to switch off, on a
        deployment that started cleanly. Every other value in this element is refused at
        load with a named message; this one is not an exception to that rule.
        """
        if key not in self.params:
            return default
        value = self.params[key]
        if not isinstance(value, bool):
            raise ConfigurationError(
                f"recognize element {self.name!r}: {key} must be true or false, "
                f"got {value!r}"
            )
        return value

    def _parse_classes(self, declared: Any) -> tuple[str, ...] | None:
        """``params: classes:`` as a tuple of labels, or ``None`` for "every row".

        Verbatim in shape from :meth:`~shipinfer.topology.elements.track.ShipvisionTrack.
        _parse_classes`, which reads the same ``Detections.labels`` one slice earlier: two
        elements in one package selecting detection rows two different ways is the asymmetry
        that makes a chain file's ``classes:`` mean something different depending on where it
        is written.

        ``None`` and not ``()``: an empty list in a chain file means "query nothing", which
        is a strange thing to ask for but an unambiguous one, and conflating it with an
        absent key would make a typo silently query everything.
        """
        if declared is None:
            return None
        if isinstance(declared, str) or not isinstance(declared, Sequence):
            raise ConfigurationError(
                f"recognize element {self.name!r}: `params: classes:` must be a list of "
                f"detection labels, got {type(declared).__name__}"
            )
        return tuple(str(entry) for entry in declared)

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

        self._gallery = self._build(reid, self._capacity)
        if loaded is not None:
            self._fill(loaded)
        if self._enrol_enabled:
            self._enrolled = self._build(reid, self._enrol_capacity)
        if context.metrics is not None:
            self._metrics = _RecognizeMetrics.build(context.metrics)
        self._warn_about_what_will_be_quiet(loaded)

    def _build(self, reid: Any, capacity: int | None) -> Any:
        """One gallery of the configured implementation, bounded as the chain asked.

        Called twice when ``enrol: true``: the curated store takes ``capacity:`` and the
        minted store takes ``enrol_capacity:``. Same implementation for both, because the
        two are searched with the same query and merged by score — a deployment that wants
        an exact search over its curated ships wants one over what it minted too.

        Raises:
            ConfigurationError: the implementation is unknown, or it does not accept a
                keyword this element was told to pass (``per_identity`` with ``centroid``
                selected is the case that will actually happen).
        """
        options: dict[str, Any] = {}
        if capacity is not None:
            options["capacity"] = capacity
        if self._per_identity is not None:
            options["per_identity"] = self._per_identity
        if self._dim is not None:
            options["dim"] = self._dim
        try:
            return reid.GALLERIES.build(self._gallery_impl, **options)
        except Exception as exc:
            # `GALLERIES.build` raises *shipvision's* ConfigurationError, a different class
            # from ours, and it would otherwise leave `open()` as a foreign exception type.
            # Re-raised with the names that exist, which is the half an operator who
            # misspelled `centroid` actually needs.
            raise ConfigurationError(
                f"recognize element {self.name!r}: cannot build gallery "
                f"{self._gallery_impl!r} ({exc}); available: {reid.GALLERIES.names()}"
            ) from exc

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
        self._enrolled = None
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
        **neither** — no file and no declaration — the width is not knowable at open at all,
        and the first vector to arrive fixes it inside the gallery; that is what the warning
        below says, and it is the case a deployment removes by declaring ``dim``.

        An **empty** file is still compared, and deliberately: ``np.savez`` keeps the shape,
        :func:`~shipinfer.topology.gallery_store.load_gallery_file` refuses a zero-width
        array, so a ``(0, 512)`` archive states its width as loudly as a full one does. A
        gallery that was emptied and is about to be re-enrolled from *this* chain's embedder
        is exactly the deployment where a stale width would otherwise be found on the first
        frame after somebody enrolled into it.
        """
        if self._dim is None or loaded is None:
            return
        if loaded.dim != self._dim:
            raise ConfigurationError(
                f"recognize element {self.name!r}: {loaded.path} holds {loaded.dim}-d "
                f"vectors and `dim: {self._dim}` was declared. Two models are feeding one "
                "gallery; re-enrol the gallery with the embedder this chain runs"
            )

    def _fill(self, loaded: GalleryFile) -> None:
        """Add every row of the file to the **curated** store, then say what capacity dropped.

        The only writer this store ever has. Enrolment writes to :attr:`_enrolled`, which is
        what makes "a minted identity cannot evict a curated one" a structural property
        rather than a rule someone has to keep obeying.

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
        if self._gallery_impl not in _EXACT_EXCLUSION:
            _LOG.warning(
                "recognize %r uses the %r gallery, whose `exclude_camera` this element "
                "cannot vouch for: %s excludes exactly (one entry, one camera), and a "
                "gallery that folds an identity's views into one vector can only record the "
                "camera it saw most recently, so a query may be answered by an entry that "
                "already contains its own camera's view. That is the self-match the "
                "protocol exists to prevent, and it inflates every score. Write "
                "`gallery: flat` unless this deployment has measured the trade",
                self.name,
                self._gallery_impl,
                sorted(_EXACT_EXCLUSION),
            )

    # -- the walk ------------------------------------------------------------------------

    def _do_process(self, item: ChainItem) -> ChainItem:
        """One query per selected row; file ``{row index: (identity, similarity)}``.

        Returns:
            The successor item, payload untouched, with ``meta["identities"]`` added: a
            **mapping from detection row index** holding one entry for each row this element
            queried, either ``(identity, similarity)`` or ``(None, None)``.

            Three states, and the third is the one worth reading twice. A row that matched is
            ``(identity, similarity)``. A row that was queried and matched nobody is
            ``(None, None)``. A row that was **not queried** — no vector was filed for it, or
            ``params: classes:`` did not select it — is **absent from the mapping**, because
            it is not this element's row to answer: the runner's fan-in merges the branches of
            a chain by unioning their metadata, and a placeholder here would collide with the
            answer another branch's recogniser has for the same detection.

            **No ``caps=``.** The payload is handed on unchanged, so the cap it carries is
            the cap it arrived with; see the module docstring for what stamping one here
            would relabel.

        An **empty** ``vectors`` is not an error: it is a frame that had no ships in it once
        the embedder selected its rows, and it files an empty mapping, makes no query and
        counts nothing — an unembedded row is not an "unknown", and counting it as one would
        make the unknown rate a function of how many people walked past. A **missing**
        ``vectors`` is a different thing entirely — this element was placed where nothing
        embeds — and raises, because the alternative is a chain that recognises nobody and
        reports no failure.

        Raises:
            ValidationError: ``meta["vectors"]`` is absent, or is a shape this element cannot
                align with detection rows, or ``classes:`` was configured and the item
                carries no ``detections`` to read a label from. All three are wiring, and all
                three name the fix.
            ServerStateError: called before :meth:`Element.open` — the base class's refusal.
        """
        if "vectors" not in item.meta:
            raise ValidationError(
                f"recognize element {self.name!r} found no `vectors` in the metadata for "
                f"{item.key}. It queries a gallery with the embeddings an `embed` element "
                "produced, so it must sit after one on every path that reaches it; an empty "
                "mapping would mean 'no ships in this frame', which is not the same thing"
            )
        detections = item.meta.get("detections")
        rows = self._rows(item.meta["vectors"], detections, item)
        rows = self._selected(rows, detections, item)
        camera = item.context.camera_id
        confidences = _confidences(detections)

        observe = None if self._metrics is None else self._metrics.latency_us.observe
        identities: dict[int, tuple[str | None, float | None]] = {}
        queried = 0
        matched = 0
        enrolled = 0
        try:
            for index, vector in rows:
                # `exclude_camera` on every identity query, no way to turn it off. Rule 1.
                accepted = self._best_match(vector, exclude_camera=camera, observe=observe)
                queried += 1
                if accepted is not None:
                    matched += 1
                    identities[index] = (accepted.identity, float(accepted.score))
                    continue
                minted = self._enrol(vector, index, item, confidences, observe)
                if minted is not None:
                    enrolled += 1
                # A minted identity IS this row's identity — it was just declared — and the
                # similarity behind it is `None` because it matched nothing. Rule 2 holds
                # either way: what is never filed for an unknown row is `0`.
                identities[index] = (minted, None)
        finally:
            # In a `finally` so one row raising does not discard the frame's counters. The
            # queries already made were made, and an operator reading a latency histogram
            # against a query count that silently lost a frame is reading a ratio.
            self._record(camera, queried, matched, enrolled)
        return item.derive(identities=identities)

    def _selected(
        self, rows: tuple[tuple[int, np.ndarray], ...], detections: Any, item: ChainItem
    ) -> tuple[tuple[int, np.ndarray], ...]:
        """The rows ``params: classes:`` admits — all of them when it is unset.

        The same selection :meth:`~shipinfer.topology.elements.track.ShipvisionTrack._selected`
        makes over the same ``Detections.labels``, and it is here for the same reason: an
        upstream ``when:`` guard gates the *element per item*, never the rows inside it, so
        an admitted frame arrives carrying every detection it has. Filtering here rather than
        trusting the embedder to have filtered is also what makes the row indices this
        element files disjoint from another branch's by construction.

        Args:
            rows: ``(row index, vector)`` pairs, already attributed by :meth:`_rows`.
            detections: ``meta["detections"]``, or ``None``.
            item: named in the refusal, because a chain has several elements and several
                cameras and "no detections" without a frame tag is not actionable.

        Raises:
            ValidationError: ``classes:`` is configured and the item carries no labels. A
                filter with nothing to filter on cannot silently pass everything: that would
                query every person in the frame against a ship gallery and file an identity
                for them.
        """
        if self._classes is None:
            return rows
        labels = getattr(detections, "labels", None)
        if labels is None:
            raise ValidationError(
                f"recognize element {self.name!r} has `classes: {list(self._classes)}` and "
                f"{item.key} carries no `detections` to read a row's label from. Row "
                "selection is by the detector's own labels, so a chain that filters classes "
                "here needs a detect element that decodes its boxes (`meta['detections']`) "
                "ahead of it"
            )
        wanted = self._classes
        return tuple((index, vector) for index, vector in rows if labels[index] in wanted)

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

        The count is cross-checked against ``meta["detections"]`` — C3's decoded
        :class:`~shipinfer.topology.elements.detections.Detections`, which
        :class:`~shipinfer.topology.elements.pool.PoolDetect` files today. The check is still
        written as "if the key is there, it must agree" rather than as a requirement, because
        a chain may legitimately embed without a decoding detector in front (a fixed-crop
        source, a test), and the dependency is one ``len()``. ``classes:`` is the one setting
        that turns it into a requirement, and :meth:`_selected` says so.

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

    def _best_match(
        self, vector: np.ndarray, *, exclude_camera: str | None, observe: Any
    ) -> Any:
        """The better accepted match across the stores this element holds, or ``None``.

        One store — and one query — unless ``enrol: true`` built a second one. Merging is by
        score and nothing else: both stores were asked with the same threshold and the same
        exclusion, so the two ``accepted`` values are directly comparable and the higher one
        is the answer the single-store element would have given. Only ``accepted`` is merged,
        not the ranking: this element publishes one identity per row and a merged ``top_k``
        would be a ranked list nobody reads.

        Args:
            exclude_camera: the item's own camera. Rule 1, on every store.
            observe: the histogram's bound ``observe``, or ``None``.
        """
        accepted = self._query(
            self._gallery, vector, exclude_camera=exclude_camera, observe=observe
        ).accepted
        if self._enrolled is None:
            return accepted
        minted = self._query(
            self._enrolled, vector, exclude_camera=exclude_camera, observe=observe
        ).accepted
        if accepted is None or minted is None:
            return accepted if minted is None else minted
        return minted if minted.score > accepted.score else accepted

    def _known_anywhere(self, vector: np.ndarray, observe: Any) -> bool:
        """Whether this appearance is already in **either** store, from any camera at all.

        Deliberately without the exclusion; :meth:`_enrol` argues why at length.
        """
        for store in (self._gallery, self._enrolled):
            if store is None:
                continue
            if self._query(store, vector, exclude_camera=None, observe=observe).accepted:
                return True
        return False

    def _query(
        self,
        store: Any,
        vector: np.ndarray,
        *,
        exclude_camera: str | None,
        observe: Any,
    ) -> Any:
        """One gallery query, timed. The only place this element calls the library.

        Args:
            store: which of the two galleries to ask.
            exclude_camera: the camera whose rows are dropped before ranking. Always the
                item's own for an identity query; see :meth:`_enrol` for the one other
                caller and why its answer to this is different.
            observe: the histogram's bound ``observe``, or ``None`` when the runner offered
                no metrics registry. Bound by the caller before its loop so that the
                per-row cost of "are we counting?" is one predictable branch.

        Raises:
            ValidationError: the query width disagrees with the gallery's. That is rule 4
                arriving late — two embedders feeding one gallery on a chain that declared no
                ``dim:`` — and it reaches here as ``shipvision.errors.DimensionMismatchError``,
                which is not one of ours, so the runner would charge it to "this element has
                a bug" rather than to the wiring that caused it.
        """
        started = time.perf_counter()
        try:
            result = store.query(
                vector,
                top_k=self._top_k,
                threshold=self._threshold,
                exclude_camera=exclude_camera,
            )
        except ShipInferError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"recognize element {self.name!r}: the {self._gallery_impl!r} gallery "
                f"refused a query ({exc}). A width mismatch here is two embedders feeding "
                f"one gallery; declare `dim:` on this element to have it refused at "
                f"start-up instead of on a frame"
            ) from exc
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

        **What it never does is cost the operator a curated identity.** Both shipped
        galleries evict the least recently *observed* entry, and this element never re-adds
        on a match, so a file-loaded identity's observation clock is frozen at load and it is
        the first thing a full gallery drops. Measured on the original single-store version:
        capacity 8, four curated identities, forty stranger frames from one camera — two
        seconds at 20 fps — left ``curated survivors: []``. That is ADR-005's own failure, a
        bounded shared buffer evicting the entry nobody is refreshing, reproduced inside the
        gallery. So the minted entries live in a **second** store built at ``open`` and
        queried beside the first: whatever its bound discards is something this element
        minted, and the curated store has no writer after ``_fill`` at all.

        The alternative — refusing to enrol once the single store is full — needs the
        gallery's ``capacity`` attribute, and ``capacity`` is not part of
        ``BaseGallery``: both shipped implementations happen to expose it, and reading it
        would make this element's central safety property depend on something the contract
        does not promise. Two stores need only ``add``, ``query`` and ``__len__``, which the
        ABC does declare, so the property holds for any implementation a deployment selects.

        **The floor is a detection confidence, and it comes from C3's ``Detections``**, which
        :class:`~shipinfer.topology.elements.pool.PoolDetect` files today. An item that
        carries none — a chain that embeds without a decoding detector ahead of it — has no
        per-row quality to read, and a row whose quality is unknown is **not** enrolled: the
        safe answer is the one that changes nothing. That would be silent, so the first time
        it happens the element says so, once.

        The minted identity is ``<prefix>:<camera>:<frame>:<row>`` — for example
        ``auto:cam-03:184102:2``. Three properties, each of which is a thing an operator
        needs: it is **unique across the fleet** without coordination (camera ids are, frame
        ids are per-camera monotonic, the row index separates two ships in one frame), it is
        **greppable**, so an id that a human enrolled is distinguishable at a glance from one
        the server minted at 3 a.m., and it **names its own provenance**, so the crop behind a
        bad prototype can be found in the recording.

        **What the opt-in really costs**, stated because the earlier claim of "one extra
        query" was not true: turning it on adds a second store, so *every* row costs two
        gallery queries instead of one, and an unmatched row costs two more for the
        membership check below — four rather than one. Both stores are bounded and the
        queries are gemms against them, so the cost is arithmetic rather than growth, but it
        is not free and a deployment at 1000 fps should size for it.

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
        if self._known_anywhere(vector, observe):
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
        self._enrolled.add(
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

        ``queries`` is the number of rows this element actually asked about, which is not the
        number of detections in the frame: a row nobody embedded, and a row ``classes:``
        excluded, are neither queries nor unknowns. Counting them as unknowns would make the
        unknown rate — the number an operator watches after a model change — a function of
        how many people walked past the ship, which is the one thing it must not be.

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
