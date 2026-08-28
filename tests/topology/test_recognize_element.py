"""The ``recognize`` element that queries a gallery: what it files, and what it refuses.

Green **with or without** ``3rdparty/shipvision``, which is the shape ``test_bridge.py``
established and the reason it is worth repeating here: CI does not check the submodule out
(``.claude/CLAUDE.md``), so the tests that need it present ask for it with
:func:`pytest.importorskip`, and the one that needs it *absent* arranges the absence rather
than assuming it. The properties that need no gallery at all — the caps, the two model
declarations, every refusal in :mod:`shipinfer.topology.gallery_store` — run either way,
which is most of this file.

Four of these tests are the ones the element exists for, and each was checked by reverting
the line it covers (the numbers are in the slice report):

* ``exclude_camera`` is passed on every query. Dropped, the cross-camera test answers
  ``ship-a`` — the identity's own camera's enrolment, an exact match, and a score that would
  make the recogniser look perfect while measuring the tracker.
* an unknown row is ``None``. Filed as ``0`` instead, the assertion that it is not an int
  fails — and in production it would be indistinguishable from gallery id 0.
* the gallery file is validated. With the checks removed, a NaN row loads clean and every
  subsequent score is NaN.
* a dim mismatch stops the deploy at ``open()``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.metrics import MetricsRegistry
from shipinfer.core.request import RequestContext
from shipinfer.topology import ChainSpec, Topology, bridge, gallery_store
from shipinfer.topology.base import ChainItem, ElementContext, ElementKind
from shipinfer.topology.caps import parse_caps
from shipinfer.topology.elements.pool import PoolRecognize
from shipinfer.topology.elements.recognize import GalleryRecognize
from shipinfer.topology.registry import create_element, registry_for

BGR = parse_caps(("bgr@cpu",))[0]

#: A four-wide embedding space, small enough to write the vectors down and reason about the
#: cosine similarities by eye.
DIM = 4

#: The probe every cross-camera test sends: ``ship-a``'s own appearance, from ``ship-a``'s
#: own camera.
PROBE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

#: ``cos(PROBE, NEARLY) == 0.9``, so the answer a *correctly excluded* query gives is a
#: number the test can name rather than "something less than 1".
NEARLY = np.array([0.9, np.sqrt(1.0 - 0.81), 0.0, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def clear_bridge_caches() -> Iterator[None]:
    """No test inherits another's memoised shipvision module, in either direction."""
    bridge.load_reid.cache_clear()
    bridge.load_types.cache_clear()
    yield
    bridge.load_reid.cache_clear()
    bridge.load_types.cache_clear()


def write_gallery(
    root: Path,
    name: str = "recognize",
    *,
    vectors: np.ndarray,
    identities: Sequence[str],
    cameras: Sequence[str] | None = None,
    version: int = 1,
) -> Path:
    """Write one ``<root>/<name>/<version>/gallery.npz``, the layout ADR-006 implies.

    A fixture that writes the *real* file rather than monkeypatching the loader: the format
    is a contract with whatever enrols the identities offline, and a test that stubbed the
    read would pin nothing about it.

    The default entry name is ``recognize`` because the element's default is its own slot
    name, so a test that does not care writes nothing about it and the default is exercised.
    """
    directory = root / name / str(version)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / gallery_store.GALLERY_FILE
    payload: dict[str, Any] = {
        "vectors": np.asarray(vectors),
        "identities": np.array(list(identities), dtype="<U24"),
    }
    if cameras is not None:
        payload["camera_ids"] = np.array(list(cameras), dtype="<U24")
    np.savez(path, **payload)
    return path


def three_identities_two_cameras(root: Path) -> Path:
    """Three ships, each enrolled from ``cam-A`` and from ``cam-B``. Six rows.

    Laid out so that one probe separates a working ``exclude_camera`` from a broken one with
    no ambiguity at all:

    ==========  =======  ================================  ==================
    identity    camera   vector                            cos with ``PROBE``
    ==========  =======  ================================  ==================
    ship-a      cam-A    ``PROBE``                         1.0
    ship-a      cam-B    ``[0, 0, 1, 0]``                  0.0
    ship-b      cam-A    ``[0, 1, 0, 0]``                  0.0
    ship-b      cam-B    ``NEARLY``                        0.9
    ship-c      cam-A    ``[0, 0, 0, 1]``                  0.0
    ship-c      cam-B    ``[0, 0, -1, 0]``                 0.0
    ==========  =======  ================================  ==================

    A query of ``PROBE`` from ``cam-A`` must answer ``ship-b`` at 0.9: its own camera's rows
    are excluded, including the exact match. Without the exclusion it answers ``ship-a`` at
    1.0 — the self-match that measures the tracker and inflates every score.
    """
    vectors = np.array(
        [
            PROBE,
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            NEARLY,
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return write_gallery(
        root,
        vectors=vectors,
        identities=["ship-a", "ship-a", "ship-b", "ship-b", "ship-c", "ship-c"],
        cameras=["cam-A", "cam-B", "cam-A", "cam-B", "cam-A", "cam-B"],
    )


def build(name: str = "recognize", **params: Any) -> GalleryRecognize:
    """The element, built the way the chain loader builds it."""
    element = create_element(ElementKind.RECOGNIZE, "shipvision", name, params)
    assert isinstance(element, GalleryRecognize)
    return element


def item(
    vectors: Any,
    *,
    camera: str = "cam-A",
    frame: int = 7,
    payload: object = "frame-handle",
    **meta: Any,
) -> ChainItem:
    """One chain item carrying an embedder's output, and a payload to hand on untouched."""
    return ChainItem(
        context=RequestContext(camera_id=camera, frame_id=frame),
        caps=BGR,
        payload=payload,
        meta={"vectors": vectors, **meta},
    )


@dataclass
class FakeDetections:
    """The two members of C3's ``Detections`` this element reads: ``len`` and ``scores``.

    Two, and that is itself the evidence the dependency is as small as the plan asked for. It
    is a stand-in and not the real class because C3 has not landed on this branch; when it
    does, this double keeps working, because these are the attributes it declares.
    """

    scores: Sequence[float]

    def __len__(self) -> int:
        return len(self.scores)


class TestTheDeclarations:
    """What the loader and the runner read off the class, before anything is opened."""

    def test_it_is_registered_under_the_recognize_kind_as_shipvision(self) -> None:
        registry = registry_for(ElementKind.RECOGNIZE)

        assert registry.get("shipvision") is GalleryRecognize
        assert GalleryRecognize.impl == "shipvision"
        assert GalleryRecognize.kind is ElementKind.RECOGNIZE

    def test_the_caps_are_the_pool_element_s_verbatim(self) -> None:
        """Copied, not re-derived — and the wildcard is the load-bearing half.

        ``produces = ("*@*",)`` is the claim that the payload is handed on unchanged, which
        is what makes a ``bgr@cpu`` chain and an ``nv12@gpu`` chain both legal through this
        element and a device-to-host download still refusable behind it. A concrete cap here
        would relabel the frame.
        """
        assert GalleryRecognize.accepts == ("nv12@gpu", "tensor@gpu", "bgr@cpu")
        assert GalleryRecognize.produces == ("*@*",)
        assert build().is_sink is False

    def test_it_needs_neither_a_model_name_nor_a_model_pool(self) -> None:
        """The divergence C2's split was built for, on the element that ships it.

        Both ``False``: there is no artefact to name and nothing to submit. ``PoolRecognize``
        one registry entry away answers both ``True``, which is why one attribute could not
        have covered the kind.
        """
        assert GalleryRecognize.requires_model_name is False
        assert GalleryRecognize.needs_model is False
        assert PoolRecognize.requires_model_name is True
        assert PoolRecognize.needs_model is True

    def test_a_chain_may_name_it_with_no_model_while_the_pool_impl_may_not(self) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml("""
                name: gallery
                elements:
                  decode:    {impl: mock}
                  recognize: {impl: shipvision}
                  output:    {impl: mock}
                """))

        assert isinstance(chain.node("recognize").element, GalleryRecognize)
        assert chain.node("recognize").element.model is None

        with pytest.raises(ConfigurationError, match="must name a `model:"):
            Topology.from_spec(ChainSpec.from_yaml("""
                    elements:
                      decode:    {impl: mock}
                      recognize: {impl: pool}
                      output:    {impl: mock}
                    """))

    def test_a_typo_in_params_is_refused_at_load_not_ignored(self) -> None:
        """A silently dropped key means the element ran with a default nobody chose.

        The whole file is about not answering plausibly, and ``treshold: 0.9`` accepted and
        ignored is the same failure one level up: the deployment believes it set a threshold.
        """
        with pytest.raises(ConfigurationError, match="unknown params"):
            build(treshold=0.9)

    @pytest.mark.parametrize(("key", "value"), [("top_k", 0), ("capacity", -1), ("dim", 1.5)])
    def test_a_count_that_is_not_one_is_refused_where_the_chain_is_loaded(
        self, key: str, value: Any
    ) -> None:
        with pytest.raises(ConfigurationError, match=key):
            build(**{key: value})


class TestWithoutTheSubmodule:
    """A host that never checked ``3rdparty/shipvision`` out — arranged, not assumed."""

    @pytest.fixture()
    def masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        for name in list(sys.modules):
            if name == "shipvision" or name.startswith("shipvision."):
                monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "shipvision", None)

    def test_the_chain_still_loads_so_the_failure_is_about_the_host_not_the_yaml(
        self, masked: None
    ) -> None:
        """Registration is eager; only the runtime is lazy.

        The distinction is the whole reason ``topology/bridge.py`` exists: an operator on a
        machine with no submodule gets "install this" and not "unknown element", which are
        two different problems with two different fixes.
        """
        chain = Topology.from_spec(ChainSpec.from_yaml("""
                elements:
                  decode:    {impl: mock}
                  recognize: {impl: shipvision}
                  output:    {impl: mock}
                """))

        assert isinstance(chain.node("recognize").element, GalleryRecognize)

    def test_open_refuses_with_the_command_that_fixes_it(self, masked: None) -> None:
        element = build()

        with pytest.raises(ConfigurationError) as caught:
            element.open(ElementContext())

        message = str(caught.value)
        assert "shipvision.reid" in message
        assert "git submodule update --init 3rdparty/shipvision" in message
        assert "pip install -e 3rdparty/shipvision" in message
        assert element.is_open is False, "a failed open leaves the element closed"


class TestTheGalleryFileOnDisk:
    """:mod:`shipinfer.topology.gallery_store` — pure numpy, and refused at load or not at all.

    None of these needs shipvision: the point of putting the format in ``topology`` is that
    the file is shipinfer's concept (ADR-006), so its validation runs on any host.
    """

    def test_it_reads_the_newest_version_directory(self, tmp_path: Path) -> None:
        write_gallery(tmp_path, vectors=np.eye(2, dtype=np.float32)[:1], identities=["old"])
        write_gallery(
            tmp_path, vectors=np.eye(2, dtype=np.float32)[:1], identities=["new"], version=2
        )

        resolved = gallery_store.resolve_gallery_path(tmp_path, "recognize")

        assert resolved.parent.name == "2"
        assert gallery_store.load_gallery_file(resolved).identities == ("new",)

    def test_an_explicit_version_wins_and_a_missing_one_is_named(self, tmp_path: Path) -> None:
        write_gallery(tmp_path, vectors=np.eye(2, dtype=np.float32)[:1], identities=["old"])
        write_gallery(
            tmp_path, vectors=np.eye(2, dtype=np.float32)[:1], identities=["new"], version=2
        )

        chosen = gallery_store.resolve_gallery_path(tmp_path, "recognize", version=1)
        assert gallery_store.load_gallery_file(chosen).identities == ("old",)

        with pytest.raises(ConfigurationError, match=r"no version 5 \(available: \[1, 2\]\)"):
            gallery_store.resolve_gallery_path(tmp_path, "recognize", version=5)

    def test_a_missing_entry_names_the_layout_that_would_have_worked(
        self, tmp_path: Path
    ) -> None:
        write_gallery(
            tmp_path, "ship_gallery", vectors=np.eye(2, dtype=np.float32)[:1], identities=["a"]
        )

        with pytest.raises(ConfigurationError) as caught:
            gallery_store.resolve_gallery_path(tmp_path, "person_gallery")

        assert "ship_gallery" in str(caught.value), "it lists what is there"
        assert str(Path("person_gallery") / "1" / "gallery.npz") in str(caught.value)

    def test_an_entry_with_no_version_directory_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "ship_gallery").mkdir()

        with pytest.raises(ConfigurationError, match="no numbered version directory"):
            gallery_store.resolve_gallery_path(tmp_path, "ship_gallery")

    def test_the_camera_column_is_optional_and_an_empty_cell_means_none(
        self, tmp_path: Path
    ) -> None:
        """A numpy text array has no ``None``, so ``""`` is the only spelling a writer has."""
        path = write_gallery(
            tmp_path,
            vectors=np.eye(2, dtype=np.float32),
            identities=["a", "b"],
            cameras=["cam-1", ""],
        )

        assert gallery_store.load_gallery_file(path).camera_ids == ("cam-1", None)

    def test_an_empty_gallery_file_is_a_state_not_an_error(self, tmp_path: Path) -> None:
        path = write_gallery(
            tmp_path, vectors=np.zeros((0, 8), dtype=np.float32), identities=[]
        )

        loaded = gallery_store.load_gallery_file(path)

        assert len(loaded) == 0
        assert (
            loaded.dim == 8
        ), "the width survives an empty file, which is what the dim check reads"

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            pytest.param(
                {"vectors": np.zeros((2, 2, 2), dtype=np.float32), "identities": ["a", "b"]},
                "must be (N, d)",
                id="three-dimensional",
            ),
            pytest.param(
                {"vectors": np.eye(2, dtype=np.int32), "identities": ["a", "b"]},
                "an embedding is float",
                id="integer-dtype",
            ),
            pytest.param(
                {
                    "vectors": np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
                    "identities": ["a", "b"],
                },
                "row 1 of 'vectors' is not finite",
                id="not-finite",
            ),
            pytest.param(
                {
                    "vectors": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
                    "identities": ["a", "b"],
                },
                "row 1 of 'vectors' is all zeros",
                id="zero-vector",
            ),
            pytest.param(
                {"vectors": np.eye(2, dtype=np.float32), "identities": ["only-one"]},
                "parallel",
                id="length-mismatch",
            ),
            pytest.param(
                {"vectors": np.eye(2, dtype=np.float32), "identities": np.array([1, 2])},
                "is text",
                id="numeric-identities",
            ),
            pytest.param(
                {"vectors": np.eye(2, dtype=np.float32), "identities": ["a", ""]},
                "row 1 is empty",
                id="empty-identity",
            ),
            pytest.param(
                {"identities": ["a"]},
                "no 'vectors' array",
                id="no-vectors",
            ),
        ],
    )
    def test_a_malformed_archive_is_refused_naming_the_file_and_the_row(
        self, tmp_path: Path, payload: dict[str, Any], expected: str
    ) -> None:
        """Every one of these loads clean without the checks, and answers plausibly after.

        A NaN row poisons every score in the gallery (``shipvision.reid.distance.normalize``
        says so); a zero row sits at cosine 0 from everything, which is a *plausible* answer
        to every query; a length mismatch files a vector under another ship's name. None of
        them raises anywhere else, which is why they are refused here.
        """
        directory = tmp_path / "ship_gallery" / "1"
        directory.mkdir(parents=True)
        path = directory / gallery_store.GALLERY_FILE
        np.savez(path, **{key: np.asarray(value) for key, value in payload.items()})

        with pytest.raises(ConfigurationError) as caught:
            gallery_store.load_gallery_file(path)

        assert expected in str(caught.value)
        assert str(path) in str(caught.value), "the refusal names the file to go and fix"

    def test_a_file_that_is_not_an_archive_is_refused_rather_than_traced(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "gallery.npz"
        path.write_bytes(b"not a zip")

        with pytest.raises(ConfigurationError, match=r"not a readable \.npz archive"):
            gallery_store.load_gallery_file(path)

    def test_a_missing_file_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="does not exist"):
            gallery_store.load_gallery_file(tmp_path / "gallery.npz")


@pytest.fixture()
def reid() -> Any:
    """``shipvision.reid``, or a skip. Every test below queries a real gallery."""
    return pytest.importorskip("shipvision.reid")


class TestTheQuery:
    """What the element files, with a real ``shipvision`` gallery underneath it."""

    def test_a_query_from_camera_a_is_answered_by_the_camera_b_entry(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """``exclude_camera``, proven by the answer rather than by the call.

        ``ship-a``'s own camera holds an exact match for the probe. The correct answer is
        ``ship-b`` at 0.9, from ``cam-B``; the answer without the exclusion is ``ship-a`` at
        1.0. Reverting the ``exclude_camera=`` argument turns this test red on both halves,
        which is what makes it evidence rather than decoration.

        ``flat`` and not the default ``centroid`` for this one: a centroid holds one vector
        per identity and therefore one camera — the most recent — so the six rows would fold
        into three and the case would no longer be about per-row exclusion.
        """
        element = build(gallery="flat", gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            filed = element.process(item(np.array([PROBE]), camera="cam-A")).meta["identities"]
        finally:
            element.close()

        identity, score = filed[0]
        assert identity == "ship-b", "the probe's own camera's rows were excluded"
        assert score == pytest.approx(0.9, abs=1e-3)

    def test_the_same_probe_from_a_third_camera_finds_the_exact_match(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The control for the test above: the exclusion removes rows, it does not break ranking.

        Without this, "``ship-b`` came back" would be equally consistent with an element that
        never finds ``ship-a`` at all.
        """
        element = build(gallery="flat", gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            filed = element.process(item(np.array([PROBE]), camera="cam-Z")).meta["identities"]
        finally:
            element.close()

        assert filed == [("ship-a", pytest.approx(1.0))]

    def test_an_unmatched_row_is_none_and_never_zero(self, reid: Any, tmp_path: Path) -> None:
        """0 is a legitimate gallery id (``pipeline/schema.py``), so it cannot mean 'nobody'.

        The identity assertion is ``is None`` and the type assertion is separate on purpose:
        ``0 == None`` is False, so an equality check alone would pass for a ``0`` **and** for
        a ``""``, and both are values a downstream consumer would publish as an identity.
        """
        element = build(gallery="flat", gallery_dir=str(tmp_path), threshold=0.95, dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            filed = element.process(item(np.array([PROBE]), camera="cam-A")).meta["identities"]
        finally:
            element.close()

        identity, similarity = filed[0]
        assert identity is None
        assert similarity is None
        assert not isinstance(identity, int), "an unknown is not 0, and 0 is a real id"

    def test_the_payload_and_the_cap_are_handed_on_untouched(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """``produces: *@*`` means the outbound cap is the inbound one — so stamp nothing.

        A ``caps=`` here would relabel whatever the chain is carrying as this element's first
        declared cap, and the download arch.md §8 refuses would become invisible one element
        further down with every edge still reported valid.
        """
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            incoming = item(np.array([PROBE]), camera="cam-A")
            result = element.process(incoming)
        finally:
            element.close()

        assert result.caps is incoming.caps
        assert result.payload is incoming.payload
        assert result.context is incoming.context, "the (camera, frame) tag rides on"
        assert incoming.meta.get("identities") is None, "the input item is not mutated"

    def test_an_empty_frame_files_an_empty_list_and_is_not_an_error(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """ "No ships in this frame" is not a failure and must not read like one."""
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            filed = element.process(item(np.zeros((0, DIM), np.float32))).meta["identities"]
        finally:
            element.close()

        assert filed == []

    def test_a_branch_embedder_files_a_mapping_from_row_index(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The shape ``embed_ship`` needs: it embedded rows 0 and 2 of a four-row frame.

        The identities come back aligned with the *detection* rows it was given, which is the
        alignment that keeps an embedding attached to the object it came from.
        """
        element = build(gallery="flat", gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            filed = element.process(
                item(
                    {2: NEARLY, 0: np.array([0.0, 1.0, 0.0, 0.0], np.float32)},
                    camera="cam-B",
                    detections=FakeDetections(scores=[0.9, 0.9, 0.9, 0.9]),
                )
            ).meta["identities"]
        finally:
            element.close()

        assert [identity for identity, _ in filed] == [
            "ship-b",
            "ship-a",
        ], "row 0 first, row 2 second — the mapping's keys are row indices, not arrival order"

    def test_a_raw_model_response_is_refused_and_the_message_names_the_gap(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """``PoolEmbed`` files ``response.outputs`` verbatim: ``{tensor_name: Tensor}``.

        Guessing which of that tensor's rows belongs to which detection is exactly the
        mistake that puts an embedding on the wrong object, so it is refused with the missing
        step named. The scatter-back is slice C8's.
        """
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            with pytest.raises(ValidationError) as caught:
                element.process(item({"embeddings": np.zeros((3, DIM), np.float32)}))
        finally:
            element.close()

        message = str(caught.value)
        assert "embeddings" in message
        assert "pool" in message and "C8" in message

    def test_a_positional_array_that_disagrees_with_the_detections_is_refused(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """Two vectors for four detections is not "the first two"; it is unaligned.

        The check is C3's hook and is written as "if the key is there, it must agree", so it
        costs nothing on this branch and fails loudly rather than drifting once it lands.
        """
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            with pytest.raises(ValidationError, match="2 vector\\(s\\) for 4 detection"):
                element.process(
                    item(
                        np.zeros((2, DIM), np.float32),
                        detections=FakeDetections(scores=[0.9] * 4),
                    )
                )
        finally:
            element.close()

    def test_no_vectors_at_all_is_a_wiring_failure_not_an_empty_answer(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """A recognize element with nothing embedding ahead of it recognises nobody, forever.

        Distinct from the empty-list case above, and the message says which: an empty list is
        "no ships in this frame", a missing key is "this element is in the wrong place".
        """
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            bare = ChainItem(
                context=RequestContext(camera_id="cam-A", frame_id=1), caps=BGR, payload=None
            )
            with pytest.raises(ValidationError, match="found no `vectors`"):
                element.process(bare)
        finally:
            element.close()


class TestWhatTheGalleryIsAsked:
    """The arguments, recorded — the half a behavioural assertion cannot see.

    A spy registered in ``shipvision``'s own registry rather than a monkeypatched attribute,
    because that is the seam the element actually uses (``GALLERIES.build``) and it keeps the
    test honest about how a deployment would substitute an implementation.
    """

    @pytest.fixture()
    def spy(self, reid: Any) -> type:
        if "spy" not in reid.GALLERIES.names():

            @reid.GALLERIES.register("spy")
            class SpyGallery(reid.BaseGallery):
                """Records every query, answers nothing."""

                def __init__(self, *, capacity: int = 8, dim: int | None = None) -> None:
                    self.capacity = capacity
                    self.calls: list[dict[str, Any]] = []
                    self._dim = dim

                @property
                def dim(self) -> int | None:
                    return self._dim

                def add(self, embedding: Any) -> int:
                    return 0

                def query(self, vector: Any, **kwargs: Any) -> Any:
                    self.calls.append(kwargs)
                    return reid.QueryResult(matches=())

                def remove_identity(self, identity: str) -> int:
                    return 0

                def clear(self) -> None:
                    return None

                def __len__(self) -> int:
                    return 0

                @property
                def identities(self) -> tuple[str, ...]:
                    return ()

        return reid.GALLERIES.get("spy")

    def test_every_query_carries_the_item_s_own_camera_and_the_configured_top_k(
        self, spy: type
    ) -> None:
        element = build(gallery="spy", top_k=3, threshold=0.4)
        element.open(ElementContext())

        try:
            element.process(item(np.zeros((2, DIM), np.float32), camera="cam-07"))
            calls = element._gallery.calls
        finally:
            element.close()

        assert calls == [
            {"top_k": 3, "threshold": 0.4, "exclude_camera": "cam-07"},
            {"top_k": 3, "threshold": 0.4, "exclude_camera": "cam-07"},
        ], "one query per row, each excluding the camera that asked"


class TestOpen:
    """What stops a deploy, and what only warns."""

    def test_a_width_that_disagrees_with_the_declared_one_stops_the_deploy(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """Two models feeding one gallery, caught before a frame exists.

        Its only other symptom is a similarity matrix that cannot be formed — or one that
        can, because a broadcast succeeded (``BaseGallery.dim``).
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), dim=128)

        with pytest.raises(ConfigurationError) as caught:
            element.open(ElementContext())

        message = str(caught.value)
        assert "4-d" in message and "128" in message
        assert "gallery.npz" in message

    def test_an_unknown_gallery_implementation_lists_the_ones_that_exist(
        self, reid: Any
    ) -> None:
        element = build(gallery="centroied")

        with pytest.raises(ConfigurationError, match="centroid"):
            element.open(ElementContext())

    def test_an_empty_gallery_opens_with_one_warning_and_answers_none(
        self, reid: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A legitimate state — nobody enrolled yet — that reads exactly like a working one.

        Refusing it would mean a chain could not be brought up before its identities existed.
        Being quiet about it would mean a deployment that recognises nobody for a week and
        reports nothing, which is the failure this whole project was started over.
        """
        element = build()
        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.recognize"):
            element.open(ElementContext())

        empty = [r for r in caplog.records if "EMPTY gallery" in r.getMessage()]
        assert len(empty) == 1, "once, at open — not once per frame"

        try:
            filed = element.process(item(np.array([PROBE]))).meta["identities"]
        finally:
            element.close()

        assert filed == [(None, None)]

    def test_a_capacity_that_drops_identities_at_load_says_so(
        self, reid: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Six rows, three identities, room for two: one identity is silently gone otherwise."""
        three_identities_two_cameras(tmp_path)
        element = build(gallery="flat", gallery_dir=str(tmp_path), capacity=2, dim=DIM)

        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.recognize"):
            element.open(ElementContext())
        element.close()

        assert any("capacity" in record.getMessage() for record in caplog.records)

    def test_a_gallery_with_no_camera_column_warns_that_exclusion_cannot_work(
        self, reid: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``exclude_camera`` cannot exclude a row that does not say where it came from."""
        write_gallery(
            tmp_path, vectors=np.eye(DIM, dtype=np.float32)[:2], identities=["a", "b"]
        )
        element = build(gallery_dir=str(tmp_path), dim=DIM)

        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.recognize"):
            element.open(ElementContext())
        element.close()

        assert any("exclude_camera" in record.getMessage() for record in caplog.records)

    def test_an_explicit_file_path_wins_over_the_repository_triple(
        self, reid: Any, tmp_path: Path
    ) -> None:
        path = write_gallery(
            tmp_path,
            "elsewhere",
            vectors=np.eye(DIM, dtype=np.float32)[:1],
            identities=["only"],
        )
        element = build(gallery_file=str(path), gallery_dir=str(tmp_path), dim=DIM)
        element.open(ElementContext())

        try:
            assert list(element._gallery.identities) == ["only"]
        finally:
            element.close()

    def test_the_entry_name_defaults_to_the_slot_and_gallery_name_overrides_it(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """A slot called ``recognize`` looks for ``<gallery_dir>/recognize/…`` and says so.

        The default is a convention rather than a guess — it is how the model repository
        already names things — and a deployment whose entry is called something else changes
        one key instead of restructuring a directory.
        """
        three_identities_two_cameras(tmp_path)
        write_gallery(
            tmp_path,
            "ship_gallery",
            vectors=np.eye(DIM, dtype=np.float32)[:1],
            identities=["renamed"],
        )

        by_slot = build("recognize", gallery_dir=str(tmp_path), dim=DIM)
        by_slot.open(ElementContext())
        try:
            assert sorted(by_slot._gallery.identities) == ["ship-a", "ship-b", "ship-c"]
        finally:
            by_slot.close()

        named = build(
            "recognize", gallery_dir=str(tmp_path), gallery_name="ship_gallery", dim=DIM
        )
        named.open(ElementContext())
        try:
            assert list(named._gallery.identities) == ["renamed"]
        finally:
            named.close()

    def test_close_drops_the_gallery_and_a_reopen_reloads_it(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The state is the gallery, so ``close`` releasing it is what makes a restart clean."""
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), dim=DIM)

        element.open(ElementContext())
        element.close()
        assert element._gallery is None

        element.open(ElementContext())
        try:
            assert len(element._gallery) == 3, "a centroid per identity, reloaded"
        finally:
            element.close()


class TestMetrics:
    """What an operator sees, and the one case where they must see nothing."""

    def test_the_counters_reconcile_and_name_the_camera_that_asked(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """``queries == matches + unknowns``, per camera and per element.

        Labelled by camera because the number that matters in this system is never the shard
        total (ADR-005) — "unknowns are climbing on camera 7" is the sentence an operator
        acts on, and a shard-wide sum destroys the evidence for it. Labelled by element
        because a chain may hold two ``recognize`` slots, and two slots sharing a counter is
        a counter that answers neither.
        """
        three_identities_two_cameras(tmp_path)
        registry = MetricsRegistry()
        element = build(
            "recognize", gallery="flat", gallery_dir=str(tmp_path), threshold=0.8, dim=DIM
        )
        element.open(ElementContext(metrics=registry))

        try:
            element.process(
                item(np.array([PROBE, np.full(DIM, 0.5, np.float32)]), camera="cam-A")
            )
        finally:
            element.close()

        labels = {"camera": "cam-A", "element": "recognize"}
        collected = {metric.name: metric for metric in registry.collect()}
        queries = collected["shipinfer_recognize_queries_total"]
        matches = collected["shipinfer_recognize_matches_total"]
        unknowns = collected["shipinfer_recognize_unknown_total"]

        assert queries.value(**labels) == 2
        assert matches.value(**labels) == 1, "the probe found ship-b across cameras"
        assert unknowns.value(**labels) == 1
        assert queries.value(camera="cam-B", element="recognize") == 0, "no cross-talk"
        assert list(collected["shipinfer_recognize_query_latency_us"].samples())

    def test_a_runner_that_offers_no_registry_gets_no_metrics_at_all(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """An element must count nothing rather than mint a private registry nobody scrapes.

        A metric on a registry no exporter reads is worse than an absent one, because it
        reads as evidence (:class:`~shipinfer.topology.base.ElementContext`).
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        element.open(ElementContext())

        try:
            assert element.process(item(np.array([PROBE]))).meta["identities"]
        finally:
            element.close()


class TestEnrolment:
    """Opt-in, off, and bounded — in that order of importance."""

    #: A direction no enrolled ship points in: cosine 0.5 with each axis of the fixture, so
    #: it is unmatched at any sane threshold and becomes a *new* identity if anything enrols
    #: it. The whole enrolment story can then be told by asking twice.
    STRANGER = np.full(DIM, 0.5, dtype=np.float32)

    def test_an_unmatched_row_teaches_the_gallery_nothing_by_default(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """Asked twice, the same stranger is a stranger both times.

        A behavioural assertion rather than a length check, because the property is what the
        *next frame* sees. The default is the whole point: a gallery that enrols every
        stranger it meets grows one identity per crop of the same ship, and the rising
        identity count reads like coverage.
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery="flat", gallery_dir=str(tmp_path), threshold=0.9, dim=DIM)
        element.open(ElementContext())

        try:
            first = element.process(
                item(
                    np.array([self.STRANGER]), camera="cam-A", detections=FakeDetections([1.0])
                )
            ).meta["identities"]
            second = element.process(
                item(
                    np.array([self.STRANGER]), camera="cam-Z", detections=FakeDetections([1.0])
                )
            ).meta["identities"]
        finally:
            element.close()

        assert first == [(None, None)]
        assert second == [(None, None)], "nothing was learned from the first frame"

    def test_when_it_is_on_the_next_camera_to_see_the_same_ship_recognises_it(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """Enrolment, proven by the thing it is for: a cross-camera match that could not
        have happened before.

        The minted id names its own provenance — ``<prefix>:<camera>:<frame>:<row>`` — so an
        identity the server minted at 3 a.m. is distinguishable at a glance from one a human
        enrolled, and the crop behind a bad prototype can be found in the recording.

        The second half is the one worth reading twice: the *same* camera asking again still
        gets ``None``, because the enrolled entry carries ``camera_id="cam-A"`` and
        ``exclude_camera`` drops it. That is the protocol working on a row this element wrote
        itself.
        """
        three_identities_two_cameras(tmp_path)
        registry = MetricsRegistry()
        element = build(
            gallery="flat",
            gallery_dir=str(tmp_path),
            threshold=0.9,
            dim=DIM,
            enrol=True,
            enrol_min_confidence=0.5,
        )
        element.open(ElementContext(metrics=registry))

        try:
            minted = element.process(
                item(
                    np.array([self.STRANGER]),
                    camera="cam-A",
                    frame=184102,
                    detections=FakeDetections([0.8]),
                )
            ).meta["identities"]
            elsewhere = element.process(
                item(
                    np.array([self.STRANGER]), camera="cam-B", detections=FakeDetections([0.8])
                )
            ).meta["identities"]
            again = element.process(
                item(
                    np.array([self.STRANGER]), camera="cam-A", detections=FakeDetections([0.8])
                )
            ).meta["identities"]
        finally:
            element.close()

        assert minted == [("auto:cam-A:184102:0", None)], "identity yes, similarity no"
        assert elsewhere == [("auto:cam-A:184102:0", pytest.approx(1.0))]
        assert again == [(None, None)], "its own camera's enrolment is still excluded"

        enrolments = {metric.name: metric for metric in registry.collect()}[
            "shipinfer_recognize_enrolments_total"
        ]
        assert enrolments.total() == 1, (
            "one enrolment for three unmatched-by-its-own-camera frames: the membership "
            "check is what keeps a camera from re-enrolling the ship it just enrolled"
        )

    def test_a_row_the_detector_was_unsure_of_is_not_enrolled(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The two mistakes are not symmetric: refusing a real new ship loses one crop, and
        enrolling a blurred half-occluded one puts a bad prototype in front of every future
        query."""
        three_identities_two_cameras(tmp_path)
        element = build(
            gallery="flat",
            gallery_dir=str(tmp_path),
            threshold=0.9,
            dim=DIM,
            enrol=True,
            enrol_min_confidence=0.9,
        )
        element.open(ElementContext())

        try:
            filed = element.process(
                item(np.array([self.STRANGER]), detections=FakeDetections([0.4]))
            ).meta["identities"]
            later = element.process(
                item(
                    np.array([self.STRANGER]), camera="cam-Z", detections=FakeDetections([0.4])
                )
            ).meta["identities"]
        finally:
            element.close()

        assert filed == [(None, None)]
        assert later == [(None, None)], "the blurred crop did not become an identity"

    def test_a_row_with_no_confidence_is_not_enrolled_and_says_so_once(
        self, reid: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Until C3 lands, no item carries a per-row confidence — so nothing is enrolled.

        Silent, that is a deployment with enrolment configured and switched on that never
        enrols anybody. One warning, not one per frame: this runs at a thousand frames a
        second.
        """
        element = build(enrol=True, enrol_min_confidence=0.1)
        element.open(ElementContext())

        try:
            with caplog.at_level(logging.WARNING, logger="shipinfer.topology.recognize"):
                for _ in range(3):
                    element.process(item(np.array([PROBE])))
            assert len(element._gallery) == 0
        finally:
            element.close()

        warned = [r for r in caplog.records if "no per-row confidence" in r.getMessage()]
        assert len(warned) == 1

    def test_enrolment_stays_inside_the_gallery_s_capacity(self, reid: Any) -> None:
        """The bound is the library's and this test is what says the element does not defeat it.

        A gallery that grows without one is a memory leak with a plausible name
        (``shipvision.reid.gallery.base``).
        """
        element = build(
            gallery="flat", capacity=2, dim=DIM, enrol=True, enrol_min_confidence=0.5
        )
        element.open(ElementContext())

        try:
            for frame in range(5):
                element.process(
                    item(
                        np.array([np.eye(DIM, dtype=np.float32)[frame % DIM]]),
                        frame=frame,
                        detections=FakeDetections(scores=[0.9]),
                    )
                )
            assert len(element._gallery) == 2
        finally:
            element.close()


class TestInAChain:
    """The element in the wiring it will actually be deployed in."""

    def chain(self) -> Topology:
        return Topology.from_spec(ChainSpec.from_yaml("""
                name: ship_recognition
                elements:
                  decode:     {impl: mock}
                  detect:     {impl: mock}
                  embed_ship: {impl: mock, when: class == ship}
                  recognize:  {impl: shipvision, when: class == ship, after: embed_ship}
                  output:     {impl: mock}
                """))

    def test_detect_to_embed_to_recognize_negotiates_on_the_inbound_cap(self) -> None:
        """``*@*`` is resolved from what arrives, which is the whole property.

        The mock chain is ``nv12@gpu`` end to end, so every edge through this element is
        ``nv12@gpu`` — including the one *out* of it, which is the wildcard being resolved
        rather than stamped. In phase D the same chain is still legal with an ``nv12@gpu``
        decoder in front, and a ``bgr@cpu`` chain resolves the same wildcard the other way.
        """
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in self.chain().edges}

        assert caps[("detect", "embed_ship")] == "nv12@gpu"
        assert caps[("embed_ship", "recognize")] == "nv12@gpu"
        assert caps[("recognize", "output")] == "nv12@gpu"

    def test_the_wildcard_makes_the_when_bypass_carry_exactly_what_the_element_would(
        self,
    ) -> None:
        """A ``when:`` element's predecessor-to-successor pair is checked too, and here it is free.

        The loader validates that bypass because an item the condition rejects really is
        handed from ``embed_ship`` straight to ``output``
        (:func:`~shipinfer.topology.chain._negotiate_edges`, and
        :meth:`~shipinfer.topology.chain.ElementNode.admits` for the runtime half). For *this*
        element the check can never fail, and the reason is the whole point of the wildcard:
        ``produces: *@*`` is resolved from the inbound cap, so what the bypass carries and
        what the element would have handed on are the same cap.

        Pinned because it is why ``when: class == ship`` on a gallery query costs the loader
        nothing — and because a concrete ``produces`` here would quietly stop that being true,
        which is a thing a reviewer of a future change should be told by a red test rather
        than by reading this file.
        """
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in self.chain().edges}

        assert caps[("embed_ship", "recognize")] == caps[("recognize", "output")]

    def test_the_guard_gates_the_element_and_not_the_rows_inside_it(self) -> None:
        """``when: class == ship`` is evaluated once per *item*, on ``meta["class"]``.

        Worth pinning because the natural reading of "recognize only ships" is per-row, and
        it is not: a frame the condition rejects skips this element entirely, and a frame it
        admits is queried for **every** vector it carries. Per-row class filtering needs C3's
        per-row labels and is not this slice's.
        """
        node = self.chain().node("recognize")

        assert node.admits(item(np.zeros((1, DIM), np.float32), **{"class": "ship"})) is True
        assert node.admits(item(np.zeros((1, DIM), np.float32), **{"class": "person"})) is False
        assert (
            node.admits(item(np.zeros((1, DIM), np.float32))) is False
        ), "a missing class satisfies neither operator"
