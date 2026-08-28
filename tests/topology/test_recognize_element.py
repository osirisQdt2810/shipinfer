"""The ``recognize`` element that queries a gallery: what it files, and what it refuses.

Green **with or without** ``3rdparty/shipvision``, which is the shape ``test_bridge.py``
established and the reason it is worth repeating here: CI does not check the submodule out
(``.claude/CLAUDE.md``), so the tests that need it present ask for it with
:func:`pytest.importorskip`, and the one that needs it *absent* arranges the absence rather
than assuming it. The properties that need no gallery at all — the caps, the two model
declarations, every refusal in :mod:`shipinfer.topology.gallery_store` — run either way,
which is most of this file.

Seven of these tests are the ones the element exists for, and each was checked by reverting
the line it covers (the numbers are in the slice report):

* ``exclude_camera`` is passed on every query, **and the default gallery honours it per
  entry**. Dropped, the cross-camera test answers ``ship-a`` — the identity's own camera's
  enrolment, an exact match, and a score that would make the recogniser look perfect while
  measuring the tracker. Put the default back to ``centroid`` and the same test answers
  ``ship-a`` at 0.994 with the argument still in place, which is why the default is pinned
  by a test of its own rather than by a comment.
* an unknown row is ``None``. Filed as ``0`` instead, the assertion that it is not an int
  fails — and in production it would be indistinguishable from gallery id 0.
* identities are keyed by **detection row**. Filed as a positional list, a frame whose ship
  rows are 0 and 2 files two entries that name neither, and a frame with no ships files a
  value byte-identical to a frame with no detections.
* enrolment writes to the second store. Pointed back at the curated one, forty stranger
  frames evict every identity the operator enrolled.
* the gallery file is validated. With the checks removed, a NaN row loads clean and every
  subsequent score is NaN.
* the archive is read with ``allow_pickle=False``. Flipped, an object array in the ``.npz``
  executes on load, before any validation runs — the test proves the side effect does not.
* a dim mismatch stops the deploy at ``open()``.
"""

from __future__ import annotations

import logging
import os
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
from shipinfer.topology.elements.detections import Detections
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


#: Wide enough that four curated ships and forty strangers can all be *distinct basis
#: vectors*: mutually orthogonal, so "did enrolment evict a curated identity" is answered by
#: arithmetic rather than by a question about how random vectors happen to land in 4-d.
WIDE = 64


def curated_fleet(root: Path, *, count: int = 4, camera: str = "cam-A") -> Path:
    """``count`` ships an operator enrolled offline, one view each, from one camera.

    The gallery an ``enrol: true`` deployment is standing on: identities somebody curated,
    which the server must still be able to recognise after a week of meeting strangers.
    """
    return write_gallery(
        root,
        vectors=np.eye(WIDE, dtype=np.float32)[:count],
        identities=[f"ship-{index}" for index in range(count)],
        cameras=[camera] * count,
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
    """The two members of ``Detections`` the *unfiltered* path reads: ``len`` and ``scores``.

    Two, and that is itself the evidence the dependency is as small as the plan asked for.
    C3's real :class:`~shipinfer.topology.elements.detections.Detections` has landed and
    :func:`detected` below builds one; this double stays because a double that carries
    exactly the two attributes the code touches is what *proves* the dependency is those two.
    Any test that reads a **label** — the ``classes:`` path — uses the real class instead,
    because ``labels`` is a field with an invariant (one per row, checked in
    ``__post_init__``) and a stand-in for it would pin nothing.
    """

    scores: Sequence[float]

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class UnsizedDetections:
    """A ``detections`` with ``labels`` and no ``__len__`` — the reviewer's first double.

    Legitimate as far as this element is concerned: ``meta["detections"]`` is somebody else's
    key and a value it cannot size is treated as absent, which is what lets a chain embed
    with no decoding detector in front of it. The consequence is that the shared reader has
    no row count to bound the vectors against, so a ``classes:`` filter is the first thing to
    index the labels — and with one label for three rows it is also the first thing to run
    off the end.
    """

    labels: Sequence[str]


@dataclass
class ShortLabels:
    """Sized, and still carrying fewer labels than rows — the reviewer's second double.

    C3's real ``Detections.__post_init__`` refuses this, which is why the element is right to
    trust it; nothing refuses it for a value that merely looks like one.
    """

    scores: Sequence[float]
    labels: Sequence[str]

    def __len__(self) -> int:
        return len(self.scores)


class WatchedScores:
    """``FakeDetections`` that records how many times anything read its ``scores``.

    A counter and not a timer: "this value is not built when nobody reads it" is a property,
    and a property is asserted by observing the read rather than by measuring what skipping
    it saved on a shared box.
    """

    def __init__(self, *, scores: Sequence[float]) -> None:
        self._scores = list(scores)
        self.reads = 0

    def __len__(self) -> int:
        return len(self._scores)

    @property
    def scores(self) -> Sequence[float]:
        self.reads += 1
        return self._scores


#: The demo repository's COCO numbering, the same table ``DecodeParams`` ships with. The class
#: id has to agree with the label or the ``Detections`` is one no decoder could emit.
CLASS_IDS = {"person": 0, "ship": 8}


def detected(*labels: str, scores: Sequence[float] | None = None) -> Detections:
    """C3's real ``Detections`` for a frame of ``labels``, one 10x10 box each.

    The real class and not a double, because these tests are about ``labels`` — the field
    ``params: classes:`` selects on — and because a row's label, class id and score have to
    stay parallel, which is an invariant this class enforces and a dataclass of two fields
    cannot.
    """
    count = len(labels)
    return Detections(
        boxes=np.tile(np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32), (count, 1)),
        scores=np.asarray(list(scores) if scores is not None else [0.9] * count, np.float32),
        class_ids=np.asarray([CLASS_IDS.get(label, 99) for label in labels], np.int32),
        labels=tuple(labels),
    )


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

    @pytest.mark.parametrize("value", ["off", "no", 0, [], "true"], ids=repr)
    def test_a_switch_that_is_not_a_switch_is_refused_rather_than_coerced(
        self, value: Any
    ) -> None:
        """``bool(value)`` accepts anything, and ``enrol: "off"`` is truthy.

        Which means an operator switching enrolment off in the most natural spelling there is
        would get exactly the behaviour they were trying to prevent, on a deployment that
        started cleanly and logged nothing. Every other value in this element is refused at
        load with a named message; this one was the exception and is not any more.
        """
        with pytest.raises(ConfigurationError, match="enrol must be true or false"):
            build(enrol=value)

    @pytest.mark.parametrize("value", ["ship", 8, {"ship": True}], ids=repr)
    def test_a_class_filter_that_is_not_a_list_of_labels_is_refused(self, value: Any) -> None:
        """``classes: ship`` is the natural typo, and a string is a ``Sequence`` of letters.

        Accepted, it would filter on ``"s"``, ``"h"``, ``"i"``… and match no row at all — an
        element that queries nothing and reports nothing. Refused with the same message
        ``ShipvisionTrack`` gives for the same key, because a chain file's ``classes:`` must
        not mean two things in two slots.
        """
        with pytest.raises(ConfigurationError, match="`params: classes:` must be a list"):
            build(classes=value)


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

    def test_a_pickled_object_array_is_refused_without_being_executed(
        self, tmp_path: Path
    ) -> None:
        """``allow_pickle=False`` is a security claim, and this is the measurement behind it.

        A ``.npz`` is a zip, and an object array inside one is a pickle: loading it *runs*
        whatever the archive says, before a single one of this module's validations gets a
        look at the data. A model repository is a directory an operator syncs from somewhere
        else (the module docstring argues exactly this), so the archive is not trusted input.

        The detonator's ``__reduce__`` names ``os.system`` and a ``touch``, which is the
        crudest possible proof and deliberately so: with ``allow_pickle=True`` the marker
        file appears, measured — the refusal that goes red without this test is a refusal
        that *runs the payload and then complains*. The assertion is therefore two things,
        and the second is the one that matters: the load was refused, **and** nothing ran.
        """
        marker = tmp_path / "the-pickle-ran"

        class Detonator:
            def __reduce__(self) -> tuple[Any, tuple[str]]:
                return (os.system, (f"touch {marker}",))

        payload = np.empty(1, dtype=object)
        payload[0] = Detonator()
        directory = tmp_path / "ship_gallery" / "1"
        directory.mkdir(parents=True)
        path = directory / gallery_store.GALLERY_FILE
        np.savez(path, vectors=payload, identities=np.array(["a"]))

        with pytest.raises(ConfigurationError) as caught:
            gallery_store.load_gallery_file(path)

        # First, and deliberately first: with `allow_pickle=True` this file is *still*
        # refused — by the shape check, because an object array is `(1,)` and not `(N, d)` —
        # so a test that only asserted the refusal would stay green over a loader that runs
        # the payload and then complains. `os.system` blocks, so the marker is there or the
        # code never ran.
        assert not marker.exists(), "the archive's pickle executed before it was refused"
        assert "allow_pickle" in str(caught.value) or "Object array" in str(caught.value)

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

        **The default gallery, named nowhere in this test**, and that is the half worth
        reading twice. Rule 1 is only as strong as the implementation underneath it: a
        gallery that folds an identity's views into one vector can record only the camera it
        saw most recently, so this same file queried through ``gallery: centroid`` answers
        ``ship-a`` at 0.994 with the ``exclude_camera=`` argument still in place. A test that
        opted into ``flat`` would prove the argument is passed and say nothing about what a
        deployment that wrote no ``gallery:`` key actually gets.
        """
        element = build(gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
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
        never finds ``ship-a`` at all. The default gallery here too, for the same reason.
        """
        element = build(gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext())

        try:
            filed = element.process(item(np.array([PROBE]), camera="cam-Z")).meta["identities"]
        finally:
            element.close()

        assert filed == {0: ("ship-a", pytest.approx(1.0))}

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

    @pytest.mark.parametrize("empty", [{}, np.zeros((0, DIM), np.float32)], ids=["map", "arr"])
    def test_an_empty_frame_files_an_empty_mapping_and_counts_nothing(
        self, reid: Any, tmp_path: Path, empty: Any
    ) -> None:
        """ "No ships in this frame" is not a failure and must not read like one.

        Both spellings of empty arrive in production: an ``(0, d)`` array from an embedder
        that ran on a frame with no detections, and an empty **mapping** from a branch
        embedder that was handed a frame of people and selected no ship rows. Neither is a
        query, and — the half a length assertion would miss — neither is an *unknown*. If an
        unembedded row were counted as unknown, the unknown rate an operator watches after a
        model change would move with how many people walked past the ship.
        """
        registry = MetricsRegistry()
        element = build(gallery_dir=str(tmp_path), dim=DIM)
        three_identities_two_cameras(tmp_path)
        element.open(ElementContext(metrics=registry))

        try:
            filed = element.process(item(empty, detections=detected())).meta["identities"]
        finally:
            element.close()

        assert filed == {}
        counted = {metric.name: metric for metric in registry.collect()}
        assert counted["shipinfer_recognize_queries_total"].total() == 0
        assert counted["shipinfer_recognize_unknown_total"].total() == 0

    def test_a_branch_embedder_files_a_mapping_from_row_index(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The shape ``embed_ship`` needs: it embedded rows 0 and 2 of a four-row frame.

        The identities come back **keyed by the detection row**, which is the alignment that
        keeps an identity attached to the object it came from. Filed as a positional list —
        the shape this element shipped with — the answer would be
        ``[("ship-b", 1.0), ("ship-a", 0.9)]``: two entries that name neither row 0 nor row
        2, and that a consumer can only read by assuming the embedder embedded the first two
        detections. It did not.

        Rows 1 and 3 are **absent**, not ``(None, None)``. Nothing embedded them, so this
        element has no answer for them and must not file one: the runner merges the branches
        of a chain by unioning their metadata, and a placeholder here is what would collide
        with the person branch's answer for the same detection.
        """
        element = build(gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
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

        assert set(filed) == {0, 2}, "the keys are the detector's row indices"
        assert filed[0][0] == "ship-b"
        assert filed[2][0] == "ship-a"

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


class TestTheClassFilter:
    """``params: {classes: [...]}`` — which detection rows this element asks about.

    Separate from ``when:``, which gates the whole element on one item
    (:meth:`TestInAChain.test_the_guard_gates_the_element_and_not_the_rows_inside_it`), and
    the pair is the reason both exist: a frame admitted by the guard still carries every
    detection the detector found, people included, and a ship gallery asked about a person
    answers *something*.

    Mirrors :class:`~shipinfer.topology.elements.track.ShipvisionTrack`'s key of the same
    name, over the same ``Detections.labels``, so a chain file's ``classes:`` means one thing
    wherever it is written.
    """

    def rows(self) -> np.ndarray:
        """Four rows whose *person* vectors would match loudly if they were ever queried.

        Rows 1 and 3 are ``PROBE`` — ``ship-a``'s own appearance. A filter that silently let
        them through would file ``ship-a`` for two people and the assertion below would name
        it; a filter that dropped the wrong rows would lose the two real answers.
        """
        return np.array([[0.0, 1.0, 0.0, 0.0], PROBE, NEARLY, PROBE], dtype=np.float32)

    def test_only_the_named_class_is_queried_and_the_rest_are_absent(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The person rows are not queried, not counted, and not in the mapping.

        "Not in the mapping" is the load-bearing third: filing ``(None, None)`` for a person
        would be this element asserting that nobody recognises that detection, which is false
        — the person branch does — and the runner's fan-in would have two branches offering
        different values for the same key.
        """
        three_identities_two_cameras(tmp_path)
        registry = MetricsRegistry()
        element = build(gallery_dir=str(tmp_path), threshold=0.5, dim=DIM, classes=["ship"])
        element.open(ElementContext(metrics=registry))

        try:
            filed = element.process(
                item(
                    self.rows(),
                    camera="cam-B",
                    detections=detected("ship", "person", "ship", "person"),
                )
            ).meta["identities"]
        finally:
            element.close()

        assert set(filed) == {0, 2}, "the person rows were never this element's to answer"
        assert filed[0][0] == "ship-b"
        assert filed[2][0] == "ship-a"
        counted = {metric.name: metric for metric in registry.collect()}
        assert counted["shipinfer_recognize_queries_total"].total() == 2
        assert counted["shipinfer_recognize_unknown_total"].total() == 0

    def test_two_slots_filtering_two_classes_produce_disjoint_keys(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The property the runner's fan-in needs, stated where it can be checked.

        A ship recogniser and a person recogniser sit on two branches of one chain and rejoin
        at the tracker. Their ``identities`` mappings are merged there, and a merge is only
        well-defined if the two never claim the same detection row. Row selection by class is
        what makes that true by construction rather than by a comment in the runner.
        """
        three_identities_two_cameras(tmp_path)
        ships = build(
            "recognize_ship",
            gallery_dir=str(tmp_path),
            gallery_name="recognize",
            dim=DIM,
            classes=["ship"],
        )
        people = build(
            "recognize_person",
            gallery_dir=str(tmp_path),
            gallery_name="recognize",
            dim=DIM,
            classes=["person"],
        )
        detections = detected("ship", "person", "ship", "person")
        ships.open(ElementContext())
        people.open(ElementContext())

        try:
            by_ship = ships.process(
                item(self.rows(), camera="cam-B", detections=detections)
            ).meta["identities"]
            by_person = people.process(
                item(self.rows(), camera="cam-B", detections=detections)
            ).meta["identities"]
        finally:
            ships.close()
            people.close()

        assert set(by_ship) == {0, 2}
        assert set(by_person) == {1, 3}
        assert not set(by_ship) & set(by_person), "two branches, one row each, no collision"
        assert set(by_ship) | set(by_person) == {0, 1, 2, 3}

    def test_a_class_filter_with_no_detections_to_read_is_a_wiring_failure(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """A filter with nothing to filter on must not quietly pass everything.

        Passing everything is the dangerous default here, not the safe one: it queries every
        person in the frame against a ship gallery and files an identity for them, which is
        the failure ``topology/ship_person.yaml``'s ``when:`` clauses were added for.
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), dim=DIM, classes=["ship"])
        element.open(ElementContext())

        try:
            with pytest.raises(ValidationError, match="carries no `detections`"):
                element.process(item(np.array([PROBE])))
        finally:
            element.close()

    @pytest.mark.parametrize(
        "detections",
        [
            UnsizedDetections(labels=("ship",)),
            ShortLabels(scores=[0.9] * 3, labels=("ship",)),
        ],
        ids=["unsized", "short-labels"],
    )
    def test_labels_that_do_not_cover_the_rows_is_a_refusal_and_not_an_index_error(
        self, reid: Any, tmp_path: Path, detections: Any
    ) -> None:
        """Unreachable with C3's real ``Detections``; reachable with what this element accepts.

        The element duck-types ``meta["detections"]`` on purpose — a value it cannot size is
        treated as absent rather than refused, because it is not this element's key to
        validate — and that is exactly the door a short ``labels`` walks through: with no
        ``len()`` the shared reader has no row bound to check the vectors against, so
        ``labels[index]`` is the first thing to notice. A bare ``IndexError`` reaching the
        runner is charged to "this element has a bug"; this is a chain wired to a detector
        that does not carry one label per box, and the two have different fixes.
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), dim=DIM, classes=["ship"])
        element.open(ElementContext())

        try:
            with pytest.raises(ValidationError, match="`classes: \\['ship'\\]`") as caught:
                element.process(item(np.tile(PROBE, (3, 1)), detections=detections))
        finally:
            element.close()

        assert "label(s)" in str(caught.value)


class TestARefusalFromTheLibrary:
    """A ``shipvision`` error reaching the walk must arrive as one of ours."""

    def test_a_query_of_the_wrong_width_is_a_wiring_failure_and_not_a_bug(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """Rule 4, arriving late because the chain declared no ``dim:``.

        ``open()`` refuses a width mismatch it can see, and with no ``dim:`` declared there is
        nothing to compare the file against until a vector turns up. When one does,
        ``shipvision`` raises ``DimensionMismatchError`` — not a ``ShipInferError`` — and the
        runner would charge it to "this element has a bug" rather than to the two embedders
        feeding one gallery that actually caused it. The re-raise carries the fix.
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path))
        element.open(ElementContext())

        try:
            with pytest.raises(ValidationError, match="declare `dim:`"):
                element.process(item(np.zeros((1, 8), np.float32)))
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

        assert filed == {0: (None, None)}

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

    def test_a_gallery_that_cannot_exclude_per_view_says_so_at_open(
        self, reid: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rule 1 is only as strong as the implementation underneath it, so name the weak one.

        ``centroid`` is a legitimate choice — its memory scales with the fleet rather than
        with dwell time, and the library documents the trade honestly. What is not legitimate
        is making that choice *silently*, because the symptom is a re-identification number
        that looks better than the system is.
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery="centroid", gallery_dir=str(tmp_path), dim=DIM)

        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.recognize"):
            element.open(ElementContext())
        element.close()

        warned = [r for r in caplog.records if "exclude_camera" in r.getMessage()]
        assert len(warned) == 1, "once, at open"
        assert "flat" in warned[0].getMessage(), "the warning names the fix"

    def test_the_default_gallery_does_not_warn_about_its_own_exclusion(
        self, reid: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The other half: a warning every deployment sees is a warning nobody reads."""
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), dim=DIM)

        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.recognize"):
            element.open(ElementContext())
        element.close()

        assert not [r for r in caplog.records if "exclude_camera" in r.getMessage()]

    def test_the_warned_about_gallery_really_does_answer_the_self_match(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The measurement behind the warning, so it is evidence rather than an opinion.

        Same file, same probe, same ``exclude_camera=cam-A`` — ``flat`` answers ``ship-b``
        (rule 1 honoured) and ``centroid`` answers ``ship-a``, the identity whose own camera
        is asking. The fold left one vector carrying the camera of the *most recent*
        observation (``cam-B``), so the ``cam-A`` view inside it is not excluded by anything.

        This asserts a **library** behaviour on purpose. If ``shipvision``'s centroid ever
        excludes per view, this test goes red — and that is the signal that the default and
        the warning above have become stale, which is exactly when somebody should look.
        """
        three_identities_two_cameras(tmp_path)
        folded = build(gallery="centroid", gallery_dir=str(tmp_path), threshold=0.5, dim=DIM)
        folded.open(ElementContext())

        try:
            filed = folded.process(item(np.array([PROBE]), camera="cam-A")).meta["identities"]
        finally:
            folded.close()

        assert filed[0][0] == "ship-a", "the fold cannot exclude the view that is asking"
        assert filed[0][1] > 0.99, "and it scores it as a near-perfect match"

    def test_an_empty_file_still_states_a_width_and_a_mismatch_stops_the_deploy(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """A gallery emptied for re-enrolment keeps its dtype and shape, so it keeps its claim.

        That is the deployment where a stale width would otherwise be found on the first
        frame after somebody enrolled into it — long after the deploy that introduced it.
        """
        write_gallery(tmp_path, vectors=np.zeros((0, 8), dtype=np.float32), identities=[])
        element = build(gallery_dir=str(tmp_path), dim=DIM)

        with pytest.raises(ConfigurationError) as caught:
            element.open(ElementContext())

        assert "8-d" in str(caught.value) and "4" in str(caught.value)

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
            assert len(element._gallery) == 6, "every row of the file, reloaded"
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

        assert first == {0: (None, None)}
        assert second == {0: (None, None)}, "nothing was learned from the first frame"
        assert element._enrolled is None, "no second store is built when enrolment is off"

    @pytest.mark.parametrize("enrol", [False, True], ids=["off", "on"])
    def test_the_confidences_are_read_only_when_something_will_read_them(
        self, reid: Any, tmp_path: Path, enrol: bool
    ) -> None:
        """``_confidences`` is a ``float()`` per row, and with ``enrol: false`` nobody reads it.

        ``_enrol`` returns before it looks at the value whenever enrolment is off, which is
        the default, so building it on every frame is per-row work on the path that runs at
        fifteen thousand rows a second — the one place this file departed from its own
        resolve-once standard (CONVENTIONS §2.5). Asserted by counting the reads rather than
        by timing them, because a timing assertion on a shared box measures the box.
        """
        three_identities_two_cameras(tmp_path)
        element = build(gallery_dir=str(tmp_path), threshold=0.9, dim=DIM, enrol=enrol)
        element.open(ElementContext())
        watched = WatchedScores(scores=[1.0])

        try:
            element.process(item(np.array([self.STRANGER]), detections=watched))
        finally:
            element.close()

        assert watched.reads == (1 if enrol else 0)

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

        assert minted == {0: ("auto:cam-A:184102:0", None)}, "identity yes, similarity no"
        assert elsewhere == {0: ("auto:cam-A:184102:0", pytest.approx(1.0))}
        assert again == {0: (None, None)}, "its own camera's enrolment is still excluded"

        enrolments = {metric.name: metric for metric in registry.collect()}[
            "shipinfer_recognize_enrolments_total"
        ]
        assert enrolments.total() == 1, (
            "one enrolment for three unmatched-by-its-own-camera frames: the membership "
            "check is what keeps a camera from re-enrolling the ship it just enrolled"
        )

    def test_forty_stranger_frames_do_not_cost_the_operator_a_curated_identity(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The failure this element shipped with, and the property that replaces it.

        Both shipped galleries evict the least recently *observed* entry, and this element
        never re-adds on a match — so a file-loaded identity's observation clock is frozen at
        load and it is the first thing a full gallery drops. Measured on the single-store
        version: capacity 8, four curated identities, forty stranger frames from one camera
        (two seconds at 20 fps) left ``curated survivors: []``. At the shipped default
        capacity a 1000 fps fleet turns the whole gallery over in about ten seconds of
        uncorrelated crops. That is ADR-005's own failure — a bounded shared buffer evicting
        the entry nobody is refreshing — reproduced inside the gallery.

        Forty *orthogonal* strangers, so this is arithmetic and not a question about how
        random vectors land: each is unmatched, each mints, and none of them resembles a
        curated ship enough to be declined by the membership check.

        The last assertion is the one that matters to an operator: after all of it, a curated
        ship is still recognised. A surviving *name* would also be satisfied by a gallery
        holding a vector that had been quietly overwritten.
        """
        curated_fleet(tmp_path)
        element = build(
            gallery_dir=str(tmp_path),
            capacity=8,
            enrol_capacity=8,
            dim=WIDE,
            threshold=0.5,
            enrol=True,
            enrol_min_confidence=0.5,
        )
        element.open(ElementContext())

        try:
            for frame in range(40):
                element.process(
                    item(
                        np.array([np.eye(WIDE, dtype=np.float32)[4 + frame]]),
                        camera="cam-A",
                        frame=frame,
                        detections=FakeDetections(scores=[0.9]),
                    )
                )
            survivors = sorted(element._gallery.identities)
            still_known = element.process(
                item(
                    np.array([np.eye(WIDE, dtype=np.float32)[0]]),
                    camera="cam-Z",
                    detections=FakeDetections(scores=[0.9]),
                )
            ).meta["identities"]
        finally:
            element.close()

        assert survivors == ["ship-0", "ship-1", "ship-2", "ship-3"]
        assert still_known[0][0] == "ship-0", "and the vector behind the name is still its own"

    def test_the_bound_enrolment_runs_into_is_the_one_it_filled_itself(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The other half: bounded still, and bounded where the growth actually is.

        Forty mints into a store of eight leaves eight — the eviction happened, it was the
        library's, and every entry it discarded was one this element had minted. The counter
        says forty because forty rows really were enrolled; a deployment reading
        ``enrolments_total`` against ``len`` is reading turnover, which is the number that
        tells it ``enrol_capacity`` is too small.
        """
        curated_fleet(tmp_path)
        registry = MetricsRegistry()
        element = build(
            gallery_dir=str(tmp_path),
            enrol_capacity=8,
            dim=WIDE,
            threshold=0.5,
            enrol=True,
            enrol_min_confidence=0.5,
        )
        element.open(ElementContext(metrics=registry))

        try:
            for frame in range(40):
                element.process(
                    item(
                        np.array([np.eye(WIDE, dtype=np.float32)[4 + frame]]),
                        camera="cam-A",
                        frame=frame,
                        detections=FakeDetections(scores=[0.9]),
                    )
                )
            minted = len(element._enrolled)
            curated = len(element._gallery)
        finally:
            element.close()

        assert minted == 8, "the minted store held its own bound"
        assert curated == 4, "and nothing was taken out of the curated one to make room"
        enrolments = {metric.name: metric for metric in registry.collect()}[
            "shipinfer_recognize_enrolments_total"
        ]
        assert enrolments.total() == 40

    def test_a_camera_does_not_re_enrol_the_ship_it_just_enrolled(
        self, reid: Any, tmp_path: Path
    ) -> None:
        """The one deliberate hole in rule 1, pinned on its own rather than as a side effect.

        Two questions, two different answers to ``exclude_camera``. "Which identity may I
        publish for this row?" excludes the row's own camera, because a same-camera match
        measures the tracker. "Is this appearance in the gallery at all?" must **not**,
        because a camera can never match what it itself enrolled — so with the exclusion in
        place the same ship is enrolled again on every frame it is seen, and a gallery that
        grows per frame is the memory leak ``BaseGallery`` warns about.

        Ten frames of one ship on one camera: one enrolment, and the published answer stays
        ``None`` throughout, because the identity exists and this camera may not claim it.
        Give the membership check the exclusion back and this is ten enrolments.
        """
        curated_fleet(tmp_path)
        registry = MetricsRegistry()
        element = build(
            gallery_dir=str(tmp_path),
            dim=WIDE,
            threshold=0.5,
            enrol=True,
            enrol_min_confidence=0.5,
        )
        element.open(ElementContext(metrics=registry))
        stranger = np.eye(WIDE, dtype=np.float32)[9]

        try:
            filed = [
                element.process(
                    item(
                        np.array([stranger]),
                        camera="cam-A",
                        frame=frame,
                        detections=FakeDetections(scores=[0.9]),
                    )
                ).meta["identities"]
                for frame in range(10)
            ]
            minted = len(element._enrolled)
        finally:
            element.close()

        assert minted == 1, "ten frames of one ship are one identity, not ten"
        assert filed[0] == {0: ("auto:cam-A:0:0", None)}, "the frame that minted it"
        assert all(
            answer == {0: (None, None)} for answer in filed[1:]
        ), "its own camera may not claim what it enrolled — the published answer stays None"

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

        assert filed == {0: (None, None)}
        assert later == {0: (None, None)}, "the blurred crop did not become an identity"

    def test_a_row_with_no_confidence_is_not_enrolled_and_says_so_once(
        self, reid: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A chain that embeds with no decoding detector ahead of it enrols nobody.

        ``meta["detections"]`` is C3's and is filed by ``PoolDetect`` today, so this is the
        chain that does not have one — a fixed-crop source, a test rig — rather than a
        temporary state of the branch. A row whose quality is unknown is not enrolled,
        because the safe answer is the one that changes nothing.

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
            assert len(element._enrolled) == 0
        finally:
            element.close()

        warned = [r for r in caplog.records if "no per-row confidence" in r.getMessage()]
        assert len(warned) == 1

    def test_enrolment_stays_inside_the_enrolled_store_s_capacity(self, reid: Any) -> None:
        """The bound is the library's and this test is what says the element does not defeat it.

        A gallery that grows without one is a memory leak with a plausible name
        (``shipvision.reid.gallery.base``). ``enrol_capacity:`` is the knob that bounds the
        minted store specifically, so that a deployment can hold ten thousand curated ships
        and still cap what the server teaches itself.
        """
        element = build(
            gallery="flat", enrol_capacity=2, dim=DIM, enrol=True, enrol_min_confidence=0.5
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
            assert len(element._enrolled) == 2
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
                  embed_ship: {impl: mock, params: {classes: [ship]}}
                  recognize:  {impl: shipvision, params: {classes: [ship]}, after: embed_ship}
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

        Pinned because a concrete ``produces`` here would quietly stop that being true, which
        a reviewer should be told by a red test rather than by reading this file.
        """
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in self.chain().edges}

        assert caps[("embed_ship", "recognize")] == caps[("recognize", "output")]

    def test_row_selection_is_params_classes_and_not_a_guard(self) -> None:
        """Which rows are queried is ``params: {classes: [...]}``, over C3's labels.

        ``when:`` guards the *element*, once per item, on ``meta["class"]`` — which nothing
        in the chain sets, so a slot that tried to select rows with it would select none.
        The two are not interchangeable and only one of them is per row.
        """
        node = self.chain().node("recognize")

        assert node.element.params["classes"] == ["ship"]
        assert node.admits(item(np.zeros((1, DIM), np.float32))) is True
