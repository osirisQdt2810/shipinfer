"""Reading an enrolled gallery off disk — shipinfer's half of a shipvision gallery.

``shipvision.reid`` owns the gallery: the bounded matrix, the gemm, the same-camera
exclusion protocol. What it deliberately does **not** own is persistence — it has no
``save`` and no ``load``, because *where a deployment keeps its enrolled identities* is a
deployment question and the library refuses to answer it. That half is this module, and it
lives here rather than in ``3rdparty/`` for the same reason ADR-006 gives for the model
repository: the on-disk layout is shipinfer's concept.

**The layout is the model repository's, unchanged (ADR-006).** A gallery is an entry beside
the models it is fed by::

    model_repository/
      ship_embedder/       config.yaml  1/model.plan
      ship_gallery/                     1/gallery.npz     <- one of these

so the same directory an operator already rsyncs carries the identities, the same
"highest numbered version directory wins" rule applies, and rolling a gallery back is
``2/`` → ``1/`` rather than a file swap. There is no ``config.yaml``: a gallery is not a
model, nothing loads it into an instance group, and
:class:`~shipinfer.repository.ModelRepository` skips a directory that has none with a log
line rather than refusing the repository.

**Why ``.npz`` and not YAML or JSON.** The payload is an ``(N, d)`` float32 matrix — a
thousand ships at 512 floats is 2 MB of numbers, which is a 20 MB text file that parses in
seconds and does not round-trip exactly. ``numpy`` is already a dependency of every pure
layer, ``np.savez`` is one call, and the archive keeps the array's dtype and shape instead
of making the loader guess them. It is read with ``allow_pickle=False``, always: a ``.npz``
is a zip, a pickled object array inside one is arbitrary code, and a model repository is a
directory an operator syncs from somewhere else.

**Everything is validated here, once, at load.** The refusals below all name the file and
the row, because the alternative is a gallery that accepts a malformed vector and answers
plausibly: a non-finite value propagates through the similarity matrix into *every* score
(``shipvision.reid.distance.normalize`` says so), and a zero row sits at cosine 0 from
everything, which is a plausible answer to every query. Both would surface as "recognition
got worse", weeks later, with no error anywhere.

The module imports numpy and nothing else from outside ``core`` — ``topology`` is a pure
layer (ADR-017), it may not import :mod:`shipinfer.repository`, and it must not import
shipvision at module scope. So this reads *files* and returns *arrays*; turning those into
``Embedding`` objects and into a gallery is
:mod:`shipinfer.topology.elements.recognize`'s job, inside ``_do_open``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "GALLERY_FILE",
    "GalleryFile",
    "load_gallery_file",
    "resolve_gallery_path",
]

#: The artefact's name inside a version directory. Fixed rather than configurable: a
#: repository entry is found by convention in this project (``config.yaml``, ``model.plan``),
#: and a second spelling would mean two ways to have a gallery that is not found.
GALLERY_FILE = "gallery.npz"

#: A version directory, exactly as :mod:`shipinfer.repository.model_repository` matches one.
#: Restated rather than imported because ``topology`` may not import ``repository``; it is
#: five characters of regex and the alternative is a layering hole.
_VERSION_DIR = re.compile(r"^\d+$")

#: Array names inside the archive. ``camera_ids`` is optional; the other two are not.
_VECTORS = "vectors"
_IDENTITIES = "identities"
_CAMERA_IDS = "camera_ids"


@dataclass(frozen=True, slots=True)
class GalleryFile:
    """One validated gallery archive: parallel arrays, ready to be added to a gallery.

    Parallel arrays and not a list of objects, for the reason
    ``pipeline/graph/detections.py`` gives: everything downstream is batched, and the one
    thing that must never be lost is the *alignment* between a vector and its label.

    Args:
        vectors: ``(N, d)`` float32, C-contiguous. **Not** normalised — the gallery
            normalises on ``add`` and doing it twice would be a second opinion about the
            same invariant.
        identities: one label per row, in row order.
        camera_ids: the camera each row was seen from, or ``None`` where the file did not
            say. Load-bearing rather than decoration: it is what
            ``query(exclude_camera=...)`` matches against, so a row with no camera can never
            be excluded from a query and will flatter a same-camera match.
        path: where this came from, so a refusal one layer up can name it.
    """

    vectors: np.ndarray
    identities: tuple[str, ...]
    camera_ids: tuple[str | None, ...]
    path: Path

    @property
    def dim(self) -> int:
        """The embedding width every row carries. Known even for an empty file."""
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def __repr__(self) -> str:
        distinct = len(set(self.identities))
        return f"<GalleryFile rows={len(self)} identities={distinct} dim={self.dim}>"


def resolve_gallery_path(root: Path | str, name: str, *, version: int | None = None) -> Path:
    """Find ``<root>/<name>/<version>/gallery.npz``, newest version by default.

    Args:
        root: the model repository directory. The *same* one the models are in — a gallery
            is an entry beside them (ADR-006), not a second repository to configure.
        name: the entry's directory name.
        version: which version directory, or ``None`` for the highest, which is what
            Triton's default ``version_policy: latest`` means and what
            :class:`~shipinfer.repository.ModelRepository` already does for a model.

    Returns:
        The path to the archive. Existence is checked here, so a caller that got a path
        back has a file to read.

    Raises:
        ConfigurationError: at every step, naming what was looked for, what was found in
            its place, and the layout that would have worked. A gallery that silently
            resolves to nothing is a deployment that recognises nobody and says so nowhere,
            which is the whole failure this project was started over.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise ConfigurationError(
            f"gallery repository {root_path} is not a directory; `gallery_dir:` names the "
            "model repository the gallery entry lives in"
        )
    entry = root_path / name
    if not entry.is_dir():
        present = sorted(p.name for p in root_path.iterdir() if p.is_dir())
        raise ConfigurationError(
            f"no gallery entry {name!r} in {root_path} (entries: {present}); expected "
            f"{entry / '1' / GALLERY_FILE}"
        )

    found = sorted(
        int(p.name) for p in entry.iterdir() if p.is_dir() and _VERSION_DIR.match(p.name)
    )
    if not found:
        raise ConfigurationError(
            f"{entry}: no numbered version directory (expected e.g. {entry / '1'})"
        )
    if version is None:
        chosen = found[-1]
    elif version in found:
        chosen = version
    else:
        raise ConfigurationError(f"{entry}: no version {version} (available: {found})")

    path = entry / str(chosen) / GALLERY_FILE
    if not path.is_file():
        present = sorted(p.name for p in path.parent.iterdir())
        raise ConfigurationError(
            f"{path} not found (present: {present}); a gallery entry is a version "
            f"directory containing {GALLERY_FILE!r}"
        )
    return path


def load_gallery_file(path: Path | str) -> GalleryFile:
    """Read and validate one ``gallery.npz``.

    The archive holds ``vectors`` ``(N, d)`` float, ``identities`` ``(N,)`` text and,
    optionally, ``camera_ids`` ``(N,)`` text. An empty string in ``camera_ids`` means "the
    file did not say", because a fixed-width numpy text array has no ``None``.

    Written by whatever enrolled the identities — a notebook, an offline batch job,
    ``np.savez(path, vectors=v, identities=ids, camera_ids=cams)``. There is deliberately no
    writer here yet: nothing in the server enrols to disk (see
    :class:`~shipinfer.topology.elements.recognize.GalleryRecognize` on what ``enrol: true``
    does and does not persist), and a save path with no caller is a format with no test.

    Returns:
        The validated arrays. An **empty** file (``N == 0``) is accepted and is not an
        error: a deployment that has not enrolled anyone yet is an ordinary state, and the
        element says so with one warning rather than refusing to start.

    Raises:
        ConfigurationError: the file is unreadable, is missing an array, or holds one whose
            shape, dtype or values cannot be an embedding. Every message names the path,
            and the value ones name the row.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigurationError(f"gallery file {file_path} does not exist")
    try:
        with np.load(file_path, allow_pickle=False) as archive:
            names = list(archive.files)
            missing = [key for key in (_VECTORS, _IDENTITIES) if key not in names]
            if missing:
                raise ConfigurationError(
                    f"{file_path}: no {missing[0]!r} array (found: {sorted(names)}); a "
                    f"gallery archive holds {_VECTORS!r} (N, d) and {_IDENTITIES!r} (N,)"
                )
            vectors = archive[_VECTORS]
            identities = archive[_IDENTITIES]
            cameras = archive[_CAMERA_IDS] if _CAMERA_IDS in names else None
    except (OSError, ValueError) as exc:
        # `allow_pickle=False` turns an object array into a ValueError, and a truncated or
        # non-zip file into an OSError. Both mean "this is not a gallery archive", and the
        # operator needs the path far more than the numpy traceback.
        raise ConfigurationError(f"{file_path} is not a readable .npz archive ({exc})") from exc

    vectors = _validated_vectors(vectors, file_path)
    rows = int(vectors.shape[0])
    return GalleryFile(
        vectors=vectors,
        identities=_validated_labels(identities, rows, file_path),
        camera_ids=_validated_cameras(cameras, rows, file_path),
        path=file_path,
    )


def _validated_vectors(vectors: np.ndarray, path: Path) -> np.ndarray:
    """The ``(N, d)`` float32 matrix, or a refusal that names the row that broke it."""
    if vectors.ndim != 2:
        raise ConfigurationError(
            f"{path}: {_VECTORS!r} must be (N, d), got shape {vectors.shape}"
        )
    if not np.issubdtype(vectors.dtype, np.floating):
        raise ConfigurationError(
            f"{path}: {_VECTORS!r} has dtype {vectors.dtype}, and an embedding is float; an "
            "integer array here is a packing mistake, not a quantised gallery"
        )
    if vectors.shape[1] == 0:
        raise ConfigurationError(f"{path}: {_VECTORS!r} is zero-width, so it holds no vectors")

    finite = np.isfinite(vectors).all(axis=1)
    if not finite.all():
        raise ConfigurationError(
            f"{path}: row {int(np.argmin(finite))} of {_VECTORS!r} is not finite. One NaN "
            "propagates through the similarity matrix into every score, so the whole "
            "gallery would answer plausibly and wrongly"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise ConfigurationError(
            f"{path}: row {int(np.argmin(norms))} of {_VECTORS!r} is all zeros. It has no "
            "direction: it would sit at cosine 0 from every query, which is a plausible "
            "answer to all of them"
        )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _decoded_text(array: np.ndarray, rows: int, key: str, path: Path) -> tuple[str, ...]:
    """One text cell per row, decoded to ``str``. The shape and dtype half, shared.

    Text and not numbers, for both columns: an identity and a camera id both cross Kafka and
    a log line as strings, and a gallery whose labels arrive as ``int64`` would publish
    ``np.int64(7)`` where the rest of the system publishes ``"ship-7"``.
    """
    if array.ndim != 1 or array.shape[0] != rows:
        raise ConfigurationError(
            f"{path}: {key!r} is {array.shape} for {rows} vector(s); the arrays are "
            "parallel and a mismatch is how a vector ends up under another ship's name"
        )
    if array.dtype.kind not in ("U", "S"):
        raise ConfigurationError(
            f"{path}: {key!r} has dtype {array.dtype}, and {key} is text; write a numeric "
            "id as a string so it survives the wire unchanged"
        )
    return tuple(
        value.decode() if isinstance(value, bytes) else str(value) for value in array.tolist()
    )


def _validated_labels(labels: np.ndarray, rows: int, path: Path) -> tuple[str, ...]:
    """One **non-empty** identity per row."""
    decoded = _decoded_text(labels, rows, _IDENTITIES, path)
    for index, value in enumerate(decoded):
        if not value:
            raise ConfigurationError(
                f"{path}: {_IDENTITIES!r} row {index} is empty, and an unnamed row can only "
                "be published as an identity nobody can look up"
            )
    return decoded


def _validated_cameras(
    cameras: np.ndarray | None, rows: int, path: Path
) -> tuple[str | None, ...]:
    """The optional camera column. Absent means every row is ``None``, and that is legal.

    Legal but weakening, which is why the element warns about it rather than this function
    refusing: ``exclude_camera`` cannot exclude a row that does not say where it came from,
    so such a row can be matched by the very camera that produced it — the self-match the
    protocol exists to prevent (``shipvision.reid.gallery.base``). An empty cell means the
    same thing, because a fixed-width numpy text array has no ``None`` to write.
    """
    if cameras is None:
        return (None,) * rows
    return tuple(value or None for value in _decoded_text(cameras, rows, _CAMERA_IDS, path))
