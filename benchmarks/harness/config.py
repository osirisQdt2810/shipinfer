"""One configuration object, shared by both systems under test.

The two systems are wired very differently — the baseline is a C++ binary with two
independent single-model pipelines, ShipInfer is a Python server running a five-stage DAG —
but a comparison is only a comparison if the *load* is identical. So the load lives here,
once, and each driver translates it into its own vocabulary:

======================  ==================================  ==============================
:class:`BenchConfig`    ``sim_pipeline_v2``                 ShipInfer
======================  ==================================  ==============================
``cameras``             ``--num-{det,seg}-source-workers``  ``ingest.cameras``, half on
                        (half each)                         person frames, half on ship
``fps``                 ``--{det,seg}-source-fps``          ``CameraConfig.fps``
``gpus``                ``--gpu-ids``                       ``devices.visible_gpus``
``batch``               ``--{det,seg}-batch-size``          the engine's ``max_batch_size``
``buffer_capacity``     ``--{det,seg}-buffer-capacity``     ``pipeline.queue_capacity``
``workers_per_module``  ``--num-{det,seg}-workers``         ``instance_group.count`` in the
                                                            model's own ``config.yaml``
======================  ==================================  ==============================

Two of those translations are inexact and the harness says so rather than papering over
them, because they bound what the final number means:

* **``batch`` is advisory for ShipInfer.** A TensorRT plan carries its own batch dimension,
  so the server's batch size is a property of the artefact, not of this config. The field is
  kept because it is what the baseline is *told*, and both must be told the same thing.
* **``workers_per_module`` is advisory for ShipInfer.** The baseline spawns exactly that many
  inference threads per module; ShipInfer's instance count comes from each model's
  ``config.yaml`` so that one repository runs unchanged on a 4-GPU box and a 16-GPU node
  (ADR-006). The harness records what it actually expanded to.

``cameras`` is split in half on purpose. The baseline runs person frames through the
detector and ship frames through the segmenter as two unrelated streams, so at 50 workers it
offers 25x20 = 500 img/s to each. ShipInfer runs one DAG over all 50 cameras, so it offers
1000 frames/s to the detector and a data-dependent number of crops downstream. Same total
offered load, different distribution — and :mod:`benchmarks.harness.analysis` reports per
module precisely because those two are not the same experiment reduced to one number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["BenchConfig", "Resolution"]

#: The frame sets that ship with ``benchmarks/baseline``. Both are 1920x1080 JPEGs; ``4k``
#: is the same content upscaled by the baseline's own ``change_image_resolution.py``.
Resolution = str

_RESOLUTION_FOLDERS: dict[str, tuple[str, str]] = {
    "2k": ("person_2K", "ship_2K"),
    "4k": ("person_4K", "ship_4K"),
}


def _repo_root() -> Path:
    """The repository root, whether the harness runs from the host or from ``/work``."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """The load, the hardware and the artefacts — everything a run needs to be repeatable.

    ``seconds`` and ``warmup_s`` are separate because the first ten seconds of either system
    are start-up, not steady state: the baseline's 50 source threads stagger their first
    decode and its four TensorRT runners deserialise their engines while the sources are
    already producing. Fitting a growth rate through that ramp measures the ramp. See
    :func:`benchmarks.harness.analysis.analyse`.
    """

    #: Total source workers across both data streams. The baseline gets ``cameras // 2`` per
    #: module; ShipInfer gets ``cameras`` replay cameras over one DAG.
    cameras: int = 50
    #: Frames per second *per* source worker.
    fps: float = 20.0
    #: Physical CUDA device ordinals. Four on this box, because that is what is usable.
    gpus: tuple[int, ...] = (2, 3, 4, 5)
    #: Batch the inference workers are asked to assemble.
    batch: int = 8
    #: Wall seconds to hold the load, including ``warmup_s``.
    seconds: float = 70.0
    #: Leading seconds discarded before the growth fit. 10 s covers engine deserialisation
    #: and the source threads' stagger on both systems.
    warmup_s: float = 10.0
    #: How often the buffer occupancy is sampled. 1.0 s is the baseline's own cadence and
    #: changing it would make the two logs incomparable.
    sample_interval_s: float = 1.0
    #: Inference worker threads per module, for the baseline. Advisory for ShipInfer — see
    #: the module docstring.
    workers_per_module: int = 4
    #: Buffer bound. The baseline's default is 65536, deep enough that a saturated 70 s run
    #: accumulates without ever hitting the cap, which is what makes the growth rate rather
    #: than the drop rate the thing being measured.
    buffer_capacity: int = 65536
    #: Frames ShipInfer may have in flight through the DAG at once. This is the analogue of
    #: the baseline's inference-thread count and it has to be large: a pipeline worker blocks
    #: on every stage, so filling one detector batch of 8 needs at least 8 workers and
    #: keeping eight detector instances busy needs an order of magnitude more.
    pipeline_workers: int = 96
    #: ``2k`` or ``4k``.
    resolution: Resolution = "2k"

    #: Where the frames come from. Defaults resolve under ``benchmarks/baseline/data``.
    person_frames: Path | None = None
    ship_frames: Path | None = None
    #: The plans the baseline binary loads. ShipInfer loads its own from
    #: ``model_repository/<name>/1/model.plan``, built from the same ONNX.
    det_engine: Path | None = None
    seg_engine: Path | None = None
    model_repository: Path | None = None
    #: Where JSONL logs, console captures and ``summary.json`` land.
    out_dir: Path = field(default_factory=lambda: _repo_root() / ".artifacts" / "bench")

    # -- derived ------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.cameras < 2:
            raise ValueError("cameras must be >= 2: the baseline splits them between modules")
        if self.cameras % 2:
            raise ValueError(
                f"cameras must be even so the two baseline streams get the same offered "
                f"rate; got {self.cameras}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not self.gpus:
            raise ValueError("at least one GPU is needed")
        if self.resolution not in _RESOLUTION_FOLDERS:
            raise ValueError(
                f"resolution must be one of {sorted(_RESOLUTION_FOLDERS)}, "
                f"got {self.resolution!r}"
            )
        if self.warmup_s >= self.seconds:
            raise ValueError(
                f"warmup_s={self.warmup_s} leaves no steady tail in a {self.seconds}s run"
            )

    @property
    def sources_per_module(self) -> int:
        """Source workers feeding one baseline module."""
        return self.cameras // 2

    @property
    def offered_per_module(self) -> float:
        """Images per second offered to one baseline module."""
        return self.sources_per_module * self.fps

    @property
    def offered_total(self) -> float:
        """Images per second offered to the whole system, on either side."""
        return self.cameras * self.fps

    @property
    def steady_seconds(self) -> float:
        return self.seconds - self.warmup_s

    # -- path resolution ----------------------------------------------------------------

    def resolved(self) -> BenchConfig:
        """Fill in every ``None`` path from the repository layout. Idempotent.

        Kept separate from ``__post_init__`` so a config can be constructed and inspected on
        a machine where the artefacts are absent — a test does exactly that.
        """
        root = _repo_root()
        person_dir, ship_dir = _RESOLUTION_FOLDERS[self.resolution]
        data = root / "benchmarks" / "baseline" / "data"
        return replace(
            self,
            person_frames=self.person_frames or data / person_dir,
            ship_frames=self.ship_frames or data / ship_dir,
            det_engine=self.det_engine or root / "models" / "yolo26n_fp32.engine",
            seg_engine=self.seg_engine or root / "models" / "yolo26n-seg_fp32.engine",
            model_repository=self.model_repository or root / "model_repository",
        )

    def require_inputs(self) -> None:
        """Fail before a run rather than after 70 s of measuring nothing.

        Raises:
            FileNotFoundError: naming the missing artefact and how to produce it.
        """
        resolved = self.resolved()
        for label, path, remedy in (
            (
                "person frames",
                resolved.person_frames,
                "git submodule update --init benchmarks/baseline",
            ),
            (
                "ship frames",
                resolved.ship_frames,
                "git submodule update --init benchmarks/baseline",
            ),
            ("detector engine", resolved.det_engine, "scripts/build_engines.py"),
            ("segmenter engine", resolved.seg_engine, "scripts/build_engines.py"),
            ("model repository", resolved.model_repository, "scripts/build_engines.py"),
        ):
            assert path is not None  # resolved() filled it
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} missing at {path} — produce it with `{remedy}`"
                )

    # -- reporting ----------------------------------------------------------------------

    def cuda_visible_devices(self) -> str:
        """``CUDA_VISIBLE_DEVICES`` for a run restricted to :attr:`gpus`.

        ShipInfer indexes devices 0..n-1 *inside* that restriction, which is the only way to
        pin it to four of eight GPUs without teaching every layer about physical ordinals.
        The baseline takes physical ids on the command line instead, so its driver passes
        ``0,1,2,3`` once this variable is set. Both end up on the same silicon; the harness
        records which mapping each one used.
        """
        return ",".join(str(g) for g in self.gpus)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe, and the thing written into every log's metadata line."""
        resolved = self.resolved()
        return {
            "cameras": self.cameras,
            "fps": self.fps,
            "gpus": list(self.gpus),
            "batch": self.batch,
            "seconds": self.seconds,
            "warmup_s": self.warmup_s,
            "sample_interval_s": self.sample_interval_s,
            "workers_per_module": self.workers_per_module,
            "buffer_capacity": self.buffer_capacity,
            "pipeline_workers": self.pipeline_workers,
            "resolution": self.resolution,
            "sources_per_module": self.sources_per_module,
            "offered_per_module": self.offered_per_module,
            "offered_total": self.offered_total,
            "person_frames": str(resolved.person_frames),
            "ship_frames": str(resolved.ship_frames),
            "det_engine": str(resolved.det_engine),
            "seg_engine": str(resolved.seg_engine),
            "model_repository": str(resolved.model_repository),
            "in_container": Path("/.dockerenv").exists(),
            "hostname": os.uname().nodename,
        }
