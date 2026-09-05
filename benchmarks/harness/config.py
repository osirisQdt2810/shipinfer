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
``instances_per_gpu``   ``--num-{det,seg}-workers``         ``instance_group.count`` in the
                        (x the GPU count)                   model's own ``config.yaml``
======================  ==================================  ==============================

Two of those translations are inexact and the harness says so rather than papering over
them, because they bound what the final number means:

* **``batch`` is advisory for ShipInfer.** A TensorRT plan carries its own batch dimension,
  so the server's batch size is a property of the artefact, not of this config. The field is
  kept because it is what the baseline is *told*, and both must be told the same thing.
* **``instances_per_gpu`` is READ from the repository**, no longer kept in step by hand. Our
  count comes from each model's ``config.yaml`` (ADR-006) and the baseline takes a *total* on
  its command line, so this field is the translation — and it is the one number here that can
  silently make the comparison unfair. It has been wrong twice: ``4`` total across four GPUs
  against our ``count: 2`` per GPU, and later ``seg: 1`` against a repository saying ``2``.
  ``concurrency_note`` prints both figures so a reader can check rather than trust.

``cameras`` is split in half on purpose. The baseline runs person frames through the
detector and ship frames through the segmenter as two unrelated streams, so at 50 workers it
offers 25x20 = 500 img/s to each. ShipInfer runs one DAG over all 50 cameras, so it offers
1000 frames/s to the detector and a data-dependent number of crops downstream. Same total
offered load, different distribution — and :mod:`benchmarks.harness.analysis` reports per
module precisely because those two are not the same experiment reduced to one number.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

__all__ = ["MODULE_MODELS", "BenchConfig", "Resolution", "read_instances_per_gpu"]

#: The frame sets that ship with ``benchmarks/baseline``. Both are 1920x1080 JPEGs; ``4k``
#: is the same content upscaled by the baseline's own ``change_image_resolution.py``.
Resolution = str

#: The shapes the ShipInfer side can take; see :attr:`BenchConfig.topology`.
TOPOLOGIES = ("single", "fleet", "service")

_RESOLUTION_FOLDERS: dict[str, tuple[str, str]] = {
    "2k": ("person_2K", "ship_2K"),
    "4k": ("person_4K", "ship_4K"),
}


def _digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks — a plan is 100+ MB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    """The repository root, whether the harness runs from the host or from ``/work``."""
    return Path(__file__).resolve().parents[2]


#: The baseline's module vocabulary -> the repository model each one stands for. The baseline
#: knows two stages; we run four, and only these two are the ones it is given threads for.
MODULE_MODELS: Mapping[str, str] = {"det": "ship_detector", "seg": "ship_segmenter"}


def read_instances_per_gpu(repository: Path) -> dict[str, int]:
    """Each baseline module's per-GPU instance count, READ FROM THE REPOSITORY.

    `instance_groups` is where the number is decided (ADR-006: one repository runs unchanged
    on a 4-GPU box and a 16-GPU node), the baseline takes a total on its command line, and
    this file's header calls the translation "the one number in this file that can silently
    make the comparison unfair". It was kept in step by hand and drifted -- so it is read.

    `repository/resolved.py` is the reader, the same one `shipinfer plan` uses, and it works
    on a driverless box because an instance COUNT is config rather than hardware.

    REFUSES rather than substituting a number: "no repository", "one that will not parse" and
    "one that says 2" are three different events, and a *plausible* wrong concurrency is the
    whole subject here.

    Raises:
        ConfigurationError: unreadable, or a module's model declares no device instances.
    """
    from shipinfer.core.errors import ConfigurationError
    from shipinfer.repository import ModelRepository
    from shipinfer.repository.resolved import model_runtimes

    try:
        runtimes = model_runtimes(ModelRepository.load(repository))
    except Exception as error:
        raise ConfigurationError(
            f"cannot read {repository} for the baseline's per-GPU instance counts ({error}). "
            f"That number decides how much concurrency the baseline is given, so guessing it "
            f"is how the comparison became unfair the last two times"
        ) from None
    missing = [
        model
        for model in MODULE_MODELS.values()
        if model not in runtimes or runtimes[model].instances is None
    ]
    if missing:
        raise ConfigurationError(
            f"{missing} declare no device instances in {repository}; the baseline's worker "
            f"count is not derivable from a repository that asks for none"
        )
    return {module: runtimes[model].instances for module, model in MODULE_MODELS.items()}


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
    # doc: long the field is empty by default and that needs its reason
    #: Inference instances **per GPU**, per module: how the baseline is given the same
    #: concurrency we give ourselves. Not a tuning knob.
    #:
    #: EMPTY BY DEFAULT, and `resolved()` fills it from `model_repository/*/config.yaml`. The
    #: literal it replaced said `seg: 1` while the repository said `count: 2`, so on a 7-GPU
    #: box the baseline got seven segmenter threads where we ran fourteen.
    #:
    #: Set it explicitly to override; `resolved()` leaves a non-empty mapping alone.
    instances_per_gpu: Mapping[str, int] = field(default_factory=dict)
    #: OpenMP threads, applied to **both** systems or to neither. Previously pinned to 1 on
    #: the baseline process only, while our torch pre-processing had the whole box — and the
    #: baseline letterboxes on the CPU inside those same threads, so the pin plausibly *was*
    #: the wall the 854.7 img/s measurement hit. ``None`` leaves both unpinned.
    omp_threads: int | None = None
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

    #: Where frames come from: ``replay`` decodes JPEGs from disk, ``rtsp`` pulls H.264 over a
    #: real socket from ``scripts/rtsp_serve.py``.
    #:
    #: R55 makes RTSP mandatory for the benchmark and not only for the tests, and the reason
    #: is not pedantry: replay measures the inference plane with the decode path *removed*.
    #: The deployment reads fifty RTSP cameras, so NVDEC, the jitter buffer, reconnects and
    #: the NV12 path are all part of the system's real cost and none of them appear in a
    #: replay run. A replay number is therefore an upper bound on the RTSP one, and the two
    #: must never be compared as though they measured the same thing — which is why the source
    #: is recorded in the run's metadata rather than left implicit.
    source: str = "replay"
    #: The port `rtsp_serve` listens on when ``source == "rtsp"``.
    rtsp_port: int = 8554
    #: How the ShipInfer side is laid out across processes. ``single`` is one process over
    #: every GPU (the in-process server this harness always ran); ``fleet`` and ``service`` are
    #: one process per shard through the real launcher, the latter with the cross-process tier
    #: for the crop-stage models (T3). The baseline is unaffected: it has one shape.
    topology: str = "single"
    #: Shard processes under ``fleet``/``service``; 0 means one per GPU.
    shards: int = 0
    #: An explicit cameras-per-shard split, in camera order, one GPU per shard. Empty means the
    #: launcher's own plan (LPT by offered fps, which is balanced by construction). The split is
    #: how the harness models *the plan was right when it was made and the crowd moved*: a
    #: crowded shard next to quiet ones, which is the case `fleet` cannot fix and `service`
    #: exists for.
    shard_cameras: tuple[int, ...] = ()
    #: This process's slice of the cameras, by id. Empty means all of them. A shard child is
    #: given its slice here and offers exactly ``len(camera_ids) x fps``.
    camera_ids: tuple[str, ...] = ()

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
        if self.topology not in TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {sorted(TOPOLOGIES)}, got {self.topology!r}"
            )
        if self.shards < 0:
            raise ValueError(f"shards must be >= 0, got {self.shards}")
        if self.shard_cameras:
            if self.topology == "single":
                raise ValueError(
                    "shard_cameras needs a multi-process topology (fleet or service)"
                )
            if any(n <= 0 for n in self.shard_cameras):
                raise ValueError(
                    f"every shard needs at least one camera, got {self.shard_cameras}"
                )
            if sum(self.shard_cameras) != self.cameras:
                raise ValueError(
                    f"shard_cameras {self.shard_cameras} sums to {sum(self.shard_cameras)}, "
                    f"not the {self.cameras} cameras of the run"
                )
            if self.shards and len(self.shard_cameras) != self.shards:
                raise ValueError(
                    f"shard_cameras names {len(self.shard_cameras)} shards but shards={self.shards}"
                )
            if len(self.shard_cameras) > len(self.gpus):
                raise ValueError(
                    f"{len(self.shard_cameras)} shards on {len(self.gpus)} GPUs: an explicit split "
                    f"gives each shard one GPU"
                )

    def module_instances(self) -> Mapping[str, int]:
        """``instances_per_gpu``, filled from the repository if the caller left it empty.

        Not only in `resolved()`: an unresolved config still has to answer this, and a
        `concurrency_note` that printed nothing would remove the line the reader checks the
        fairness with -- worse than the stale literal it replaced.
        """
        if self.instances_per_gpu:
            return self.instances_per_gpu
        repository = self.model_repository or _repo_root() / "model_repository"
        return read_instances_per_gpu(repository)

    def workers_for(self, module: str) -> int:
        """Baseline inference threads for one module: instances per GPU x GPUs.

        The baseline takes a total, we configure per GPU, and the comparison is only fair
        if the two describe the same machine.
        """
        return max(1, self.module_instances().get(module, 1) * len(self.gpus))

    @property
    def concurrency_note(self) -> str:
        """One line for the report, so the reader can check the fairness themselves."""
        # Read ONCE and used for both halves: `workers_for` re-enters `module_instances()`,
        # so the two numbers on this line came from two independent reads of the same file.
        instances = self.module_instances()
        gpus = len(self.gpus)
        per_gpu = ", ".join(f"{k}={v}/gpu" for k, v in sorted(instances.items()))
        total = ", ".join(f"{k}={max(1, v * gpus)}" for k, v in sorted(instances.items()))
        pin = "unpinned" if self.omp_threads is None else f"OMP_NUM_THREADS={self.omp_threads}"
        return f"concurrency: {per_gpu} (baseline totals {total}); both sides {pin}"

    @property
    def sources_per_module(self) -> int:
        """Source workers feeding one baseline module."""
        return self.cameras // 2

    @property
    def offered_per_module(self) -> float:
        """Images per second offered to one baseline module."""
        return self.sources_per_module * self.fps

    @property
    def camera_count(self) -> int:
        """Cameras *this process* drives: its slice under a sharded run, all of them otherwise."""
        return len(self.camera_ids) if self.camera_ids else self.cameras

    @property
    def offered_total(self) -> float:
        """Images per second offered to the whole system, on either side — or, for a shard
        child, to this process: its slice of the cameras times the frame rate."""
        return self.camera_count * self.fps

    @property
    def steady_seconds(self) -> float:
        return self.seconds - self.warmup_s

    def at_offer(self, multiplier: float) -> BenchConfig:
        """The same run at ``multiplier`` times the offered rate — one rung of a sweep.

        **Scaled by fps, not by camera count.** The number of cameras is the topology: it
        sets how many source workers each side runs, how the baseline's two queues are fed,
        and how the fair queue's lanes are populated, so changing it changes the experiment
        rather than the load. Changing the per-camera frame rate leaves all of that fixed
        and moves only the thing being swept.

        ``out_dir`` is left alone; the sweep gives each rung its own subdirectory, so the
        occupancy logs of two rungs cannot land on top of each other.
        """
        if multiplier <= 0:
            raise ValueError(f"a sweep rung must offer something: got x{multiplier}")
        return replace(self, fps=self.fps * multiplier)

    # -- path resolution ----------------------------------------------------------------

    def resolved(self) -> BenchConfig:
        """Fill in every ``None`` path from the repository layout. Idempotent.

        Kept separate from ``__post_init__`` so a config can be constructed and inspected on
        a machine where the artefacts are absent — a test does exactly that.
        """
        root = _repo_root()
        person_dir, ship_dir = _RESOLUTION_FOLDERS[self.resolution]
        data = root / "benchmarks" / "baseline" / "data"
        repository = self.model_repository or root / "model_repository"
        # TOLERANT here, strict at the point of use: `resolved()` is documented as usable
        # where the artefacts are absent, so it leaves the field EMPTY and
        # `module_instances()` refuses when somebody needs the number. What never happens
        # either way is a plausible made-up count.
        try:
            resolved_instances = read_instances_per_gpu(repository)
        except Exception:
            resolved_instances = {}
        return replace(
            self,
            person_frames=self.person_frames or data / person_dir,
            ship_frames=self.ship_frames or data / ship_dir,
            det_engine=self.det_engine or root / "models" / "yolo26n_fp32.engine",
            seg_engine=self.seg_engine or root / "models" / "yolo26n-seg_fp32.engine",
            model_repository=repository,
            instances_per_gpu=self.instances_per_gpu or resolved_instances,
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
        self.require_same_engines()

    #: Which flat engine the baseline loads, against the plan the server loads for the same
    #: model. Both sides must run the *same file* or the comparison measures the engines.
    _ENGINE_PAIRS = (("ship_detector", "det_engine"), ("ship_segmenter", "seg_engine"))

    def require_same_engines(self) -> None:
        """Refuse unless each side's engine is byte-identical to the other's.

        Existence was all that was checked, and existence is not the property that matters.
        A plan built fp16 against the baseline's fp32 `yolo26n_fp32.engine` is roughly a 2x
        "architecture" win that nothing in the harness could detect — and the precision is
        not recoverable from a serialised plan without loading it, so the check is on the
        bytes. `scripts/build_engines.py` copies one file into both places, which is what
        makes this hold.

        Raises:
            RuntimeError: the two sides would load different engines for a model.
        """
        resolved = self.resolved()
        repository = resolved.model_repository
        if repository is None:
            return
        for model, attribute in self._ENGINE_PAIRS:
            flat = getattr(resolved, attribute)
            plan = repository / model / "1" / "model.plan"
            if flat is None or not flat.is_file():
                continue
            if not plan.is_file():
                # Fails closed. Skipping an absent plan made the guard useless in exactly
                # the case it exists for: `autobuild` then builds the server its *own*
                # engine from ONNX after this check has already passed, so the two sides
                # run different plans and the one property this method claims to enforce
                # is the one that silently does not hold.
                raise RuntimeError(
                    f"{model}: the baseline loads {flat.name} but the server has no plan at "
                    f"{plan}. It would build its own from ONNX, and a comparison across two "
                    f"engines measures the engines. Run "
                    f"`python scripts/build_engines.py --force` to put one file in both."
                )
            if _digest(flat) != _digest(plan):
                raise RuntimeError(
                    f"{model}: the baseline loads {flat.name} and the server loads "
                    f"{plan.relative_to(repository.parent)}, and they are different files. "
                    f"A comparison across two engines measures the engines. Rebuild both "
                    f"from one ONNX with `python scripts/build_engines.py --force`."
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
        instances = resolved.module_instances()
        return {
            "cameras": self.cameras,
            "fps": self.fps,
            "gpus": list(self.gpus),
            "batch": self.batch,
            "seconds": self.seconds,
            "warmup_s": self.warmup_s,
            "sample_interval_s": self.sample_interval_s,
            # Through `module_instances()`, not the raw field: this is "the thing written
            # into every log's metadata line", and an UNRESOLVED config would otherwise record
            # an empty concurrency -- the exact hazard `module_instances` exists for, one
            # method along, and a second source for the number this PR consolidated.
            "instances_per_gpu": dict(instances),
            "baseline_workers": {k: self.workers_for(k) for k in instances},
            "omp_threads": self.omp_threads,
            "buffer_capacity": self.buffer_capacity,
            "pipeline_workers": self.pipeline_workers,
            "resolution": self.resolution,
            "source": self.source,
            "rtsp_port": self.rtsp_port,
            "topology": self.topology,
            "shards": self.shards,
            "shard_cameras": list(self.shard_cameras),
            "camera_ids": list(self.camera_ids),
            "out_dir": str(self.out_dir),
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
            # Recorded because this box is shared and a CPU-bound Python pipeline suffers
            # far more from a noisy neighbour than the baseline's GPU-bound C++ binary does.
            # A run taken at `load average 35` on 48 cores is not comparable with one taken
            # at 2, and without this in the log nobody can tell which they are reading.
            "load_average": [round(v, 2) for v in os.getloadavg()],
            "cpu_count": os.cpu_count(),
        }

    _PATH_FIELDS = (
        "person_frames",
        "ship_frames",
        "det_engine",
        "seg_engine",
        "model_repository",
        "out_dir",
    )
    _TUPLE_FIELDS = ("gpus", "shard_cameras", "camera_ids")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchConfig:
        """The inverse of :meth:`as_dict` for the fields that are configuration.

        Derived keys (``offered_total``, ``baseline_workers``, the host facts) are ignored: a
        shard child rebuilds its parent's configuration from the JSON the parent wrote, then
        narrows it to its own slice.
        """
        names = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in names or value is None:
                continue
            if key in cls._PATH_FIELDS:
                kwargs[key] = Path(value)
            elif key in cls._TUPLE_FIELDS:
                kwargs[key] = tuple(value)
            elif key == "instances_per_gpu":
                kwargs[key] = dict(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
