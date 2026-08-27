"""``shipinfer deepstream`` — run one shard's DeepStream graph."""

from __future__ import annotations

from pathlib import Path

from shipinfer.cli.common import build_settings, console

__all__ = ["deepstream"]


def deepstream(
    repository: Path,
    *,
    gpus: str | None = None,
    config_dir: Path | None = None,
    log_level: str = "INFO",
    dry_run: bool = False,
) -> int:
    """Decode, detect, track and embed this shard's cameras inside one DeepStream graph.

    The child of ``shipinfer fleet --topology deepstream``, and usable on its own — the fleet
    tells it which cameras and which GPU through the environment, exactly as it tells a
    ``serve`` shard, so running one by hand is running what the fleet runs.

    ``--dry-run`` generates the nvinfer, tracker and label files, says where they are, and
    stops. It is exempt from the container gate on purpose: it reads a repository and writes
    text, touching no device and producing no measurement, and the machine an operator reviews
    a config on is usually not the machine that will run it. The real run is gated, like
    ``serve`` and ``bench``.
    """
    out = console()
    settings = build_settings(repository, gpus=gpus, log_level=log_level)

    # Imported here rather than at module scope: `shipinfer --help` and `shipinfer repo ls`
    # must not pay for the pipeline package, which is the house rule for every command.
    from shipinfer.pipeline.deepstream import DeepStreamPipeline

    pipeline = DeepStreamPipeline(settings, config_root=config_dir)
    if dry_run:
        generated = pipeline.write_configs(require_engine=False)
        out.print(
            f"shard {pipeline.shard_index} on gpu {pipeline.gpu_id}, "
            f"{len(pipeline.cameras)} camera(s) -> {generated.root}"
        )
        for path in generated.paths():
            out.print(f"  {path}")
        if generated.missing_files:
            # Not an error here and an error on a real start: an engine is host-specific and
            # built per machine, so a control box legitimately has none. Saying so is the point
            # — a config naming a plan that does not exist is a shard that will not start.
            out.print(
                f"[yellow]warning[/yellow]: {len(generated.missing_files)} artefact(s) named "
                f"by these configs are not on this machine: "
                f"{', '.join(str(p) for p in generated.missing_files)}"
            )
        return 0

    # The gate lives in the command, not only in the shell hook: this one drives a GPU.
    from shipinfer.runtime.containment import require_container

    require_container("`shipinfer deepstream`")
    return pipeline.run()
