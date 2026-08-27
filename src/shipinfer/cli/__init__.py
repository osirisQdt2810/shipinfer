"""The ``shipinfer`` command line.

Thin on purpose: every command is a few lines that parse flags and call into the library.
Anything a command can do, a Python caller can do the same way — a CLI that owns behaviour
becomes the only way to use the system, and this one is a library first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer

__all__ = ["build_app", "main"]

_DEFAULT_REPO = Path("model_repository")


def build_app() -> typer.Typer:
    """Construct the typer application.

    Built inside a function so ``import shipinfer.cli`` does not require typer — the
    library must stay usable on a host where only the runtime extras are installed.
    """
    import typer

    from shipinfer.cli import commands

    app = typer.Typer(
        name="shipinfer",
        help="Triton-shaped, hackable multi-GPU inference server.",
        no_args_is_help=True,
        add_completion=False,
    )
    repo_app = typer.Typer(help="Inspect a model repository.", no_args_is_help=True)
    app.add_typer(repo_app, name="repo")

    repo_option = typer.Option(
        _DEFAULT_REPO, "--repository", "-r", help="Model repository path."
    )
    gpus_option = typer.Option(
        None, "--gpus", help="Comma-separated device indices, e.g. 0,1,2."
    )
    policy_option = typer.Option(None, "--policy", help="Placement policy name.")
    log_option = typer.Option("INFO", "--log-level", "-l")

    @app.command()
    def doctor() -> None:
        """Report devices, CUDA provider and native extension status."""
        raise typer.Exit(commands.doctor())

    @app.command()
    def backends() -> None:
        """List registered execution backends."""
        raise typer.Exit(commands.list_backends())

    @app.command()
    def policies() -> None:
        """List registered placement policies."""
        raise typer.Exit(commands.list_policies())

    @app.command()
    def queues() -> None:
        """List registered request queues."""
        raise typer.Exit(commands.list_queues())

    @app.command()
    def topologies() -> None:
        """List registered process topologies."""
        raise typer.Exit(commands.list_topologies())

    @repo_app.command("ls")
    def repo_ls(repository: Path = repo_option) -> None:
        """List the models in a repository."""
        raise typer.Exit(commands.repo_list(repository))

    @repo_app.command("show")
    def repo_show(name: str, repository: Path = repo_option) -> None:
        """Print one model's resolved configuration."""
        raise typer.Exit(commands.repo_show(repository, name))

    @app.command()
    def serve(
        repository: Path = repo_option,
        gpus: str = gpus_option,
        policy: str = policy_option,
        http: bool = typer.Option(False, "--http", help="Expose the KServe v2 HTTP API."),
        host: str = typer.Option("0.0.0.0", "--host"),
        port: int = typer.Option(8000, "--port"),
        log_level: str = log_option,
    ) -> None:
        """Load a repository and serve until interrupted."""
        raise typer.Exit(
            commands.serve(
                repository,
                gpus=gpus,
                policy=policy,
                http=http,
                host=host,
                port=port,
                log_level=log_level,
            )
        )

    @app.command()
    def fleet(
        repository: Path = repo_option,
        shards: int | None = typer.Option(
            None,
            "--shards",
            "-n",
            help="How many processes to split the fleet across (default: one per visible GPU).",
        ),
        gpus: str = gpus_option,
        policy: str = policy_option,
        topology: str | None = typer.Option(
            None,
            "--topology",
            help="Which topology to run: a name from `shipinfer topologies` (default: settings).",
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Print the plan and stop, without spawning anything."
        ),
        drain_s: float | None = typer.Option(
            None, "--drain", help="Seconds a shard gets to drain before it is killed."
        ),
        log_level: str = log_option,
    ) -> None:
        """Split the fleet across several processes, under one topology, and supervise them."""
        raise typer.Exit(
            commands.fleet(
                repository,
                shards=shards,
                gpus=gpus,
                policy=policy,
                topology=topology,
                dry_run=dry_run,
                drain_s=drain_s,
                log_level=log_level,
            )
        )

    @app.command()
    def deepstream(
        repository: Path = repo_option,
        gpus: str = gpus_option,
        config_dir: Path | None = typer.Option(
            None, "--config-dir", help="Where the generated DeepStream configs are written."
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Generate the configs, print where they are, and stop."
        ),
        log_level: str = log_option,
    ) -> None:
        """Run one shard's cameras through a DeepStream graph."""
        raise typer.Exit(
            commands.deepstream(
                repository,
                gpus=gpus,
                config_dir=config_dir,
                dry_run=dry_run,
                log_level=log_level,
            )
        )

    @app.command()
    def bench(
        model: str,
        repository: Path = repo_option,
        cameras: int = typer.Option(50, help="Synthetic camera count."),
        fps: int = typer.Option(20, help="Frames per second per camera."),
        seconds: float = typer.Option(5.0, help="How long to drive load."),
        skew: float = typer.Option(1.0, help="How much more traffic camera 0 sends."),
        in_flight: int = typer.Option(256, help="Maximum outstanding requests."),
        policy: str = policy_option,
        gpus: str = gpus_option,
        log_level: str = typer.Option("WARNING", "--log-level", "-l"),
    ) -> None:
        """Drive synthetic load and report balance, fairness and latency."""
        raise typer.Exit(
            commands.bench(
                repository,
                model,
                cameras=cameras,
                fps=fps,
                seconds=seconds,
                skew=skew,
                in_flight=in_flight,
                policy=policy,
                gpus=gpus,
                log_level=log_level,
            )
        )

    return app


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``shipinfer`` script and ``python -m shipinfer``."""
    try:
        app = build_app()
    except ImportError:
        print(
            'the CLI needs typer: pip install "shipinfer[cli]"',
            file=sys.stderr,
        )
        return 2
    app(args=argv)
    return 0
