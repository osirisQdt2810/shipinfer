"""CLI commands — one module per command group."""

from shipinfer.cli.commands.bench import bench
from shipinfer.cli.commands.doctor import doctor
from shipinfer.cli.commands.registries import list_backends, list_policies, list_queues
from shipinfer.cli.commands.repo import repo_list, repo_show
from shipinfer.cli.commands.serve import serve

__all__ = [
    "bench",
    "doctor",
    "list_backends",
    "list_policies",
    "list_queues",
    "repo_list",
    "repo_show",
    "serve",
]
