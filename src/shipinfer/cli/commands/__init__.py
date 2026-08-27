"""CLI commands — one module per command group."""

from shipinfer.cli.commands.bench import bench
from shipinfer.cli.commands.deepstream import deepstream
from shipinfer.cli.commands.doctor import doctor
from shipinfer.cli.commands.fleet import fleet
from shipinfer.cli.commands.registries import (
    list_backends,
    list_policies,
    list_queues,
    list_topologies,
)
from shipinfer.cli.commands.repo import repo_list, repo_show
from shipinfer.cli.commands.serve import serve

__all__ = [
    "bench",
    "deepstream",
    "doctor",
    "fleet",
    "list_backends",
    "list_policies",
    "list_queues",
    "list_topologies",
    "repo_list",
    "repo_show",
    "serve",
]
