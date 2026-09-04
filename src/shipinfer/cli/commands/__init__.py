"""CLI commands — one module per command group."""

from shipinfer.cli.commands.bench import bench
from shipinfer.cli.commands.deepstream import deepstream
from shipinfer.cli.commands.doctor import doctor
from shipinfer.cli.commands.plan import plan
from shipinfer.cli.commands.registries import (
    list_backends,
    list_policies,
    list_queues,
    list_runners,
)
from shipinfer.cli.commands.repo import repo_list, repo_show
from shipinfer.cli.commands.run import run
from shipinfer.cli.commands.serve import serve

__all__ = [
    "bench",
    "deepstream",
    "doctor",
    "list_backends",
    "list_policies",
    "list_queues",
    "list_runners",
    "plan",
    "repo_list",
    "repo_show",
    "run",
    "serve",
]
