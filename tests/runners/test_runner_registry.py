"""The runner registry: one name per implementation, one door that builds it.

Three runners will execute the same chain (arch.md §1), and an operator picks one by name.
The properties worth pinning are the ones that make adding the second and third a new file
rather than an edit: the registered name is the class's own ``name``, an alias resolves to
it, and a typo is answered with the list of what exists.

Named ``test_runner_registry.py`` rather than ``test_registry.py`` because
``tests/core/test_registry.py`` already exists and pytest's default import mode gives two
same-named modules in non-package directories the same module name — the same collision
``tests/topology/test_chain.py`` explains in its own docstring.
"""

from __future__ import annotations

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import ServerSettings
from shipinfer.runners import RUNNERS, InprocessRunner, Runner, build_runner
from shipinfer.topology import ChainSpec, Topology

CHAIN = """
name: two_step
elements:
  decode: {impl: replay}
  output: {impl: none}
"""


@pytest.fixture()
def chain() -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(CHAIN))


class TestTheRegistry:
    def test_the_inprocess_runner_is_registered_under_its_own_name(self) -> None:
        """The name in the settings tree and the name in a log line cannot drift."""
        assert "inprocess" in RUNNERS
        assert RUNNERS.get("inprocess") is InprocessRunner
        assert InprocessRunner.name == "inprocess"

    def test_the_alias_resolves_to_the_canonical_name(self) -> None:
        assert RUNNERS.canonical("single") == "inprocess"
        assert RUNNERS.get("single") is InprocessRunner

    def test_every_registration_is_a_runner(self) -> None:
        """The registry's base check, which is what makes ``build_runner``'s return type true."""
        assert all(issubclass(entry.resolve(), Runner) for entry in RUNNERS)

    def test_a_class_that_is_not_a_runner_is_refused_at_registration(self) -> None:
        with pytest.raises(TypeError, match="does not subclass Runner"):

            @RUNNERS.register("not-a-runner")
            class Impostor:
                pass

    def test_describe_gives_a_line_per_runner(self) -> None:
        """What a ``--list`` flag prints; an undocumented runner is a bug in its docstring."""
        assert all(name and text for name, text in RUNNERS.describe())


class TestBuildRunner:
    def test_it_builds_the_named_runner_over_the_topology(self, chain: Topology) -> None:
        settings = ServerSettings()

        runner = build_runner("inprocess", chain, settings, shard_id=3)

        assert isinstance(runner, InprocessRunner)
        assert runner.topology is chain
        assert runner.settings is settings
        assert runner.shard_id == 3
        assert not runner.is_running

    def test_it_passes_implementation_options_through(self, chain: Topology) -> None:
        runner = build_runner("single", chain, workers=7)

        assert isinstance(runner, InprocessRunner)
        assert runner.workers == 7

    def test_omitting_the_settings_takes_the_defaults(self, chain: Topology) -> None:
        """A runner must be buildable from a chain alone — that is what a test and a laptop do."""
        runner = build_runner("inprocess", chain)

        assert runner.settings.pipeline.workers >= 1

    def test_an_unknown_name_lists_the_ones_that_exist(self, chain: Topology) -> None:
        with pytest.raises(ConfigurationError) as caught:
            build_runner("inprocesss", chain)

        assert "inprocesss" in str(caught.value)
        assert "inprocess" in str(caught.value), "the message must list what there is"
