"""Configuring the logger must be undoable, because tests and embedders both undo it.

`configure` turns `propagate` off so an embedder's root logger does not print every record
twice. `shutdown` has to put it back: an embedder who stops us otherwise never gets their
logging back, and inside a test session the effect is worse than untidy — records reach no
handler here and are not passed up either, so `caplog` goes blind for everything that runs
afterwards. Six container-tier tests were asserting on log records and finding none.
"""

from __future__ import annotations

import logging

import pytest

from shipinfer.core.logging import configure, get_logger, shutdown

_ROOT = "shipinfer"


@pytest.fixture(autouse=True)
def _leave_logging_as_we_found_it():
    """This file toggles the very state it is about; do not leak it into the next test."""
    logger = logging.getLogger(_ROOT)
    before = (logger.propagate, logger.level, list(logger.handlers))
    yield
    shutdown()
    logger.propagate, logger.level = before[0], before[1]
    for handler in before[2]:
        logger.addHandler(handler)


class TestTheIsolationFixtureClearsWhatConfigureRemembers:
    """The module globals, not just the logger — the half the first fixture stopped short of.

    A test reaching `cli.common.build_settings` calls `configure(force=True)` and never
    `shutdown()`, so `_prior_propagate` survives it; the next `configure`'s leading `shutdown`
    then restores THAT value over the one the current test just set. Asserted rather than
    trusted, because the fixture is autouse and so invisible from here.
    """

    def test_no_test_starts_with_state_left_by_a_previous_one(self) -> None:
        from shipinfer.core import logging as core_logging

        assert core_logging._prior_propagate is None, (
            "a previous test left _prior_propagate set; configure() will restore its value "
            "over this test's own"
        )
        assert core_logging._active_sink is None, (
            "a previous test left _active_sink set; configure() without force early-returns "
            "and attaches no handler at all"
        )


class TestShutdownGivesPropagationBack:
    def test_configure_turns_it_off_and_shutdown_turns_it_on(self) -> None:
        logger = logging.getLogger(_ROOT)
        logger.propagate = True

        configure(sink="stream", force=True)
        assert logger.propagate is False, "configure must stop records escaping to the root"

        shutdown()
        assert logger.propagate is True, (
            "shutdown left propagate off; an embedder who stops us never gets their logging "
            "back, and caplog stays blind for the rest of a test session"
        )

    def test_it_restores_what_the_embedder_had_not_what_logging_defaults_to(self) -> None:
        """An embedder who had already turned it off keeps it off."""
        logger = logging.getLogger(_ROOT)
        logger.propagate = False

        configure(sink="stream", force=True)
        shutdown()

        assert logger.propagate is False

    def test_a_record_reaches_a_root_handler_after_configure_and_shutdown(self, caplog) -> None:
        """The operational shape of the bug, in the terms a test hits it in.

        Not a restatement of the flag: this asserts the consequence, so a future change that
        keeps the flag correct by accident but breaks delivery still fails here.
        """
        configure(sink="stream", force=True)
        shutdown()

        with caplog.at_level(logging.ERROR, logger=f"{_ROOT}.runners.walk"):
            get_logger("runners.walk").error("a record a test would assert on")

        assert [
            r for r in caplog.records if "would assert on" in r.getMessage()
        ], "caplog saw nothing: records are neither handled here nor propagated up"

    def test_shutdown_without_configure_is_safe(self) -> None:
        """It is documented as safe to call when nothing is configured, and it must not
        invent a propagate value it was never given."""
        logger = logging.getLogger(_ROOT)
        logger.propagate = False

        shutdown()

        assert logger.propagate is False, "nothing was configured; nothing to restore"
