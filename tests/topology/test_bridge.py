"""The shipvision bridge: one refusal, one message, and no import until it is asked for.

Green **with or without** the submodule, which is the point of the file. CI does not check
``3rdparty/shipvision`` out (``.claude/CLAUDE.md``), so the tests that need it to be missing
make it missing rather than assuming it is, and the tests that need it present skip.

``None`` in ``sys.modules`` is how the absence is arranged: CPython's import machinery treats
it as "this module is known not to be importable" and raises ``ImportError`` for it, which is
the same exception a real absent submodule raises and the same one the bridge catches. A
``monkeypatch`` fixture undoes it, and the caches are cleared either side so no test leaves a
memoised module behind for the next one.
"""

from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.topology import bridge

#: Every loader, with the dotted name it must name in its refusal.
LOADERS = [
    (bridge.load_mot, "shipvision.mot"),
    (bridge.load_mtmc, "shipvision.mtmc"),
    (bridge.load_reid, "shipvision.reid"),
    (bridge.load_types, "shipvision.types"),
]


@pytest.fixture(autouse=True)
def clear_caches() -> Iterator[None]:
    """No test inherits another's memoised module, in either direction."""
    for loader, _ in LOADERS:
        loader.cache_clear()
    yield
    for loader, _ in LOADERS:
        loader.cache_clear()


@pytest.fixture()
def without_shipvision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that never checked the submodule out — arranged, not assumed."""
    for name in list(sys.modules):
        if name == "shipvision" or name.startswith("shipvision."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "shipvision", None)


class TestTheRefusalNamesTheFix:
    @pytest.mark.parametrize(
        ("loader", "subpackage"), LOADERS, ids=[name for _, name in LOADERS]
    )
    def test_a_missing_submodule_is_a_typed_refusal_carrying_the_command(
        self, loader, subpackage: str, without_shipvision: None
    ) -> None:
        """The one message, four times.

        ``ConfigurationError`` and not ``ImportError``: an operator whose deployment names
        ``impl: shipvision`` has a *configuration* problem with a known fix, and the fix is in
        the sentence rather than in whatever documentation they can find. The wording is
        ``pipeline/graph/tracking.py``'s, so somebody who has met it once recognises it.
        """
        with pytest.raises(ConfigurationError) as caught:
            loader()

        message = str(caught.value)
        assert subpackage in message, "the refusal says which subpackage is missing"
        assert "git submodule update --init 3rdparty/shipvision" in message
        assert "pip install -e 3rdparty/shipvision" in message

    def test_the_original_import_error_is_chained_not_swallowed(
        self, without_shipvision: None
    ) -> None:
        """ "No such module" and "the extension is built for another Python" read alike
        otherwise, and only the second is worth a rebuild."""
        with pytest.raises(ConfigurationError) as caught:
            bridge.load_mot()

        assert isinstance(caught.value.__cause__, ImportError)

    def test_the_failure_is_not_memoised_so_a_late_install_is_picked_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """:func:`functools.lru_cache` caches a return value and never an exception.

        Which is the behaviour worth having: a cached refusal would outlive the ``pip install``
        that fixed it and make a restart the only cure.

        The attempts are **counted**, and that is the assertion that carries the property. Two
        failing calls both raising proves only that both raised; a memoised refusal would do
        that too, and would do it without ever touching the import machinery again. So
        ``__import__`` is wrapped for the duration and every ``shipvision`` name it is asked
        for is recorded — two calls, two real attempts — and the third call, made after the
        module is put back, is what says the cache did not poison the recovery.
        """
        attempts: list[str] = []
        real_import = builtins.__import__

        def counting_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "shipvision" or name.startswith("shipvision."):
                attempts.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "shipvision", None)
        monkeypatch.setattr(builtins, "__import__", counting_import)
        for _ in range(2):
            with pytest.raises(ConfigurationError):
                bridge.load_mot()

        assert attempts == [
            "shipvision",
            "shipvision",
        ], "the second call replayed a cached refusal instead of trying the import again"

        monkeypatch.undo()
        pytest.importorskip("shipvision.mot")
        assert bridge.load_mot() is not None, "the third call, after the install, succeeds"


class TestTheSuccessIsMemoised:
    def test_two_calls_return_the_same_module_object(self) -> None:
        """One import, not one per frame. An element calls this from ``_do_open``, and a
        fifty-camera fleet opens its chain once per shard rather than once."""
        pytest.importorskip("shipvision.mot")

        assert bridge.load_mot() is bridge.load_mot()
        assert bridge.load_mot().__name__ == "shipvision.mot"

    def test_each_loader_answers_with_its_own_subpackage(self) -> None:
        pytest.importorskip("shipvision.types")

        assert bridge.load_types().__name__ == "shipvision.types"


class TestAvailability:
    def test_it_is_false_when_the_submodule_is_absent(self, without_shipvision: None) -> None:
        assert bridge.shipvision_available() is False

    def test_it_is_true_when_the_submodule_is_installed(self) -> None:
        pytest.importorskip("shipvision.types")

        assert bridge.shipvision_available() is True

    def test_it_agrees_with_what_the_loader_does(self) -> None:
        """The two must not be able to disagree: a caller that skips on ``False`` and a
        caller that catches the refusal have to be answering the same question."""
        available = bridge.shipvision_available()
        try:
            bridge.load_types()
        except ConfigurationError:
            assert available is False
        else:
            assert available is True


class TestNothingIsImportedUntilItIsAsked:
    def test_importing_the_bridge_imports_no_shipvision(self) -> None:
        """The property the whole module exists for, at its own seam.

        ``tests/test_architecture.py`` makes the same assertion for ``shipinfer.topology`` as
        a whole and is the enforcement point; this one is narrower and fails first, naming the
        module that broke it. A subprocess because ``shipvision`` is already in this
        interpreter's ``sys.modules`` on any host that has it.
        """
        import subprocess

        code = (
            "import sys, shipinfer.topology.bridge as b; "
            "assert callable(b.load_mot); "
            "eager = [m for m in sys.modules if m == 'shipvision' "
            "or m.startswith('shipvision.')]; "
            "assert not eager, eager"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
