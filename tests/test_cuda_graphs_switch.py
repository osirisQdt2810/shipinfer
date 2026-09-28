"""`SHIPINFER_CUDA_GRAPHS` is one switch for two planes, so it must be spelled one way.

The Python plane reads it through `envs.py`; the C++ plane through `env_on_off` in
`csrc/shipinfer/core/env.h`, which `test_env_on_off.cpp` exercises. This file pins what the
compilers cannot: both accept exactly ``on``/``off``, and every container recipe that runs
the C++ plane forwards the variable without choosing a value for it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from shipinfer import envs
from shipinfer.core.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = "SHIPINFER_CUDA_GRAPHS"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class TestOneSpellingOnBothPlanes:
    @pytest.mark.parametrize(
        ("value", "read"), [("on", "on"), ("off", "off"), ("", ""), (" on ", "on")]
    )
    def test_the_python_plane_takes_what_the_cpp_plane_takes(
        self, value: str, read: str, monkeypatch
    ) -> None:
        monkeypatch.setenv(VARIABLE, value)
        assert read == envs.SHIPINFER_CUDA_GRAPHS

    @pytest.mark.parametrize("value", ["1", "0", "true", "OFF"])
    def test_and_refuses_what_the_cpp_plane_refuses(self, value, monkeypatch) -> None:
        monkeypatch.setenv(VARIABLE, value)
        with pytest.raises(ConfigurationError):
            _ = envs.SHIPINFER_CUDA_GRAPHS

    def test_the_cpp_plane_reads_it_as_an_on_off_switch_defaulting_on(self) -> None:
        env_header = _read("csrc/shipinfer/core/env.h")
        assert re.search(r'spelled == "on"\) return true;', env_header)
        assert re.search(r'spelled == "off"\) return false;', env_header)
        bench = _read("csrc/shipinfer/cli/bench.cpp")
        assert f'env_on_off("{VARIABLE}", true)' in bench, (
            "the C++ plane's default is ON (ADR-024); reading it any other way would turn "
            "graphs off on that plane alone"
        )


class TestTheContainerRecipesForwardIt:
    @pytest.mark.parametrize("recipe", ["deploy/rootless/cpp.sh", "deploy/rootless/profile.sh"])
    def test_forwarded_as_the_caller_set_it(self, recipe: str) -> None:
        # `-e VAR` with no value: a recipe that wrote `-e VAR=off` would choose the plane's
        # default for it, which is how profiles once measured a configuration no run used.
        forwarded = re.findall(rf"-e {VARIABLE}(\S*)", _read(recipe))
        assert forwarded == [""], f"{recipe} forwards {VARIABLE} as {forwarded}"
