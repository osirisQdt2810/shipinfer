#!/usr/bin/env bash
# Everything a CI runner needs before the tests can run, in one place.
#
# The workflows call this instead of `pip install` directly, so a repository that needs
# more than an editable install (a CMake build, a model download, a system package) does
# not need its own copy of the workflow.
set -euo pipefail
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e ".[dev]"
