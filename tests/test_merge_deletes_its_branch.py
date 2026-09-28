"""The merge job deletes the branch it merged, and only when that is shown to be safe.

The repo's `delete_branch_on_merge` fires for a merge a person makes, not for one made with
GITHUB_TOKEN, so 257 of 275 workflow-merged branches were left on the remote (V183).
EXECUTED, not spelled: each case runs the step's own `run:` script under `bash -e`, as the
runner does, with a stub `gh` first on PATH that answers from the case and logs every call.
The stub prints what `gh ... --jq` would; the jq expressions were checked against the real API.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-pipeline.yml"
DELETE = "api -X DELETE repos/o/r/git/refs/heads/feat/x --silent"

#: Stands in for `gh`: logs its argv, then answers the call the way the case says to.
GH = r"""#!/bin/bash
printf '%s\n' "$*" >> "$GH_LOG"
case "$*" in
  "pr view "*) printf '%s\n' "$GH_LABELS" ;;
  "pr merge "*) exit "${GH_MERGE_RC:-0}" ;;
  "workflow run "*) exit "${GH_DISPATCH_RC:-0}" ;;
  "pr list "*) [ "${GH_LIST_RC:-0}" = 0 ] || exit "$GH_LIST_RC"
               printf '%s\n' "${GH_STACKED:-}" ;;
  "api -X DELETE "*) exit "${GH_DELETE_RC:-0}" ;;
  "api repos/"*) [ "${GH_TIP_RC:-0}" = 0 ] || exit "$GH_TIP_RC"
                 printf '%s\n' "$GH_TIP" ;;
  *) echo "stub gh: unexpected call: $*" >&2; exit 99 ;;
esac
"""

#: What the event payload would carry, keyed by the env names the steps declare.
EVENT = {
    "GH_TOKEN": "t",
    "PR": "7",
    "REPO": "o/r",
    "HEAD_SHA": "abc123",
    "HEAD_REF": "feat/x",
    "HEAD_REPO": "o/r",
    "DEFAULT_BRANCH": "main",
}


def _steps() -> list[dict]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["merge"]["steps"]


def _step(step_id: str) -> dict:
    return next(s for s in _steps() if s.get("id") == step_id)


def _run(step_id: str, tmp_path: Path, **case: str) -> tuple[int, str, list[str], str]:
    """Run one step's script; returns (rc, stdout, gh calls, GITHUB_OUTPUT)."""
    step = _step(step_id)
    stub = tmp_path / "bin" / "gh"
    stub.parent.mkdir()
    stub.write_text(GH, encoding="utf-8")
    stub.chmod(0o755)
    script, log, out = tmp_path / "step.sh", tmp_path / "gh.log", tmp_path / "output"
    script.write_text(step["run"], encoding="utf-8")
    log.touch()
    out.touch()
    # ONLY the declared env, so `set -u` fails a script that reads a name the step never set.
    env = {name: case.pop(name, EVENT[name]) for name in step["env"]}
    env |= {"PATH": f"{stub.parent}:/usr/bin:/bin", "GH_LOG": str(log)}
    env |= {"GITHUB_OUTPUT": str(out), **case}
    done = subprocess.run(["bash", "-e", str(script)], capture_output=True, text=True, env=env)
    return done.returncode, done.stdout, log.read_text().splitlines(), out.read_text()


class TestTheMergeStepSaysWhatItMerged:
    def test_a_merge_is_reported_before_the_dispatch(self, tmp_path: Path) -> None:
        rc, _, calls, out = _run("merge", tmp_path, GH_LABELS="automerge")
        assert rc == 0
        assert "pr merge 7 --repo o/r --squash --match-head-commit abc123" in calls
        assert "workflow run ci.yml --repo o/r --ref main" in calls
        assert "merged=true" in out.splitlines()

    def test_a_failed_dispatch_still_reports_the_merge(self, tmp_path: Path) -> None:
        # The merge already happened, so the branch must still go.
        rc, _, _, out = _run("merge", tmp_path, GH_LABELS="automerge", GH_DISPATCH_RC="1")
        assert rc != 0
        assert "merged=true" in out.splitlines()

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param({"GH_LABELS": "bug"}, id="no-automerge-label"),
            pytest.param({"GH_LABELS": "automerge", "GH_MERGE_RC": "1"}, id="merge-refused"),
        ],
    )
    def test_no_merge_reports_nothing(self, tmp_path: Path, case: dict) -> None:
        _, _, _, out = _run("merge", tmp_path, **case)
        assert "merged=true" not in out


class TestTheBranchGoesOnlyWhenThatIsSafe:
    def test_the_merged_branch_is_deleted(self, tmp_path: Path) -> None:
        rc, stdout, calls, _ = _run("delete-branch", tmp_path, GH_TIP="abc123")
        assert rc == 0, stdout
        assert calls[-1] == DELETE
        assert "Deleted branch feat/x." in stdout

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param({"GH_STACKED": "301 302", "GH_TIP": "abc123"}, id="stacked-on"),
            pytest.param({"GH_TIP": "fff999"}, id="it-moved-after-the-merge"),
        ],
    )
    def test_it_is_kept_when_github_says_no(self, tmp_path: Path, case: dict) -> None:
        rc, stdout, calls, _ = _run("delete-branch", tmp_path, **case)
        assert (rc, DELETE in calls) == (0, False)
        assert "Kept branch feat/x:" in stdout

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param({"HEAD_REPO": "fork/r"}, id="a-fork"),
            pytest.param({"HEAD_REF": "main"}, id="the-default-branch"),
            pytest.param({"HEAD_REF": "feat/x#1"}, id="a-fragment-would-name-feat-x"),
            pytest.param({"HEAD_REF": "feat/%78"}, id="an-escape"),
            pytest.param({"HEAD_REF": "feat/../main"}, id="a-dot-dot"),
        ],
    )
    def test_it_is_kept_without_asking_github(self, tmp_path: Path, case: dict) -> None:
        rc, stdout, calls, _ = _run("delete-branch", tmp_path, **case)
        assert (rc, calls) == (0, [])
        assert "Kept branch" in stdout

    @pytest.mark.parametrize(
        "case",
        # The tip matches, so ONLY the unanswered question stands between it and a delete.
        [{"GH_LIST_RC": "1", "GH_TIP": "abc123"}, {"GH_TIP_RC": "1"}],
        ids=["no-pr-list", "no-tip"],
    )
    def test_an_unanswered_check_keeps_the_branch(self, tmp_path: Path, case: dict) -> None:
        rc, _, calls, _ = _run("delete-branch", tmp_path, **case)
        assert (rc != 0, DELETE in calls) == (True, False)

    def test_a_failed_step_does_not_fail_the_job(self, tmp_path: Path) -> None:
        rc, _, calls, _ = _run("delete-branch", tmp_path, GH_TIP="abc123", GH_DELETE_RC="1")
        assert (rc != 0, calls[-1]) == (True, DELETE)
        assert _step("delete-branch")["continue-on-error"] is True


class TestTheStepIsWiredToTheMerge:
    def test_it_runs_after_any_merge_and_only_after_one(self) -> None:
        ids = [s.get("id") for s in _steps()]
        assert ids.index("delete-branch") == ids.index("merge") + 1
        # `!cancelled()`, not the implicit `success()`: a failed dispatch follows a real merge.
        wanted = "${{ !cancelled() && steps.merge.outputs.merged == 'true' }}"
        assert _step("delete-branch")["if"] == wanted

    def test_it_reads_the_head_from_the_event(self) -> None:
        env = _step("delete-branch")["env"]
        assert env["HEAD_REF"] == "${{ github.event.pull_request.head.ref }}"
        assert env["HEAD_REPO"] == "${{ github.event.pull_request.head.repo.full_name }}"
        assert env["HEAD_SHA"] == _step("merge")["env"]["HEAD_SHA"]
