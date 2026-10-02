from pathlib import Path
import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from tasks import github_digest as gd

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


REPOS = [
    {"name": "alpha", "pushedAt": _iso(NOW - timedelta(days=1))},
    {"name": "beta", "pushedAt": _iso(NOW - timedelta(days=10))},
    {"name": "gamma", "pushedAt": _iso(NOW - timedelta(days=2))},
    {"name": "cold-repo", "pushedAt": _iso(NOW - timedelta(days=60))},
]

ISSUES = {
    "alpha": [
        {
            "number": 1,
            "title": "Fix widget",
            "labels": [{"name": "status/in-progress"}],
            "updatedAt": _iso(NOW - timedelta(days=5)),
        },
        {
            "number": 2,
            "title": "Add sprocket",
            "labels": [{"name": "status/todo"}],
            "updatedAt": _iso(NOW - timedelta(hours=1)),
        },
        {
            "number": 3,
            "title": "Fresh in-progress",
            "labels": [{"name": "status/in-progress"}],
            "updatedAt": _iso(NOW - timedelta(hours=2)),
        },
    ],
    "beta": [],
}

PRS = {
    "alpha": [
        {"number": 10, "title": "Widget patch", "isDraft": False, "reviewDecision": "APPROVED"},
        {"number": 11, "title": "WIP sprocket", "isDraft": True, "reviewDecision": ""},
    ],
    "beta": [],
    "gamma": [],
}

RUNS = {
    "alpha": [
        {"name": "CI", "conclusion": "failure", "createdAt": _iso(NOW - timedelta(hours=2)), "url": "https://x/1"},
        {"name": "CI", "conclusion": "failure", "createdAt": _iso(NOW - timedelta(hours=30)), "url": "https://x/2"},
        {"name": "CI", "conclusion": "success", "createdAt": _iso(NOW - timedelta(hours=1)), "url": "https://x/3"},
    ],
    "beta": [],
    "gamma": [],
}

# gamma's issue list is broken (deleted mid-run, rate-limited, whatever) —
# this is the failure the other calls and other repos must survive.
GAMMA_ISSUE_ERROR = "GraphQL: Could not resolve to a Repository with the name 'octocat/gamma'. (repository)"
FAILING_CALLS = {("issue", "gamma")}


def _fake_run(cmd, **kwargs):
    assert cmd[0] == "gh"
    subcommand = cmd[1]
    if subcommand == "repo":
        payload = REPOS
    else:
        repo = cmd[4].split("/")[1]
        if (subcommand, repo) in FAILING_CALLS:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr=GAMMA_ISSUE_ERROR)
        payload = {"issue": ISSUES, "pr": PRS, "run": RUNS}[subcommand][repo]
    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network."""

    def __init__(self):
        self.calls: list[str] = []

    def ensure_server(self) -> None:
        pass

    def chat(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.calls.append(prompt)
        assert system is not None and "untrusted" in system
        return "line one\nline two\nline three\nline four\nline five"


@pytest.fixture(autouse=True)
def stub_subprocess(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run)


def test_active_repos_filters_by_push_window():
    names = {r["name"] for r in gd.active_repos(NOW)}
    assert names == {"alpha", "beta", "gamma"}


def test_status_label():
    assert gd.status_label([{"name": "bug"}, {"name": "status/blocked"}]) == "status/blocked"
    assert gd.status_label([{"name": "bug"}]) is None


def test_stale_in_progress_only_flags_untouched_in_progress_issues():
    stale = gd.stale_in_progress(ISSUES["alpha"], NOW)
    assert [i["number"] for i in stale] == [1]


def test_repo_failed_runs_only_counts_last_24h_failures():
    runs, error = gd.repo_failed_runs("alpha", NOW)
    assert error is None
    assert [r["url"] for r in runs] == ["https://x/1"]


def test_repo_issues_isolates_a_failed_gh_call():
    issues, error = gd.repo_issues("gamma")
    assert issues == []
    assert error == GAMMA_ISSUE_ERROR


def test_gather_aggregates_per_repo():
    data = gd.gather(NOW)
    by_name = {d["name"]: d for d in data}
    assert set(by_name) == {"alpha", "beta", "gamma"}

    alpha = by_name["alpha"]
    assert len(alpha["issues"]) == 3
    assert len(alpha["prs"]) == 2
    assert len(alpha["failed_runs"]) == 1
    assert [i["number"] for i in alpha["stale"]] == [1]
    assert alpha["status_counts"] == {"status/in-progress": 2, "status/todo": 1}
    assert alpha["errors"] == []

    beta = by_name["beta"]
    assert beta["issues"] == [] and beta["prs"] == [] and beta["failed_runs"] == []
    assert beta["errors"] == []


def test_gather_isolates_a_failed_repo_call():
    """gamma's issue call fails; alpha and beta must still come back whole,
    and gamma must still be reported (with its issue-derived fields empty
    and the error recorded) rather than dropped or aborting the run."""
    data = gd.gather(NOW)
    by_name = {d["name"]: d for d in data}

    gamma = by_name["gamma"]
    assert gamma["issues"] == []
    assert gamma["stale"] == []
    assert gamma["status_counts"] == {}
    assert gamma["errors"] == [GAMMA_ISSUE_ERROR]
    # the pr/run calls for gamma weren't affected
    assert gamma["prs"] == []
    assert gamma["failed_runs"] == []

    # alpha, unrelated to gamma's failure, is unaffected
    alpha = by_name["alpha"]
    assert len(alpha["issues"]) == 3
    assert alpha["errors"] == []


def test_main_writes_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gd, "LMStudio", FakeLMStudio)
    monkeypatch.setattr(gd, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    gd.main()

    out = capsys.readouterr().out.strip()
    path = Path(out)
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "GitHub nightly digest" in html
    assert "alpha" in html and "beta" in html and "gamma" in html
    assert "line one" in html
    assert "CI failures" in html
    assert "fetch errors" in html
    assert "Could not resolve to a Repository" in html


def test_first_line_handles_missing_stderr():
    assert gd._first_line("") == "gh command failed"
    assert gd._first_line(None) == "gh command failed"
    assert gd._first_line("boom\nmore detail\n") == "boom"
