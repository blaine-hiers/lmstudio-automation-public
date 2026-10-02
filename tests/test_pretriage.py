import json

import pytest

from tasks import pretriage
from tasks.pretriage import Candidate, Suggestion, build_report, classify, find_candidates, write_json


def _issue(number, labels, body="body text", repo="octocat/redline"):
    return {
        "repository": {"nameWithOwner": repo},
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://github.com/{repo}/issues/{number}",
        "body": body,
        "labels": [{"name": name} for name in labels],
    }


def test_find_candidates_flags_missing_type_tier_or_backlog(monkeypatch):
    issues = [
        _issue(1, ["bug", "tier/sonnet"]),  # fully triaged: excluded
        _issue(2, ["tier/sonnet"]),  # missing type: included
        _issue(3, ["bug"]),  # missing tier: included
        _issue(4, ["bug", "tier/sonnet", "status/backlog"]),  # backlog: included
    ]
    monkeypatch.setattr(pretriage, "_gh_json", lambda *args: issues)

    candidates, truncated = find_candidates("octocat", 50)

    assert [c.number for c in candidates] == [2, 3, 4]
    assert all(isinstance(c, Candidate) for c in candidates)
    assert truncated is False


def test_find_candidates_empty_queue(monkeypatch):
    monkeypatch.setattr(pretriage, "_gh_json", lambda *args: [])
    candidates, truncated = find_candidates("octocat", 50)
    assert candidates == []
    assert truncated is False


def test_find_candidates_flags_truncation_when_raw_count_hits_limit(monkeypatch):
    # Two open issues, and the search cap (limit=2) is exactly the raw count
    # returned -- a sign there may be more issues beyond what gh gave back.
    issues = [_issue(1, []), _issue(2, [])]
    monkeypatch.setattr(pretriage, "_gh_json", lambda *args: issues)

    _, truncated = find_candidates("octocat", 2)

    assert truncated is True


def test_find_candidates_not_truncated_when_below_limit(monkeypatch):
    issues = [_issue(1, [])]
    monkeypatch.setattr(pretriage, "_gh_json", lambda *args: issues)

    _, truncated = find_candidates("octocat", 50)

    assert truncated is False


class FakeLM:
    def __init__(self, result=None, *, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.calls.append((prompt, schema, system, name))
        if self._error is not None:
            raise self._error
        return self._result


def test_classify_builds_suggestion_from_extract_result():
    candidate = Candidate(
        repo="octocat/redline", number=7, title="Worker rewrites state file",
        url="https://github.com/octocat/redline/issues/7",
        body="Steps to reproduce...", labels=["bug"],
    )
    lm = FakeLM({
        "type": "bug",
        "tier": "sonnet",
        "workable": True,
        "missing": [],
        "reason": "Repro and expected behavior are both stated.",
    })

    suggestion = classify(lm, candidate, system="SYSTEM PROMPT")

    assert isinstance(suggestion, Suggestion)
    assert suggestion.repo == "octocat/redline"
    assert suggestion.number == 7
    assert suggestion.type == "bug"
    assert suggestion.tier == "sonnet"
    assert suggestion.workable is True
    assert suggestion.missing == []
    assert suggestion.error is None
    assert lm.calls[0][2] == "SYSTEM PROMPT"
    assert lm.calls[0][1] == pretriage.SCHEMA
    assert "Worker rewrites state file" in lm.calls[0][0]


def test_classify_marks_unworkable_with_missing_list():
    candidate = Candidate(
        repo="octocat/redline", number=8, title="Fix the thing",
        url="https://github.com/octocat/redline/issues/8",
        body="It's broken sometimes.", labels=[],
    )
    lm = FakeLM({
        "type": "bug",
        "tier": "sonnet",
        "workable": False,
        "missing": ["reproduction steps", "expected behavior"],
        "reason": "No repro or expected behavior given.",
    })

    suggestion = classify(lm, candidate, system="SYSTEM PROMPT")

    assert suggestion.workable is False
    assert suggestion.missing == ["reproduction steps", "expected behavior"]


def test_classify_isolates_extract_exception():
    candidate = Candidate(
        repo="octocat/redline", number=9, title="Flaky one",
        url="https://github.com/octocat/redline/issues/9",
        body="whatever", labels=[],
    )
    lm = FakeLM(error=RuntimeError("server returned empty content"))

    suggestion = classify(lm, candidate, system="SYSTEM PROMPT")

    assert suggestion.error == "server returned empty content"
    assert suggestion.type is None
    assert suggestion.tier is None
    assert suggestion.workable is None
    assert suggestion.missing is None
    assert suggestion.reason is None
    # identifying fields still survive a failed classification
    assert suggestion.repo == "octocat/redline"
    assert suggestion.number == 9


def test_classify_isolates_malformed_result_missing_keys():
    candidate = Candidate(
        repo="octocat/redline", number=10, title="Malformed",
        url="https://github.com/octocat/redline/issues/10",
        body="whatever", labels=[],
    )
    # No "reason" key -- as if the model returned truncated/invalid JSON.
    lm = FakeLM({"type": "bug", "tier": "sonnet", "workable": True, "missing": []})

    suggestion = classify(lm, candidate, system="SYSTEM PROMPT")

    assert suggestion.error is not None
    assert suggestion.type is None
    assert suggestion.reason is None


def test_classify_truncates_long_body_in_prompt():
    long_body = "x" * (pretriage.BODY_LIMIT + 500)
    candidate = Candidate(
        repo="octocat/redline", number=11, title="Huge issue",
        url="https://github.com/octocat/redline/issues/11",
        body=long_body, labels=[],
    )
    lm = FakeLM({
        "type": "bug", "tier": "sonnet", "workable": True, "missing": [], "reason": "ok",
    })

    classify(lm, candidate, system="SYSTEM PROMPT")

    prompt = lm.calls[0][0]
    assert len(prompt) < len(long_body)
    assert "truncated" in prompt
    # the candidate itself is untouched
    assert len(candidate.body) == pretriage.BODY_LIMIT + 500


def test_classify_does_not_truncate_short_body():
    candidate = Candidate(
        repo="octocat/redline", number=12, title="Small issue",
        url="https://github.com/octocat/redline/issues/12",
        body="short body", labels=[],
    )
    lm = FakeLM({
        "type": "bug", "tier": "sonnet", "workable": True, "missing": [], "reason": "ok",
    })

    classify(lm, candidate, system="SYSTEM PROMPT")

    assert "truncated" not in lm.calls[0][0]
    assert "short body" in lm.calls[0][0]


def test_build_report_empty_queue_renders_clean_state():
    r = build_report([], "octocat")
    html = r.render()
    assert "queue is clean" in html
    assert "Suggestions" not in html


def test_build_report_lists_candidates_and_needs_input():
    suggestions = [
        Suggestion(
            repo="octocat/redline", number=1, title="A workable one",
            url="https://github.com/octocat/redline/issues/1", labels=["bug"],
            type="bug", tier="sonnet", workable=True, missing=[], reason="Clear repro.",
        ),
        Suggestion(
            repo="octocat/redline", number=2, title="<script>alert(1)</script>",
            url="https://github.com/octocat/redline/issues/2", labels=[],
            type="feature", tier="opus", workable=False,
            missing=["acceptance criteria"], reason="No acceptance criteria.",
        ),
    ]
    r = build_report(suggestions, "octocat")
    html = r.render()

    assert "2 flagged" in html
    assert "redline#2" in html
    assert "No acceptance criteria." in html
    # untrusted issue title must be escaped, never raw
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_report_shows_truncated_warning():
    r = build_report([], "octocat", truncated=True)
    html = r.render()
    assert "truncated" in html
    assert "higher --limit" in html


def test_build_report_shows_failures_separately_from_suggestions():
    suggestions = [
        Suggestion(
            repo="octocat/redline", number=1, title="Fine",
            url="https://github.com/octocat/redline/issues/1", labels=[],
            type="bug", tier="sonnet", workable=True, missing=[], reason="Clear.",
        ),
        Suggestion(
            repo="octocat/redline", number=2, title="Broke",
            url="https://github.com/octocat/redline/issues/2", labels=[],
            type=None, tier=None, workable=None, missing=None, reason=None,
            error="server returned empty content",
        ),
    ]
    r = build_report(suggestions, "octocat")
    html = r.render()

    assert "Failed to classify" in html
    assert "server returned empty content" in html
    assert "redline#2" in html
    # the failed issue must not show up in the classified table as though it succeeded
    assert '<td class="num">1</td>' in html  # from the successful row's "#" column


def test_write_json_round_trips_suggestions(tmp_path):
    suggestions = [
        Suggestion(
            repo="octocat/redline", number=1, title="A workable one",
            url="https://github.com/octocat/redline/issues/1", labels=["bug"],
            type="bug", tier="sonnet", workable=True, missing=[], reason="Clear repro.",
        ),
    ]
    path = write_json(suggestions, tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "truncated": False,
        "suggestions": [
            {
                "repo": "octocat/redline", "number": 1, "title": "A workable one",
                "url": "https://github.com/octocat/redline/issues/1", "labels": ["bug"],
                "type": "bug", "tier": "sonnet", "workable": True, "missing": [],
                "reason": "Clear repro.", "error": None,
            }
        ],
    }


def test_write_json_includes_truncated_flag_and_error_field(tmp_path):
    suggestions = [
        Suggestion(
            repo="octocat/redline", number=2, title="Broke",
            url="https://github.com/octocat/redline/issues/2", labels=[],
            type=None, tier=None, workable=None, missing=None, reason=None,
            error="boom",
        ),
    ]
    path = write_json(suggestions, tmp_path, truncated=True)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["truncated"] is True
    assert data["suggestions"][0]["error"] == "boom"
    assert data["suggestions"][0]["type"] is None


def test_gh_json_raises_when_gh_missing(monkeypatch):
    monkeypatch.setattr(pretriage.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="gh CLI not found"):
        pretriage._gh_json("search", "issues")
