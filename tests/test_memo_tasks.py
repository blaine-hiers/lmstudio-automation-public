"""Memory between runs (#46), per wired task, end to end through main().

Every test drives the real task code against a fake Gmail/gh/filesystem and
a fake LM that counts model calls, then checks the acceptance criteria:
- a second run on unchanged input makes ZERO model calls and renders an
  identical report body (everything outside <footer>, with file-name
  timestamps normalised);
- a changed input recomputes only that item;
- --fresh recomputes everything; a model switch invalidates everything;
- a failed call is not remembered.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lmsa.sources.gmail import Message
from tasks import (
    deadlines,
    file_sorter,
    followup_radar,
    inbox_brief,
    labeler,
    newsletter_zine,
    phish_check,
    pretriage,
    reply_drafts,
    travel,
)

NOW = datetime.now(timezone.utc)


# -- fakes -----------------------------------------------------------------------

class CountingLM:
    """Stands in for LMStudio. `answer(prompt, name)` returns the extract()
    result for one prompt (or raises); every extract/chat call is counted."""

    def __init__(self, out_dir, answer, *, chat_model="model-a"):
        self.cfg = SimpleNamespace(output_dir=out_dir, chat_model=chat_model, vision_model="vision-a")
        self.answer = answer
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.prompts.append(prompt)
        return self.answer(prompt, name)

    def chat(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.prompts.append(prompt)
        return "Thanks, will do."


class FakeGmail:
    def __init__(self, messages, sent_messages=()):
        self.messages = list(messages)
        self.sent_messages = list(sent_messages)
        self.cfg = SimpleNamespace(imap_user="me@example.com")

    def __call__(self, cfg=None):  # so the instance can replace the Gmail class
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def sent(self):
        return "SENT"

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        pool = self.sent_messages if folder == "SENT" else self.messages
        return pool[:limit]


def msg(msg_id, subject, *, text="Body text.", thread_id=None, date=None, from_addr="alice@example.com",
        to=("me@example.com",), list_unsubscribe="", list_id=""):
    return Message(
        msg_id=msg_id, thread_id=thread_id or f"t-{msg_id}", labels=[], flags=[],
        from_name="Alice Example", from_addr=from_addr, to=list(to), reply_to="",
        subject=subject, date=date or NOW - timedelta(hours=2),
        list_unsubscribe=list_unsubscribe, list_id=list_id, auth_results="",
        text=text, attachments=[], web_url=f"https://mail.google.com/mail/u/0/#all/{msg_id}",
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    out = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out))
    monkeypatch.setenv("LMSA_IMAP_USER", "me@example.com")
    return SimpleNamespace(out=out, db=tmp_path / "lmsa.sqlite", tmp=tmp_path)


_STAMP = re.compile(r"\d{8}-\d{4,6}")


def run_task(monkeypatch, module, env, *args) -> str:
    """Run module.main() with argv, return the report HTML and delete it
    (two runs in the same minute would otherwise share a file name)."""
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()
    [path] = [p for p in env.out.rglob("*.html") if p.name != "index.html"]
    html = path.read_text(encoding="utf-8")
    path.unlink()
    return html


def body(html: str) -> str:
    """The report minus its footer (timestamps + the memory line) and with
    any file-name timestamps normalised."""
    return _STAMP.sub("<stamp>", re.sub(r"<footer>.*?</footer>", "", html, flags=re.S))


def footer(html: str) -> str:
    return re.search(r"<footer>(.*?)</footer>", html, flags=re.S).group(1)


def last_summary(env) -> dict:
    with sqlite3.connect(env.db) as conn:
        (summary,) = conn.execute("SELECT summary_json FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(summary)


def assert_second_run_is_free(monkeypatch, module, env, lm, n_items, *args) -> str:
    first = run_task(monkeypatch, module, env, *args)
    assert lm.calls == n_items
    assert last_summary(env)["memo_misses"] == n_items
    lm.prompts.clear()

    second = run_task(monkeypatch, module, env, *args)
    assert lm.calls == 0, "second run on unchanged input must make zero model calls"
    assert body(second) == body(first)
    assert f"Memory: {n_items} reused, 0 new LLM calls" in footer(second)
    assert "last successful run never" in footer(first)
    assert "last successful run 20" in footer(second)
    summary = last_summary(env)
    assert (summary["memo_hits"], summary["memo_misses"]) == (n_items, 0)
    return second


# -- inbox_brief / reply_drafts (shared triage memo) ---------------------------

def _triage_answer(prompt, name):
    if "Subject: broken" in prompt:
        raise ValueError("model returned invalid JSON")
    bucket = "act" if "Invoice" in prompt else "read"
    return {"bucket": bucket, "why": "because", "action": "do it", "due_date": None}


@pytest.mark.slow
def test_inbox_brief_reuses_classifications(monkeypatch, env):
    gmail = FakeGmail([msg("1", "Invoice due"), msg("2", "FYI")])
    lm = CountingLM(env.out, _triage_answer)
    monkeypatch.setattr(inbox_brief, "Gmail", gmail)
    monkeypatch.setattr(inbox_brief, "LMStudio", lambda: lm)

    assert_second_run_is_free(monkeypatch, inbox_brief, env, lm, 2)

    # a new message recomputes only that one
    lm.prompts.clear()
    gmail.messages.insert(0, msg("3", "Another invoice"))
    run_task(monkeypatch, inbox_brief, env)
    assert lm.prompts == ["From: Alice Example <alice@example.com>\nSubject: Another invoice\n\nBody:\nBody text."]

    # --fresh recomputes everything
    lm.prompts.clear()
    html = run_task(monkeypatch, inbox_brief, env, "--fresh")
    assert lm.calls == 3
    assert "Memory: 0 reused, 3 new LLM calls (--fresh)" in footer(html)

    # a model switch invalidates everything
    lm.prompts.clear()
    lm.cfg.chat_model = "model-b"
    run_task(monkeypatch, inbox_brief, env)
    assert lm.calls == 3


def test_inbox_brief_does_not_remember_a_failure(monkeypatch, env):
    gmail = FakeGmail([msg("1", "broken"), msg("2", "FYI")])
    lm = CountingLM(env.out, _triage_answer)
    monkeypatch.setattr(inbox_brief, "Gmail", gmail)
    monkeypatch.setattr(inbox_brief, "LMStudio", lambda: lm)

    run_task(monkeypatch, inbox_brief, env)
    lm.prompts.clear()
    html = run_task(monkeypatch, inbox_brief, env)

    assert len(lm.prompts) == 1 and "Subject: broken" in lm.prompts[0]  # retried
    assert "Failed to classify" in html


def test_reply_drafts_reuses_inbox_brief_classifications(monkeypatch, env):
    messages = [msg("1", "Invoice due"), msg("2", "FYI")]
    lm = CountingLM(env.out, _triage_answer)
    for module in (inbox_brief, reply_drafts):
        monkeypatch.setattr(module, "Gmail", FakeGmail(messages))
        monkeypatch.setattr(module, "LMStudio", lambda: lm)

    run_task(monkeypatch, inbox_brief, env)
    assert lm.calls == 2
    lm.prompts.clear()

    html = run_task(monkeypatch, reply_drafts, env)
    assert lm.calls == 0  # both already classified by inbox_brief
    assert "Memory: 2 reused, 0 new LLM calls" in footer(html)

    lm.prompts.clear()
    again = run_task(monkeypatch, reply_drafts, env)
    assert lm.calls == 0
    assert body(again) == body(html)


# -- labeler ---------------------------------------------------------------------

def test_labeler_reuses_classifications_and_the_dry_run_feeds_apply(monkeypatch, env):
    labels = env.tmp / "labels.toml"
    labels.write_text(
        '[[label]]\nname = "Receipts"\ndescription = "Purchase confirmations."\n', encoding="utf-8"
    )
    gmail = FakeGmail([msg("1", "Your receipt"), msg("2", "Hello")])
    gmail.add_calls = []
    gmail.add_labels = lambda msg_id, names: gmail.add_calls.append((msg_id, list(names)))
    lm = CountingLM(env.out, lambda p, n: {"labels": ["Receipts"] if "receipt" in p else []})
    monkeypatch.setattr(labeler, "Gmail", gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", labels)

    assert_second_run_is_free(monkeypatch, labeler, env, lm, 2)

    lm.prompts.clear()
    run_task(monkeypatch, labeler, env, "--apply")
    assert lm.calls == 0  # the dry run's answers are reused by --apply
    assert gmail.add_calls == [("1", ["Receipts"])]

    # editing the label catalog changes the prompt: everything recomputes
    labels.write_text(
        '[[label]]\nname = "Receipts"\ndescription = "Anything you paid for."\n', encoding="utf-8"
    )
    lm.prompts.clear()
    run_task(monkeypatch, labeler, env)
    assert lm.calls == 2


# -- phish_check -----------------------------------------------------------------

def test_phish_check_reuses_model_scores(monkeypatch, env):
    gmail = FakeGmail([msg("1", "Verify your account now"), msg("2", "Lunch?")])
    lm = CountingLM(
        env.out,
        lambda p, n: {"urgency_or_money_pressure": 5 if "Verify" in p else 0, "reason": "r"},
    )
    monkeypatch.setattr(phish_check, "Gmail", gmail)
    monkeypatch.setattr(phish_check, "LMStudio", lambda: lm)

    assert_second_run_is_free(monkeypatch, phish_check, env, lm, 2)

    lm.prompts.clear()
    gmail.messages.append(msg("3", "Gift cards needed"))
    run_task(monkeypatch, phish_check, env)
    assert len(lm.prompts) == 1 and "Gift cards" in lm.prompts[0]


def test_phish_check_does_not_remember_an_invalid_score(monkeypatch, env):
    gmail = FakeGmail([msg("1", "Hi")])
    lm = CountingLM(env.out, lambda p, n: {"urgency_or_money_pressure": 9, "reason": "out of range"})
    monkeypatch.setattr(phish_check, "Gmail", gmail)
    monkeypatch.setattr(phish_check, "LMStudio", lambda: lm)

    run_task(monkeypatch, phish_check, env)
    run_task(monkeypatch, phish_check, env)
    assert lm.calls == 2


# -- travel / deadlines ------------------------------------------------------------

def test_travel_reuses_extractions(monkeypatch, env):
    start = (NOW + timedelta(days=30)).replace(microsecond=0)

    def answer(prompt, name):
        if "Flight" not in prompt:
            return {"items": []}
        return {"items": [{
            "kind": "flight", "provider": "Example Air", "confirmation": "ABC123",
            "start": start.isoformat(), "end": (start + timedelta(hours=3)).isoformat(),
            "from": "AAA", "to": "BBB", "address": None, "cancelled": False,
        }]}

    gmail = FakeGmail([msg("1", "Flight confirmation"), msg("2", "Newsletter")])
    lm = CountingLM(env.out, answer)
    monkeypatch.setattr(travel, "Gmail", gmail)
    monkeypatch.setattr(travel, "LMStudio", lambda: lm)

    html = assert_second_run_is_free(monkeypatch, travel, env, lm, 2)
    assert "Example Air" in html

    lm.prompts.clear()
    gmail.messages.append(msg("3", "Hotel booking"))
    run_task(monkeypatch, travel, env)
    assert len(lm.prompts) == 1 and "Hotel booking" in lm.prompts[0]


def test_travel_does_not_remember_an_empty_response(monkeypatch, env):
    gmail = FakeGmail([msg("1", "Flight confirmation")])
    lm = CountingLM(env.out, lambda p, n: {})  # what extract() gives for an empty reply
    monkeypatch.setattr(travel, "Gmail", gmail)
    monkeypatch.setattr(travel, "LMStudio", lambda: lm)

    run_task(monkeypatch, travel, env)
    run_task(monkeypatch, travel, env)
    assert lm.calls == 2


def test_deadlines_reuses_extractions(monkeypatch, env):
    due = (NOW + timedelta(days=10)).date().isoformat()

    def answer(prompt, name):
        if "Tax" not in prompt:
            return {"items": []}
        return {"items": [{
            "title": "File taxes", "kind": "deadline", "start": due, "end": None,
            "all_day": True, "location": None,
        }]}

    gmail = FakeGmail([msg("1", "Tax reminder"), msg("2", "Hello")])
    lm = CountingLM(env.out, answer)
    monkeypatch.setattr(deadlines, "Gmail", gmail)
    monkeypatch.setattr(deadlines, "LMStudio", lambda: lm)

    html = assert_second_run_is_free(monkeypatch, deadlines, env, lm, 2)
    assert "File taxes" in html

    lm.prompts.clear()
    gmail.messages.append(msg("3", "Tax part two"))
    run_task(monkeypatch, deadlines, env)
    assert len(lm.prompts) == 1 and "Tax part two" in lm.prompts[0]


# -- newsletter_zine -------------------------------------------------------------

def test_newsletter_zine_reuses_summaries(monkeypatch, env):
    news = dict(list_unsubscribe="<mailto:u@example.com>")
    gmail = FakeGmail([msg("1", "AI weekly", **news), msg("2", "Rust weekly", **news)])
    lm = CountingLM(env.out, lambda p, n: {
        "topic": "AI" if "AI" in p else "Rust", "three_bullets": ["a", "b", "c"],
        "worth_full_read": 4, "why": "good",
    })
    monkeypatch.setattr(newsletter_zine, "Gmail", gmail)
    monkeypatch.setattr(newsletter_zine, "LMStudio", lambda: lm)

    assert_second_run_is_free(monkeypatch, newsletter_zine, env, lm, 2)

    lm.prompts.clear()
    gmail.messages.append(msg("3", "Go weekly", **news))
    run_task(monkeypatch, newsletter_zine, env)
    assert len(lm.prompts) == 1 and "Go weekly" in lm.prompts[0]


# -- followup_radar --------------------------------------------------------------

def test_followup_radar_reuses_answers_until_the_thread_changes(monkeypatch, env):
    def sent(msg_id, thread, subject, days_ago):
        return msg(msg_id, subject, thread_id=thread, date=NOW - timedelta(days=days_ago),
                   from_addr="me@example.com", to=("bob@example.com",))

    sent_messages = [sent("s1", "t1", "Need the report", 5), sent("s2", "t2", "Contract?", 6)]
    gmail = FakeGmail(list(sent_messages), sent_messages=sent_messages)
    lm = CountingLM(env.out, lambda p, n: {"expects_reply": True, "ask_summary": "Reply please."})
    monkeypatch.setattr(followup_radar, "Gmail", gmail)
    monkeypatch.setattr(followup_radar, "LMStudio", lambda: lm)

    assert_second_run_is_free(monkeypatch, followup_radar, env, lm, 2)

    # a new message of mine in thread t1 changes what the model is shown: recompute t1 only
    lm.prompts.clear()
    nudge = sent("s3", "t1", "Re: Need the report", 4)
    gmail.sent_messages.append(nudge)
    gmail.messages.append(nudge)
    run_task(monkeypatch, followup_radar, env)
    assert len(lm.prompts) == 1 and "Re: Need the report" in lm.prompts[0]

    # a reply from Bob in t2 answers it: t2 drops out, nothing is recomputed
    lm.prompts.clear()
    gmail.messages.append(msg("r1", "Re: Contract?", thread_id="t2", date=NOW - timedelta(days=1),
                              from_addr="bob@example.com"))
    html = run_task(monkeypatch, followup_radar, env)
    assert lm.calls == 0
    assert "Contract?" not in body(html)


# -- pretriage -------------------------------------------------------------------

def test_pretriage_reuses_suggestions_until_an_issue_is_edited(monkeypatch, env):
    issues = [
        {"repository": {"nameWithOwner": "o/r"}, "number": n, "title": f"Issue {n}",
         "url": f"https://github.com/o/r/issues/{n}", "labels": [], "body": "details",
         "updatedAt": "2026-09-01T00:00:00Z"}
        for n in (1, 2)
    ]
    monkeypatch.setattr(pretriage, "_gh_json", lambda *args: issues)
    lm = CountingLM(env.out, lambda p, n: {
        "type": "bug", "tier": "sonnet", "workable": True, "missing": [], "reason": "clear",
    })
    monkeypatch.setattr(pretriage, "LMStudio", lambda: lm)

    assert_second_run_is_free(monkeypatch, pretriage, env, lm, 2)

    lm.prompts.clear()
    issues[1] = {**issues[1], "title": "Issue 2 (edited)", "updatedAt": "2026-09-20T00:00:00Z"}
    run_task(monkeypatch, pretriage, env)
    assert len(lm.prompts) == 1 and "Issue 2 (edited)" in lm.prompts[0]


# -- file_sorter -----------------------------------------------------------------

def test_file_sorter_reuses_plans_until_a_file_is_touched(monkeypatch, env):
    root = env.tmp / "downloads"
    root.mkdir()
    old = time.time() - 3 * 86400
    for name in ("a.txt", "b.txt"):
        f = root / name
        f.write_text(f"contents of {name}", encoding="utf-8")
        os.utime(f, (old, old))
    lm = CountingLM(env.out, lambda p, n: {"descriptive_name": "notes", "folder": "Docs"})
    monkeypatch.setattr(file_sorter, "LMStudio", lambda: lm)

    assert_second_run_is_free(monkeypatch, file_sorter, env, lm, 2, "--root", str(root))

    lm.prompts.clear()
    touched = root / "b.txt"
    touched.write_text("new contents", encoding="utf-8")
    os.utime(touched, (old - 60, old - 60))
    run_task(monkeypatch, file_sorter, env, "--root", str(root))
    assert len(lm.prompts) == 1 and "b.txt" in lm.prompts[0]


# -- --fresh on the dashboard ----------------------------------------------------

@pytest.mark.parametrize("module", [
    inbox_brief, reply_drafts, labeler, phish_check, travel, deadlines,
    newsletter_zine, followup_radar, pretriage, file_sorter,
])
def test_fresh_is_offered_on_the_tile_without_a_confirm(module):
    from pathlib import Path

    from lmsa.task_meta import parse_source

    meta = parse_source(Path(module.__file__).read_text(encoding="utf-8"))
    [fresh] = [a for a in meta.args if a.flag == "--fresh"]
    assert fresh.kind == "bool"
    assert fresh.write is False
    assert "--fresh" not in meta.write_options


def test_followup_radar_key_is_empty_without_a_message_identity():
    # a missing X-GM-MSGID must not collapse different messages onto one key
    def cand(msg_id, thread="t1"):
        return followup_radar.Candidate(
            thread_id=thread, sent_message=msg(msg_id, "x", thread_id=thread),
            recipient="bob@example.com", days_waiting=3.0)

    assert followup_radar.memo_key(cand("")) == ""
    assert followup_radar.memo_key(cand("s1", thread="")) == ""
    assert followup_radar.memo_key(cand("s1")) == "t1:s1"
