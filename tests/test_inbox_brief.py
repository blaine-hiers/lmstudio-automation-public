import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from lmsa.digest import RunRow
from tasks import inbox_brief as ib
from tasks.inbox_brief import (
    MESSAGE_LIMIT,
    Triaged,
    build_report,
    classify_all,
    classify_one,
    fetch_messages,
    summary_counts,
)


def _message(**overrides):
    base = dict(
        msg_id="1", thread_id="1", labels=[], flags=[],
        from_name="Alice Example", from_addr="alice@example.com",
        to=["bot@example.com"], reply_to="", subject="Quarterly numbers",
        date=None, list_unsubscribe="", list_id="", auth_results="",
        text="Body text.", attachments=[],
        web_url="https://mail.google.com/mail/u/0/#all/1",
    )
    base.update(overrides)
    from lmsa.sources.gmail import Message
    return Message(**base)


def _triaged(bucket="read", **overrides):
    base = dict(
        msg=_message(), bucket=bucket, why="because", action="do it",
        due_date=None, error=None,
    )
    base.update(overrides)
    return Triaged(**base)


# --- query / fetch -----------------------------------------------------

def test_query_defaults_to_newer_than_1d():
    assert ib._query(None) == "newer_than:1d"


def test_query_uses_hours_flag_as_unix_after():
    # newer_than: has no hour unit in Gmail search; after:<epoch seconds> does.
    assert ib._query(6, now=1_000_000) == "after:978400"


class _FakeGmail:
    def __init__(self, messages):
        self._messages = messages
        self.queries = []
        self.limits = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        self.queries.append(query)
        self.limits.append(limit)
        return self._messages[:limit]


def test_fetch_messages_passes_query_and_cap(monkeypatch):
    fake = _FakeGmail([_message(msg_id=str(i)) for i in range(3)])
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: fake)

    messages, truncated = fetch_messages(None)

    assert fake.queries == ["newer_than:1d"]
    assert fake.limits == [MESSAGE_LIMIT]
    assert len(messages) == 3
    assert truncated is False


def test_fetch_messages_flags_truncation_at_cap(monkeypatch):
    fake = _FakeGmail([_message(msg_id=str(i)) for i in range(MESSAGE_LIMIT)])
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: fake)

    _, truncated = fetch_messages(None)

    assert truncated is True


# --- classify_one / classify_all / summary_counts -----------------------

def test_classify_one_wraps_classify_result(monkeypatch):
    msg = _message()
    monkeypatch.setattr(
        ib, "classify",
        lambda m, lm, memo=None: {"bucket": "act", "why": "pay up", "action": "pay it", "due_date": "2026-10-01"},
    )

    result = classify_one(msg, lm=object())

    assert result == Triaged(
        msg=msg, bucket="act", why="pay up", action="pay it", due_date="2026-10-01",
    )


def test_classify_one_isolates_a_failure(monkeypatch):
    msg = _message()
    monkeypatch.setattr(ib, "classify", lambda m, lm, memo=None: (_ for _ in ()).throw(RuntimeError("boom")))

    result = classify_one(msg, lm=object())

    assert result.error == "boom"
    assert result.bucket is None
    assert result.msg is msg


def test_classify_all_isolates_one_bad_message_among_good_ones(monkeypatch):
    good = _message(msg_id="good")
    bad = _message(msg_id="bad")

    def fake_classify(m, lm, memo=None):
        if m.msg_id == "bad":
            raise RuntimeError("boom")
        return {"bucket": "read", "why": "fyi", "action": "none", "due_date": None}

    monkeypatch.setattr(ib, "classify", fake_classify)

    results = classify_all([good, bad], lm=object())

    assert results[0].bucket == "read"
    assert results[0].error is None
    assert results[1].bucket is None
    assert results[1].error == "boom"


def test_summary_counts_tallies_each_bucket_and_failures():
    triaged = [
        _triaged(bucket="act"),
        _triaged(bucket="act"),
        _triaged(bucket="reply"),
        _triaged(bucket="ignore"),
        _triaged(bucket=None, error="boom"),
    ]

    counts = summary_counts(triaged)

    assert counts == {
        "act": 2, "reply": 1, "read": 0, "ignore": 1, "failed": 1, "total": 5,
    }


# --- build_report --------------------------------------------------------

def test_build_report_routes_buckets_to_the_right_component():
    triaged = [
        _triaged(bucket="act", due_date="2026-09-30"),
        _triaged(bucket="reply"),
        _triaged(bucket="read"),
        _triaged(bucket="ignore"),
    ]

    html = build_report(triaged, query="newer_than:1d", truncated=False).render()

    assert "Act" in html
    assert "Reply" in html
    assert "Read" in html
    assert "Ignore" in html  # tile label -- ignored messages get no section, just the count
    assert "due 2026-09-30" in html


def test_build_report_escapes_untrusted_email_content():
    triaged = [_triaged(bucket="act", msg=_message(subject="<script>alert(1)</script>"))]

    html = build_report(triaged, query="newer_than:1d", truncated=False).render()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_report_shows_truncation_warning():
    html = build_report([], query="newer_than:1d", truncated=True).render()
    assert "truncated" in html
    assert "narrower --hours" in html


def test_build_report_shows_failures_separately():
    triaged = [
        _triaged(bucket="read"),
        _triaged(bucket=None, why=None, action=None, due_date=None, error="server returned empty content"),
    ]

    html = build_report(triaged, query="newer_than:1d", truncated=False).render()

    assert "Failed to classify" in html
    assert "server returned empty content" in html


def test_build_report_empty_inbox_renders_clean_state():
    html = build_report([], query="newer_than:1d", truncated=False).render()
    assert "inbox is quiet" in html


# --- main() end to end, everything faked ---------------------------------

class _FakeLM:
    def __init__(self, output_dir):
        self.cfg = SimpleNamespace(output_dir=output_dir)

    def ensure_server(self):
        pass


def test_main_runs_end_to_end_and_records_the_run(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))

    messages = [_message(msg_id="1", subject="Invoice due"), _message(msg_id="2", subject="FYI")]
    fake_gmail = _FakeGmail(messages)
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(ib, "LMStudio", lambda: _FakeLM(out_dir))

    def fake_classify(m, lm, memo=None):
        if m.msg_id == "1":
            return {"bucket": "act", "why": "pay it", "action": "pay", "due_date": "2026-10-01"}
        return {"bucket": "ignore", "why": "noise", "action": "none", "due_date": None}

    monkeypatch.setattr(ib, "classify", fake_classify)
    monkeypatch.setattr("sys.argv", ["inbox_brief"])

    ib.main()

    [written] = [p for p in out_dir.rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "Invoice due" in html
    assert "pay it" in html

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, summary_json FROM runs").fetchone()
    assert row[0] == "ok"
    assert '"act": 1' in row[1]
    assert '"ignore": 1' in row[1]


# --- digest() (#104: master report hook) ----------------------------------


class _FakeLMWithMemo:
    """Like `_FakeLM`, but with a real `.extract()` keyed off the message
    subject in the prompt -- so the *real* `lmsa.triage.classify()` runs and
    its `cached()` call actually writes the "triage" memo (`llm_memo`), the
    same as a real run. digest() then reads that memo back; it never calls
    this (or anything else) itself."""

    def __init__(self, output_dir, answers_by_subject: dict):
        self.cfg = SimpleNamespace(output_dir=output_dir, chat_model="fake-model")
        self._answers = answers_by_subject

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, name="result"):
        for subject, answer in self._answers.items():
            if subject in prompt:
                return answer
        raise AssertionError(f"no fake answer configured for prompt: {prompt[:80]!r}")


DIGEST_ANSWERS = {
    "Contract to sign": {"bucket": "act", "why": "sign the contract", "action": "sign", "due_date": "2020-01-01"},
    "Quick question": {"bucket": "reply", "why": "answer the question", "action": "reply", "due_date": "2020-01-01"},
    "Newsletter": {"bucket": "ignore", "why": "noise", "action": "none", "due_date": None},
}


def _run_main_for_digest(monkeypatch, tmp_path, *, answers=None):
    """Runs main() end to end (faked Gmail, real classify()/memo against a
    faked model) so handled_seen/llm_memo hold real stored rows, then hands
    back the resulting RunRow."""
    out_dir = tmp_path / "output"
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))

    messages = [
        _message(msg_id="1", subject="Contract to sign", from_addr="alice@example.com"),
        _message(msg_id="2", subject="Quick question", from_addr="bob@example.com"),
        _message(msg_id="3", subject="Newsletter", from_addr="ads@example.com"),
    ]
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: _FakeGmail(messages))
    monkeypatch.setattr(ib, "LMStudio", lambda: _FakeLMWithMemo(out_dir, answers or DIGEST_ANSWERS))
    monkeypatch.setattr("sys.argv", ["inbox_brief"])
    ib.main()

    with sqlite3.connect(db_path) as conn:
        cols = conn.execute(
            "SELECT id, task, started, status, finished, report_path, summary_json FROM runs"
        ).fetchone()
    return RunRow(id=cols[0], task=cols[1], started=cols[2], status=cols[3],
                  finished=cols[4], report_path=cols[5], summary_json=cols[6])


def test_digest_orders_act_before_reply_with_stored_why(monkeypatch, tmp_path):
    run = _run_main_for_digest(monkeypatch, tmp_path)

    d = ib.digest(run, max_items=5)

    assert [t.label for t in d.tiles] == ["New", "Important", "Needs reply"]
    assert [t.value for t in d.tiles] == [3, 1, 1]
    assert [i.title for i in d.items] == ["Contract to sign", "Quick question"]
    assert d.items[0].blurb == "alice@example.com · sign the contract"
    assert d.items[0].url == "https://mail.google.com/mail/u/0/#all/1"
    assert d.items[0].tone == "hot"
    assert d.items[1].tone == "warn"
    # Both due 2020-01-01 (well before this run); only the reply is flagged.
    assert d.callouts == ["1 reply is overdue"]


def test_digest_respects_max_items(monkeypatch, tmp_path):
    run = _run_main_for_digest(monkeypatch, tmp_path)

    d = ib.digest(run, max_items=1)

    assert len(d.items) == 1
    assert d.items[0].title == "Contract to sign"


def test_digest_drops_checked_off_messages(monkeypatch, tmp_path):
    run = _run_main_for_digest(monkeypatch, tmp_path)
    with ib.Store() as store:
        ib.handled.mark(store.conn, "1", source_task="inbox_brief")

    d = ib.digest(run, max_items=5)

    assert [i.title for i in d.items] == ["Quick question"]


def test_digest_survives_a_later_run_that_fails_after_reshowing_the_card(monkeypatch, tmp_path):
    """Regression (#104 review round 2): `handled_seen` is keyed by message,
    not (task, run) -- `lmsa.handled.seen()` moves a re-shown card's row to
    the newest run's id (`ON CONFLICT ... SET run_id`), even if that run
    then fails. Ok run 1 shows "Contract to sign"; run 2 re-fetches the same
    open mail (moving the row's run_id to run 2) and then fails right after
    `handled.seen()` commits (main() calls it before `add_footer()`/
    `report.write()`). The master report falls back to run 1 (the latest
    ok run) -- its digest must still show the still-open card, not drop it
    just because a later, failed run touched the row."""
    run1 = _run_main_for_digest(monkeypatch, tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("simulated failure after handled.seen()")

    monkeypatch.setattr(ib, "add_footer", _boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        ib.main()  # run 2: re-shows the same mail, moves handled_seen.run_id, then fails

    with sqlite3.connect(os.environ["LMSA_DB"]) as conn:
        run2_status = conn.execute(
            "SELECT status FROM runs WHERE id != ? ORDER BY id DESC LIMIT 1", (run1.id,)
        ).fetchone()[0]
    assert run2_status == "error"

    d = ib.digest(run1, max_items=5)  # digest of the *old*, still-latest-ok run

    assert [i.title for i in d.items] == ["Contract to sign", "Quick question"]


def test_digest_tiles_agree_with_the_open_items_listed(monkeypatch, tmp_path):
    """Important / Needs reply count the open cards the digest lists, so a
    later run adding an open card can't leave the tiles under-stating the
    items shown beneath them (#104 PR review)."""
    run1 = _run_main_for_digest(monkeypatch, tmp_path)
    with sqlite3.connect(os.environ["LMSA_DB"]) as conn:
        conn.execute(
            "INSERT INTO handled_seen (source_task, message_id, run_id, subject, sender, web_url, first_seen, last_seen) "
            "VALUES ('inbox_brief', 'later-act', ?, 'Later contract', 'b@example.com', '', "
            "'2026-09-28T07:00:00', '2026-09-28T07:00:00')",
            (run1.id + 1,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO llm_memo (task, item_key, version, value_json, created, last_seen) "
            "VALUES (?, 'later-act', 'test', ?, '2026-09-28T07:00:00', '2026-09-28T07:00:00')",
            (ib.MEMO_TASK, json.dumps({"bucket": "act", "why": "sign it", "due_date": ""})),
        )
    d = ib.digest(run1, max_items=10)
    tiles = {t.label: t.value for t in d.tiles}
    assert tiles["Important"] == sum(1 for i in d.items if i.tone == "hot") == 2
    assert tiles["Needs reply"] == sum(1 for i in d.items if i.tone == "warn")


def test_digest_never_touches_a_model_or_network_client(monkeypatch, tmp_path):
    run = _run_main_for_digest(monkeypatch, tmp_path)

    class _Explosive:
        def __getattr__(self, name):
            raise AssertionError(f"digest() must not use {name!r} -- no model/network calls")

    monkeypatch.setattr(ib, "LMStudio", lambda: _Explosive())
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: _Explosive())

    d = ib.digest(run, max_items=5)
    assert not d.is_empty()


def test_digest_escapes_hostile_stored_text(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    hostile = "<script>alert(1)</script>"
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: _FakeGmail(
        [_message(msg_id="1", subject=hostile, from_addr="alice@example.com")]
    ))
    monkeypatch.setattr(ib, "LMStudio", lambda: _FakeLMWithMemo(
        out_dir, {hostile: {"bucket": "act", "why": hostile, "action": "x", "due_date": None}}
    ))
    monkeypatch.setattr("sys.argv", ["inbox_brief"])
    ib.main()
    with sqlite3.connect(db_path) as conn:
        cols = conn.execute(
            "SELECT id, task, started, status, finished, report_path, summary_json FROM runs"
        ).fetchone()
    run = RunRow(id=cols[0], task=cols[1], started=cols[2], status=cols[3],
                  finished=cols[4], report_path=cols[5], summary_json=cols[6])

    d = ib.digest(run, max_items=5)
    assert d.items[0].title == hostile  # digest() hands back plain, unescaped text ...

    # ... same as any other Card title/body: the renderer (`Card.render()`,
    # which every digest item goes through -- master_report.build_full_report)
    # is what's responsible for escaping it before it reaches HTML.
    html = ib.Card(title=d.items[0].title, body=d.items[0].blurb).render()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
