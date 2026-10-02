"""Unit tests for tasks.followup_radar.

`find_unanswered` must never query Gmail per thread -- it works entirely
off one Sent fetch and one All Mail headers-only fetch, both already in
hand, and decides "answered" by comparing dates/addresses in memory.
"""

from __future__ import annotations

from types import SimpleNamespace

from datetime import datetime, timedelta, timezone

import pytest

from lmsa.sources.gmail import Message
from tasks import followup_radar as radar
from tasks.followup_radar import (
    Candidate,
    build_report,
    classify,
    find_unanswered,
    own_addresses,
)

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def _msg(
    *,
    msg_id="m1",
    thread_id="t1",
    from_addr="bob@example.com",
    from_name="Bob",
    to=("me@example.com",),
    subject="Subj",
    date=None,
    text="body text",
    web_url="https://mail.google.com/mail/u/0/#all/1",
    reply_to="",
) -> Message:
    return Message(
        msg_id=msg_id,
        thread_id=thread_id,
        labels=[],
        flags=[],
        from_name=from_name,
        from_addr=from_addr,
        to=list(to),
        reply_to=reply_to,
        subject=subject,
        date=date,
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text=text,
        attachments=[],
        web_url=web_url,
    )


# -- own_addresses --------------------------------------------------------------


def test_own_addresses_includes_imap_user_and_sent_aliases():
    sent = [
        _msg(from_addr="Me@Example.com"),
        _msg(from_addr="me+alias@example.com"),
    ]
    addrs = own_addresses(sent, "me@example.com")
    assert addrs == {"me@example.com", "me+alias@example.com"}


def test_own_addresses_handles_empty_sent_and_blank_imap_user():
    assert own_addresses([], "") == set()
    assert own_addresses([], "  ") == set()


# -- find_unanswered --------------------------------------------------------------


def test_find_unanswered_flags_thread_with_no_later_reply():
    sent = [_msg(thread_id="t1", date=NOW - timedelta(days=5), to=["bob@example.com"])]
    all_mail = [sent[0]]  # Sent items also appear in All Mail
    own = {"me@example.com"}

    candidates = find_unanswered(sent, all_mail, own, NOW, min_days=3)

    assert len(candidates) == 1
    c = candidates[0]
    assert isinstance(c, Candidate)
    assert c.thread_id == "t1"
    assert c.recipient == "bob@example.com"
    assert c.days_waiting == pytest.approx(5.0, abs=0.01)


def test_find_unanswered_excludes_thread_with_later_reply_from_someone_else():
    sent_msg = _msg(thread_id="t1", date=NOW - timedelta(days=5))
    reply = _msg(
        thread_id="t1",
        from_addr="bob@example.com",
        date=NOW - timedelta(days=2),
    )
    own = {"me@example.com"}

    candidates = find_unanswered([sent_msg], [sent_msg, reply], own, NOW, min_days=3)

    assert candidates == []


def test_find_unanswered_ignores_a_later_message_that_is_from_self():
    # e.g. a second message you sent yourself into the thread (a self-CC,
    # or a duplicate of the same Sent item echoed back in All Mail) must
    # not count as "someone answered".
    sent_msg = _msg(thread_id="t1", date=NOW - timedelta(days=5))
    later_self = _msg(
        thread_id="t1", from_addr="me@example.com", date=NOW - timedelta(days=1)
    )
    own = {"me@example.com"}

    candidates = find_unanswered([sent_msg], [sent_msg, later_self], own, NOW, min_days=3)

    assert len(candidates) == 1


def test_find_unanswered_excludes_threads_under_the_day_threshold():
    sent_msg = _msg(thread_id="t1", date=NOW - timedelta(days=1))
    own = {"me@example.com"}

    candidates = find_unanswered([sent_msg], [sent_msg], own, NOW, min_days=3)

    assert candidates == []


def test_find_unanswered_uses_the_latest_sent_message_per_thread():
    older = _msg(thread_id="t1", msg_id="a", date=NOW - timedelta(days=10), subject="old")
    newer = _msg(thread_id="t1", msg_id="b", date=NOW - timedelta(days=5), subject="new")
    own = {"me@example.com"}

    candidates = find_unanswered([older, newer], [older, newer], own, NOW, min_days=3)

    assert len(candidates) == 1
    assert candidates[0].sent_message.subject == "new"
    assert candidates[0].days_waiting == pytest.approx(5.0, abs=0.01)


def test_find_unanswered_skips_messages_missing_thread_id_or_date():
    no_thread = _msg(thread_id="", date=NOW - timedelta(days=5))
    no_date = _msg(thread_id="t2", date=None)
    own = {"me@example.com"}

    candidates = find_unanswered([no_thread, no_date], [], own, NOW, min_days=3)

    assert candidates == []


def test_find_unanswered_recipient_falls_back_to_reply_to_then_unknown():
    with_to = _msg(thread_id="t1", date=NOW - timedelta(days=5), to=["bob@example.com"])
    no_to = _msg(thread_id="t2", date=NOW - timedelta(days=5), to=[], reply_to="carol@example.com")
    no_to_no_reply = _msg(thread_id="t3", date=NOW - timedelta(days=5), to=[])
    own = {"me@example.com"}

    candidates = find_unanswered(
        [with_to, no_to, no_to_no_reply], [with_to, no_to, no_to_no_reply], own, NOW, min_days=3
    )
    by_thread = {c.thread_id: c.recipient for c in candidates}

    assert by_thread["t1"] == "bob@example.com"
    assert by_thread["t2"] == "carol@example.com"
    assert by_thread["t3"] == "unknown"


# -- classify --------------------------------------------------------------------


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


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        thread_id="t1",
        sent_message=_msg(thread_id="t1", subject="Need the report", text="Can you send the Q3 numbers?"),
        recipient="bob@example.com",
        days_waiting=5.0,
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_classify_expects_reply_true_carries_ask_summary():
    lm = FakeLM({"expects_reply": True, "ask_summary": "Send the Q3 numbers."})
    candidate = _candidate()

    result = classify(lm, candidate)

    assert result.expects_reply is True
    assert result.ask_summary == "Send the Q3 numbers."
    assert result.error is None
    assert result.days_waiting == 5.0
    assert "Need the report" in lm.calls[0][0]
    assert lm.calls[0][2] == radar.SYSTEM


def test_classify_expects_reply_false_drops_ask_summary():
    lm = FakeLM({"expects_reply": False, "ask_summary": "irrelevant"})

    result = classify(lm, _candidate())

    assert result.expects_reply is False
    assert result.ask_summary == ""


def test_classify_isolates_extract_exception():
    lm = FakeLM(error=RuntimeError("server returned empty content"))

    result = classify(lm, _candidate())

    assert result.error == "server returned empty content"
    assert result.expects_reply is False


def test_classify_truncates_long_body_in_prompt():
    long_text = "x" * (radar.BODY_LIMIT + 500)
    candidate = _candidate(sent_message=_msg(thread_id="t1", text=long_text))
    lm = FakeLM({"expects_reply": True, "ask_summary": "ok"})

    classify(lm, candidate)

    prompt = lm.calls[0][0]
    assert len(prompt) < len(long_text)
    assert "truncated" in prompt


# -- build_report --------------------------------------------------------------


def _followup(**overrides):
    from tasks.followup_radar import FollowUp

    defaults = dict(
        thread_id="t1",
        recipient="bob@example.com",
        subject="Need the report",
        days_waiting=5.0,
        web_url="https://mail.google.com/mail/u/0/#all/1",
        expects_reply=True,
        ask_summary="Send the Q3 numbers.",
    )
    defaults.update(overrides)
    return FollowUp(**defaults)


def test_build_report_empty_state():
    r = build_report([], [], days=3)
    html = r.render()
    assert "all caught up" in html
    assert "Waiting on a reply" not in html


def test_build_report_lists_waiting_cards_sorted_by_age_descending():
    a = _followup(thread_id="t1", subject="Short wait", days_waiting=4.0)
    b = _followup(thread_id="t2", subject="Long wait", days_waiting=9.0)

    r = build_report([a, b], [], days=3)
    html = r.render()

    assert html.index("Long wait") < html.index("Short wait")
    assert "1 waiting" not in html  # sanity: don't collide with the "waiting" tile text
    assert "2 waiting" in html


def test_build_report_tone_escalates_warn_to_hot_after_seven_days():
    warm = _followup(thread_id="t1", days_waiting=7.0)
    hot = _followup(thread_id="t2", days_waiting=7.1)

    r = build_report([warm, hot], [], days=3)
    html = r.render()

    assert 'class="card t-hot"' in html
    assert 'class="card t-warn"' in html


def test_build_report_escapes_untrusted_subject_and_recipient():
    f = _followup(subject="<script>alert(1)</script>", recipient="<img onerror=x>")
    r = build_report([f], [], days=3)
    html = r.render()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img onerror=x>" not in html


def test_build_report_shows_failures_separately():
    ok = _followup(thread_id="t1")
    failed = _followup(thread_id="t2", expects_reply=False, ask_summary="", error="server timeout")

    r = build_report([ok], [failed], days=3)
    html = r.render()

    assert "Classification errors" in html
    assert "server timeout" in html


def test_build_report_shows_truncation_warning():
    r = build_report([], [], days=3, sent_truncated=True, all_mail_truncated=False)
    html = r.render()
    assert "truncated" in html
    assert "Sent" in html


# -- main() end to end -----------------------------------------------------------


def test_main_makes_exactly_one_sent_and_one_all_mail_query(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    sent_msg = _msg(
        thread_id="t1",
        date=now - timedelta(days=5),
        to=["bob@example.com"],
        from_addr="me@example.com",
        subject="Need the report",
        text="Can you send the Q3 numbers?",
    )

    class FakeGmail:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def sent(self):
            return "[Gmail]/Sent Mail"

        def search(self, query, folder=None, limit=200, *, headers_only=False):
            self.calls.append(
                {"query": query, "folder": folder, "limit": limit, "headers_only": headers_only}
            )
            return [sent_msg] if not headers_only else [sent_msg]

    fake_gmail = FakeGmail()
    monkeypatch.setattr(radar, "Gmail", lambda: fake_gmail)

    class FakeLM:
        def ensure_server(self):
            pass

        def extract(self, prompt, schema, *, system=None, name="result"):
            return {"expects_reply": True, "ask_summary": "Send the Q3 numbers."}

    monkeypatch.setattr(radar, "LMStudio", lambda: FakeLM())
    monkeypatch.setenv("LMSA_IMAP_USER", "me@example.com")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    radar.main([])

    assert len(fake_gmail.calls) == 2
    sent_call = next(c for c in fake_gmail.calls if not c["headers_only"])
    all_mail_call = next(c for c in fake_gmail.calls if c["headers_only"])
    assert sent_call["folder"] == "[Gmail]/Sent Mail"

    [written] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "1 waiting" in html
    assert "Need the report" in html
    assert "Send the Q3 numbers." in html


def test_main_isolates_a_classification_failure(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    sent_msg = _msg(
        thread_id="t1",
        date=now - timedelta(days=5),
        from_addr="me@example.com",
        subject="Broken one",
    )

    class FakeGmail:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def sent(self):
            return "[Gmail]/Sent Mail"

        def search(self, query, folder=None, limit=200, *, headers_only=False):
            return [sent_msg]

    monkeypatch.setattr(radar, "Gmail", lambda: FakeGmail())

    class FailingLM:
        def ensure_server(self):
            pass

        def extract(self, prompt, schema, *, system=None, name="result"):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(radar, "LMStudio", lambda: FailingLM())
    monkeypatch.setenv("LMSA_IMAP_USER", "me@example.com")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    radar.main([])  # must not raise

    [written] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "model unavailable" in html
    assert "all caught up" in html  # the failed thread never counts as "waiting"


def test_bounce_and_out_of_office_do_not_count_as_replies():
    from datetime import datetime, timedelta, timezone
    from tasks.followup_radar import _has_later_reply

    sent_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    later = sent_at + timedelta(hours=1)

    def msg(addr, subject):
        return SimpleNamespace(from_addr=addr, subject=subject, date=later)

    own = {"me@example.com"}
    assert not _has_later_reply([msg("mailer-daemon@example.com", "Delivery Status Notification (Failure)")], sent_at, own)
    assert not _has_later_reply([msg("pat@example.com", "Automatic reply: budget")], sent_at, own)
    assert not _has_later_reply([msg("pat@example.com", "Out of Office: back Monday")], sent_at, own)
    assert _has_later_reply([msg("pat@example.com", "Re: budget")], sent_at, own)
