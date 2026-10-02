"""tasks.reply_drafts: draft (never send) replies for the triage "reply" bucket.

Fakes Gmail (search + append_draft) and the chat model, per this repo's
testing convention -- no live IMAP, no live LM Studio. `test_gmail_source.py`
already proves `append_draft` only ever issues IMAP APPEND, and that the
Gmail source unfolds Message-ID/References/In-Reply-To; the tests here prove
this task calls append_draft (or doesn't, in dry-run) at the right times,
never drafts to itself or an automated address, and gets threading and the
partial-write recovery story right.
"""

from __future__ import annotations

import email.policy
import sqlite3
import sys
from datetime import datetime, timezone
from email import message_from_bytes
from types import SimpleNamespace

import pytest

import tasks.reply_drafts as rd
from lmsa.sources.gmail import Message
from lmsa.store import Store


def _message(**overrides) -> Message:
    base = dict(
        msg_id="1", thread_id="1", labels=[], flags=[],
        from_name="Alice Example", from_addr="alice@example.com",
        to=["bot@example.com"], reply_to="", subject="Can we push the demo?",
        date=None, list_unsubscribe="", list_id="", auth_results="",
        text="Can we push Thursday's demo to Friday?", attachments=[],
        web_url="https://mail.google.com/mail/u/0/#all/1",
        message_id="<orig@example.com>", references="", in_reply_to="",
        reply_to_all=(),
    )
    base.update(overrides)
    return Message(**base)


class _FakeGmail:
    """Exposes search/append_draft/sent/cfg -- the surface this task actually uses."""

    def __init__(self, messages: list[Message], *, sent: list[Message] | None = None, imap_user: str = "bot@example.com"):
        self._messages = messages
        self._sent = sent or []
        self.cfg = SimpleNamespace(imap_user=imap_user)
        self.searches: list[tuple] = []
        self.appended: list[bytes] = []
        self.fail_append = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def sent(self) -> str:
        return "[Gmail]/Sent Mail"

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        self.searches.append((query, folder, limit, headers_only))
        if folder == "[Gmail]/Sent Mail":
            return self._sent[:limit]
        return self._messages[:limit]

    def drafts_folder(self) -> str:
        if getattr(self, "no_drafts_folder", False):
            raise RuntimeError(r"no folder advertises the \Drafts special-use flag")
        return "[Gmail]/Drafts"

    def append_draft(self, message_bytes: bytes, folder: str | None = None) -> None:
        if getattr(self, "abort_after_store", False):
            from imapclient.exceptions import IMAPClientAbortError
            self.appended.append(message_bytes)  # the server stored it...
            raise IMAPClientAbortError("socket EOF")  # ...then the connection dropped
        if getattr(self, "fail_append_no", False):
            from imapclient.exceptions import IMAPClientError
            raise IMAPClientError("APPEND NO [TRYCREATE]")
        if self.fail_append:
            raise RuntimeError("IMAP APPEND failed")
        self.appended.append(message_bytes)


class _FakeLM:
    def __init__(self, output_dir=None, reply_text: str = "Friday works for me, see you then."):
        self.cfg = SimpleNamespace(output_dir=output_dir)
        self.reply_text = reply_text
        self.chat_calls: list[dict] = []

    def ensure_server(self):
        pass

    def chat(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.chat_calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        return self.reply_text


def _db(tmp_path) -> Store:
    return Store(tmp_path / "lmsa.sqlite")


def _parse(raw: bytes):
    return message_from_bytes(raw, policy=email.policy.default)


# --- is_automated_sender / reply_subject --------------------------------------


@pytest.mark.parametrize(
    "addr",
    [
        "noreply@example.com", "no-reply@example.com", "do-not-reply@example.com",
        "DoNotReply@Example.com", "notifications@example.com", "mailer-daemon@example.com",
        "postmaster@example.com",
    ],
)
def test_is_automated_sender_matches_common_patterns(addr):
    assert rd.is_automated_sender(addr) is True


def test_is_automated_sender_false_for_a_person():
    assert rd.is_automated_sender("alice@example.com") is False


def test_reply_subject_adds_re_prefix_once():
    assert rd.reply_subject("Can we push the demo?") == "Re: Can we push the demo?"


@pytest.mark.parametrize("subject", ["Re: Already replying", "RE:Already replying", "re : Already replying"])
def test_reply_subject_does_not_double_re(subject):
    once = rd.reply_subject(subject)
    assert rd.reply_subject(once) == once  # idempotent -- a second pass adds no second prefix


def test_reply_subject_handles_empty_subject():
    assert rd.reply_subject("") == "Re:"


# --- own_addresses / is_own_mail (finding #2: don't draft to yourself) ---------


def test_own_addresses_includes_imap_user_and_sent_from_addrs():
    sent = [_message(from_addr="bot@example.com"), _message(from_addr="Bot+Alias@Example.com")]
    own = rd.own_addresses(sent, "bot@example.com")
    assert own == {"bot@example.com", "bot+alias@example.com"}


def test_own_addresses_handles_empty_sent_and_user():
    assert rd.own_addresses([], "") == set()
    assert rd.own_addresses([], "bot@example.com") == {"bot@example.com"}


def test_is_own_mail_true_for_sent_label():
    msg = _message(from_addr="alice@example.com", labels=["\\Sent"])
    assert rd.is_own_mail(msg, own=set()) is True


def test_is_own_mail_true_for_draft_label_or_flag():
    assert rd.is_own_mail(_message(labels=["\\Draft"]), own=set()) is True
    assert rd.is_own_mail(_message(flags=["\\Draft"]), own=set()) is True
    assert rd.is_own_mail(_message(labels=["\\Drafts"]), own=set()) is True


def test_is_own_mail_true_for_own_from_addr():
    msg = _message(from_addr="bot@example.com")
    assert rd.is_own_mail(msg, own={"bot@example.com"}) is True


def test_is_own_mail_false_for_an_ordinary_message():
    msg = _message(from_addr="alice@example.com", labels=["\\Inbox"])
    assert rd.is_own_mail(msg, own={"bot@example.com"}) is False


def test_process_reply_bucket_excludes_own_mail_before_classifying(tmp_path, monkeypatch):
    own_msg = _message(msg_id="own", from_addr="bot@example.com")
    sent_msg = _message(msg_id="sent", labels=["\\Sent"])
    person_msg = _message(msg_id="person", from_addr="alice@example.com")

    classify_calls = []

    def fake_classify(msg, lm, memo=None):
        classify_calls.append(msg.msg_id)
        return {"bucket": "reply", "why": "", "action": "", "due_date": None}

    monkeypatch.setattr(rd, "classify", fake_classify)

    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        classified, drafts, skipped_own = rd.process_reply_bucket(
            gmail, db.conn, lm, [own_msg, sent_msg, person_msg], own={"bot@example.com"}, apply=False
        )

    assert skipped_own == 2
    assert classify_calls == ["person"]  # never classified the owner's own mail
    assert len(classified) == 1
    assert len(drafts) == 1
    assert drafts[0].msg.msg_id == "person"


# --- resolved_recipients / reply_to_mismatches_sender (findings #3, #6, #7) ----


def test_resolved_recipients_uses_all_reply_to_addresses():
    msg = _message(reply_to_all=("a@example.com", "b@example.com"))
    assert rd.resolved_recipients(msg) == ["a@example.com", "b@example.com"]


def test_resolved_recipients_falls_back_to_from_when_no_reply_to():
    msg = _message(from_addr="alice@example.com", reply_to_all=())
    assert rd.resolved_recipients(msg) == ["alice@example.com"]


def test_reply_to_mismatches_sender_true_for_different_domain():
    msg = _message(from_addr="alice@example.com", reply_to_all=("alice@evil.example",))
    assert rd.reply_to_mismatches_sender(msg) is True


def test_reply_to_mismatches_sender_false_for_same_domain():
    msg = _message(from_addr="alice@example.com", reply_to_all=("alice+support@example.com",))
    assert rd.reply_to_mismatches_sender(msg) is False


def test_reply_to_mismatches_sender_false_when_no_reply_to():
    msg = _message(from_addr="alice@example.com", reply_to_all=())
    assert rd.reply_to_mismatches_sender(msg) is False


def test_draft_one_flags_reply_to_mismatch_but_still_drafts(tmp_path):
    """Finding #7: mismatch is surfaced, never silently redirected or blocked."""
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        msg = _message(from_addr="alice@example.com", reply_to_all=("alice@evil.example",))
        d = rd.draft_one(gmail, db.conn, lm, msg, apply=False)

    assert d.reply_to_mismatch is True
    assert d.status == "dry_run"  # still drafted -- flagged, not blocked
    assert d.to == "alice@evil.example"


def test_draft_one_drafts_to_all_reply_to_addresses(tmp_path):
    """Finding #6."""
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        msg = _message(reply_to_all=("a@example.com", "b@example.com"))
        d = rd.draft_one(gmail, db.conn, lm, msg, apply=False)

    assert d.to == "a@example.com, b@example.com"


def test_draft_one_checks_automated_on_resolved_recipient_not_just_from(tmp_path):
    """Finding #3: From is a real person, but Reply-To is automated -- the
    reply would actually land on the automated address."""
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        msg = _message(from_addr="alice@example.com", reply_to_all=("noreply@example.com",))
        d = rd.draft_one(gmail, db.conn, lm, msg, apply=True)

    assert d.status == "skipped_automated"
    assert lm.chat_calls == []
    assert gmail.appended == []


# --- build_eml / build_references (findings #1, #5) -----------------------------


def test_build_eml_sets_threading_headers_from_message_id_and_references():
    msg = _message(message_id="<orig@example.com>", references="<root@example.com>")
    raw = rd.build_eml(msg, from_addr="bot@example.com", to="alice@example.com", subject="Re: hi", body="Sure.")
    parsed = _parse(raw)

    assert parsed["To"] == "alice@example.com"
    assert parsed["From"] == "bot@example.com"
    assert parsed["Subject"] == "Re: hi"
    assert parsed["In-Reply-To"] == "<orig@example.com>"
    assert parsed["References"] == "<root@example.com> <orig@example.com>"
    assert "Sure." in parsed.get_content()


def test_build_eml_references_falls_back_to_in_reply_to_when_original_has_no_references():
    """RFC 5322 sec 3.6.4: References = parent's References, else parent's
    In-Reply-To, then the parent's own Message-ID appended."""
    msg = _message(message_id="<orig@example.com>", references="", in_reply_to="<parent@example.com>")
    raw = rd.build_eml(msg, from_addr="bot@example.com", to="alice@example.com", subject="Re: hi", body="Sure.")
    parsed = _parse(raw)
    assert parsed["References"] == "<parent@example.com> <orig@example.com>"


def test_build_eml_references_falls_back_to_message_id_alone_when_original_has_neither():
    msg = _message(message_id="<orig@example.com>", references="", in_reply_to="")
    raw = rd.build_eml(msg, from_addr="bot@example.com", to="alice@example.com", subject="Re: hi", body="Sure.")
    parsed = _parse(raw)
    assert parsed["References"] == "<orig@example.com>"


def test_build_eml_omits_threading_headers_when_original_has_no_message_id():
    msg = _message(message_id="", references="", in_reply_to="")
    raw = rd.build_eml(msg, from_addr="bot@example.com", to="alice@example.com", subject="Re: hi", body="Sure.")
    parsed = _parse(raw)
    assert parsed["In-Reply-To"] is None
    assert parsed["References"] is None


def test_build_eml_handles_a_crlf_folded_references_header_without_raising():
    """The exact bug from review finding #1: a folded References must not
    make EmailMessage raise "Header values may not contain linefeed"."""
    msg = _message(
        message_id="<abc123@example.com>",
        references="<root@example.com>\r\n <mid1@example.com>\r\n <mid2@example.com>",
    )
    raw = rd.build_eml(msg, from_addr="bot@example.com", to="alice@example.com", subject="Re: hi", body="Sure.")
    parsed = _parse(raw)
    assert "\r" not in parsed["References"] and "\n" not in parsed["References"]
    assert parsed["References"] == (
        "<root@example.com> <mid1@example.com> <mid2@example.com> <abc123@example.com>"
    )


def test_build_references_normalises_ids_missing_angle_brackets():
    msg = _message(message_id="orig@example.com", references="root@example.com")
    assert rd.build_references(msg) == "<root@example.com> <orig@example.com>"


def test_build_references_dedupes_a_repeated_message_id():
    msg = _message(message_id="<a@example.com>", references="<root@example.com> <a@example.com>")
    assert rd.build_references(msg) == "<root@example.com> <a@example.com>"


def test_build_references_trims_a_very_long_chain_keeping_first_and_last():
    ids = " ".join(f"<{i}@example.com>" for i in range(30))
    msg = _message(message_id="<newest@example.com>", references=ids)
    result = rd.build_references(msg).split()

    assert len(result) == rd.REFERENCES_MAX_IDS
    assert result[0] == "<0@example.com>"  # thread root kept
    assert result[-1] == "<newest@example.com>"  # this message's own id kept, appended last


# --- appending/drafted store + recovery (finding #4) ----------------------------


def test_draft_status_none_when_never_attempted(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        status = rd.draft_status(db.conn, "1")
    assert status is None


def test_mark_appending_then_mark_drafted_transitions_status(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        now = datetime.now(timezone.utc)
        rd.mark_appending(db.conn, "1", now=now)
        mid_status = rd.draft_status(db.conn, "1")
        rd.mark_drafted(db.conn, "1", now=now)
        final_status = rd.draft_status(db.conn, "1")

    assert mid_status == "appending"
    assert final_status == "drafted"


def test_draft_one_apply_calls_append_draft_and_marks_drafted(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        d = rd.draft_one(gmail, db.conn, lm, _message(msg_id="42"), apply=True)
        status = rd.draft_status(db.conn, "42")

    assert d.status == "drafted"
    assert len(gmail.appended) == 1
    parsed = _parse(gmail.appended[0])
    assert parsed["To"] == "alice@example.com"
    assert status == "drafted"


def test_draft_one_append_failure_leaves_status_appending_not_error_retryable(tmp_path):
    """Finding #4: an append that raises must not look like nothing happened
    -- the row stays 'appending' so a later run treats it as possibly
    already drafted rather than blindly retrying (risking a duplicate)."""
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        gmail.fail_append = True
        lm = _FakeLM()
        d = rd.draft_one(gmail, db.conn, lm, _message(msg_id="7"), apply=True)
        status = rd.draft_status(db.conn, "7")

    assert d.status == "error"
    assert status == "appending"


def test_draft_one_mark_drafted_failure_after_successful_append_is_recoverable(tmp_path, monkeypatch):
    """Finding #4, the other half: append succeeds but the DB write that
    would flip 'appending' -> 'drafted' fails. Must not silently look
    fully successful, and must not risk a duplicate append next run."""
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()

        def boom(*a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(rd, "mark_drafted", boom)
        d = rd.draft_one(gmail, db.conn, lm, _message(msg_id="9"), apply=True)
        status = rd.draft_status(db.conn, "9")

    assert d.status == "error"
    assert len(gmail.appended) == 1  # the append really did happen
    assert status == "appending"  # never silently marked drafted, never lost either


def test_draft_one_appending_status_is_possibly_drafted_and_never_reappends(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        rd.mark_appending(db.conn, "1", now=datetime.now(timezone.utc))
        gmail = _FakeGmail([])
        lm = _FakeLM()
        d = rd.draft_one(gmail, db.conn, lm, _message(msg_id="1"), apply=True)

    assert d.status == "possibly_drafted"
    assert lm.chat_calls == []
    assert gmail.appended == []  # never re-attempted automatically


def test_draft_one_dry_run_never_calls_append_draft_or_writes_status(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        d = rd.draft_one(gmail, db.conn, lm, _message(), apply=False)
        status = rd.draft_status(db.conn, "1")

    assert d.status == "dry_run"
    assert d.body == "Friday works for me, see you then."
    assert gmail.appended == []
    assert status is None


def test_draft_one_re_run_after_apply_is_idempotent(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        msg = _message(msg_id="99")
        rd.draft_one(gmail, db.conn, lm, msg, apply=True)
        second = rd.draft_one(gmail, db.conn, lm, msg, apply=True)

    assert second.status == "already_drafted"
    assert len(gmail.appended) == 1


def test_draft_one_isolates_a_missing_address(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        msg = _message(from_addr="", reply_to_all=())
        d = rd.draft_one(gmail, db.conn, lm, msg, apply=True)

    assert d.status == "error"
    assert "reply-to" in d.error.lower()
    assert gmail.appended == []


def test_draft_one_isolates_a_model_failure(tmp_path):
    class _BoomLM(_FakeLM):
        def chat(self, *a, **k):
            raise RuntimeError("model server error")

    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        d = rd.draft_one(gmail, db.conn, _BoomLM(), _message(), apply=True)

    assert d.status == "error"
    assert "model server error" in d.error
    assert gmail.appended == []


def test_draft_one_truncates_a_very_long_model_reply(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM(reply_text="x" * (rd.REPLY_MAX_CHARS + 500))
        d = rd.draft_one(gmail, db.conn, lm, _message(), apply=False)

    assert len(d.body) <= rd.REPLY_MAX_CHARS + len("\n\n[... truncated, 500 more characters omitted ...]")
    assert "truncated" in d.body


# --- draft_reply_text bounds the prompt input -----------------------------------


def test_draft_reply_text_truncates_a_long_body_before_prompting():
    lm = _FakeLM()
    long_body = "y" * (rd.INPUT_LIMIT + 1000)
    rd.draft_reply_text(lm, _message(text=long_body))

    prompt = lm.chat_calls[0]["prompt"]
    assert len(prompt) < len(long_body)
    assert "truncated" in prompt


def test_draft_reply_text_passes_system_prompt_marking_data_not_instructions():
    lm = _FakeLM()
    rd.draft_reply_text(lm, _message())
    assert "not instructions" in lm.chat_calls[0]["system"]


# --- process_reply_bucket -------------------------------------------------------


def test_process_reply_bucket_only_drafts_the_reply_bucket(tmp_path, monkeypatch):
    reply_msg = _message(msg_id="r1", subject="Reply me")
    act_msg = _message(msg_id="a1", subject="Pay invoice")

    def fake_classify(msg, lm, memo=None):
        return {"bucket": "reply" if msg.msg_id == "r1" else "act", "why": "", "action": "", "due_date": None}

    monkeypatch.setattr(rd, "classify", fake_classify)

    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        classified, drafts, skipped_own = rd.process_reply_bucket(
            gmail, db.conn, lm, [reply_msg, act_msg], own=set(), apply=False
        )

    assert len(classified) == 2
    assert skipped_own == 0
    assert len(drafts) == 1
    assert drafts[0].msg.msg_id == "r1"


def test_process_reply_bucket_isolates_a_classify_failure(tmp_path, monkeypatch):
    good = _message(msg_id="good")
    bad = _message(msg_id="bad")

    def fake_classify(msg, lm, memo=None):
        if msg.msg_id == "bad":
            raise RuntimeError("boom")
        return {"bucket": "reply", "why": "", "action": "", "due_date": None}

    monkeypatch.setattr(rd, "classify", fake_classify)

    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        lm = _FakeLM()
        classified, drafts, _skipped_own = rd.process_reply_bucket(
            gmail, db.conn, lm, [good, bad], own=set(), apply=False
        )

    errors = [e for _m, _b, e in classified if e]
    assert errors == ["boom"]
    assert len(drafts) == 1  # the bad one never reached drafting


# --- summary_counts / build_report ----------------------------------------------


def test_summary_counts_tallies_each_status():
    classified = [(_message(msg_id="1"), "reply", None), (_message(msg_id="2"), None, "boom")]
    drafts = [rd.Draft(msg=_message(msg_id="1"), status="drafted")]
    counts = rd.summary_counts(classified, drafts, skipped_own=3)
    assert counts["messages"] == 5
    assert counts["skipped_own_mail"] == 3
    assert counts["drafted"] == 1
    assert counts["errors"] == 1  # the classify failure


def test_build_report_dry_run_notes_no_writes():
    drafts = [rd.Draft(msg=_message(), to="alice@example.com", subject="Re: hi", body="Sure.", status="dry_run")]
    html = rd.build_report(
        [(_message(), "reply", None)], drafts, query="newer_than:1d", truncated=False, apply=False
    ).render()
    assert "Dry-run" in html
    assert "Sure." in html


def test_build_report_escapes_untrusted_draft_and_original_text():
    msg = _message(text="<script>alert(1)</script>")
    drafts = [rd.Draft(msg=msg, to="a@example.com", subject="Re: hi", body="<img onerror=alert(1)>", status="dry_run")]
    html = rd.build_report([(msg, "reply", None)], drafts, query="q", truncated=False, apply=False).render()
    assert "<script>alert(1)</script>" not in html
    assert "<img onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in html


def test_build_report_shows_already_drafted_and_skipped_without_leaking_body():
    already = rd.Draft(msg=_message(msg_id="1"), status="already_drafted")
    skipped = rd.Draft(msg=_message(msg_id="2", from_addr="noreply@example.com"), status="skipped_automated")
    html = rd.build_report([], [already, skipped], query="q", truncated=False, apply=True).render()
    assert "already exists" in html
    assert "automated" in html


def test_build_report_shows_possibly_drafted_needing_manual_check():
    possibly = rd.Draft(msg=_message(msg_id="1"), status="possibly_drafted")
    html = rd.build_report([], [possibly], query="q", truncated=False, apply=True).render()
    assert "Needs manual check" in html
    assert "check Gmail Drafts" in html.lower() or "check gmail drafts" in html.lower()


def test_build_report_flags_reply_to_mismatch_card():
    d = rd.Draft(
        msg=_message(from_addr="alice@example.com", reply_to_all=("alice@evil.example",)),
        to="alice@evil.example", subject="Re: hi", body="Sure.", status="dry_run", reply_to_mismatch=True,
    )
    html = rd.build_report([(_message(), "reply", None)], [d], query="q", truncated=False, apply=False).render()
    assert "reply-to differs from sender" in html


def test_build_report_notes_skipped_own_mail():
    html = rd.build_report([], [], query="q", truncated=False, apply=False, skipped_own=4).render()
    assert "own mail" in html.lower()
    assert "4 message" in html


def test_build_report_empty_reply_bucket():
    html = rd.build_report([], [], query="newer_than:1d", truncated=False, apply=False).render()
    assert "No messages in the reply bucket" in html


def test_build_report_shows_truncation_warning():
    html = rd.build_report([], [], query="q", truncated=True, apply=False).render()
    assert "truncated" in html.lower()


# --- main() end to end -----------------------------------------------------------


def test_main_dry_run_never_appends_and_records_the_run(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))

    msg = _message(msg_id="1", subject="Push the demo?")
    fake_gmail = _FakeGmail([msg])
    monkeypatch.setattr(rd, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(rd, "LMStudio", lambda: _FakeLM(out_dir))
    monkeypatch.setattr(rd, "classify", lambda m, lm, memo=None: {"bucket": "reply", "why": "", "action": "", "due_date": None})
    monkeypatch.setattr(sys, "argv", ["reply_drafts"])

    rd.main()

    assert fake_gmail.appended == []  # dry-run: never writes
    [written] = [p for p in out_dir.rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "Push the demo?" in html
    assert "Friday works for me" in html

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, summary_json FROM runs").fetchone()
    assert row[0] == "ok"
    assert '"dry_run": 1' in row[1]
    assert '"drafted": 0' in row[1]


def test_main_apply_appends_and_is_idempotent_on_rerun(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))

    msg = _message(msg_id="1", subject="Push the demo?")
    fake_gmail = _FakeGmail([msg])
    monkeypatch.setattr(rd, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(rd, "LMStudio", lambda: _FakeLM(out_dir))
    monkeypatch.setattr(rd, "classify", lambda m, lm, memo=None: {"bucket": "reply", "why": "", "action": "", "due_date": None})
    monkeypatch.setattr(sys, "argv", ["reply_drafts", "--apply"])

    rd.main()
    assert len(fake_gmail.appended) == 1

    # Re-run: same message, same msg_id -- must not append a second draft.
    rd.main()
    assert len(fake_gmail.appended) == 1


def test_main_never_drafts_a_reply_to_its_own_sent_mail_or_drafts(monkeypatch, tmp_path):
    """End-to-end regression for finding #2: All Mail includes Sent and
    Drafts, and neither should ever get a reply drafted at it."""
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    own_sent = _message(msg_id="s1", labels=["\\Sent"], from_addr="bot@example.com", subject="Sent by me")
    own_draft = _message(msg_id="d1", labels=["\\Draft"], from_addr="bot@example.com", subject="A draft")
    from_self = _message(msg_id="f1", from_addr="bot@example.com", subject="Self note")
    real = _message(msg_id="r1", from_addr="alice@example.com", subject="Please reply")

    fake_gmail = _FakeGmail([own_sent, own_draft, from_self, real], sent=[])
    monkeypatch.setattr(rd, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(rd, "LMStudio", lambda: _FakeLM(out_dir))

    classified_ids = []

    def fake_classify(msg, lm, memo=None):
        classified_ids.append(msg.msg_id)
        return {"bucket": "reply", "why": "", "action": "", "due_date": None}

    monkeypatch.setattr(rd, "classify", fake_classify)
    monkeypatch.setattr(sys, "argv", ["reply_drafts", "--apply"])

    rd.main()

    assert classified_ids == ["r1"]  # only the real message from a stranger was ever classified
    assert len(fake_gmail.appended) == 1
    parsed = _parse(fake_gmail.appended[0])
    assert parsed["To"] == "alice@example.com"


def test_main_flags_truncation_at_message_limit(monkeypatch, tmp_path):
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    messages = [_message(msg_id=str(i)) for i in range(rd.MESSAGE_LIMIT)]
    fake_gmail = _FakeGmail(messages)
    monkeypatch.setattr(rd, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(rd, "LMStudio", lambda: _FakeLM(out_dir))
    monkeypatch.setattr(rd, "classify", lambda m, lm, memo=None: {"bucket": "ignore", "why": "", "action": "", "due_date": None})
    monkeypatch.setattr(sys, "argv", ["reply_drafts"])

    rd.main()

    [written] = [p for p in out_dir.rglob("*.html") if p.name != "index.html"]
    assert "truncated" in written.read_text(encoding="utf-8").lower()



def test_definite_append_failure_clears_the_intent_row(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        gmail.fail_append_no = True
        d = rd.draft_one(gmail, db.conn, _FakeLM(), _message(msg_id="8"), apply=True)
        assert d.status == "error"
        assert rd.draft_status(db.conn, "8") is None  # retried normally next run


def test_missing_drafts_folder_leaves_no_intent_row(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        gmail.no_drafts_folder = True
        d = rd.draft_one(gmail, db.conn, _FakeLM(), _message(msg_id="9"), apply=True)
        assert d.status == "error"
        assert rd.draft_status(db.conn, "9") is None


def test_clear_status_unsticks_a_possibly_drafted_message(tmp_path):
    from datetime import datetime, timezone

    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        rd.mark_appending(db.conn, "10", now=datetime.now(timezone.utc))
        rd.clear_status(db.conn, "10")
        assert rd.draft_status(db.conn, "10") is None


def test_too_many_reply_to_addresses_is_not_drafted(tmp_path):
    many = tuple(f"p{i}@example.com" for i in range(rd.MAX_RECIPIENTS + 1))
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        d = rd.draft_one(gmail, db.conn, _FakeLM(), _message(msg_id="11", reply_to_all=many), apply=True)
        assert d.status == "error" and "limit" in d.error
        assert gmail.appended == []



def test_dropped_connection_mid_append_never_makes_a_duplicate_draft(tmp_path):
    with _db(tmp_path) as db:
        rd.ensure_schema(db.conn)
        gmail = _FakeGmail([])
        gmail.abort_after_store = True
        first = rd.draft_one(gmail, db.conn, _FakeLM(), _message(msg_id="12"), apply=True)
        assert first.status == "error"
        assert rd.draft_status(db.conn, "12") == "appending"  # outcome unknown: kept
        gmail.abort_after_store = False
        second = rd.draft_one(gmail, db.conn, _FakeLM(), _message(msg_id="12"), apply=True)
        assert second.status == "possibly_drafted"
        assert len(gmail.appended) == 1
