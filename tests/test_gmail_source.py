"""Unit tests for lmsa.sources.gmail.

Two layers of fake, on purpose:

- Most tests fake at the `imapclient.IMAPClient` level (a stand-in object
  with the same methods our Gmail class calls). That's the right boundary
  for testing *our* logic (readonly + PEEK requested, newest-first
  ordering, label/date/address normalisation, etc.) without re-testing
  imapclient's own wire-protocol handling.
- `test_wire_level_readonly_and_peek` instead drives a REAL imapclient
  IMAPClient, backed by a real (unmodified) imaplib.IMAP4 connected to a
  scripted fake socket. That proves the bytes actually placed on the wire
  ask for EXAMINE (not SELECT) and BODY.PEEK[] (not BODY[]) -- the two
  guarantees ("never marks read") that matter most and that a purely
  IMAPClient-level fake can't verify.
"""

import imaplib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import imapclient
import imapclient.imap4 as imap4
from imapclient.exceptions import IMAPClientAbortError, IMAPClientError
from imapclient.imap_utf7 import encode as encode_utf7

from lmsa.config import Settings
from lmsa.sources.gmail import (
    AttachmentNotFoundError,
    AttachmentTooLargeError,
    _decode_header_value,
    Gmail,
    GmailAuthError,
    is_reserved_label,
    LabelWriteError,
)

PLAIN_RAW = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bob@example.com, carol@example.com\r\n"
    b"Reply-To: alice+reply@example.com\r\n"
    b"Subject: Weekly digest\r\n"
    b"Date: Mon, 22 Sep 2025 08:00:00 -0400\r\n"
    b"List-Id: <digest.example.com>\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Hello there, plain text body.\r\n"
)

NAIVE_DATE_RAW = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bob@example.com\r\n"
    b"Subject: Unknown zone\r\n"
    b"Date: Mon, 22 Sep 2025 08:00:00 -0000\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Body.\r\n"
)

HTML_ONLY_RAW = (
    b"From: Newsletter <news@example.com>\r\n"
    b"To: bot@example.com\r\n"
    b"Subject: Sale\r\n"
    b"Date: Tue, 23 Sep 2025 09:00:00 -0400\r\n"
    b"Content-Type: text/html; charset=utf-8\r\n"
    b"\r\n"
    b"<html><body><script>evil()</script><p>Big <b>sale</b> today &amp; more</p></body></html>\r\n"
)

HEADERS_ONLY_TAIL = b"Subject: Weekly digest\r\n\r\n"

THREADED_RAW = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bob@example.com\r\n"
    b"Subject: Weekly digest\r\n"
    b"Date: Mon, 22 Sep 2025 08:00:00 -0400\r\n"
    b"Message-ID: <abc123@example.com>\r\n"
    b"References: <root@example.com> <mid@example.com>\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Hello there, plain text body.\r\n"
)

# A References header folded across physical lines (RFC 5322 sec 2.2.3): a
# CRLF followed by leading whitespace on the continuation line. Real threads
# with more than a couple of hops get sent this way by most mail clients.
CRLF_FOLDED_RAW = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bob@example.com\r\n"
    b"Subject: Thread\r\n"
    b"Date: Mon, 22 Sep 2025 08:00:00 -0400\r\n"
    b"Message-ID: <abc123@example.com>\r\n"
    b"References: <root@example.com>\r\n"
    b" <mid1@example.com>\r\n"
    b" <mid2@example.com>\r\n"
    b"In-Reply-To: <mid2@example.com>\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Body.\r\n"
)

# Same fold, but with a bare LF continuation (no CR) -- some senders/relays
# normalise line endings and this should unfold identically.
LF_FOLDED_RAW = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bob@example.com\r\n"
    b"Subject: Thread\r\n"
    b"Date: Mon, 22 Sep 2025 08:00:00 -0400\r\n"
    b"Message-ID: <abc123@example.com>\r\n"
    b"References: <root@example.com>\n <mid1@example.com>\n\t<mid2@example.com>\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Body.\r\n"
)

MULTI_REPLY_TO_RAW = (
    b"From: Alice Example <alice@example.com>\r\n"
    b"To: bob@example.com\r\n"
    b"Reply-To: alice+support@example.com, alice+sales@example.com\r\n"
    b"Subject: Thread\r\n"
    b"Date: Mon, 22 Sep 2025 08:00:00 -0400\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Body.\r\n"
)


def _cfg(**overrides) -> Settings:
    base = dict(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        chat_model="chat",
        embed_model="embed",
        vision_model="vision",
        output_dir="/tmp/out",
        timeout=300.0,
        imap_host="imap.gmail.com",
        imap_port=993,
        imap_user="bot@example.com",
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Fake at the IMAPClient level
# ---------------------------------------------------------------------------


class FakeIMAPClient:
    """Stands in for imapclient.IMAPClient. Records every call for assertions."""

    def __init__(self, host, port=None, *, script, ssl=True):
        self.host = host
        self.port = port
        self.script = script
        self.calls = []
        self.wrote = False

    def login(self, user, password):
        self.calls.append(("login", user, password))
        if self.script.get("login_fail"):
            raise IMAPClientError("bad credentials")

    def select_folder(self, folder, readonly=False):
        self.calls.append(("select_folder", folder, readonly))
        if readonly and self.wrote and self.script.get("reselect_fail"):
            raise OSError("socket dropped on readonly re-select")

    def gmail_search(self, query):
        self.calls.append(("gmail_search", query))
        return list(self.script.get("uids", []))

    def fetch(self, messages, data):
        self.calls.append(("fetch", list(messages), list(data)))
        return self.script.get("fetch", {})

    def search(self, criteria):
        self.calls.append(("search", list(criteria)))
        return list(self.script.get("search_uids", self.script.get("uids", [])))

    def list_folders(self):
        self.calls.append(("list_folders",))
        return self.script.get("folders", [])

    def find_special_folder(self, flag):
        self.calls.append(("find_special_folder", flag))
        if flag == rb"\Sent":
            return self.script.get("sent_folder")
        if flag == rb"\Drafts":
            return self.script.get("drafts_folder")
        return None

    def append(self, folder, msg, flags=(), msg_time=None):
        self.calls.append(("append", folder, msg, tuple(flags), msg_time))
        if self.script.get("append_fail"):
            raise IMAPClientError("append failed")
        return b"OK"

    def add_gmail_labels(self, messages, labels, silent=False):
        self.calls.append(("add_gmail_labels", list(messages), list(labels), silent))
        self._store_outcome()

    def remove_gmail_labels(self, messages, labels, silent=False):
        self.calls.append(("remove_gmail_labels", list(messages), list(labels), silent))
        self._store_outcome()

    def _store_outcome(self):
        self.wrote = True
        if self.script.get("store_error"):
            raise self.script["store_error"]

    def shutdown(self):
        self.calls.append(("shutdown",))

    def logout(self):
        self.calls.append(("logout",))


def _install_fake(monkeypatch, script):
    created = {}

    def factory(host, port=None, ssl=True, timeout=None):
        client = FakeIMAPClient(host, port, script=script, ssl=ssl)
        created["client"] = client
        return client

    monkeypatch.setattr("lmsa.sources.gmail.imapclient.IMAPClient", factory)
    monkeypatch.setattr("lmsa.sources.gmail.secrets.resolve", lambda name: "app-password-fake")
    return created


def _fetch_item(msgid, thrid, labels=(), flags=(), body_key=b"BODY[]", raw=PLAIN_RAW):
    return {
        b"X-GM-MSGID": msgid,
        b"X-GM-THRID": thrid,
        b"X-GM-LABELS": labels,
        b"FLAGS": flags,
        body_key: raw,
    }


def test_search_parses_messages(monkeypatch):
    script = {
        "uids": [42],
        "fetch": {42: _fetch_item(555, 999, labels=(b"\\Inbox",), flags=(b"\\Seen",))},
    }
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        messages = gmail.search("newer_than:1d")

    assert len(messages) == 1
    m = messages[0]
    assert m.msg_id == "555"
    assert m.thread_id == "999"
    assert m.labels == ["\\Inbox"]
    assert m.flags == ["\\Seen"]
    assert m.from_name == "Alice Example"
    assert m.from_addr == "alice@example.com"
    assert m.to == ["bob@example.com", "carol@example.com"]
    assert m.reply_to == "alice+reply@example.com"
    assert m.subject == "Weekly digest"
    assert m.text == "Hello there, plain text body."
    assert m.web_url == f"https://mail.google.com/mail/u/0/#all/{999:x}"

    client = created["client"]
    assert ("select_folder", "[Gmail]/All Mail", True) in client.calls
    assert ("gmail_search", "newer_than:1d") in client.calls


def test_search_fetches_internaldate_as_an_aware_datetime(monkeypatch):
    item = _fetch_item(555, 999)
    # With normalise_times off, imapclient hands back the server's own offset.
    item[b"INTERNALDATE"] = datetime(2026, 9, 20, 8, 30, 0, tzinfo=timezone(timedelta(hours=-7)))
    created = _install_fake(monkeypatch, {"uids": [42], "fetch": {42: item}})

    with Gmail(_cfg()) as gmail:
        [m] = gmail.search("newer_than:1d")

    client = created["client"]
    assert client.normalise_times is False  # set at connect, before any fetch
    fetch_call = next(c for c in client.calls if c[0] == "fetch")
    assert "INTERNALDATE" in fetch_call[2]
    assert m.internal_date == datetime(2026, 9, 20, 15, 30, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw, utc", [
    # US fall-back night: both are "01:30" local wall-clock, an hour apart.
    (b'"01-Nov-2026 01:30:00 -0400"', datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)),
    (b'"01-Nov-2026 01:30:00 -0500"', datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)),
    # A summer message read in winter (and vice versa) keeps its own offset.
    (b'"20-Oct-2026 10:00:00 -0400"', datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)),
    (b'"08-Mar-2026 07:30:00 +0000"', datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc)),
])
def test_internaldate_is_exact_across_dst_whatever_the_local_offset(monkeypatch, raw, utc):
    # Pin the "local system offset" to winter time: with normalise_times on,
    # imapclient would localise every INTERNALDATE with it (an hour off for
    # summer mail). Parsed the way Gmail now asks for it (normalise off),
    # the result must not depend on the local offset at all.
    from imapclient import fixed_offset
    from imapclient.response_parser import parse_fetch_response

    from lmsa.sources.gmail import _aware_internal_date

    monkeypatch.setattr(fixed_offset.FixedOffset, "for_system", classmethod(lambda cls: cls(-300)))
    parsed = parse_fetch_response([b"1 (UID 5 INTERNALDATE " + raw + b")"], False, True)
    assert _aware_internal_date(parsed[5][b"INTERNALDATE"]) == utc


def test_naive_internaldate_is_taken_as_utc():
    from lmsa.sources.gmail import _aware_internal_date

    assert _aware_internal_date(datetime(2026, 1, 2, 3, 4)) == datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)


def test_search_meta_fetches_only_msgid_and_internaldate_in_chunks(monkeypatch):
    when = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)

    class Client(FakeIMAPClient):
        def fetch(self, messages, data):
            self.calls.append(("fetch", list(messages), list(data)))
            return {u: {b"X-GM-MSGID": 1000 + u, b"INTERNALDATE": when} for u in messages if u != 3}

    created = {}

    def factory(host, port=None, ssl=True, timeout=None):
        created["client"] = Client(host, port, script={"uids": [5, 1, 3, 2, 4]}, ssl=ssl)
        return created["client"]

    monkeypatch.setattr("lmsa.sources.gmail.imapclient.IMAPClient", factory)
    monkeypatch.setattr("lmsa.sources.gmail.secrets.resolve", lambda name: "app-password-fake")
    monkeypatch.setattr(Gmail, "META_CHUNK", 2)

    with Gmail(_cfg()) as gmail:
        metas = gmail.search_meta("after:1")

    assert [(m.msg_id, m.uid) for m in metas] == [("1001", 1), ("1002", 2), ("1004", 4), ("1005", 5)]
    assert all(m.internal_date == when for m in metas)
    fetches = [c for c in created["client"].calls if c[0] == "fetch"]
    assert [f[1] for f in fetches] == [[1, 2], [3, 4], [5]]
    assert all(f[2] == ["X-GM-MSGID", "INTERNALDATE"] for f in fetches)  # never a body
    assert all(c[2] is True for c in created["client"].calls if c[0] == "select_folder")


def test_fetch_uids_fetches_full_messages_readonly_in_order(monkeypatch):
    script = {"uids": [], "fetch": {7: _fetch_item(777, 1), 9: _fetch_item(999, 2)}}
    created = _install_fake(monkeypatch, script)
    with Gmail(_cfg()) as gmail:
        messages = gmail.fetch_uids([9, 7])
    assert [m.msg_id for m in messages] == ["999", "777"]
    fetch_call = next(c for c in created["client"].calls if c[0] == "fetch")
    assert "BODY.PEEK[]" in fetch_call[2]
    assert all(c[2] is True for c in created["client"].calls if c[0] == "select_folder")


def test_internal_date_is_none_when_not_fetched(monkeypatch):
    _install_fake(monkeypatch, {"uids": [42], "fetch": {42: _fetch_item(555, 999)}})
    with Gmail(_cfg()) as gmail:
        [m] = gmail.search("newer_than:1d")
    assert m.internal_date is None


def test_fetch_by_msgids_refetches_readonly_in_order_and_skips_missing(monkeypatch):
    class Client(FakeIMAPClient):
        def search(self, criteria):
            self.calls.append(("search", list(criteria)))
            return {555: [42], 777: [43]}.get(criteria[1], [])

    created = {}

    def factory(host, port=None, ssl=True, timeout=None):
        created["client"] = Client(host, port, script={
            "fetch": {42: _fetch_item(555, 1), 43: _fetch_item(777, 2)},
        }, ssl=ssl)
        return created["client"]

    monkeypatch.setattr("lmsa.sources.gmail.imapclient.IMAPClient", factory)
    monkeypatch.setattr("lmsa.sources.gmail.secrets.resolve", lambda name: "app-password-fake")

    with Gmail(_cfg()) as gmail:
        messages = gmail.fetch_by_msgids(["777", "999", "not-a-number", "555"])

    assert [m.msg_id for m in messages] == ["777", "555"]
    calls = created["client"].calls
    assert all(c[2] is True for c in calls if c[0] == "select_folder")  # readonly only
    fetch_call = next(c for c in calls if c[0] == "fetch")
    assert fetch_call[1] == [43, 42]
    assert "BODY.PEEK[]" in fetch_call[2]


def test_select_is_readonly(monkeypatch):
    script = {"uids": [], "fetch": {}}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.search("newer_than:1d")

    select_call = next(c for c in created["client"].calls if c[0] == "select_folder")
    assert select_call == ("select_folder", "[Gmail]/All Mail", True)


def test_fetch_requests_peek_not_bare_body(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1)}}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.search("x")

    fetch_call = next(c for c in created["client"].calls if c[0] == "fetch")
    items = fetch_call[2]
    assert "BODY.PEEK[]" in items
    assert "BODY[]" not in items


def test_headers_only_mode_uses_header_peek_and_skips_body(monkeypatch):
    script = {
        "uids": [1],
        "fetch": {1: _fetch_item(1, 1, body_key=b"BODY[HEADER]", raw=HEADERS_ONLY_TAIL)},
    }
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x", headers_only=True)

    assert msg.text == ""
    assert msg.attachments == []
    assert msg.subject == "Weekly digest"
    fetch_call = next(c for c in created["client"].calls if c[0] == "fetch")
    assert "BODY.PEEK[HEADER]" in fetch_call[2]


def test_search_returns_newest_first_and_respects_limit(monkeypatch):
    uids = [10, 55, 3, 91, 2]
    script = {
        "uids": uids,
        "fetch": {uid: _fetch_item(uid, uid) for uid in uids},
    }
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        messages = gmail.search("x", limit=2)

    # newest (highest UID) first, truncated to `limit` -- not the oldest kept
    assert [m.msg_id for m in messages] == ["91", "55"]
    fetch_call = next(c for c in created["client"].calls if c[0] == "fetch")
    assert fetch_call[1] == [91, 55]


def test_count_uses_search_only_never_fetches(monkeypatch):
    script = {"uids": [1, 2, 3], "fetch": {}}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        assert gmail.count("newer_than:1d") == 3

    assert not any(c[0] == "fetch" for c in created["client"].calls)
    assert ("gmail_search", "newer_than:1d") in created["client"].calls


def test_html_only_body_converted_to_text(monkeypatch):
    script = {
        "uids": [2],
        "fetch": {2: _fetch_item(2, 2, raw=HTML_ONLY_RAW)},
    }
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("sale")

    assert "evil()" not in msg.text
    assert "<" not in msg.text
    assert "Big sale today & more" in msg.text


def test_multi_recipient_to_header(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=PLAIN_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.to == ["bob@example.com", "carol@example.com"]


def test_naive_date_is_treated_as_utc(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=NAIVE_DATE_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.date is not None
    assert msg.date.tzinfo is not None
    from datetime import timezone

    assert msg.date.utcoffset() == timezone.utc.utcoffset(None)


def test_aware_date_offset_is_preserved(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=PLAIN_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.date is not None
    assert msg.date.utcoffset().total_seconds() == -4 * 3600


def test_labels_decoded_from_modified_utf7(monkeypatch):
    accented = encode_utf7("Café")
    script = {
        "uids": [1],
        "fetch": {1: _fetch_item(1, 1, labels=(b"\\Important", accented))},
    }
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.labels == ["\\Important", "Café"]


def test_header_object_value_is_normalised_to_str():
    from email.header import Header

    assert _decode_header_value(Header("hello", "ascii")) == "hello"
    assert _decode_header_value(None) == ""


def test_folders_and_sent(monkeypatch):
    script = {
        "uids": [],
        "fetch": {},
        "folders": [
            ((b"\\HasNoChildren",), "/", "INBOX"),
            ((b"\\HasNoChildren", b"\\Sent"), "/", "[Gmail]/Sent Mail"),
        ],
        "sent_folder": "[Gmail]/Sent Mail",
    }
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        assert "INBOX" in gmail.folders()
        assert gmail.sent() == "[Gmail]/Sent Mail"

    assert ("find_special_folder", rb"\Sent") in created["client"].calls


def test_sent_raises_when_no_special_use_flag(monkeypatch):
    script = {"uids": [], "fetch": {}, "sent_folder": None}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(RuntimeError, match="Sent"):
            gmail.sent()


# ---------------------------------------------------------------------------
# Write path: add_labels / remove_labels (lmstudio-automation#16)
# ---------------------------------------------------------------------------


def test_add_labels_selects_writable_then_restores_readonly(monkeypatch):
    script = {"uids": [], "fetch": {}, "search_uids": [77]}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.add_labels("555", ["Receipts"])

    calls = created["client"].calls
    assert ("select_folder", "[Gmail]/All Mail", False) in calls
    assert ("search", ["X-GM-MSGID", 555]) in calls
    assert ("add_gmail_labels", [77], ["Receipts"], True) in calls
    select_calls = [c for c in calls if c[0] == "select_folder"]
    # last select must restore readonly, whatever came before it
    assert select_calls[-1] == ("select_folder", "[Gmail]/All Mail", True)


def test_remove_labels_uses_remove_gmail_labels_not_add(monkeypatch):
    script = {"uids": [], "fetch": {}, "search_uids": [77]}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.remove_labels("555", ["Receipts"])

    calls = created["client"].calls
    assert ("remove_gmail_labels", [77], ["Receipts"], True) in calls
    assert not any(c[0] == "add_gmail_labels" for c in calls)


def test_add_labels_never_stores_flags_or_expunges(monkeypatch):
    """The write path only ever touches X-GM-LABELS -- never \\Seen or any
    other flag, and never EXPUNGE/COPY/MOVE (those methods aren't even on
    the fake, so calling them would raise AttributeError)."""
    script = {"uids": [], "fetch": {}, "search_uids": [77]}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.add_labels("555", ["Receipts"])

    assert not any(c[0] in ("add_flags", "remove_flags", "expunge", "copy") for c in created["client"].calls)


def test_add_labels_raises_when_message_not_found_and_still_restores_readonly(monkeypatch):
    script = {"uids": [], "fetch": {}, "search_uids": []}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(LookupError, match="555"):
            gmail.add_labels("555", ["Receipts"])

    select_calls = [c for c in created["client"].calls if c[0] == "select_folder"]
    assert select_calls[-1] == ("select_folder", "[Gmail]/All Mail", True)
    assert not any(c[0] == "add_gmail_labels" for c in created["client"].calls)


def test_add_labels_rejects_non_numeric_msg_id(monkeypatch):
    script = {"uids": [], "fetch": {}, "search_uids": []}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(ValueError, match="X-GM-MSGID"):
            gmail.add_labels("not-a-number", ["Receipts"])


def test_add_labels_empty_list_is_a_noop(monkeypatch):
    script = {"uids": [], "fetch": {}, "search_uids": [77]}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.add_labels("555", [])

    # never even selects the folder writable for an empty label set
    assert not any(c[0] == "select_folder" and c[2] is False for c in created["client"].calls)


RESERVED = [
    "\\Trash", "\\Spam", "\\Inbox", "\\Important", "\\Starred", "\\Draft", "\\Sent",
    "CATEGORY_PROMOTIONS", "category_social",
    "INBOX", "inbox", "Spam", "TRASH", "Draft", "SENT", "Starred", "IMPORTANT", "Unread", "chat",
    "[Gmail]/Trash", "[Gmail]/All Mail", "[gmail]/spam", "",
]


@pytest.mark.parametrize("label", RESERVED)
def test_is_reserved_label_rejects_gmail_system_labels(label):
    assert is_reserved_label(label) is True


@pytest.mark.parametrize("label", ["Receipts", "Action Needed", "Inbox Zero", "Parent/Child", "Sent Items"])
def test_is_reserved_label_allows_user_labels(label):
    assert is_reserved_label(label) is False


@pytest.mark.parametrize("method", ["add_labels", "remove_labels"])
@pytest.mark.parametrize("label", RESERVED)
def test_label_writes_refuse_system_labels_before_any_network_call(monkeypatch, method, label):
    """Adding \\Trash/\\Spam would trash/spam mail and removing \\Inbox would
    archive it -- the public write API refuses them before touching IMAP."""
    script = {"uids": [], "fetch": {}, "search_uids": [77]}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        calls_before = list(created["client"].calls)
        with pytest.raises(ValueError, match="system label"):
            getattr(gmail, method)("555", ["Receipts", label])

    # nothing at all was sent between login and logout
    assert created["client"].calls == calls_before + [("logout",)]


def test_add_labels_reselect_failure_after_store_is_success_and_drops_session(monkeypatch):
    """STORE returned, so the label IS on the message: add_labels must not
    raise (or the caller would never record it for undo), and the session
    must not stay usable in a possibly-writable state."""
    script = {"uids": [], "fetch": {}, "search_uids": [77], "reselect_fail": True}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.add_labels("555", ["Receipts"])  # no exception
        with pytest.raises(RuntimeError, match="closed"):
            gmail.search("anything")

    calls = created["client"].calls
    assert calls[1:] == [
        ("select_folder", "[Gmail]/All Mail", False),
        ("search", ["X-GM-MSGID", 555]),
        ("add_gmail_labels", [77], ["Receipts"], True),
        ("select_folder", "[Gmail]/All Mail", True),
        ("shutdown",),
    ]  # and nothing after the shutdown -- not even the logout


def test_store_error_is_not_masked_by_reselect_failure(monkeypatch):
    script = {
        "uids": [], "fetch": {}, "search_uids": [77],
        "reselect_fail": True, "store_error": OSError("connection reset mid-STORE"),
    }
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(LabelWriteError, match="connection reset mid-STORE") as info:
            gmail.add_labels("555", ["Receipts"])
        with pytest.raises(RuntimeError, match="closed"):
            gmail.add_labels("555", ["Receipts"])

    assert info.value.maybe_applied is True  # socket error: outcome unknown
    assert isinstance(info.value.__cause__, OSError)
    assert created["client"].calls[-1] == ("shutdown",)


@pytest.mark.parametrize(
    "error,maybe_applied",
    [
        (IMAPClientError("STORE NO [CANNOT] refused"), False),  # tagged NO: server refused
        (IMAPClientAbortError("socket error: EOF"), True),  # connection dropped: unknown
        (TimeoutError("timed out"), True),
    ],
)
def test_store_failure_reports_whether_it_may_have_applied(monkeypatch, error, maybe_applied):
    script = {"uids": [], "fetch": {}, "search_uids": [77], "store_error": error}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(LabelWriteError) as info:
            gmail.remove_labels("555", ["Receipts"])

    assert info.value.maybe_applied is maybe_applied
    # re-select succeeded, so the session was restored readonly, not dropped
    calls = created["client"].calls
    assert ("select_folder", "[Gmail]/All Mail", True) in calls
    assert ("shutdown",) not in calls


def test_missing_credentials_error_message(monkeypatch):
    monkeypatch.setattr("lmsa.sources.gmail.secrets.resolve", lambda name: None)

    with pytest.raises(GmailAuthError, match="credentials missing"):
        with Gmail(_cfg(imap_user="")):
            pass


def test_login_failure_does_not_leak_password(monkeypatch):
    script = {"login_fail": True, "uids": [], "fetch": {}}
    _install_fake(monkeypatch, script)

    with pytest.raises(GmailAuthError) as excinfo:
        with Gmail(_cfg()):
            pass

    assert "app-password-fake" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# message_id / references (added additively for #17 reply threading)
# ---------------------------------------------------------------------------


def test_message_id_and_references_are_parsed(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=THREADED_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.message_id == "<abc123@example.com>"
    assert msg.references == "<root@example.com> <mid@example.com>"


def test_message_id_and_references_default_to_empty_string(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=PLAIN_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.message_id == ""
    assert msg.references == ""


def test_crlf_folded_references_is_unfolded_to_one_line(monkeypatch):
    """Review finding #1: a folded References must come back as a plain,
    linefeed-free string -- it gets set verbatim on an outgoing draft's
    header later, and EmailMessage raises on an embedded CR/LF."""
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=CRLF_FOLDED_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert "\r" not in msg.references and "\n" not in msg.references
    assert msg.references == "<root@example.com> <mid1@example.com> <mid2@example.com>"
    assert "\r" not in msg.in_reply_to and "\n" not in msg.in_reply_to
    assert msg.in_reply_to == "<mid2@example.com>"
    assert "\r" not in msg.message_id and "\n" not in msg.message_id


def test_lf_folded_references_is_unfolded_to_one_line(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=LF_FOLDED_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert "\r" not in msg.references and "\n" not in msg.references
    assert msg.references == "<root@example.com> <mid1@example.com> <mid2@example.com>"


def test_a_folded_references_header_can_be_set_on_an_outgoing_email_message(monkeypatch):
    """The actual failure mode the review caught: build_eml-style usage of
    a folded References must not raise "Header values may not contain
    linefeed"."""
    from email.message import EmailMessage as StdEmailMessage

    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=CRLF_FOLDED_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    out = StdEmailMessage()
    out["References"] = msg.references  # must not raise
    out["In-Reply-To"] = msg.in_reply_to  # must not raise


def test_reply_to_all_captures_every_address(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=MULTI_REPLY_TO_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.reply_to_all == ("alice+support@example.com", "alice+sales@example.com")
    assert msg.reply_to == "alice+support@example.com"  # unchanged: first address, for old callers


def test_reply_to_all_empty_when_no_reply_to_header(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=PLAIN_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")

    assert msg.reply_to_all == ("alice+reply@example.com",)


# ---------------------------------------------------------------------------
# append_draft: the one write path (#17). Proves only APPEND (with \Draft)
# is issued -- never a select_folder, and no STORE/EXPUNGE/COPY/MOVE.
# ---------------------------------------------------------------------------


def test_append_draft_resolves_drafts_folder_and_appends_with_draft_flag(monkeypatch):
    script = {"uids": [], "fetch": {}, "drafts_folder": "[Gmail]/Drafts"}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.append_draft(b"raw message bytes")

    client = created["client"]
    assert ("find_special_folder", rb"\Drafts") in client.calls
    assert ("append", "[Gmail]/Drafts", b"raw message bytes", (imapclient.DRAFT,), None) in client.calls


def test_append_draft_never_selects_a_folder(monkeypatch):
    script = {"uids": [], "fetch": {}, "drafts_folder": "[Gmail]/Drafts"}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        gmail.append_draft(b"raw message bytes")

    assert not any(c[0] == "select_folder" for c in created["client"].calls)
    assert not any(c[0] in ("store", "expunge", "copy", "move") for c in created["client"].calls)


def test_append_draft_raises_when_no_drafts_special_use_folder(monkeypatch):
    script = {"uids": [], "fetch": {}, "drafts_folder": None}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(RuntimeError, match="Drafts"):
            gmail.append_draft(b"raw message bytes")


def test_append_draft_propagates_append_failure(monkeypatch):
    script = {"uids": [], "fetch": {}, "drafts_folder": "[Gmail]/Drafts", "append_fail": True}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        with pytest.raises(IMAPClientError):
            gmail.append_draft(b"raw message bytes")


def test_no_smtp_module_used_anywhere_in_gmail_source():
    """Grep-verifiable guarantee: this module can draft, but it can never send."""
    import lmsa.sources.gmail as g

    src = Path(g.__file__).read_text(encoding="utf-8")
    assert "smtplib" not in src
    assert ".sendmail(" not in src
    assert ".send_message(" not in src
# attachment_bytes
# ---------------------------------------------------------------------------


def _multipart_raw(parts, *, subject="Your receipt", msgid_hdr=None):
    """A multipart/mixed raw message with a plain-text body plus `parts`
    (name, mime, payload) as attachments, built with stdlib `email` so the
    MIME structure (boundaries, base64 body encoding) is realistic."""
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart()
    msg["From"] = "Billing <billing@example.com>"
    msg["To"] = "bot@example.com"
    msg["Subject"] = subject
    msg["Date"] = "Tue, 23 Sep 2025 09:00:00 -0400"
    msg.attach(MIMEText("Thanks for your purchase.", "plain"))
    for name, mime, payload in parts:
        _maintype, _sep, subtype = mime.partition("/")
        part = MIMEApplication(payload, _subtype=subtype or "octet-stream")
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
    return msg.as_bytes()


PDF_BYTES = b"%PDF-1.4 fake pdf content for testing"
JPG_BYTES = b"\xff\xd8\xfffake jpeg content for testing"

TWO_ATTACHMENTS_RAW = _multipart_raw([
    ("receipt.pdf", "application/pdf", PDF_BYTES),
    ("photo.jpg", "image/jpeg", JPG_BYTES),
])


def test_attachment_bytes_returns_the_matching_part_by_index(monkeypatch):
    script = {
        "uids": [42],
        "fetch": {42: _fetch_item(555, 999, raw=TWO_ATTACHMENTS_RAW)},
    }
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")
        assert [a.name for a in msg.attachments] == ["receipt.pdf", "photo.jpg"]

        assert gmail.attachment_bytes(msg, 0) == PDF_BYTES
        assert gmail.attachment_bytes(msg, 1) == JPG_BYTES

    calls = created["client"].calls
    assert ("search", ["X-GM-MSGID", "555"]) in calls
    fetch_calls = [c for c in calls if c[0] == "fetch"]
    # the two attachment_bytes() refetches both asked for BODY.PEEK[] (readonly), not BODY[]
    for call in fetch_calls[1:]:
        assert call[2] == ["BODY.PEEK[]"]


def test_attachment_bytes_out_of_range_index_raises(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=TWO_ATTACHMENTS_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")
        with pytest.raises(IndexError):
            gmail.attachment_bytes(msg, 5)


def test_attachment_bytes_rejects_declared_size_over_cap_without_fetching(monkeypatch):
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=TWO_ATTACHMENTS_RAW)}}
    created = _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")
        with pytest.raises(AttachmentTooLargeError):
            gmail.attachment_bytes(msg, 0, max_size=1)

    # rejected on the already-known declared size -- never re-selected/re-searched
    assert not any(c[0] == "search" for c in created["client"].calls)


def test_attachment_bytes_rejects_actual_payload_over_cap(monkeypatch):
    # A declared size that understates the real payload (a lying/miscounted
    # Content-Length) must still be caught after the real bytes come back.
    script = {"uids": [1], "fetch": {1: _fetch_item(1, 1, raw=TWO_ATTACHMENTS_RAW)}}
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")
        oversized = type(msg.attachments[0])(name=msg.attachments[0].name, mime=msg.attachments[0].mime, size=1)
        msg = type(msg)(**{**msg.__dict__, "attachments": [oversized, msg.attachments[1]]})
        with pytest.raises(AttachmentTooLargeError):
            gmail.attachment_bytes(msg, 0, max_size=len(PDF_BYTES) - 1)


def test_attachment_bytes_message_not_found(monkeypatch):
    script = {
        "uids": [1],
        "fetch": {1: _fetch_item(1, 1, raw=TWO_ATTACHMENTS_RAW)},
        "search_uids": [],
    }
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")
        with pytest.raises(AttachmentNotFoundError, match="not found"):
            gmail.attachment_bytes(msg, 0)


def test_attachment_bytes_message_shrank_since_search(monkeypatch):
    # Between search() and attachment_bytes(), the message now has fewer
    # attachment parts than the index it was found at (e.g. re-sent/edited).
    one_attachment_raw = _multipart_raw([("receipt.pdf", "application/pdf", PDF_BYTES)])
    script = {
        "uids": [1],
        "fetch": {1: _fetch_item(1, 1, raw=TWO_ATTACHMENTS_RAW)},
    }
    _install_fake(monkeypatch, script)

    with Gmail(_cfg()) as gmail:
        [msg] = gmail.search("x")
        # simulate the refetch now returning a message with only one part
        script["fetch"] = {1: _fetch_item(1, 1, raw=one_attachment_raw)}
        with pytest.raises(AttachmentNotFoundError):
            gmail.attachment_bytes(msg, 1)


# ---------------------------------------------------------------------------
# Wire-level: real imapclient.IMAPClient + real (unmodified) imaplib.IMAP4,
# talking to a scripted fake socket. Proves EXAMINE (not SELECT) and
# BODY.PEEK[] (not BODY[]) are what actually goes out on the wire.
# ---------------------------------------------------------------------------

CRLF = b"\r\n"


class _FakeSocket:
    """A socket whose reads come from a pre-scripted buffer and whose
    writes are recorded verbatim for inspection."""

    def __init__(self, script_bytes: bytes):
        self._buf = script_bytes
        self.sent = bytearray()

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def makefile(self, _mode):
        return self

    def readline(self, *_a, **_k) -> bytes:
        idx = self._buf.find(CRLF)
        if idx == -1:
            line, self._buf = self._buf, b""
            return line
        line, self._buf = self._buf[: idx + 2], self._buf[idx + 2 :]
        return line

    def read(self, size: int) -> bytes:
        data, self._buf = self._buf[:size], self._buf[size:]
        return data

    def recv(self, size: int) -> bytes:  # Python 3.14's imaplib reads via recv()
        return self.read(size)

    def shutdown(self, _how) -> None:  # and calls shutdown() on logout/close
        pass

    def close(self) -> None:
        pass

    def settimeout(self, _value) -> None:
        pass


def test_wire_level_readonly_and_peek(monkeypatch):
    monkeypatch.setattr(imaplib.random, "randint", lambda a, b: 4096)
    tagpre = imaplib.Int2AP(4096)

    script = (
        b"* OK IMAP4rev1 Server ready\r\n"
        b"* CAPABILITY IMAP4rev1 X-GM-EXT-1\r\n"
        + tagpre + b"0 OK CAPABILITY completed\r\n"
        + tagpre + b"1 OK LOGIN completed\r\n"
        + b"* 3 EXISTS\r\n"
        + b"* 0 RECENT\r\n"
        + b"* FLAGS (\\Seen \\Answered)\r\n"
        + b"* OK [PERMANENTFLAGS (\\Seen \\Answered)] permitted\r\n"
        + tagpre + b"2 OK [READ-ONLY] EXAMINE completed\r\n"
        # deliberately no response for the FETCH that follows: we only
        # need the bytes the client already sent before it blocks on read.
    )
    fake_socket = _FakeSocket(script)
    monkeypatch.setattr(
        imap4.IMAP4WithTimeout, "_create_socket", lambda self, timeout=None: fake_socket
    )

    client = imapclient.IMAPClient("fake-host", ssl=False)
    client.login("bot@example.com", "app-password-fake")
    client.select_folder("[Gmail]/All Mail", readonly=True)

    with pytest.raises(imaplib.IMAP4.error):
        client.fetch([1], ["BODY.PEEK[]"])

    sent = bytes(fake_socket.sent)
    assert b'EXAMINE "[Gmail]/All Mail"' in sent
    assert b"SELECT " not in sent
    assert b"BODY.PEEK[]" in sent
    # the exact bytes "BODY[]" (no PEEK) never went out -- would mark messages read
    assert b" BODY[]" not in sent and b"(BODY[])" not in sent


def test_raw_utf8_headers_decode_without_replacement_chars():
    from email import message_from_bytes
    from lmsa.sources.gmail import _decode_header_value

    msg = message_from_bytes("Subject: héllo\nFrom: Jörg <j@example.com>\n\nx".encode("utf-8"))
    assert _decode_header_value(msg.get("Subject")) == "héllo"
    assert _decode_header_value(msg.get("From")) == "Jörg <j@example.com>"
    assert _decode_header_value("Hello =?utf-8?q?W=C3=B6rld?=") == "Hello Wörld"


def test_labels_are_always_strings():
    from lmsa.sources.gmail import _decode_labels

    assert _decode_labels([rb"\Inbox", 2024, None, b"&AOQ-bc"]) == [r"\Inbox", "2024", "NIL", "äbc"]


def test_connect_uses_timeout_and_shuts_down_on_login_failure(monkeypatch):
    import pytest
    from imapclient.exceptions import LoginError
    from lmsa.sources import gmail as g

    seen = {}

    class FakeClient:
        def __init__(self, host, port=None, ssl=True, timeout=None):
            seen["timeout"] = timeout

        def login(self, user, password):
            raise LoginError("nope")

        def shutdown(self):
            seen["shutdown"] = True

    monkeypatch.setattr(g.imapclient, "IMAPClient", FakeClient)
    monkeypatch.setattr(g.secrets, "resolve", lambda name: "invented-pass")
    cfg = g.settings()
    cfg = type(cfg)(**{**cfg.__dict__, "imap_user": "someone@example.com", "timeout": 42.0})
    with pytest.raises(g.GmailAuthError):
        g.Gmail(cfg).__enter__()
    assert seen == {"timeout": 42.0, "shutdown": True}



def test_raw_utf8_reply_to_address_decodes_cleanly():
    from lmsa.sources.gmail import _address_list
    from email import message_from_bytes

    msg = message_from_bytes("Reply-To: Jörg <jörg@exämple.com>\n\nx".encode("utf-8"))
    assert _address_list(msg.get_all("Reply-To")) == ["jörg@exämple.com"]


def test_encoded_word_cannot_smuggle_an_extra_address():
    from lmsa.sources.gmail import _address_list

    assert _address_list(["=?utf-8?q?evil=40attacker.example=2C?= <ok@example.com>"]) == ["ok@example.com"]
