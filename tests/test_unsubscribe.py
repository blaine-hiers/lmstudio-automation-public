"""tasks.unsubscribe: headers-only unsubscribe hit list.

Fakes Gmail at the `tasks.unsubscribe.Gmail` boundary (per the module's own
testing note) rather than at imapclient level -- this task's own logic is
what's under test, not the IMAP wire format, which `test_gmail_source.py`
already covers.
"""

from __future__ import annotations

import sys

import pytest

import tasks.unsubscribe as unsub
from lmsa.report import Link
from lmsa.sources.gmail import Message

SENT_FOLDER = "[Gmail]/Sent Mail"


def _msg(
    from_addr: str,
    from_name: str = "",
    *,
    seen: bool = True,
    list_unsubscribe: str = "",
    to: list[str] | None = None,
) -> Message:
    return Message(
        msg_id="1",
        thread_id="1",
        labels=[],
        flags=["\\Seen"] if seen else [],
        from_name=from_name,
        from_addr=from_addr,
        to=to or [],
        reply_to="",
        subject="",
        date=None,
        list_unsubscribe=list_unsubscribe,
        list_id="",
        auth_results="",
        text="",
        attachments=[],
        web_url="",
    )


class _FakeGmail:
    """Exposes search/sent, the surface `tasks.unsubscribe` actually calls."""

    def __init__(self, inbound: list[Message], sent: list[Message]):
        self.inbound = inbound
        self.sent_msgs = sent
        self.calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def sent(self) -> str:
        return SENT_FOLDER

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        self.calls.append((query, folder, limit, headers_only))
        return self.sent_msgs if folder == SENT_FOLDER else self.inbound


# --- parse_list_unsubscribe -------------------------------------------------


def test_parse_prefers_https_over_mailto():
    header = "<mailto:unsub@example.com>, <https://example.com/unsub?id=1>"
    assert unsub.parse_list_unsubscribe(header) == "https://example.com/unsub?id=1"


def test_parse_falls_back_to_mailto_only():
    assert unsub.parse_list_unsubscribe("<mailto:unsub@example.com>") == "mailto:unsub@example.com"


def test_parse_handles_multiple_https_entries_keeps_first():
    header = "<https://a.example.com/one>, <https://a.example.com/two>"
    assert unsub.parse_list_unsubscribe(header) == "https://a.example.com/one"


@pytest.mark.parametrize("header", ["", "garbage no angle brackets", "   "])
def test_parse_returns_none_for_empty_or_malformed(header):
    assert unsub.parse_list_unsubscribe(header) is None


# --- collect_stats -----------------------------------------------------------


def test_score_is_volume_times_unread_share():
    messages = [
        _msg("news@example.com", "Newsletter", seen=False),
        _msg("news@example.com", "Newsletter", seen=False),
        _msg("news@example.com", "Newsletter", seen=True),
    ]
    stats, failures = unsub.collect_stats(messages, sent_to=set())
    s = stats["news@example.com"]
    assert failures == 0
    assert s.count == 3
    assert s.unread == 2
    assert s.unread_share == pytest.approx(2 / 3)
    assert s.score == pytest.approx(3 * (2 / 3))


def test_score_zeroed_for_anyone_ever_emailed():
    messages = [_msg("friend@example.com", seen=False)] * 5
    stats, _ = unsub.collect_stats(messages, sent_to={"friend@example.com"})
    s = stats["friend@example.com"]
    assert s.ever_sent_to is True
    assert s.score == 0.0


def test_sender_address_normalized_case_insensitive():
    messages = [_msg("News@Example.com", seen=False), _msg("news@example.com", seen=False)]
    stats, _ = unsub.collect_stats(messages, sent_to=set())
    assert list(stats) == ["news@example.com"]
    assert stats["news@example.com"].count == 2


def test_list_unsubscribe_presence_and_link_captured():
    messages = [
        _msg("news@example.com", list_unsubscribe="<https://example.com/unsub>"),
        _msg("news@example.com", list_unsubscribe=""),
    ]
    stats, _ = unsub.collect_stats(messages, sent_to=set())
    s = stats["news@example.com"]
    assert s.has_unsubscribe is True
    assert s.unsubscribe_href == "https://example.com/unsub"


def test_message_missing_from_addr_is_skipped_not_fatal():
    messages = [_msg(""), _msg("news@example.com", seen=False)]
    stats, failures = unsub.collect_stats(messages, sent_to=set())
    assert failures == 1
    assert "news@example.com" in stats
    assert "" not in stats


def test_one_bad_message_does_not_abort_the_scan(monkeypatch):
    """A message whose flags attribute misbehaves must not sink the whole loop."""
    good = _msg("news@example.com", seen=False)
    bad = _msg("bad@example.com")
    object.__setattr__(bad, "flags", None)  # "\\Seen" in None raises TypeError

    stats, failures = unsub.collect_stats([bad, good], sent_to=set())
    assert failures == 1
    assert stats["news@example.com"].count == 1


# --- sent_recipients ---------------------------------------------------------


def test_sent_recipients_builds_set_from_single_scan():
    fake = _FakeGmail(
        inbound=[],
        sent=[_msg("me@example.com", to=["a@example.com", "b@example.com"])],
    )
    recipients, truncated = unsub.sent_recipients(fake)
    assert recipients == {"a@example.com", "b@example.com"}
    assert truncated is False
    # exactly one search call against the Sent folder -- never per-sender
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == SENT_FOLDER


def test_sent_recipients_flags_truncation(monkeypatch):
    monkeypatch.setattr(unsub, "SENT_LIMIT", 1)
    fake = _FakeGmail(inbound=[], sent=[_msg("me@example.com", to=["a@example.com"])])
    _, truncated = unsub.sent_recipients(fake)
    assert truncated is True


# --- build_report -------------------------------------------------------------


def test_build_report_ranks_and_renders_unsubscribe_link():
    stats, _ = unsub.collect_stats(
        [
            _msg(
                "spam@example.com", "Spam Co", seen=False,
                list_unsubscribe="<https://example.com/unsub>",
            ),
        ]
        * 10,
        sent_to=set(),
    )
    r = unsub.build_report(
        stats, days=90, scanned=10, failures=0,
        inbound_truncated=False, sent_truncated=False, top_n=25,
    )
    html = r.render()
    assert "Spam Co" in html
    assert 'href="https://example.com/unsub"' in html
    assert ">unsubscribe<" in html


def test_build_report_drops_javascript_scheme_link():
    stats, _ = unsub.collect_stats(
        [
            _msg(
                "spam@example.com", "Spam Co", seen=False,
                list_unsubscribe="<javascript:alert(1)>",
            ),
        ]
        * 5,
        sent_to=set(),
    )
    r = unsub.build_report(
        stats, days=90, scanned=5, failures=0,
        inbound_truncated=False, sent_truncated=False, top_n=25,
    )
    html = r.render()
    assert "javascript:" not in html
    assert ">offered<" in html  # header was present, just an unsafe scheme


def test_build_report_excludes_zero_score_senders_from_hit_list():
    stats, _ = unsub.collect_stats(
        [_msg("friend@example.com", "Friend", seen=False)] * 5,
        sent_to={"friend@example.com"},
    )
    r = unsub.build_report(
        stats, days=90, scanned=5, failures=0,
        inbound_truncated=False, sent_truncated=False, top_n=25,
    )
    html = r.render()
    assert "is clean" in html.lower() or "0 candidates" in html


def test_build_report_notes_truncation_and_failures():
    r = unsub.build_report(
        {}, days=90, scanned=5000, failures=3,
        inbound_truncated=True, sent_truncated=True, top_n=25,
    )
    html = r.render()
    assert "cap" in html
    assert "3 message" in html


def test_build_report_caps_table_to_top_n():
    messages = []
    for i in range(3):
        messages += [_msg(f"sender{i}@example.com", f"Sender {i}", seen=False)] * (10 - i)
    stats, _ = unsub.collect_stats(messages, sent_to=set())
    r = unsub.build_report(
        stats, days=90, scanned=len(messages), failures=0,
        inbound_truncated=False, sent_truncated=False, top_n=2,
    )
    html = r.render()
    assert "Sender 0" in html and "Sender 1" in html
    assert "Sender 2" not in html
    assert "top 2 of 3" in html


# --- end to end via main() ----------------------------------------------------


def test_main_queries_last_90_days_excluding_sent_and_writes_report(monkeypatch, tmp_path):
    inbound = [
        _msg(
            "newsletter@example.com", "Big Newsletter", seen=False,
            list_unsubscribe="<https://example.com/unsub>",
        ),
        _msg("newsletter@example.com", "Big Newsletter", seen=False),
        _msg("friend@example.com", "Friend", seen=False),
    ]
    sent = [_msg("me@example.com", to=["friend@example.com"])]
    fake = _FakeGmail(inbound=inbound, sent=sent)

    monkeypatch.setattr(unsub, "Gmail", lambda cfg=None: fake)
    monkeypatch.setenv("LMSA_IMAP_USER", "bot@example.com")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr(sys, "argv", ["unsubscribe"])

    unsub.main()

    inbound_call = next(c for c in fake.calls if c[1] != SENT_FOLDER)
    assert inbound_call[0] == "newer_than:90d -in:sent"
    assert inbound_call[3] is True  # headers_only

    [written] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "Big Newsletter" in html
    assert "https://example.com/unsub" in html
    # friend@example.com was written to -- must not appear in the hit list bars/table
    assert "Friend" not in html
