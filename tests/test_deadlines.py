"""tasks.deadlines: per-message isolation, strict date parsing that drops
(never crashes on) unparseable items, dropping past items, deduping on
(normalized title, date), and the end-to-end .ics + report output."""

from datetime import datetime, timezone

import icalendar
import pytest

import tasks.deadlines as d
from lmsa.sources.gmail import Message

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def make_message(msg_id="m1", subject="Subject", text="body text", web_url="https://mail.google.com/x"):
    return Message(
        msg_id=msg_id,
        thread_id="t1",
        labels=[],
        flags=[],
        from_name="Alice",
        from_addr="alice@example.com",
        to=["bob@example.com"],
        reply_to="",
        subject=subject,
        date=NOW,
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text=text,
        attachments=[],
        web_url=web_url,
    )


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network.

    `replies` is a list of either a dict (the extract() result) or an
    Exception instance/class to raise -- one entry consumed per call, in
    order, so each test message can be given its own scripted outcome.
    """

    def __init__(self, replies):
        self.calls: list[str] = []
        self._replies = list(replies)

    def ensure_server(self) -> None:
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append(prompt)
        assert system is not None and "untrusted" in system
        reply = self._replies.pop(0)
        if isinstance(reply, Exception) or (isinstance(reply, type) and issubclass(reply, Exception)):
            raise reply
        return reply


def item(title="Report due", kind="deadline", start="2026-10-01", end=None, all_day=True, location=None):
    return {"title": title, "kind": kind, "start": start, "end": end, "all_day": all_day, "location": location}


# -- gather(): per-message isolation ------------------------------------------


def test_gather_isolates_a_failing_message_and_keeps_the_rest():
    lm = FakeLMStudio(
        replies=[
            RuntimeError("server down"),
            {"items": [item(title="Still works")]},
        ]
    )
    messages = [make_message(msg_id="bad"), make_message(msg_id="good")]

    items, unparseable, failed = d.gather(lm, messages)

    assert len(items) == 1
    assert items[0].title == "Still works"
    assert len(failed) == 1
    assert failed[0].msg_id == "bad"
    assert "server down" in failed[0].error
    assert not unparseable


def test_gather_drops_only_the_unparseable_item_not_the_whole_message():
    lm = FakeLMStudio(
        replies=[
            {
                "items": [
                    item(title="Good item", start="2026-10-01"),
                    item(title="Bad item", start="not a date"),
                ]
            }
        ]
    )
    items, unparseable, failed = d.gather(lm, [make_message()])

    assert [i.title for i in items] == ["Good item"]
    assert [u.title for u in unparseable] == ["Bad item"]
    assert unparseable[0].raw_start == "not a date"
    assert not failed


def test_gather_truncates_long_bodies_before_prompting():
    long_text = "x" * 10_000
    lm = FakeLMStudio(replies=[{"items": []}])
    d.gather(lm, [make_message(text=long_text)])
    assert len(lm.calls[0]) < len(long_text) + 500
    assert "truncated" in lm.calls[0]


# -- date parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "value,all_day,expected",
    [
        ("2026-10-01", True, __import__("datetime").date(2026, 10, 1)),
        ("2026-10-01T14:30:00+00:00", True, __import__("datetime").date(2026, 10, 1)),
        ("2026-10-01T14:30:00+00:00", False, datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc)),
        ("2026-10-01", False, datetime(2026, 10, 1, 0, 0)),  # bare date on a timed item -> midnight
    ],
)
def test_parse_when_strict_success(value, all_day, expected):
    assert d._parse_when(value, all_day) == expected


@pytest.mark.parametrize(
    "value,all_day",
    [
        ("not a date", True),
        ("not a date", False),
        ("", True),
        (None, False),
    ],
)
def test_parse_when_strict_failure_returns_none_not_raise(value, all_day):
    assert d._parse_when(value, all_day) is None


# -- drop_past -------------------------------------------------------------------


def test_drop_past_splits_on_now():
    past_item = d._build_item(item(title="Past", start="2026-09-01"), make_message())
    future_item = d._build_item(item(title="Future", start="2026-10-01"), make_message())
    upcoming, past = d.drop_past([past_item, future_item], NOW)
    assert [i.title for i in upcoming] == ["Future"]
    assert [i.title for i in past] == ["Past"]


def test_drop_past_keeps_a_timed_item_later_today():
    later_today = d._build_item(
        item(title="Later today", start="2026-09-24T18:00:00+00:00", all_day=False), make_message()
    )
    upcoming, past = d.drop_past([later_today], NOW)
    assert [i.title for i in upcoming] == ["Later today"]
    assert not past


# -- dedupe ----------------------------------------------------------------------


def test_dedupe_drops_same_title_and_date_keeping_the_first_seen():
    first = d._build_item(item(title="Report Due", start="2026-10-01"), make_message(msg_id="newer"))
    second = d._build_item(item(title="report   due", start="2026-10-01"), make_message(msg_id="older"))
    different_date = d._build_item(item(title="Report Due", start="2026-10-02"), make_message(msg_id="x"))

    kept, dupes = d.dedupe([first, second, different_date])

    assert kept == [first, different_date]
    assert dupes == [second]


# -- end-to-end main() -----------------------------------------------------------


def test_main_writes_ics_and_report_with_upcoming_items_only(monkeypatch, tmp_path):
    # Dates are chosen far from the real "now" the test runs under (main()
    # uses the real clock, same as every other task's main() test in this
    # repo) so the past/future split can't flake near a boundary.
    messages = [
        make_message(msg_id="m-past", subject="Old invoice", text="Paid on 2020-01-01."),
        make_message(msg_id="m-future", subject="Contract renewal", text="Renewal due 2030-01-15."),
        make_message(msg_id="m-dup", subject="Reminder: renewal", text="Don't forget, renewal due soon."),
        make_message(msg_id="m-bad", subject="Broken", text="garbled"),
    ]
    lm = FakeLMStudio(
        replies=[
            {"items": [item(title="Invoice paid", kind="deadline", start="2020-01-01")]},
            {"items": [item(title="Contract renewal", kind="deadline", start="2030-01-15")]},
            {"items": [item(title="Contract Renewal", kind="deadline", start="2030-01-15")]},
            RuntimeError("boom"),
        ]
    )

    class FakeGmail:
        def __init__(self, cfg=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def search(self, query, *args, **kwargs):
            assert query == d.SEARCH_QUERY
            return messages

    monkeypatch.setattr(d, "Gmail", FakeGmail)
    monkeypatch.setattr(d, "LMStudio", lambda: lm)
    monkeypatch.setattr(d, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    d.main([])

    ics_files = list(tmp_path.rglob("deadlines-*.ics"))
    assert len(ics_files) == 1
    cal = icalendar.Calendar.from_ical(ics_files[0].read_bytes())
    events = cal.walk("VEVENT")
    assert len(events) == 1  # past item dropped, duplicate renewal dropped
    assert str(events[0].get("SUMMARY")) == "Contract renewal"

    [report] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    report_html = report.read_text(encoding="utf-8")
    assert "Contract renewal" in report_html
    assert "Invoice paid" not in report_html  # past item dropped from the upcoming listing entirely
    assert ics_files[0].name in report_html
    assert "boom" in report_html  # failed message surfaced, not swallowed


def test_drop_past_keeps_an_all_day_item_dated_today():
    today = d._build_item(item(title="Due today", start="2026-09-24", all_day=True), make_message())
    upcoming, past = d.drop_past([today], NOW)
    assert [i.title for i in upcoming] == ["Due today"]
    assert not past
