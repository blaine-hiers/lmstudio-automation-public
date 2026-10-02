"""RFC 5545 compliance for `lmsa.ics`: folding at 75 octets, TEXT escaping,
CRLF line endings, all-day vs timed VEVENTs, and UID stability."""

from datetime import date, datetime, timezone

import icalendar
import pytest

from lmsa.ics import Event, make_uid, render_ics, write_ics


def test_render_uses_crlf_line_endings():
    text = render_ics([Event(uid="u1", title="Meeting", start=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc))])
    assert "\r\n" in text
    assert "\n" not in text.replace("\r\n", "")


def test_no_content_line_exceeds_75_octets():
    text = render_ics(
        [
            Event(
                uid="u1",
                title="A very long meeting title that should definitely need folding across several lines of output",
                start=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
                description="Also a fairly long description, long enough on its own to need folding too.",
            )
        ]
    )
    for line in text.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75, line


def test_folding_never_splits_a_multibyte_character():
    # A title built so a 3-byte UTF-8 character ('€') lands right at the
    # fold boundary: enough ASCII padding to push the fold point mid-char.
    title = ("x" * 66) + "€€€€€€€€€€"
    event = Event(uid="u1", title=title, start=datetime(2026, 1, 5, tzinfo=timezone.utc))
    text = render_ics([event])
    # Every physical line must itself be valid UTF-8 (a split multi-byte
    # character would make .encode("utf-8") followed by round-tripping
    # through the fold logic produce an unequal or undecodable result).
    for line in text.split("\r\n"):
        line.encode("utf-8")  # raises if this line is not a valid string, which can't happen; real check below
        assert len(line.encode("utf-8")) <= 75
    # And round-tripping through a real parser must recover the exact title.
    cal = icalendar.Calendar.from_ical(text)
    [vevent] = cal.walk("VEVENT")
    assert str(vevent.get("SUMMARY")) == title


@pytest.mark.parametrize(
    "raw",
    [
        "Semicolons; and, commas",
        "Backslash \\ literal",
        "Multi\nline\ntitle",
    ],
)
def test_text_escaping_round_trips_through_a_real_parser(raw):
    event = Event(uid="u1", title=raw, start=datetime(2026, 1, 5, tzinfo=timezone.utc))
    text = render_ics([event])
    cal = icalendar.Calendar.from_ical(text)
    [vevent] = cal.walk("VEVENT")
    assert str(vevent.get("SUMMARY")) == raw


def test_timed_event_is_utc():
    # Aware, non-UTC input -> converted to UTC on the wire.
    from datetime import timedelta, tzinfo

    class Plus2(tzinfo):
        def utcoffset(self, dt):
            return timedelta(hours=2)

        def dst(self, dt):
            return timedelta(0)

        def tzname(self, dt):
            return "+02:00"

    event = Event(uid="u1", title="x", start=datetime(2026, 1, 5, 14, 0, tzinfo=Plus2()))
    text = render_ics([event])
    assert "DTSTART:20260105T120000Z" in text.replace("\r\n ", "")


def test_naive_timed_datetime_is_treated_as_local_system_time():
    naive = datetime(2026, 1, 5, 14, 0)
    expected_utc = naive.astimezone(timezone.utc)
    event = Event(uid="u1", title="x", start=naive)
    text = render_ics([event])
    assert expected_utc.strftime("DTSTART:%Y%m%dT%H%M%SZ") in text.replace("\r\n ", "")


def test_all_day_event_uses_value_date_and_exclusive_dtend_by_default():
    event = Event(uid="u1", title="Deadline", start=date(2026, 1, 5), all_day=True)
    text = render_ics([event])
    cal = icalendar.Calendar.from_ical(text)
    [vevent] = cal.walk("VEVENT")
    assert vevent.get("DTSTART").dt == date(2026, 1, 5)
    assert vevent.get("DTEND").dt == date(2026, 1, 6)  # exclusive: the day after


def test_all_day_event_honors_an_explicit_end():
    event = Event(uid="u1", title="Conference", start=date(2026, 1, 5), end=date(2026, 1, 8), all_day=True)
    text = render_ics([event])
    cal = icalendar.Calendar.from_ical(text)
    [vevent] = cal.walk("VEVENT")
    assert vevent.get("DTEND").dt == date(2026, 1, 8)


def test_dtstamp_is_utc():
    event = Event(uid="u1", title="x", start=date(2026, 1, 5), all_day=True)
    text = render_ics([event])
    cal = icalendar.Calendar.from_ical(text)
    [vevent] = cal.walk("VEVENT")
    assert vevent.get("DTSTAMP").dt.tzinfo is not None
    assert vevent.get("DTSTAMP").dt.utcoffset().total_seconds() == 0


def test_make_uid_is_deterministic_and_distinguishes_its_parts():
    assert make_uid("a", "b", "c") == make_uid("a", "b", "c")
    assert make_uid("ab", "c") != make_uid("a", "bc")


def test_write_ics_writes_the_file_and_returns_its_path(tmp_path):
    path = write_ics([Event(uid="u1", title="x", start=date(2026, 1, 5), all_day=True)], tmp_path / "out" / "x.ics")
    assert path.exists()
    assert path.read_bytes().startswith(b"BEGIN:VCALENDAR\r\n")


def test_calendar_parses_as_valid_icalendar_with_multiple_events():
    events = [
        Event(uid=make_uid("m1", "t1"), title="Deadline one", start=datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)),
        Event(uid=make_uid("m2", "t2"), title="All-day event", start=date(2026, 1, 6), all_day=True, location="Room A"),
    ]
    cal = icalendar.Calendar.from_ical(render_ics(events))
    assert len(cal.walk("VEVENT")) == 2


def test_bare_carriage_return_is_escaped_not_emitted():
    ev = Event(uid=make_uid("cr"), title="Weird\rTitle", start=date(2026, 10, 1), all_day=True)
    body = render_ics([ev])
    assert "SUMMARY:Weird\\nTitle\r\n" in body
    assert "\r" not in body.replace("\r\n", "")
