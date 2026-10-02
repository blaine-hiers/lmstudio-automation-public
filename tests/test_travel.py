"""tasks.travel: per-message/per-item isolation, confirmation-number
supersede + cancellation resolution (newest email wins, order-independent),
deterministic 48h trip chaining over overlapping/unsorted input, hotel-gap
detection, stable re-run UIDs, and the end-to-end .ics + report output."""

from datetime import datetime, timedelta, timezone

import icalendar
import pytest

import tasks.travel as t
from lmsa.sources.gmail import Message

UTC = timezone.utc
NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)


def make_message(msg_id="m1", subject="Subject", text="body text", date=NOW, web_url="https://mail.google.com/x"):
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
        date=date,
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


def item(
    kind="flight",
    provider="Example Air",
    confirmation="ABC123",
    start="2026-10-01T14:00:00",
    end=None,
    from_=None,
    to=None,
    address=None,
    cancelled=False,
):
    return {
        "kind": kind,
        "provider": provider,
        "confirmation": confirmation,
        "start": start,
        "end": end,
        "from": from_,
        "to": to,
        "address": address,
        "cancelled": cancelled,
    }


def make_segment(
    kind="flight",
    provider="Example Air",
    confirmation="ABC123",
    start=NOW,
    end=None,
    cancelled=False,
    source_msg_id="m1",
    source_subject="Subject",
    source_date=NOW,
):
    return t.Segment(
        kind=kind,
        provider=provider,
        confirmation=confirmation,
        start=start,
        end=end,
        origin=None,
        destination=None,
        address=None,
        cancelled=cancelled,
        tz_known=start.tzinfo is not None,
        source_msg_id=source_msg_id,
        source_subject=source_subject,
        source_date=source_date,
        source_url="https://mail.google.com/x",
    )


# -- gather(): per-message / per-item isolation --------------------------------


def test_gather_isolates_a_failing_message_and_keeps_the_rest():
    lm = FakeLMStudio(
        replies=[
            RuntimeError("server down"),
            {"items": [item(provider="Still works")]},
        ]
    )
    messages = [make_message(msg_id="bad"), make_message(msg_id="good")]

    segments, unparseable, failed = t.gather(lm, messages)

    assert len(segments) == 1
    assert segments[0].provider == "Still works"
    assert len(failed) == 1
    assert failed[0].msg_id == "bad"
    assert "server down" in failed[0].error
    assert not unparseable


def test_gather_drops_only_the_unparseable_item_not_the_whole_message():
    lm = FakeLMStudio(
        replies=[
            {
                "items": [
                    item(provider="Good item", start="2026-10-01T10:00:00"),
                    item(provider="Bad item", start="not a date"),
                ]
            }
        ]
    )
    segments, unparseable, failed = t.gather(lm, [make_message()])

    assert [s.provider for s in segments] == ["Good item"]
    assert [u.provider for u in unparseable] == ["Bad item"]
    assert unparseable[0].raw_start == "not a date"
    assert "unparseable start" in unparseable[0].reason
    assert not failed


def test_gather_records_an_invalid_kind_instead_of_guessing_one():
    # Previously an invalid/missing kind silently defaulted to "flight",
    # which could mistag a hotel and trip up _find_gaps's "no hotel" check.
    lm = FakeLMStudio(replies=[{"items": [item(kind="bus", provider="Mystery bus")]}])
    segments, unparseable, failed = t.gather(lm, [make_message()])

    assert not segments
    assert [u.provider for u in unparseable] == ["Mystery bus"]
    assert "invalid kind" in unparseable[0].reason
    assert not failed


def test_gather_truncates_long_bodies_before_prompting():
    long_text = "x" * 10_000
    lm = FakeLMStudio(replies=[{"items": []}])
    t.gather(lm, [make_message(text=long_text)])
    assert len(lm.calls[0]) < len(long_text) + 500
    assert "truncated" in lm.calls[0]


# -- start/end parsing (offset presence -> tz_known) ----------------------------


def test_build_segment_marks_tz_known_only_when_an_offset_is_given():
    with_offset = t._build_segment(item(start="2026-10-01T14:00:00+00:00"), make_message())
    without_offset = t._build_segment(item(start="2026-10-01T14:00:00"), make_message())
    assert with_offset.tz_known is True
    assert without_offset.tz_known is False


def test_build_segment_raises_bad_item_on_unparseable_date():
    with pytest.raises(t._BadItem):
        t._build_segment(item(start="not a date"), make_message())


def test_build_segment_raises_bad_item_on_invalid_kind():
    with pytest.raises(t._BadItem):
        t._build_segment(item(kind="bus"), make_message())


# -- resolve_segments(): supersede + cancellation, order-independent -----------


def test_resolve_segments_newer_email_supersedes_older_regardless_of_list_order():
    older = make_segment(provider="Old data", source_msg_id="old", source_date=NOW - timedelta(days=5))
    newer = make_segment(provider="New data", source_msg_id="new", source_date=NOW)

    kept_a, _ = t.resolve_segments([older, newer])
    kept_b, _ = t.resolve_segments([newer, older])  # shuffled input order

    assert [s.provider for s in kept_a] == ["New data"]
    assert [s.provider for s in kept_b] == ["New data"]


def test_resolve_segments_cancellation_from_the_newest_email_removes_the_segment():
    booked = make_segment(provider="Booked", source_msg_id="a", source_date=NOW - timedelta(days=2))
    cancel = make_segment(
        provider="Cancel notice", cancelled=True, source_msg_id="b", source_date=NOW
    )

    kept, cancelled = t.resolve_segments([booked, cancel])

    assert kept == []
    assert [s.provider for s in cancelled] == ["Cancel notice"]


def test_resolve_segments_an_older_cancellation_is_superseded_by_a_newer_rebooking():
    cancel = make_segment(
        provider="Cancel notice", cancelled=True, source_msg_id="a", source_date=NOW - timedelta(days=2)
    )
    rebooked = make_segment(provider="Rebooked", source_msg_id="b", source_date=NOW)

    kept, cancelled = t.resolve_segments([cancel, rebooked])

    assert [s.provider for s in kept] == ["Rebooked"]
    assert cancelled == []


def test_resolve_segments_keeps_segments_with_no_confirmation_independently():
    a = make_segment(provider="A", confirmation="", source_msg_id="a")
    b = make_segment(provider="B", confirmation="", source_msg_id="b")

    kept, cancelled = t.resolve_segments([a, b])

    assert {s.provider for s in kept} == {"A", "B"}
    assert cancelled == []


def test_resolve_segments_confirmation_matching_is_case_and_whitespace_insensitive():
    older = make_segment(provider="Old", confirmation="abc 123", source_date=NOW - timedelta(days=1))
    newer = make_segment(provider="New", confirmation="ABC123", source_date=NOW)

    kept, _ = t.resolve_segments([older, newer])

    assert [s.provider for s in kept] == ["New"]


def test_resolve_segments_cancellation_wins_a_same_message_tie_regardless_of_array_order():
    # Two items for the same (kind, confirmation) from the SAME message
    # (equal source_date and msg_id) can't be ordered by date at all --
    # the booking and its cancellation notice landing in one email's own
    # items array. The cancellation must always win, whichever order the
    # model happened to list them in.
    booking = make_segment(
        provider="Booking", confirmation="DUP1", cancelled=False, source_msg_id="m", source_date=NOW
    )
    cancel = make_segment(
        provider="Cancel notice", confirmation="DUP1", cancelled=True, source_msg_id="m", source_date=NOW
    )

    kept_booking_first, cancelled_booking_first = t.resolve_segments([booking, cancel])
    kept_cancel_first, cancelled_cancel_first = t.resolve_segments([cancel, booking])

    assert kept_booking_first == [] == kept_cancel_first
    assert [s.provider for s in cancelled_booking_first] == ["Cancel notice"]
    assert [s.provider for s in cancelled_cancel_first] == ["Cancel notice"]


# -- drop_past -------------------------------------------------------------------


def test_drop_past_splits_on_effective_end():
    past = make_segment(provider="Past", start=NOW - timedelta(days=10))
    future = make_segment(provider="Future", start=NOW + timedelta(days=10))
    upcoming, dropped = t.drop_past([past, future], NOW)
    assert [s.provider for s in upcoming] == ["Future"]
    assert [s.provider for s in dropped] == ["Past"]


def test_drop_past_keeps_a_segment_still_in_progress_via_its_end():
    in_progress = make_segment(
        provider="Long flight", start=NOW - timedelta(hours=2), end=NOW + timedelta(hours=2)
    )
    upcoming, dropped = t.drop_past([in_progress], NOW)
    assert [s.provider for s in upcoming] == ["Long flight"]
    assert not dropped


# -- group_trips(): deterministic 48h chaining, overlap, order-independence ----


def test_group_trips_chains_segments_within_48h():
    flight_out = make_segment(
        kind="flight", confirmation="F1", start=NOW, end=NOW + timedelta(hours=3)
    )
    hotel = make_segment(
        kind="hotel",
        confirmation="H1",
        start=NOW + timedelta(hours=20),
        end=NOW + timedelta(days=3),
    )
    flight_back = make_segment(
        kind="flight",
        confirmation="F2",
        start=NOW + timedelta(days=3, hours=10),
        end=NOW + timedelta(days=3, hours=13),
    )

    trips = t.group_trips([flight_out, hotel, flight_back])

    assert len(trips) == 1
    assert {s.confirmation for s in trips[0].segments} == {"F1", "H1", "F2"}


def test_group_trips_chains_transitively_through_a_middle_leg():
    # first->middle and middle->last are both <=48h apart, but first->last alone is not.
    first = make_segment(confirmation="A", start=NOW, end=NOW + timedelta(hours=1))
    middle = make_segment(
        confirmation="B", start=NOW + timedelta(hours=40), end=NOW + timedelta(hours=41)
    )
    last = make_segment(
        confirmation="C", start=NOW + timedelta(hours=80), end=NOW + timedelta(hours=81)
    )

    trips = t.group_trips([first, middle, last])

    assert len(trips) == 1
    assert {s.confirmation for s in trips[0].segments} == {"A", "B", "C"}


def test_group_trips_splits_when_the_gap_exceeds_48h():
    trip1 = make_segment(confirmation="A", start=NOW, end=NOW + timedelta(hours=1))
    trip2 = make_segment(
        confirmation="B", start=NOW + timedelta(hours=60), end=NOW + timedelta(hours=61)
    )

    trips = t.group_trips([trip1, trip2])

    assert len(trips) == 2
    assert {s.confirmation for s in trips[0].segments} == {"A"}
    assert {s.confirmation for s in trips[1].segments} == {"B"}


def test_group_trips_is_order_independent():
    a = make_segment(confirmation="A", start=NOW, end=NOW + timedelta(hours=1))
    b = make_segment(
        confirmation="B", start=NOW + timedelta(hours=10), end=NOW + timedelta(hours=11)
    )
    c = make_segment(
        confirmation="C", start=NOW + timedelta(hours=90), end=NOW + timedelta(hours=91)
    )

    forward = t.group_trips([a, b, c])
    shuffled = t.group_trips([c, a, b])
    reversed_ = t.group_trips([c, b, a])

    def signature(trips):
        return [sorted(s.confirmation for s in trip.segments) for trip in trips]

    assert signature(forward) == signature(shuffled) == signature(reversed_)


def test_group_trips_merges_overlapping_segments_via_running_max_end():
    # A long hotel stay overlapping two short flights, given out of order,
    # must not let the hotel's end get clobbered by a later, shorter segment.
    hotel = make_segment(
        kind="hotel", confirmation="H", start=NOW, end=NOW + timedelta(days=5)
    )
    flight1 = make_segment(
        kind="flight", confirmation="F1", start=NOW + timedelta(hours=1), end=NOW + timedelta(hours=2)
    )
    flight2 = make_segment(
        kind="flight",
        confirmation="F2",
        start=NOW + timedelta(days=4, hours=20),
        end=NOW + timedelta(days=4, hours=21),
    )

    trips = t.group_trips([flight1, hotel, flight2])

    assert len(trips) == 1
    assert {s.confirmation for s in trips[0].segments} == {"H", "F1", "F2"}


# -- hotel-gap detection ---------------------------------------------------------


def test_group_trips_flags_a_night_with_no_hotel():
    arrive = make_segment(kind="flight", confirmation="F1", start=NOW, end=NOW + timedelta(hours=1))
    depart = make_segment(
        kind="flight",
        confirmation="F2",
        start=NOW + timedelta(days=2),
        end=NOW + timedelta(days=2, hours=1),
    )
    [trip] = t.group_trips([arrive, depart])

    assert trip.gaps == [NOW.date(), (NOW + timedelta(days=1)).date()]


def test_group_trips_reports_no_gaps_when_hotel_fully_covers_the_nights():
    arrive = make_segment(kind="flight", confirmation="F1", start=NOW, end=NOW + timedelta(hours=1))
    hotel = make_segment(
        kind="hotel", confirmation="H1", start=NOW, end=NOW + timedelta(days=2)
    )
    depart = make_segment(
        kind="flight",
        confirmation="F2",
        start=NOW + timedelta(days=2),
        end=NOW + timedelta(days=2, hours=1),
    )
    [trip] = t.group_trips([arrive, hotel, depart])

    assert trip.gaps == []


def test_group_trips_a_hotel_with_no_stated_checkout_does_not_count_as_coverage():
    arrive = make_segment(kind="flight", confirmation="F1", start=NOW, end=NOW + timedelta(hours=1))
    hotel_no_end = make_segment(kind="hotel", confirmation="H1", start=NOW, end=None)
    depart = make_segment(
        kind="flight",
        confirmation="F2",
        start=NOW + timedelta(days=2),
        end=NOW + timedelta(days=2, hours=1),
    )
    [trip] = t.group_trips([arrive, hotel_no_end, depart])

    assert trip.gaps  # still flagged -- an end-less hotel can't be verified to cover anything


# -- trip_key(): stable, content-derived .ics filenames -------------------------


def test_trip_key_is_stable_across_rebuilds_of_the_same_trip():
    trip1 = [make_segment(confirmation="F1", start=NOW), make_segment(confirmation="H1", start=NOW)]
    trip2 = [make_segment(confirmation="F1", start=NOW), make_segment(confirmation="H1", start=NOW)]
    assert t.trip_key(trip1) == t.trip_key(trip2)


def test_trip_key_is_order_independent():
    a = make_segment(confirmation="F1", start=NOW)
    b = make_segment(confirmation="H1", start=NOW + timedelta(hours=5))
    assert t.trip_key([a, b]) == t.trip_key([b, a])


def test_trip_key_differs_for_a_different_trip():
    trip1 = [make_segment(confirmation="F1", start=NOW)]
    trip2 = [make_segment(confirmation="F2", start=NOW)]
    assert t.trip_key(trip1) != t.trip_key(trip2)


def test_trip_key_does_not_depend_on_wall_clock_time():
    # Two runs of the SAME trip at different, even same-second, wall-clock
    # times must map to the same key -- that's what makes a re-run
    # overwrite its own file instead of colliding with (or orphaning) an
    # unrelated trip that a timestamp+ordinal name would risk.
    segments = [make_segment(confirmation="F1", start=NOW)]
    assert t.trip_key(segments) == t.trip_key(segments)


# -- to_event(): stable UIDs across re-runs -------------------------------------


def test_to_event_uid_is_stable_across_rebuilds_for_the_same_segment():
    seg1 = make_segment(confirmation="ABC123", kind="flight", start=NOW)
    seg2 = make_segment(confirmation="ABC123", kind="flight", start=NOW)  # simulates a re-run

    assert t.to_event(seg1).uid == t.to_event(seg2).uid


def test_to_event_uid_differs_for_a_different_confirmation():
    seg1 = make_segment(confirmation="ABC123", kind="flight", start=NOW)
    seg2 = make_segment(confirmation="XYZ999", kind="flight", start=NOW)

    assert t.to_event(seg1).uid != t.to_event(seg2).uid


# -- end-to-end main() -----------------------------------------------------------


def test_main_writes_per_trip_ics_and_report(monkeypatch, tmp_path):
    # Dates chosen far from the real "now" main() uses (the real clock), so
    # the upcoming/past split can't flake near a boundary -- same convention
    # as tasks/deadlines.py's own main() test.
    far_future = "2030-01-15T10:00:00"
    messages = [
        make_message(msg_id="m-flight", subject="Your flight confirmation", text="Flight to LHR."),
        make_message(msg_id="m-hotel", subject="Hotel booking confirmed", text="Stay confirmed."),
        make_message(msg_id="m-cancel-old", subject="Old booking", text="Old car rental."),
        make_message(
            msg_id="m-cancel-new",
            subject="Cancellation",
            text="Your car rental was cancelled.",
            date=NOW + timedelta(hours=1),
        ),
        make_message(msg_id="m-bad", subject="Broken", text="garbled"),
    ]
    lm = FakeLMStudio(
        replies=[
            {
                "items": [
                    item(
                        kind="flight",
                        provider="Example Air",
                        confirmation="FL1",
                        start=far_future,
                        end="2030-01-15T14:00:00",
                        from_="JFK",
                        to="LHR",
                    )
                ]
            },
            {
                "items": [
                    item(
                        kind="hotel",
                        provider="Example Hotel",
                        confirmation="HT1",
                        start="2030-01-15T16:00:00",
                        end="2030-01-17T11:00:00",
                        address="1 Example Street",
                    )
                ]
            },
            {
                "items": [
                    item(
                        kind="car",
                        provider="Example Rental",
                        confirmation="CR1",
                        start="2030-01-15T09:00:00",
                        end="2030-01-16T09:00:00",
                    )
                ]
            },
            {
                "items": [
                    item(kind="car", provider="Example Rental", confirmation="CR1", cancelled=True)
                ]
            },
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
            assert query == t.SEARCH_QUERY
            return messages

    monkeypatch.setattr(t, "Gmail", FakeGmail)
    monkeypatch.setattr(t, "LMStudio", lambda: lm)
    monkeypatch.setattr(t, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    t.main([])

    ics_files = sorted(tmp_path.rglob("travel-*.ics"))
    assert len(ics_files) == 1  # flight + hotel chain into one trip; the cancelled car never gets one
    cal = icalendar.Calendar.from_ical(ics_files[0].read_bytes())
    events = cal.walk("VEVENT")
    summaries = {str(e.get("SUMMARY")) for e in events}
    assert any("Example Air" in s for s in summaries)
    assert any("Example Hotel" in s for s in summaries)
    assert not any("Rental" in s for s in summaries)  # cancelled -- never emitted

    [report] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    report_html = report.read_text(encoding="utf-8")
    assert "Example Air" in report_html
    assert "Example Hotel" in report_html
    assert "Example Rental" in report_html  # shown in the Cancelled section
    assert "boom" in report_html  # failed message surfaced, not swallowed
    assert ics_files[0].name in report_html


def test_main_two_distinct_trips_get_distinct_ics_files_and_a_rerun_overwrites_not_duplicates(
    monkeypatch, tmp_path
):
    messages = [
        make_message(msg_id="m-trip-a", subject="Flight A confirmation", text="Trip A flight."),
        make_message(msg_id="m-trip-b", subject="Flight B confirmation", text="Trip B flight."),
    ]

    def replies():
        return [
            {"items": [item(kind="flight", provider="Airline A", confirmation="AAA", start="2030-02-01T09:00:00")]},
            {"items": [item(kind="flight", provider="Airline B", confirmation="BBB", start="2030-06-01T09:00:00")]},
        ]

    class FakeGmail:
        def __init__(self, cfg=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def search(self, query, *args, **kwargs):
            return messages

    monkeypatch.setattr(t, "Gmail", FakeGmail)
    monkeypatch.setattr(t, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    monkeypatch.setattr(t, "LMStudio", lambda: FakeLMStudio(replies=replies()))
    t.main([])
    first_run_files = sorted(p.name for p in tmp_path.rglob("travel-*.ics"))
    assert len(first_run_files) == 2  # two trips, far enough apart to never chain

    # A second run over the identical data must overwrite the same two
    # files, not accumulate new ones under a fresh timestamp.
    monkeypatch.setattr(t, "LMStudio", lambda: FakeLMStudio(replies=replies()))
    t.main([])
    second_run_files = sorted(p.name for p in tmp_path.rglob("travel-*.ics"))
    assert second_run_files == first_run_files
