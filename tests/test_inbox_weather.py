"""Unit tests for tasks.inbox_weather.

Everything here works off headers-only Message fakes -- no body text is
ever set (mirrors the real headers_only=True fetch), and there is never
more than one model call per run (the forecast), never one per message.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from lmsa.sources.gmail import Message
from lmsa.store import Store
from tasks import inbox_weather as weather

NOW = datetime(2026, 9, 24, 18, 0, 0, tzinfo=timezone.utc)


def _msg(
    *,
    msg_id="m1",
    thread_id="t1",
    from_addr="bob@example.com",
    from_name="Bob",
    to=("me@example.com",),
    subject="Subj",
    date_=None,
    web_url="https://mail.google.com/mail/u/0/#all/1",
) -> Message:
    return Message(
        msg_id=msg_id,
        thread_id=thread_id,
        labels=[],
        flags=[],
        from_name=from_name,
        from_addr=from_addr,
        to=list(to),
        reply_to="",
        subject=subject,
        date=date_,
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text="",  # headers-only: never populated
        attachments=[],
        web_url=web_url,
    )


# -- daily_counts ----------------------------------------------------------------


def test_daily_counts_buckets_by_local_calendar_day():
    # UTC 2026-09-24 02:00 is still 2026-09-23 in a UTC-6 local zone, but
    # datetime.astimezone() with no arg converts to *this machine's* local
    # zone, which in a CI/test box is typically UTC -- so pick a date that's
    # unambiguous regardless of the runner's zone: exercise same-day bucketing.
    d = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    counts = weather.daily_counts([_msg(date_=d)], NOW, days=7)
    assert counts[d.astimezone().date()] == 1
    assert sum(counts.values()) == 1
    assert len(counts) == 7


def test_daily_counts_skips_messages_with_no_date():
    counts = weather.daily_counts([_msg(date_=None)], NOW, days=7)
    assert sum(counts.values()) == 0


def test_daily_counts_ignores_messages_outside_the_window():
    old = NOW - timedelta(days=100)
    counts = weather.daily_counts([_msg(date_=old)], NOW, days=7)
    assert sum(counts.values()) == 0


# -- weekday_hour_counts -----------------------------------------------------------


def test_weekday_hour_counts_places_message_in_local_weekday_and_hour():
    d = datetime(2026, 9, 21, 14, 30, tzinfo=timezone.utc)  # a Monday
    local = d.astimezone()
    grid = weather.weekday_hour_counts([_msg(date_=d)])
    assert grid[local.weekday()][local.hour] == 1
    assert sum(sum(row) for row in grid) == 1


def test_weekday_hour_counts_skips_messages_with_no_date():
    grid = weather.weekday_hour_counts([_msg(date_=None)])
    assert sum(sum(row) for row in grid) == 0


# -- top_senders ---------------------------------------------------------------


def test_top_senders_excludes_own_addresses_and_ranks_by_count():
    messages = [
        _msg(from_addr="alice@example.com", from_name="Alice"),
        _msg(from_addr="alice@example.com", from_name="Alice"),
        _msg(from_addr="bob@example.com", from_name="Bob"),
        _msg(from_addr="me@example.com", from_name="Me"),  # excluded: it's us
    ]
    own = {"me@example.com"}

    ranked = weather.top_senders(messages, own)

    assert ranked[0] == ("Alice", 2)
    assert ("Bob", 1) in ranked
    assert sum(count for _name, count in ranked) == 3  # the own-address message is excluded


def test_top_senders_respects_limit():
    messages = [_msg(from_addr=f"s{i}@example.com", from_name=f"S{i}") for i in range(5)]
    own: set[str] = set()
    ranked = weather.top_senders(messages, own, limit=2)
    assert len(ranked) == 2


def test_top_senders_skips_blank_from_addr():
    messages = [_msg(from_addr=""), _msg(from_addr="bob@example.com")]
    ranked = weather.top_senders(messages, set())
    assert len(ranked) == 1


# -- reply_hours -----------------------------------------------------------------


def test_reply_hours_pairs_sent_with_most_recent_prior_inbound():
    inbound_old = _msg(thread_id="t1", from_addr="bob@example.com", date_=NOW - timedelta(hours=10))
    inbound_new = _msg(thread_id="t1", from_addr="bob@example.com", date_=NOW - timedelta(hours=3))
    sent = _msg(thread_id="t1", from_addr="me@example.com", date_=NOW)
    own = {"me@example.com"}

    gaps = weather.reply_hours([sent], [inbound_old, inbound_new, sent], own)

    assert gaps == pytest.approx([3.0], abs=0.01)  # paired with the newer inbound, not the older


def test_reply_hours_skips_thread_with_no_prior_inbound():
    sent = _msg(thread_id="t1", from_addr="me@example.com", date_=NOW)
    own = {"me@example.com"}
    gaps = weather.reply_hours([sent], [sent], own)
    assert gaps == []


def test_reply_hours_excludes_self_and_automated_as_inbound():
    self_msg = _msg(thread_id="t1", from_addr="me@example.com", date_=NOW - timedelta(hours=5))
    bounce = _msg(
        thread_id="t1", from_addr="mailer-daemon@example.com", subject="Delivery Status Notification (Failure)",
        date_=NOW - timedelta(hours=4),
    )
    sent = _msg(thread_id="t1", from_addr="me@example.com", date_=NOW)
    own = {"me@example.com"}

    gaps = weather.reply_hours([sent], [self_msg, bounce, sent], own)

    assert gaps == []


def test_reply_hours_skips_sent_message_missing_date_or_thread():
    no_date = _msg(thread_id="t1", from_addr="me@example.com", date_=None)
    no_thread = _msg(thread_id="", from_addr="me@example.com", date_=NOW)
    own = {"me@example.com"}
    assert weather.reply_hours([no_date, no_thread], [], own) == []


# -- unread snapshot kv ----------------------------------------------------------


def test_unread_snapshot_round_trips_through_store(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as db:
        assert weather.load_unread_snapshots(db.conn) == []
        weather.record_unread_snapshot(db.conn, NOW - timedelta(days=7), 12)
        weather.record_unread_snapshot(db.conn, NOW, 20)

        snapshots = weather.load_unread_snapshots(db.conn)

    assert [s["unread"] for s in snapshots] == [12, 20]


def test_closest_to_week_ago_returns_none_with_no_snapshots():
    assert weather.closest_to_week_ago([], NOW) is None


def test_closest_to_week_ago_picks_nearest_to_seven_days_back():
    near = {"unread": 5, "captured_at": NOW - timedelta(days=7, hours=1)}
    far = {"unread": 99, "captured_at": NOW - timedelta(days=20)}
    result = weather.closest_to_week_ago([near, far], NOW)
    assert result["unread"] == 5


def test_closest_to_week_ago_ignores_a_one_day_old_snapshot():
    # Too recent to be "about a week ago" -- must not be reported as one.
    one_day = {"unread": 5, "captured_at": NOW - timedelta(days=1)}
    assert weather.closest_to_week_ago([one_day], NOW) is None


def test_closest_to_week_ago_returns_a_seven_day_old_snapshot():
    seven_day = {"unread": 5, "captured_at": NOW - timedelta(days=7)}
    result = weather.closest_to_week_ago([seven_day], NOW)
    assert result is not None
    assert result["unread"] == 5


def test_closest_to_week_ago_ignores_a_sixty_day_old_snapshot():
    # Too stale (e.g. after a long gap between runs) to stand in for "last week".
    sixty_day = {"unread": 5, "captured_at": NOW - timedelta(days=60)}
    assert weather.closest_to_week_ago([sixty_day], NOW) is None


def test_closest_to_week_ago_includes_the_five_and_ten_day_boundaries():
    five_day = {"unread": 1, "captured_at": NOW - timedelta(days=5)}
    ten_day = {"unread": 2, "captured_at": NOW - timedelta(days=10)}
    assert weather.closest_to_week_ago([five_day], NOW)["unread"] == 1
    assert weather.closest_to_week_ago([ten_day], NOW)["unread"] == 2


# -- snapshot pruning --------------------------------------------------------------


def test_record_unread_snapshot_prunes_entries_older_than_retention(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as db:
        weather.record_unread_snapshot(db.conn, NOW - timedelta(days=90), 1)
        weather.record_unread_snapshot(db.conn, NOW - timedelta(days=30), 2)
        weather.record_unread_snapshot(db.conn, NOW, 3)

        snapshots = weather.load_unread_snapshots(db.conn)

    # The 90-day-old snapshot is gone; the 30-day-old and current ones remain.
    assert [s["unread"] for s in snapshots] == [2, 3]


def test_record_unread_snapshot_keeps_snapshots_inside_retention(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as db:
        weather.record_unread_snapshot(db.conn, NOW - timedelta(days=59), 1)
        weather.record_unread_snapshot(db.conn, NOW, 2)

        snapshots = weather.load_unread_snapshots(db.conn)

    assert [s["unread"] for s in snapshots] == [1, 2]


# -- build_report ------------------------------------------------------------------


def _stats(**overrides) -> weather.Stats:
    defaults = dict(
        total=10,
        daily_counts={date(2026, 9, 24): 3, date(2026, 9, 23): 7},
        grid=[[0] * 24 for _ in range(7)],
        senders=[("Bob", 3)],
        reply_gaps_hours=[2.0, 4.0],
        median_reply_hours=3.0,
        unread_now=5,
        baseline_unread=None,
        baseline_captured_at=None,
        baseline_days_ago=None,
    )
    defaults.update(overrides)
    return weather.Stats(**defaults)


def test_build_report_shows_no_baseline_yet_on_first_run():
    r = weather.build_report(
        _stats(baseline_unread=None), forecast="Sunny.", forecast_error=None,
        sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "no baseline yet" in html


def test_build_report_shows_unread_delta_when_baseline_present():
    r = weather.build_report(
        _stats(unread_now=8, baseline_unread=5, baseline_days_ago=7),
        forecast="Sunny.", forecast_error=None,
        sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "+3" in html
    assert "7d ago" in html  # the real elapsed days, not a hardcoded label


def test_build_report_shows_the_true_elapsed_days_not_a_hardcoded_week():
    # A baseline just inside the tolerance window (e.g. 9 days old) must be
    # labelled with its real age, not a blanket "7d ago".
    r = weather.build_report(
        _stats(unread_now=8, baseline_unread=5, baseline_days_ago=9),
        forecast="Sunny.", forecast_error=None,
        sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "9d ago" in html
    assert "7d ago" not in html


def test_build_report_shows_forecast_text():
    r = weather.build_report(
        _stats(), forecast="Cloudy with a chance of newsletters.", forecast_error=None,
        sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "Cloudy with a chance of newsletters." in html


def test_build_report_shows_forecast_error_without_crashing():
    r = weather.build_report(
        _stats(), forecast=None, forecast_error="server returned empty content",
        sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "Forecast unavailable" in html
    assert "server returned empty content" in html


def test_build_report_shows_truncation_note():
    r = weather.build_report(
        _stats(), forecast="ok", forecast_error=None,
        sent_truncated=True, all_mail_truncated=True,
    )
    html = r.render()
    assert "truncated" in html
    assert "Sent" in html
    assert "All Mail headers" in html


def test_build_report_escapes_untrusted_sender_name():
    r = weather.build_report(
        _stats(senders=[("<script>alert(1)</script>", 4)]),
        forecast="ok", forecast_error=None, sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_report_handles_no_reply_data():
    r = weather.build_report(
        _stats(reply_gaps_hours=[], median_reply_hours=None),
        forecast="ok", forecast_error=None, sent_truncated=False, all_mail_truncated=False,
    )
    html = r.render()
    assert "no data" in html


# -- main() end to end -----------------------------------------------------------


class FakeGmail:
    def __init__(self, sent_messages, all_headers, unread=3):
        self.sent_messages = sent_messages
        self.all_headers = all_headers
        self.unread = unread
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def sent(self):
        return "[Gmail]/Sent Mail"

    def search(self, query, folder=None, limit=200, *, headers_only=False):
        self.calls.append({"query": query, "folder": folder, "limit": limit, "headers_only": headers_only})
        return self.sent_messages if folder == "[Gmail]/Sent Mail" else self.all_headers

    def count(self, query, folder=None):
        self.calls.append({"query": query, "folder": folder, "count": True})
        return self.unread


class FakeLM:
    def __init__(self, reply=None, error=None):
        self._reply = reply or {"forecast": "Clear skies, low mail pressure."}
        self._error = error
        self.calls = []

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append(prompt)
        if self._error is not None:
            raise self._error
        return self._reply


def _setup_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_IMAP_USER", "me@example.com")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))


def test_main_first_run_writes_report_with_no_baseline(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    sent = _msg(thread_id="t1", from_addr="me@example.com", date_=now - timedelta(hours=2))
    inbound = _msg(thread_id="t1", from_addr="bob@example.com", date_=now - timedelta(hours=5))
    fake_gmail = FakeGmail([sent], [sent, inbound], unread=7)
    fake_lm = FakeLM()

    monkeypatch.setattr(weather, "Gmail", lambda: fake_gmail)
    monkeypatch.setattr(weather, "LMStudio", lambda: fake_lm)
    _setup_env(monkeypatch, tmp_path)

    weather.main()

    search_calls = [c for c in fake_gmail.calls if "count" not in c]
    count_calls = [c for c in fake_gmail.calls if c.get("count")]
    assert len(search_calls) == 2
    assert len(count_calls) == 1

    [written] = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "no baseline yet" in html
    assert "Clear skies, low mail pressure." in html


def test_main_second_run_compares_against_stored_baseline(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    _setup_env(monkeypatch, tmp_path)

    # Seed a week-old snapshot as if a prior run had recorded it.
    with Store(tmp_path / "lmsa.sqlite") as db:
        weather.record_unread_snapshot(db.conn, now - timedelta(days=7), 4)

    fake_gmail = FakeGmail([], [], unread=9)
    fake_lm = FakeLM()
    monkeypatch.setattr(weather, "Gmail", lambda: fake_gmail)
    monkeypatch.setattr(weather, "LMStudio", lambda: fake_lm)

    weather.main()

    [written] = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "+5" in html  # 9 now vs 4 a week ago
    assert "no baseline yet" not in html


def test_main_survives_forecast_failure(monkeypatch, tmp_path):
    _setup_env(monkeypatch, tmp_path)
    fake_gmail = FakeGmail([], [], unread=1)
    fake_lm = FakeLM(error=RuntimeError("model unavailable"))
    monkeypatch.setattr(weather, "Gmail", lambda: fake_gmail)
    monkeypatch.setattr(weather, "LMStudio", lambda: fake_lm)

    weather.main()  # must not raise

    [written] = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "Forecast unavailable" in html
    assert "model unavailable" in html


def test_main_records_a_run_row(monkeypatch, tmp_path):
    _setup_env(monkeypatch, tmp_path)
    fake_gmail = FakeGmail([], [], unread=2)
    fake_lm = FakeLM()
    monkeypatch.setattr(weather, "Gmail", lambda: fake_gmail)
    monkeypatch.setattr(weather, "LMStudio", lambda: fake_lm)

    weather.main()

    with Store(tmp_path / "lmsa.sqlite") as db:
        row = db.conn.execute("SELECT task, status FROM runs").fetchone()
    assert row == ("tasks.inbox_weather", "ok")
