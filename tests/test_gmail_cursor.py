"""lmsa.gmail_cursor: the shared epoch-second Gmail search-bound helpers
used by tasks/job_tracker.py, tasks/receipts.py, and tasks/packages.py."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from lmsa.gmail_cursor import coerce_aware_moment, epoch_query_bound


def test_epoch_query_bound_is_whole_seconds_since_epoch():
    moment = datetime(2024, 9, 24, 12, 0, 0, tzinfo=timezone.utc)
    assert epoch_query_bound(moment) == int(moment.timestamp())


def test_coerce_aware_moment_passes_through_a_full_aware_datetime():
    raw = "2026-01-15T09:30:00+00:00"
    parsed = coerce_aware_moment(raw)
    assert parsed == datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None


def test_coerce_aware_moment_migrates_a_bare_date_to_local_midnight():
    parsed = coerce_aware_moment("2026-01-15")
    assert parsed.tzinfo is not None
    assert parsed.date() == date(2026, 1, 15)
    assert (parsed.hour, parsed.minute, parsed.second) == (0, 0, 0)


# -- shared cursor scheme (GmailCursor / RetrySet) ----------------------------

import json  # noqa: E402

from lmsa.gmail_cursor import (  # noqa: E402
    OVERLAP,
    GmailCursor,
    RetrySet,
    merge_retried,
    parse_stored_moment,
)
from lmsa.sources.gmail import Message, MessageMeta  # noqa: E402
from lmsa.store import Store  # noqa: E402


def _m(msg_id, internal=None, header=None):
    return Message(
        msg_id=msg_id, thread_id="t", labels=[], flags=[], from_name="", from_addr="a@example.com",
        to=[], reply_to="", subject=f"s{msg_id}", date=header, list_unsubscribe="", list_id="",
        auth_results="", text="", attachments=[], web_url="", internal_date=internal,
    )


T0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def test_parse_stored_moment_accepts_iso_bare_date_and_legacy_epoch():
    assert parse_stored_moment(T0.isoformat()) == T0
    assert parse_stored_moment("2026-01-15").tzinfo is not None
    assert parse_stored_moment(int(T0.timestamp())) == T0
    assert parse_stored_moment(None) is None
    assert parse_stored_moment("garbage") is None
    assert parse_stored_moment(True) is None


class _MetaGmail:
    def __init__(self, metas):
        self.metas = metas
        self.queries = []

    def search_meta(self, query):
        self.queries.append(query)
        return list(self.metas)


def _meta(msg_id, when, uid=None):
    return MessageMeta(msg_id=msg_id, internal_date=when, uid=uid or int(msg_id))


def test_scan_reaches_back_by_the_overlap_and_orders_oldest_first(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        c = GmailCursor(store.conn, "t")
        c.since = T0
        # UID order deliberately disagrees with INTERNALDATE order.
        gmail = _MetaGmail([_meta("3", T0 + timedelta(hours=1), uid=1), _meta("2", T0, uid=2),
                            _meta("1", T0 - timedelta(minutes=5), uid=9), _meta("4", T0, uid=3)])
        window = c.scan(gmail, "q", default_since=T0, limit=2, skip=lambda i: i == "4")
    assert gmail.queries == [f"q after:{int((T0 - OVERLAP).timestamp())}"]
    assert [m.msg_id for m in window.candidates] == ["1", "2", "3"]
    assert [m.msg_id for m in window.batch] == ["1", "2"] and window.pending == 1


def test_advance_never_passes_an_unprocessed_message(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        c = GmailCursor(store.conn, "t")
        c.started = T0 + timedelta(days=1)
        window = c.scan(_MetaGmail([_meta("1", T0), _meta("2", T0 + timedelta(hours=2))]),
                        "q", default_since=T0, limit=1, skip=lambda i: False)
        c.advance(window, done=lambda i: i == "1")
        assert GmailCursor(store.conn, "t").since == T0 + timedelta(hours=2)
        c.advance(window, done=lambda i: True)
        assert GmailCursor(store.conn, "t").since == T0 + timedelta(days=1)  # the run's start


def test_legacy_catch_up_state_migrates_without_skipping_anything(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        for key, value in {"last_run": (T0 + timedelta(days=1)).isoformat(),
                           "catchup_before": int(T0.timestamp()),
                           "catchup_ceiling": (T0 + timedelta(days=1)).isoformat()}.items():
            store.conn.execute("INSERT INTO kv(ns, key, value_json, updated) VALUES ('t', ?, ?, 'x')",
                               (key, json.dumps(value)))
        c = GmailCursor(store.conn, "t")
        assert c.since == T0  # min(since, catchup_before)
        c.started = T0 + timedelta(days=2)
        c.advance(c.scan(_MetaGmail([]), "q", default_since=T0, limit=5, skip=lambda i: False),
                  done=lambda i: True)
        keys = {r[0] for r in store.conn.execute("SELECT key FROM kv WHERE ns='t'")}
    assert keys == {"last_run"}


def test_retry_set_counts_attempts_gives_up_and_persists(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        r = RetrySet(store.conn, "t", max_attempts=2)
        r.failed("1", "subj", "boom")
        r.save()
        r = RetrySet(store.conn, "t", max_attempts=2)
        assert r.pending() == ["1"]
        r.failed("1", "subj", "boom again")
        assert [i.msg_id for i in r.gave_up_now] == ["1"]
        assert r.pending() == []
        r.succeeded("1")
        assert r.items == {}


def test_retry_fetch_error_costs_no_attempt_and_missing_message_does(tmp_path):
    class Boom:
        def fetch_by_msgids(self, ids):
            raise OSError("socket closed")

    class Empty:
        def fetch_by_msgids(self, ids):
            return []

    with Store(tmp_path / "db.sqlite") as store:
        r = RetrySet(store.conn, "t")
        r.failed("1", "subj", "boom")
        assert r.fetch(Boom()) == [] and "socket closed" in r.fetch_error
        assert r.items["1"].attempts == 1
        assert r.fetch(Empty()) == []
        assert r.items["1"].attempts == 2 and "no longer found" in r.items["1"].error


def test_retry_set_tolerates_a_garbage_stored_value(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.conn.execute(
            "INSERT INTO kv(ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
            ("t", "retry", json.dumps({"1": {"attempts": "x"}, "2": "nope"}), "now"),
        )
        r = RetrySet(store.conn, "t")
        assert r.pending() == ["1"] and r.items["1"].attempts == 0


def test_merge_retried_dedupes_by_msg_id_keeping_search_order():
    a, b, c = _m("a"), _m("b"), _m("c")
    assert [m.msg_id for m in merge_retried([a, b], [b, c])] == ["a", "b", "c"]
