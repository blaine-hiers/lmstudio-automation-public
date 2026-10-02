import itertools
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from lmsa.sources.gmail import Message
from lmsa.store import Store
from tasks import packages as pkg
from gmail_fakes import MetaSearchMixin

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def make_message(msg_id, subject, text="", date=NOW, from_addr="merchant@example.com"):
    return Message(
        msg_id=msg_id,
        thread_id=f"t-{msg_id}",
        labels=[],
        flags=[],
        from_name="Example Merchant",
        from_addr=from_addr,
        to=["me@example.com"],
        reply_to="",
        subject=subject,
        date=date,
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text=text,
        attachments=[],
        web_url="",
    )


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network."""

    def __init__(self, replies=None):
        self.calls: list[str] = []
        self._replies = list(replies or [])

    def ensure_server(self) -> None:
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append(prompt)
        assert system is not None and "untrusted" in system
        return self._replies.pop(0)


class KeyedFakeLMStudio:
    """`extract()` keyed by the email's subject line rather than call order --
    convenient when `main()` runs more than once (e.g. across a catch-up
    sequence) and predicting call order up front is awkward."""

    def __init__(self, by_subject: dict[str, dict]):
        self._by_subject = by_subject
        self.calls: list[str] = []

    def ensure_server(self) -> None:
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append(prompt)
        subject = prompt.split("\n", 1)[0][len("Subject: "):]
        return self._by_subject[subject]


class FakeGmail(MetaSearchMixin):
    """A `Gmail` stand-in that actually honours `after:`/`before:` in the
    query as whole-second Unix epoch timestamps (like real Gmail search),
    so tests can exercise the truncation/catch-up pagination in `main()`
    the same way it runs for real, including same-day/different-time-of-day
    cases -- mirrors tasks/receipts.py's own test fake for the same reason.
    """

    def __init__(self, messages=()):
        self.messages = list(messages)
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, limit=200, **kwargs):
        self.queries.append(query)
        after_epoch = before_epoch = None
        if "after:" in query:
            after_epoch = int(query.split("after:")[1].split()[0])
        if "before:" in query:
            before_epoch = int(query.split("before:")[1].split()[0])
        matches = [
            m for m in self.messages
            if m.date is not None
            and (after_epoch is None or m.date.timestamp() >= after_epoch)
            and (before_epoch is None or m.date.timestamp() < before_epoch)
        ]
        matches.sort(key=lambda m: m.date, reverse=True)
        return matches[:limit]


def shipment_reply(tracking="1Z999AA10123456784", carrier="UPS", merchant="Widget Co", status="shipped", eta=None):
    return {"carrier": carrier, "tracking_number": tracking, "merchant": merchant, "status": status, "eta": eta}


def non_shipment_reply():
    return {"carrier": None, "tracking_number": None, "merchant": None, "status": "ordered", "eta": None}


# -- tracking number normalisation / validation --------------------------------


def test_normalize_tracking_strips_whitespace_and_uppercases():
    assert pkg.normalize_tracking(" 1z 999 aa1 01234567 84 ") == "1Z999AA10123456784"


def test_normalize_tracking_handles_none_and_blank():
    assert pkg.normalize_tracking(None) is None
    assert pkg.normalize_tracking("   ") is None


def test_valid_tracking_accepts_alnum_and_hyphen():
    assert pkg.valid_tracking("1Z999AA10123456784")
    assert pkg.valid_tracking("1234-5678-9012")


@pytest.mark.parametrize(
    "bad",
    [
        "short",  # < 6 chars
        "a" * 41,  # > 40 chars
        "<script>alert(1)</script>",
        "javascript:alert(1)",
        "1Z999 AA1",  # space (normalize_tracking would have stripped this, but valid_tracking is defensive on its own)
        "",
    ],
)
def test_valid_tracking_rejects_bad_charset_or_length(bad):
    assert not pkg.valid_tracking(bad)


# -- carrier link allowlist -----------------------------------------------------


def test_carrier_link_known_carrier_builds_encoded_url():
    link = pkg.carrier_link("UPS", "1Z999AA10123456784")
    assert link == "https://www.ups.com/track?tracknum=1Z999AA10123456784"


def test_carrier_link_is_case_insensitive_on_carrier_name():
    assert pkg.carrier_link("ups", "1Z999AA10123456784") == pkg.carrier_link("UPS", "1Z999AA10123456784")


def test_carrier_link_unknown_carrier_returns_none():
    assert pkg.carrier_link("SomeRandomCourier", "1Z999AA10123456784") is None
    assert pkg.carrier_link(None, "1Z999AA10123456784") is None


def test_carrier_link_percent_encodes_tracking_number():
    # Slashes aren't in the valid-tracking charset, but carrier_link itself
    # must not assume that -- defense in depth via urllib.parse.quote.
    link = pkg.carrier_link("usps", "AB/CD")
    assert "AB%2FCD" in link


# -- status merge (date-driven, order-independent) -------------------------------


def test_merge_status_advances_monotonically():
    level, ndate, edate = -1, None, None
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "ordered", NOW)
    assert (level, ndate, edate) == (0, NOW, None)
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "shipped", NOW + timedelta(hours=1))
    assert level == 1
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "out_for_delivery", NOW + timedelta(hours=2))
    assert level == 2
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "delivered", NOW + timedelta(hours=3))
    assert level == 3
    assert edate is None


def test_merge_status_ignores_a_regression_to_a_lower_level():
    state = pkg.merge_status(-1, None, None, "delivered", NOW)
    assert state[0] == 3
    # An older/lower-level email arriving afterwards (e.g. processed
    # out-of-order, newest-first) must not regress the shipment -- state is
    # untouched, not just the displayed status.
    same_state = pkg.merge_status(*state, "shipped", NOW - timedelta(days=2))
    assert same_state == state


def test_merge_status_exception_can_land_at_any_level():
    level, ndate, edate = pkg.merge_status(-1, None, None, "ordered", NOW)
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "exception", NOW + timedelta(hours=1))
    assert pkg.displayed_status(level, ndate, edate) == "exception"

    level2, ndate2, edate2 = pkg.merge_status(2, NOW, None, "exception", NOW + timedelta(hours=1))
    assert level2 == 2  # exception doesn't change the level
    assert pkg.displayed_status(level2, ndate2, edate2) == "exception"


def test_merge_status_exception_with_no_prior_level_at_all():
    level, ndate, edate = pkg.merge_status(-1, None, None, "exception", NOW)
    assert level == -1
    assert pkg.displayed_status(level, ndate, edate) == "exception"


def test_merge_status_later_normal_status_clears_exception():
    level, ndate, edate = pkg.merge_status(1, NOW, None, "exception", NOW + timedelta(hours=1))
    assert pkg.displayed_status(level, ndate, edate) == "exception"
    # A normal status dated *after* the exception's own email clears it.
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "out_for_delivery", NOW + timedelta(hours=2))
    assert pkg.displayed_status(level, ndate, edate) == "out_for_delivery"


def test_merge_status_normal_status_older_than_the_exception_does_not_clear_it():
    # The exception's own email is dated *after* the incoming normal
    # status's email -- chronologically the exception is the more recent
    # event, so it must still be in force, regardless of which one this
    # function happens to see first.
    level, ndate, edate = pkg.merge_status(2, NOW, None, "exception", NOW + timedelta(hours=5))
    assert pkg.displayed_status(level, ndate, edate) == "exception"
    level, ndate, edate = pkg.merge_status(level, ndate, edate, "out_for_delivery", NOW + timedelta(hours=1))
    assert pkg.displayed_status(level, ndate, edate) == "exception"


def test_merge_status_is_order_independent_for_two_normal_statuses():
    forward = pkg.merge_status(*pkg.merge_status(-1, None, None, "shipped", NOW), "delivered", NOW + timedelta(hours=1))
    backward = pkg.merge_status(*pkg.merge_status(-1, None, None, "delivered", NOW + timedelta(hours=1)), "shipped", NOW)
    assert forward == backward
    assert forward[0] == 3


def test_merge_status_real_chronology_survives_every_processing_order():
    """shipped -> exception -> delivered, in that real chronological order.

    gmail.search() delivers newest-first, so a naive merge would apply
    these as delivered, exception, shipped -- landing on "exception" even
    though the shipment was, in reality, delivered *after* the exception
    was resolved. All 6 processing orders of the same three (status, date)
    events must land on the same place: "delivered".
    """
    t_shipped = NOW
    t_exception = NOW + timedelta(hours=5)
    t_delivered = NOW + timedelta(hours=10)
    events = [("shipped", t_shipped), ("exception", t_exception), ("delivered", t_delivered)]

    for order in itertools.permutations(events):
        level, ndate, edate = -1, None, None
        for status, when in order:
            level, ndate, edate = pkg.merge_status(level, ndate, edate, status, when)
        assert pkg.displayed_status(level, ndate, edate) == "delivered", order


def test_is_exception_compares_the_two_watermark_dates():
    assert pkg.is_exception(None, NOW) is True  # no normal status recorded yet
    assert pkg.is_exception(NOW, None) is False  # no exception ever seen
    assert pkg.is_exception(NOW, NOW - timedelta(hours=1)) is False  # exception is the older event
    assert pkg.is_exception(NOW, NOW + timedelta(hours=1)) is True  # exception is the newer event


# -- store: upsert / cursor / catch-up --------------------------------------------


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as s:
        pkg.ensure_schema(s.conn)
        yield s


def test_upsert_shipment_creates_and_reads_back(store):
    status = pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier="UPS", merchant="Widget Co", status="shipped",
        eta="2026-09-30", event_date=NOW,
    )
    assert status == "shipped"
    [row] = pkg.fetch_shipments(store.conn)
    assert row["tracking_number"] == "1Z999"
    assert row["carrier"] == "UPS"
    assert row["status"] == "shipped"
    assert row["eta"] == "2026-09-30"
    assert row["normal_status_date"] == NOW.isoformat()


def test_upsert_shipment_newest_first_processing_does_not_regress(store):
    # Simulates gmail.search()'s newest-first order: the "delivered" email
    # (newest, dated later) is upserted before the "shipped" email (older,
    # processed second in this run).
    pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier="UPS", merchant="Widget Co", status="delivered",
        eta=None, event_date=NOW,
    )
    final = pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier="UPS", merchant="Widget Co", status="shipped",
        eta=None, event_date=NOW - timedelta(days=2),
    )
    assert final == "delivered"


def test_upsert_shipment_fills_gaps_without_blanking_existing_values(store):
    pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier="UPS", merchant="Widget Co", status="ordered",
        eta="2026-09-30", event_date=NOW,
    )
    pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier=None, merchant=None, status="shipped",
        eta=None, event_date=NOW + timedelta(hours=1),
    )
    [row] = pkg.fetch_shipments(store.conn)
    assert row["carrier"] == "UPS"  # not blanked by the second, carrier-less email
    assert row["merchant"] == "Widget Co"
    assert row["eta"] == "2026-09-30"
    assert row["status"] == "shipped"


def test_upsert_shipment_does_not_touch_first_seen_on_update(store):
    pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier="UPS", merchant="Widget Co", status="ordered",
        eta=None, event_date=NOW,
    )
    [before] = pkg.fetch_shipments(store.conn)
    pkg.upsert_shipment(
        store.conn, tracking="1Z999", carrier="UPS", merchant="Widget Co", status="shipped",
        eta=None, event_date=NOW + timedelta(hours=1),
    )
    [after] = pkg.fetch_shipments(store.conn)
    assert after["first_seen"] == before["first_seen"]
    assert after["last_updated"] >= before["last_updated"]


def test_cursor_round_trips(store):
    from lmsa.gmail_cursor import Window

    assert pkg.load_cursor(store.conn) is None
    cursor = pkg._cursor(store.conn)
    cursor.started = NOW
    cursor.advance(Window(candidates=[], batch=[]), done=lambda msg_id: True)
    assert pkg.load_cursor(store.conn) == NOW


def test_cursor_survives_garbage_value(store):
    store.conn.execute(
        "INSERT INTO kv (ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
        (pkg.KV_NS, pkg.CURSOR_KEY, "not valid json {{", "now"),
    )
    store.conn.commit()
    assert pkg.load_cursor(store.conn) is None


def test_cursor_migrates_a_legacy_bare_date_value(store):
    # Older runs stored the cursor as a bare ISO date; it must still load
    # (as local midnight of that date) instead of raising or being dropped.
    store.conn.execute(
        "INSERT INTO kv (ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
        (pkg.KV_NS, pkg.CURSOR_KEY, json.dumps("2026-09-18"), "now"),
    )
    store.conn.commit()
    loaded = pkg.load_cursor(store.conn)
    assert loaded is not None
    assert loaded.tzinfo is not None
    assert loaded.date() == date(2026, 9, 18)


# -- query building -----------------------------------------------------------


# -- process_message (per-item isolation) ----------------------------------------


def test_process_message_creates_shipment(store):
    lm = FakeLMStudio(replies=[shipment_reply()])
    msg = make_message("1", "Your order has shipped", "body")
    result = pkg.process_message(lm, store.conn, msg)
    assert result["error"] is None
    assert result["skipped"] is False
    assert result["tracking_number"] == "1Z999AA10123456784"
    [row] = pkg.fetch_shipments(store.conn)
    assert row["status"] == "shipped"


def test_process_message_skips_non_shipment_email(store):
    lm = FakeLMStudio(replies=[non_shipment_reply()])
    msg = make_message("1", "Weekly newsletter", "body")
    result = pkg.process_message(lm, store.conn, msg)
    assert result["skipped"] is True
    assert result["error"] is None
    assert pkg.fetch_shipments(store.conn) == []


def test_process_message_rejects_bad_tracking_charset_without_crashing(store):
    lm = FakeLMStudio(replies=[shipment_reply(tracking="<script>bad</script>")])
    msg = make_message("1", "Your order has shipped", "body")
    result = pkg.process_message(lm, store.conn, msg)
    assert result["error"] is not None and "charset" in result["error"]
    assert pkg.fetch_shipments(store.conn) == []


def test_process_message_isolates_model_failure(store):
    class BoomLM(FakeLMStudio):
        def extract(self, *a, **k):
            raise RuntimeError("server down")

    msg = make_message("1", "Your order has shipped", "body")
    result = pkg.process_message(BoomLM(), store.conn, msg)
    assert result["error"] is not None and "server down" in result["error"]
    assert pkg.fetch_shipments(store.conn) == []


def test_process_message_rejects_out_of_enum_status_defensively(store):
    # Guards the case where extract() returns something that bypassed the
    # schema's enum constraint (e.g. a local model that ignores strict mode).
    lm = FakeLMStudio(replies=[shipment_reply(status="in_transit")])
    msg = make_message("1", "Your order has shipped", "body")
    result = pkg.process_message(lm, store.conn, msg)
    assert result["error"] is not None and "in_transit" in result["error"]
    assert pkg.fetch_shipments(store.conn) == []


def test_process_message_truncates_body_before_prompting(store):
    lm = FakeLMStudio(replies=[shipment_reply()])
    long_body = "x" * (pkg.BODY_LIMIT + 500)
    msg = make_message("1", "Your order has shipped", long_body, from_addr="merchant@shop.test")
    pkg.process_message(lm, store.conn, msg)
    # The prompt embeds the body after truncation to BODY_LIMIT chars, not
    # the full (untrusted, unbounded) body.
    assert lm.calls[0].count("x") == pkg.BODY_LIMIT


def test_process_message_uses_the_emails_own_date_for_the_merge(store):
    # A message with no parseable Date header still merges safely (falls
    # back to "now" rather than crashing on a None comparison).
    lm = FakeLMStudio(replies=[shipment_reply()])
    msg = make_message("1", "Your order has shipped", "body", date=None)
    result = pkg.process_message(lm, store.conn, msg)
    assert result["error"] is None
    [row] = pkg.fetch_shipments(store.conn)
    assert row["normal_status_date"] is not None


# -- build_report -----------------------------------------------------------------


def _shipment(
    tracking="1Z999", carrier="UPS", merchant="Widget Co", status="shipped", eta=None,
    last_updated=None, normal_status_date=None,
):
    return {
        "tracking_number": tracking,
        "carrier": carrier,
        "merchant": merchant,
        "status": status,
        "eta": eta,
        "first_seen": NOW.isoformat(),
        "last_updated": (last_updated or NOW).isoformat(),
        "normal_status_date": (normal_status_date or NOW).isoformat(),
    }


def test_build_report_tiles_counts_by_status(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    shipments = [
        _shipment(tracking="A", status="ordered"),
        _shipment(tracking="B", status="shipped"),
        _shipment(tracking="C", status="out_for_delivery"),
        _shipment(tracking="D", status="exception"),
        _shipment(tracking="E", status="delivered"),
    ]
    path = pkg.build_report(shipments, [], NOW)
    html = path.read_text(encoding="utf-8")
    for label in ("Ordered", "Shipped", "Out for delivery", "Exceptions", "Delivered (7d)"):
        assert label in html
    assert html.count('<div class="value">1</div>') == 5  # each of the 5 tiles has exactly one match


def test_build_report_delivered_collapses_to_a_count_not_a_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    shipments = [
        _shipment(tracking="RECENTDELIV", status="delivered", normal_status_date=NOW - timedelta(days=1)),
        _shipment(tracking="OLDDELIV", status="delivered", normal_status_date=NOW - timedelta(days=30)),
    ]
    path = pkg.build_report(shipments, [], NOW)
    html = path.read_text(encoding="utf-8")
    assert "RECENTDELIV" not in html  # delivered items never listed individually
    assert "OLDDELIV" not in html
    assert "Delivered (7d)" in html


def test_build_report_delivered_window_uses_email_date_not_processing_time(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    shipments = [
        # Just processed/upserted "now" (e.g. a first run catching up), but
        # the delivery email itself is 20 days old -- must NOT count.
        _shipment(
            tracking="OLDMAIL", status="delivered",
            last_updated=NOW, normal_status_date=NOW - timedelta(days=20),
        ),
        # Processed a long time ago (last_updated is stale) but the
        # delivery email is only 2 days old -- must count.
        _shipment(
            tracking="RECENTMAIL", status="delivered",
            last_updated=NOW - timedelta(days=20), normal_status_date=NOW - timedelta(days=2),
        ),
    ]
    path = pkg.build_report(shipments, [], NOW)
    html = path.read_text(encoding="utf-8")
    assert html.count('<div class="value">0</div>') == 4  # ordered/shipped/out_for_delivery/exception
    assert html.count('<div class="value">1</div>') == 1  # delivered(7d): only RECENTMAIL


def test_build_report_active_shipments_excludes_delivered(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    shipments = [
        _shipment(tracking="INTRANSIT", status="shipped"),
        _shipment(tracking="DONE", status="delivered"),
    ]
    path = pkg.build_report(shipments, [], NOW)
    html = path.read_text(encoding="utf-8")
    assert "INTRANSIT" in html
    assert "DONE" not in html


def test_build_report_known_carrier_gets_a_link_unknown_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    shipments = [
        _shipment(tracking="1Z999", carrier="UPS", status="shipped"),
        _shipment(tracking="XYZ123456", carrier="SomeRandomCourier", status="shipped"),
    ]
    path = pkg.build_report(shipments, [], NOW)
    html = path.read_text(encoding="utf-8")
    assert "ups.com/track" in html


def test_build_report_escapes_untrusted_merchant_text(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    shipments = [_shipment(tracking="1Z999", merchant='<img src=x onerror=alert(1)>', status="shipped")]
    path = pkg.build_report(shipments, [], NOW)
    html = path.read_text(encoding="utf-8")
    assert "<img" not in html
    assert "&lt;img" in html


def _pending_window(pending: int):
    from lmsa.gmail_cursor import Window
    from lmsa.sources.gmail import MessageMeta

    metas = [MessageMeta(msg_id=str(i), internal_date=None, uid=i) for i in range(pending + 1)]
    return Window(candidates=metas, batch=metas[:1])


def test_build_report_notes_pending():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        import os
        os.environ["LMSA_OUTPUT_DIR"] = d
        try:
            path = pkg.build_report([], [], NOW, window=_pending_window(5))
            html = path.read_text(encoding="utf-8")
            assert "5 more matching message(s) are pending" in html
        finally:
            del os.environ["LMSA_OUTPUT_DIR"]


def test_build_report_shows_extraction_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    results = [{"subject": "Weird email", "error": "ValueError: boom", "tracking_number": None, "skipped": False}]
    path = pkg.build_report([], results, NOW)
    html = path.read_text(encoding="utf-8")
    assert "boom" in html


# -- main() end to end -----------------------------------------------------------


def test_main_writes_report_and_tracks_a_shipment(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    fake_gmail = FakeGmail(messages=[make_message("1", "Your order has shipped", date=NOW)])
    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(pkg, "LMStudio", lambda: FakeLMStudio(replies=[shipment_reply()]))
    monkeypatch.setattr(pkg, "_now", lambda: NOW)

    pkg.main()

    reports = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert reports
    html = reports[-1].read_text(encoding="utf-8")
    assert "1Z999AA10123456784" in html

    with Store(tmp_path / "lmsa.sqlite") as store:
        assert pkg.load_cursor(store.conn) is not None


def test_main_first_run_has_no_cursor_then_second_run_uses_after_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    gmail1 = FakeGmail(messages=[])
    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: gmail1)
    monkeypatch.setattr(pkg, "LMStudio", lambda: FakeLMStudio(replies=[]))
    monkeypatch.setattr(pkg, "_now", lambda: NOW)
    pkg.main()
    # First run: the lookback window, as an epoch after: bound.
    first_after = int(gmail1.queries[0].split("after:")[1].split()[0])
    lookback = NOW - timedelta(days=pkg.FIRST_RUN_LOOKBACK_DAYS)
    assert abs(first_after - lookback.timestamp()) < 3600

    gmail2 = FakeGmail(messages=[])
    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: gmail2)
    pkg.main()
    assert "after:" in gmail2.queries[0]


def test_main_failed_run_does_not_advance_cursor(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    gmail1 = FakeGmail(messages=[make_message("1", "Your order has shipped", date=NOW)])
    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: gmail1)
    monkeypatch.setattr(pkg, "LMStudio", lambda: FakeLMStudio(replies=[shipment_reply()]))
    monkeypatch.setattr(pkg, "_now", lambda: NOW)
    pkg.main()
    with Store(tmp_path / "lmsa.sqlite") as store:
        cursor_after_success = pkg.load_cursor(store.conn)
    assert cursor_after_success is not None

    class BoomGmail:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def search_meta(self, query, folder=None):
            raise RuntimeError("IMAP connection reset")

    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: BoomGmail())
    with pytest.raises(RuntimeError):
        pkg.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        cursor_after_failure = pkg.load_cursor(store.conn)
    # A run that fails before reaching save_cursor() must leave the cursor
    # untouched -- otherwise the next run would skip whatever mail this
    # failed run never got to process.
    assert cursor_after_failure == cursor_after_success


def test_main_isolates_per_message_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    fake_gmail = FakeGmail(
        messages=[
            make_message("1", "Your order has shipped", date=NOW),
            make_message("2", "Another shipment", date=NOW),
        ]
    )
    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(
        pkg,
        "LMStudio",
        lambda: FakeLMStudio(replies=[shipment_reply(tracking="GOOD123456"), shipment_reply(tracking="<bad>")]),
    )
    monkeypatch.setattr(pkg, "_now", lambda: NOW)

    pkg.main()  # must not raise even though one message's tracking number is invalid

    with Store(tmp_path / "lmsa.sqlite") as store:
        rows = pkg.fetch_shipments(store.conn)
    assert [r["tracking_number"] for r in rows] == ["GOOD123456"]

    reports = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    html = reports[-1].read_text(encoding="utf-8")
    assert "charset" in html


def test_main_newest_first_emails_do_not_regress_a_shipments_status(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    # gmail.search() returns newest first: the "delivered" email (more
    # recent) comes before the "shipped" email (older) for the same
    # tracking number.
    fake_gmail = FakeGmail(
        messages=[
            make_message("1", "Your package was delivered", date=NOW),
            make_message("2", "Your package has shipped", date=NOW - timedelta(days=2)),
        ]
    )
    monkeypatch.setattr(pkg, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(
        pkg,
        "LMStudio",
        lambda: FakeLMStudio(
            replies=[
                shipment_reply(tracking="SAME123456", status="delivered"),
                shipment_reply(tracking="SAME123456", status="shipped"),
            ]
        ),
    )
    monkeypatch.setattr(pkg, "_now", lambda: NOW)

    pkg.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        [row] = pkg.fetch_shipments(store.conn)
    assert row["status"] == "delivered"


