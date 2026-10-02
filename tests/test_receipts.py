"""tasks.receipts: extraction, decimal-safe amounts, idempotent upserts,
per-item isolation, and cadence-based subscription/price-increase detection.

All fixtures use invented merchants and addresses (example.com); no real
mailbox or LM Studio server is touched.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import tasks.receipts as receipts
from lmsa.sources.gmail import Message
from lmsa.store import Store
from gmail_fakes import MetaSearchMixin


def _days_ago(n: int) -> str:
    """An ISO date string `n` days before now -- keeps fixtures inside the
    task's default 30-day lookback window regardless of when tests run."""
    return (datetime.now(timezone.utc) - timedelta(days=n)).date().isoformat()


def _msg(msg_id: str, subject: str = "Your receipt", day: str = "2026-01-15", **overrides) -> Message:
    fields = dict(
        msg_id=msg_id,
        thread_id=f"t-{msg_id}",
        labels=[],
        flags=[],
        from_name="Billing",
        from_addr="billing@example.com",
        to=["me@example.com"],
        reply_to="",
        subject=subject,
        date=datetime.fromisoformat(f"{day}T09:00:00+00:00"),
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text="Thanks for your purchase.",
        attachments=[],
        web_url="",
    )
    fields.update(overrides)
    return Message(**fields)


class _FakeGmail(MetaSearchMixin):
    """Filters by the `after:`/`before:` terms in the query (like real Gmail
    search) and returns newest-first, capped at `limit` -- close enough to
    the real thing to exercise the truncation/catch-up pagination in main().

    Real Gmail search accepts whole-second Unix epoch timestamps for
    `after:`/`before:`; this fake honours the same granularity so it
    actually exercises the same-day, different time-of-day cases the
    epoch-second fix is for.
    """

    def __init__(self, messages, cfg=None):
        self._messages = messages
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        self.queries.append(query)
        since_epoch = int(query.split("after:")[1].split()[0])
        before_epoch = None
        if "before:" in query:
            before_epoch = int(query.split("before:")[1].split()[0])
        matches = [
            m for m in self._messages
            if m.date.timestamp() >= since_epoch
            and (before_epoch is None or m.date.timestamp() < before_epoch)
        ]
        matches.sort(key=lambda m: m.date, reverse=True)
        return matches[:limit]


class _FakeLMStudio:
    """Stands in for `LMStudio`: `extract()` returns canned results by msg_id."""

    def __init__(self, results: dict[str, dict | Exception]):
        self._results = results
        self.cfg = type("Cfg", (), {"output_dir": None, "chat_model": "fake"})()

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        # msg_id isn't in the prompt; tests key results by subject instead
        for subject, result in self._results.items():
            if subject in prompt:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"no fake extract() result configured for prompt: {prompt[:80]!r}")


RECEIPT_RESULT = {
    "is_receipt": True,
    "merchant": "Acme Cloud",
    "amount": "19.99",
    "currency": "usd",
    "date": "2026-01-15",
    "category": "software",
    "recurring": True,
}

NOT_A_RECEIPT = {
    "is_receipt": False,
    "merchant": "",
    "amount": "0",
    "currency": "",
    "date": "",
    "category": "",
    "recurring": False,
}


def test_to_minor_units_is_decimal_safe():
    # 19.99 * 100 as a float is 1998.9999999999998 -- Decimal must not do that.
    assert receipts._to_minor_units("19.99") == 1999
    assert receipts._to_minor_units("$1,234.56") == 123456
    assert receipts._to_minor_units("10.10") == 1010


def test_extract_receipt_returns_none_for_non_receipt():
    lm = _FakeLMStudio({"Not a receipt": NOT_A_RECEIPT})
    msg = _msg("m1", subject="Not a receipt")
    assert receipts.extract_receipt(lm, msg, system="sys") is None


def test_extract_receipt_builds_row_with_minor_units_and_upper_currency():
    lm = _FakeLMStudio({"Your receipt": RECEIPT_RESULT})
    msg = _msg("m1")
    result = receipts.extract_receipt(lm, msg, system="sys")
    assert isinstance(result, receipts.ExtractedReceipt)
    assert result.amount_minor == 1999
    assert result.currency == "USD"
    assert result.merchant == "Acme Cloud"
    assert result.purchase_date == "2026-01-15"


def test_extract_receipt_records_error_without_raising():
    lm = _FakeLMStudio({"Your receipt": RuntimeError("model server down")})
    msg = _msg("m1")
    result = receipts.extract_receipt(lm, msg, system="sys")
    assert isinstance(result, receipts.ItemError)
    assert "model server down" in result.error


def test_extract_receipt_falls_back_to_message_date_on_bad_model_date():
    bad_date_result = dict(RECEIPT_RESULT, date="not a date")
    lm = _FakeLMStudio({"Your receipt": bad_date_result})
    msg = _msg("m1", day="2026-02-01")
    result = receipts.extract_receipt(lm, msg, system="sys")
    assert result.purchase_date == "2026-02-01"


def test_upsert_is_idempotent_on_rerun(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    row = receipts.ExtractedReceipt(
        msg_id="m1", merchant="Acme Cloud", amount_minor=1999, currency="USD",
        purchase_date="2026-01-15", category="software", recurring_hint=True, subject="Your receipt",
    )
    with Store(db) as store:
        receipts._ensure_schema(store)
        receipts.upsert_receipt(store, row)
        receipts.upsert_receipt(store, row)  # same msg_id again, same data
        store.conn.commit()
        rows = store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()
    assert rows[0] == 1


def test_upsert_never_overwrites_an_existing_row(tmp_path):
    # First writer owns the row in the shared `receipts` table (see
    # upsert_receipt): a second write for the same msg_id is a no-op.
    db = tmp_path / "lmsa.sqlite"
    original = receipts.ExtractedReceipt(
        msg_id="m1", merchant="Acme Cloud", amount_minor=1999, currency="USD",
        purchase_date="2026-01-15", category="software", recurring_hint=True, subject="Your receipt",
    )
    updated = receipts.ExtractedReceipt(
        msg_id="m1", merchant="Acme Cloud", amount_minor=2499, currency="USD",
        purchase_date="2026-01-15", category="software", recurring_hint=True, subject="Your receipt",
    )
    with Store(db) as store:
        receipts._ensure_schema(store)
        assert receipts.upsert_receipt(store, original) is True
        assert receipts.upsert_receipt(store, updated) is False
        store.conn.commit()
        amount = store.conn.execute("SELECT amount_minor FROM receipts WHERE msg_id='m1'").fetchone()[0]
    assert amount == 1999


def test_main_never_overwrites_a_row_the_attachment_reader_wrote_first(tmp_path, monkeypatch):
    # Order attachments -> receipts: the attachment reader's (better) vision
    # read of the invoice PDF must survive a later body read of the same email.
    db = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db))
    with Store(db) as store:
        receipts._ensure_schema(store)
        receipts.upsert_receipt(store, receipts.ExtractedReceipt(
            msg_id="m1", merchant="Vision Merchant", amount_minor=4120, currency="EUR",
            purchase_date=_days_ago(2), category="infra", recurring_hint=False, subject="Invoice attached",
        ))
        store.conn.commit()
    fake_gmail = _FakeGmail([_msg("m1", subject="Invoice attached", day=_days_ago(2))])
    fake_lm = _FakeLMStudio({"Invoice attached": {**RECEIPT_RESULT, "amount": "0.00", "merchant": "Body Guess"}})
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(receipts, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(receipts, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["receipts"])

    receipts.main()

    with Store(db) as store:
        row = store.conn.execute("SELECT merchant, amount_minor FROM receipts WHERE msg_id='m1'").fetchone()
    assert row == ("Vision Merchant", 4120)
    report = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"][0]
    assert "row already written by another task" in report.read_text(encoding="utf-8")


def _rec(merchant, currency, day, amount_minor, msg_id=None):
    return receipts.ReceiptRecord(
        msg_id=msg_id or f"{merchant}-{day}", merchant=merchant, amount_minor=amount_minor,
        currency=currency, purchase_date=day, category="software", subject="Your receipt",
    )


def test_detect_subscriptions_ignores_a_single_charge():
    records = [_rec("Acme Cloud", "USD", "2026-01-15", 1999)]
    groups, keys = receipts.detect_subscriptions(records, known_keys=set())
    assert groups == []
    assert keys == set()


def test_detect_subscriptions_flags_monthly_cadence_as_new():
    records = [
        _rec("Acme Cloud", "USD", "2026-01-15", 1999),
        _rec("Acme Cloud", "USD", "2026-02-14", 1999),
    ]
    groups, keys = receipts.detect_subscriptions(records, known_keys=set())
    assert len(groups) == 1
    group = groups[0]
    assert group.merchant == "Acme Cloud"
    assert group.cadence == "monthly"
    assert group.is_new is True
    assert group.increase_minor is None
    assert keys == {"Acme Cloud|USD"}


def test_detect_subscriptions_not_new_once_known():
    records = [
        _rec("Acme Cloud", "USD", "2026-01-15", 1999),
        _rec("Acme Cloud", "USD", "2026-02-14", 1999),
    ]
    groups, _keys = receipts.detect_subscriptions(records, known_keys={"Acme Cloud|USD"})
    assert groups[0].is_new is False


def test_detect_subscriptions_flags_price_increase():
    records = [
        _rec("Acme Cloud", "USD", "2026-01-15", 1999),
        _rec("Acme Cloud", "USD", "2026-02-14", 2499),
    ]
    groups, _keys = receipts.detect_subscriptions(records, known_keys={"Acme Cloud|USD"})
    assert groups[0].increase_minor == 500


def test_detect_subscriptions_keeps_currencies_separate():
    # Same merchant billed in two currencies must not be merged into one group.
    records = [
        _rec("Acme Cloud", "USD", "2026-01-15", 1999, msg_id="a"),
        _rec("Acme Cloud", "USD", "2026-02-14", 1999, msg_id="b"),
        _rec("Acme Cloud", "EUR", "2026-01-15", 1799, msg_id="c"),
    ]
    groups, _keys = receipts.detect_subscriptions(records, known_keys=set())
    assert len(groups) == 1
    assert groups[0].currency == "USD"


def test_main_end_to_end_is_idempotent_across_two_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    messages = [_msg("m1", subject="Your receipt", day=_days_ago(5)), _msg("m2", subject="Not a receipt", day=_days_ago(4))]
    fake_gmail = _FakeGmail(messages)
    monkeypatch.setattr(receipts, "Gmail", lambda cfg=None: fake_gmail)

    fake_lm = _FakeLMStudio({"Your receipt": RECEIPT_RESULT, "Not a receipt": NOT_A_RECEIPT})
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(receipts, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["receipts"])

    receipts.main()
    receipts.main()  # re-run: cursor has moved on, nothing new to fetch, still idempotent

    with Store(tmp_path / "lmsa.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        merchant = store.conn.execute("SELECT merchant FROM receipts").fetchone()[0]
    assert count == 1  # the one real receipt, not duplicated
    assert merchant == "Acme Cloud"

    written = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert written


def test_build_report_keeps_currencies_separate_in_tiles_and_bars():
    this_run = [
        receipts.ExtractedReceipt(
            msg_id="a", merchant="Acme Cloud", amount_minor=1999, currency="USD",
            purchase_date="2026-01-15", category="software", recurring_hint=True, subject="s",
        ),
        receipts.ExtractedReceipt(
            msg_id="b", merchant="Cafe Europa", amount_minor=500, currency="EUR",
            purchase_date="2026-01-16", category="food", recurring_hint=False, subject="s2",
        ),
    ]
    r = receipts.build_report(this_run, errors=[], subscriptions=[])
    html = r.render()
    assert "19.99 USD" in html
    assert "5.00 EUR" in html
    # Bars are split into one section per currency, not one section that
    # sums raw amounts across currencies.
    assert "Spend by merchant (USD)" in html
    assert "Spend by merchant (EUR)" in html
    assert "Spend by category (USD)" in html
    assert "Spend by category (EUR)" in html


def _pending_window(pending: int):
    from lmsa.gmail_cursor import Window
    from lmsa.sources.gmail import MessageMeta

    metas = [MessageMeta(msg_id=str(i), internal_date=None, uid=i) for i in range(pending + 1)]
    return Window(candidates=metas, batch=metas[:1])


def test_build_report_says_how_much_is_pending():
    r = receipts.build_report([], errors=[], subscriptions=[], window=_pending_window(4))
    assert "4 more matching message(s) are pending" in r.render()


def test_amount_format_rejects_scientific_notation_and_ambiguous_separators():
    for bad in ("1.5e3", "1e3", "2E2", "1.234", "not a number", ""):
        with pytest.raises(receipts.AmountFormatError):
            receipts._to_minor_units(bad)


def test_amount_format_accepts_explicit_comma_and_dot_variants():
    assert receipts._to_minor_units("1,299.00") == 129900  # comma thousands, dot decimal
    assert receipts._to_minor_units("1.299,00") == 129900  # dot thousands, comma decimal
    assert receipts._to_minor_units("1299,00") == 129900   # plain comma decimal
    assert receipts._to_minor_units("42.99") == 4299        # plain dot decimal
    assert receipts._to_minor_units("-9.99") == -999         # signed


def test_since_migrates_a_legacy_bare_date_cursor_without_crashing(tmp_path):
    # Before the epoch-second fix, `last_run` was sometimes stored as a
    # bare ISO date. `_since` must still load it (as local midnight of
    # that date) instead of raising or silently resetting the window.
    with Store(tmp_path / "lmsa.sqlite") as store:
        receipts._ensure_schema(store)
        receipts._kv_set(store, "last_run", "2026-01-15")
        store.conn.commit()
        since = receipts._since(store)
    assert since.tzinfo is not None
    assert since.date() == date(2026, 1, 15)


def test_known_subscriptions_persist_across_an_irregular_cycle(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        receipts._ensure_schema(store)

        # Run 1: first charge only -- no cadence yet, nothing known.
        recs = [_rec("Acme Cloud", "USD", "2026-01-15", 1999, msg_id="c1")]
        known = set(receipts._kv_get(store, "known_subscriptions") or [])
        groups, current = receipts.detect_subscriptions(recs, known)
        receipts._kv_set(store, "known_subscriptions", sorted(known | current))
        store.conn.commit()
        assert groups == []

        # Run 2: second charge ~30 days later -- monthly cadence, flagged new.
        recs.append(_rec("Acme Cloud", "USD", "2026-02-14", 1999, msg_id="c2"))
        known = set(receipts._kv_get(store, "known_subscriptions") or [])
        groups, current = receipts.detect_subscriptions(recs, known)
        receipts._kv_set(store, "known_subscriptions", sorted(known | current))
        store.conn.commit()
        assert groups[0].is_new is True

        # Run 3: an irregular gap pushes the median outside monthly/yearly
        # tolerance, so this run's `groups` comes back empty for this
        # merchant -- but it must stay "known" via the union so a later
        # regular charge doesn't re-fire it as "new".
        recs.append(_rec("Acme Cloud", "USD", "2026-09-01", 1999, msg_id="c3"))
        known = set(receipts._kv_get(store, "known_subscriptions") or [])
        groups, current = receipts.detect_subscriptions(recs, known)
        receipts._kv_set(store, "known_subscriptions", sorted(known | current))
        store.conn.commit()
        assert groups == []
        assert set(receipts._kv_get(store, "known_subscriptions")) == {"Acme Cloud|USD"}

        # Run 4: billing resumes monthly -- must not re-fire as "new".
        recs.append(_rec("Acme Cloud", "USD", "2026-10-01", 1999, msg_id="c4"))
        known = set(receipts._kv_get(store, "known_subscriptions") or [])
        groups, _current = receipts.detect_subscriptions(recs, known)
        assert groups[0].is_new is False
