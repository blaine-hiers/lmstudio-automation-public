import urllib.error
from datetime import datetime, timezone

import pytest

from lmsa.store import Store
from tasks import price_watch as pw

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)

PAGE_HTML = """
<html><head><style>.x{color:red}</style><script>track();</script></head>
<body><h1>Widget &amp; Co</h1><p>Price: $19.99</p><p>In stock</p></body></html>
"""


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
        if self._replies:
            return self._replies.pop(0)
        return {"price": 19.99, "currency": "$", "in_stock": True}


# -- watchlist parsing ---------------------------------------------------------


def test_load_watchlist_parses_items(tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text(
        """
        [[item]]
        name = "Widget"
        url = "https://example.com/widget"
        target_price = 9.99

        [[item]]
        url = "https://example.com/no-name"
        """,
        encoding="utf-8",
    )
    items = pw.load_watchlist(path)
    assert items[0] == pw.Item(name="Widget", url="https://example.com/widget", target_price=9.99)
    # missing `name` falls back to the url
    assert items[1] == pw.Item(name="https://example.com/no-name", url="https://example.com/no-name", target_price=None)


def test_load_watchlist_missing_file_is_empty(tmp_path):
    assert pw.load_watchlist(tmp_path / "nope.toml") == []


def test_load_watchlist_skips_entries_without_url(tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text('[[item]]\nname = "No URL"\n', encoding="utf-8")
    assert pw.load_watchlist(path) == []


def test_load_watchlist_coerces_numeric_string_target_price(tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text(
        '[[item]]\nname = "Widget"\nurl = "https://example.com/widget"\ntarget_price = "39.99"\n',
        encoding="utf-8",
    )
    items = pw.load_watchlist(path)
    assert items[0].target_price == pytest.approx(39.99)
    assert items[0].load_error is None


def test_load_watchlist_flags_unparseable_target_price_without_crashing(tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text(
        '[[item]]\nname = "Bad"\nurl = "https://example.com/bad"\ntarget_price = "not a number"\n\n'
        '[[item]]\nname = "Good"\nurl = "https://example.com/good"\ntarget_price = 10.0\n',
        encoding="utf-8",
    )
    items = pw.load_watchlist(path)
    assert items[0].load_error is not None and "not a number" in items[0].load_error
    assert items[0].target_price is None
    assert items[1].load_error is None and items[1].target_price == 10.0


def test_load_watchlist_rejects_boolean_target_price(tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text('[[item]]\nname = "Bad"\nurl = "https://example.com/bad"\ntarget_price = true\n', encoding="utf-8")
    items = pw.load_watchlist(path)
    assert items[0].load_error is not None
    assert items[0].target_price is None


def test_load_watchlist_flags_second_occurrence_of_a_duplicate_url(tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text(
        '[[item]]\nname = "First"\nurl = "https://example.com/dup"\n\n'
        '[[item]]\nname = "Second"\nurl = "https://example.com/dup"\n',
        encoding="utf-8",
    )
    items = pw.load_watchlist(path)
    assert items[0].load_error is None
    assert items[1].load_error is not None and "duplicate url" in items[1].load_error


# -- page text handling ---------------------------------------------------------


def test_strip_to_text_drops_script_style_and_tags():
    text = pw.strip_to_text(PAGE_HTML)
    assert "track()" not in text
    assert "color:red" not in text
    assert "<" not in text
    assert "Widget & Co" in text
    assert "$19.99" in text


def test_truncate_bounds_length():
    assert pw.truncate("a" * 10000, limit=100) == "a" * 100
    assert pw.truncate("short") == "short"


# -- fetch (network stubbed) ----------------------------------------------------


class _Headers:
    def get_content_charset(self):
        return "utf-8"


def test_fetch_page_uses_user_agent_and_returns_decoded_text(monkeypatch):
    captured = {}

    class Resp:
        headers = _Headers()

        def read(self):
            return "hello".encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.headers.get("User-agent")
        captured["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(pw.urllib.request, "urlopen", fake_urlopen)
    text = pw.fetch_page("https://example.com/item")
    assert text == "hello"
    assert captured["ua"] == pw.USER_AGENT
    assert captured["timeout"] == pw.FETCH_TIMEOUT


# -- per-item check / transitions -----------------------------------------------


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as s:
        pw.ensure_schema(s.conn)
        yield s


def _item(target=None):
    return pw.Item(name="Widget", url="https://example.com/widget", target_price=target)


def test_check_item_first_run_alerts_when_already_under_target(monkeypatch, store):
    # The owner just added this item specifically to be told -- there's no
    # earlier "not yet under target" run for this to be a transition from,
    # so the first-ever check alerts once instead of staying silent.
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    lm = FakeLMStudio(replies=[{"price": 19.99, "currency": "$", "in_stock": True}])

    result = pw.check_item(lm, store.conn, _item(target=25.0), NOW)

    assert result["price"] == 19.99
    assert result["under_target"] is True  # reflects current state regardless of history
    assert result["hit_target"] is True  # first check, already under target -> alert
    assert result["back_in_stock"] is False  # in stock, but no baseline yet -- unchanged for stock
    assert result["delta"] is None
    assert result["error"] is None


def test_check_item_first_run_above_target_neither_under_nor_hits(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    lm = FakeLMStudio(replies=[{"price": 30.0, "currency": "$", "in_stock": True}])

    result = pw.check_item(lm, store.conn, _item(target=25.0), NOW)

    assert result["under_target"] is False
    assert result["hit_target"] is False


def test_check_item_flags_target_hit_only_on_transition(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    item = _item(target=25.0)

    lm1 = FakeLMStudio(replies=[{"price": 30.0, "currency": "$", "in_stock": True}])
    r1 = pw.check_item(lm1, store.conn, item, NOW)
    assert r1["hit_target"] is False  # above target

    lm2 = FakeLMStudio(replies=[{"price": 19.99, "currency": "$", "in_stock": True}])
    r2 = pw.check_item(lm2, store.conn, item, NOW)
    assert r2["hit_target"] is True  # crossed below target this run
    assert r2["delta"] == pytest.approx(19.99 - 30.0)

    lm3 = FakeLMStudio(replies=[{"price": 18.0, "currency": "$", "in_stock": True}])
    r3 = pw.check_item(lm3, store.conn, item, NOW)
    assert r3["hit_target"] is False  # already under target last run -- not a new transition
    assert r3["under_target"] is True  # but current state still reflects it every run


def test_check_item_isolates_a_bad_watchlist_entry_without_affecting_others(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    good_lm = FakeLMStudio(replies=[{"price": 5.0, "currency": "$", "in_stock": True}])
    good = pw.check_item(good_lm, store.conn, _item(target=10.0), NOW)
    assert good["error"] is None and good["under_target"] is True

    bad_item = pw.Item(name="Bad", url="https://example.com/bad", load_error="invalid target_price 'nope': not a number")
    bad = pw.check_item(FakeLMStudio(), store.conn, bad_item, NOW)
    assert bad["error"] is not None and "not a number" in bad["error"]
    assert bad["price"] is None
    assert bad["under_target"] is False
    assert bad["hit_target"] is False
    # the bad entry never reached fetch/extract
    assert good_lm.calls  # sanity: the good item's model call did happen


def test_check_item_currency_change_blocks_delta_and_stale_hit_basis(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    item = _item(target=25.0)

    lm1 = FakeLMStudio(replies=[{"price": 30.0, "currency": "USD", "in_stock": True}])
    r1 = pw.check_item(lm1, store.conn, item, NOW)
    assert r1["hit_target"] is False  # above target

    # Same numeric price, but the model reads a different currency this
    # run -- 30.0 GBP vs 30.0 USD is not "no change", so no delta.
    lm2 = FakeLMStudio(replies=[{"price": 30.0, "currency": "GBP", "in_stock": True}])
    r2 = pw.check_item(lm2, store.conn, item, NOW)
    assert r2["delta"] is None
    assert r2["currency_changed"] is True

    # Now it drops under target in the new currency: since the prior
    # check's currency isn't comparable, there's no reliable "already hit"
    # basis, so this is treated like a first check and alerts.
    lm3 = FakeLMStudio(replies=[{"price": 19.99, "currency": "GBP", "in_stock": True}])
    r3 = pw.check_item(lm3, store.conn, item, NOW)
    assert r3["under_target"] is True
    assert r3["hit_target"] is True


def test_check_item_no_currency_change_when_either_side_unknown(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    item = _item()

    lm1 = FakeLMStudio(replies=[{"price": 10.0, "currency": None, "in_stock": True}])
    pw.check_item(lm1, store.conn, item, NOW)

    lm2 = FakeLMStudio(replies=[{"price": 12.0, "currency": "$", "in_stock": True}])
    r2 = pw.check_item(lm2, store.conn, item, NOW)
    assert r2["currency_changed"] is False
    assert r2["delta"] == pytest.approx(2.0)


def test_load_watchlist_duplicate_then_check_item_does_not_cross_contaminate(monkeypatch, store, tmp_path):
    path = tmp_path / "watchlist.toml"
    path.write_text(
        '[[item]]\nname = "First"\nurl = "https://example.com/widget"\ntarget_price = 10.0\n\n'
        '[[item]]\nname = "Second"\nurl = "https://example.com/widget"\ntarget_price = 999.0\n',
        encoding="utf-8",
    )
    first, second = pw.load_watchlist(path)

    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    r_first = pw.check_item(FakeLMStudio(replies=[{"price": 5.0, "currency": "$", "in_stock": True}]), store.conn, first, NOW)
    assert r_first["error"] is None
    assert r_first["under_target"] is True

    r_second = pw.check_item(FakeLMStudio(), store.conn, second, NOW)
    assert r_second["error"] is not None and "duplicate url" in r_second["error"]

    # last_check for the shared url must still reflect only the first
    # (legitimate) item's data -- the duplicate's failed check must not
    # have overwritten or blended it.
    previous = pw.last_check(store.conn, "https://example.com/widget")
    assert previous == {"price": 5.0, "currency": "$", "in_stock": True}


def test_check_item_flags_back_in_stock_only_on_transition(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    item = _item()

    lm1 = FakeLMStudio(replies=[{"price": 10.0, "currency": "$", "in_stock": False}])
    r1 = pw.check_item(lm1, store.conn, item, NOW)
    assert r1["back_in_stock"] is False

    lm2 = FakeLMStudio(replies=[{"price": 10.0, "currency": "$", "in_stock": True}])
    r2 = pw.check_item(lm2, store.conn, item, NOW)
    assert r2["back_in_stock"] is True

    lm3 = FakeLMStudio(replies=[{"price": 10.0, "currency": "$", "in_stock": True}])
    r3 = pw.check_item(lm3, store.conn, item, NOW)
    assert r3["back_in_stock"] is False  # already in stock last run


def test_check_item_isolates_fetch_failure(monkeypatch, store):
    def boom(url):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(pw, "fetch_page", boom)
    lm = FakeLMStudio()

    result = pw.check_item(lm, store.conn, _item(), NOW)

    assert result["error"] is not None and "no route to host" in result["error"]
    assert result["price"] is None
    assert result["currency_changed"] is False
    assert result["under_target"] is False
    assert result["hit_target"] is False
    assert result["back_in_stock"] is False
    # the failed check is recorded (so later runs see it), but not counted as a baseline
    assert pw.last_check(store.conn, "https://example.com/widget") is None


def test_price_history_only_counts_error_free_rows(monkeypatch, store):
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    item = _item()
    for price in (10.0, 12.0, 9.0):
        lm = FakeLMStudio(replies=[{"price": price, "currency": "$", "in_stock": True}])
        pw.check_item(lm, store.conn, item, NOW)
    # a failed check in between must not show up as a price point
    monkeypatch.setattr(pw, "fetch_page", lambda url: (_ for _ in ()).throw(OSError("down")))
    pw.check_item(FakeLMStudio(), store.conn, item, NOW)

    hist = pw.price_history(store.conn, item.url, "2000-01-01T00:00:00+00:00")
    assert hist == [10.0, 12.0, 9.0]


# -- main() end to end -----------------------------------------------------------


def test_main_writes_report_and_alerts_only_on_transition(monkeypatch, tmp_path):
    watchlist = tmp_path / "watchlist.toml"
    watchlist.write_text(
        '[[item]]\nname = "Widget"\nurl = "https://example.com/widget"\ntarget_price = 25.0\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(pw, "WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(pw, "fetch_page", lambda url: PAGE_HTML)
    monkeypatch.setattr(pw, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    notified = []
    monkeypatch.setattr(pw, "notify", lambda title, body, url=None, level="info": notified.append((title, level)))

    replies = iter(
        [
            {"price": 30.0, "currency": "$", "in_stock": True},  # run 1: above target
            {"price": 19.99, "currency": "$", "in_stock": True},  # run 2: crosses target -> alert
        ]
    )
    monkeypatch.setattr(pw, "LMStudio", lambda: FakeLMStudio(replies=[next(replies)]))

    pw.main()
    assert notified == []  # first run: price is above target, so nothing to alert on

    reports = sorted((tmp_path / "output").rglob("price-stock-watcher-*.html"))
    run1_html = reports[-1].read_text(encoding="utf-8")
    assert "At or under target" not in run1_html  # not under target yet

    pw.main()
    assert len(notified) == 1
    assert "target price hit" in notified[0][0]
    assert notified[0][1] == "urgent"

    reports = sorted((tmp_path / "output").rglob("price-stock-watcher-*.html"))
    assert reports
    html = reports[-1].read_text(encoding="utf-8")
    assert "Price &amp; stock watcher" in html or "Price & stock watcher" in html
    assert "Widget" in html
    assert "target price hit" in html
    assert "At or under target" in html  # persistent current-state callout

    # A third run at the same (still under-target) price must not alert
    # again, but the persistent "at/under target" state must still show.
    monkeypatch.setattr(pw, "LMStudio", lambda: FakeLMStudio(replies=[{"price": 19.99, "currency": "$", "in_stock": True}]))
    pw.main()
    assert len(notified) == 1  # no new alert
    reports = sorted((tmp_path / "output").rglob("price-stock-watcher-*.html"))
    run3_html = reports[-1].read_text(encoding="utf-8")
    assert "At or under target" in run3_html
    assert "New this run" not in run3_html


def test_main_isolates_per_item_failure(monkeypatch, tmp_path):
    watchlist = tmp_path / "watchlist.toml"
    watchlist.write_text(
        '[[item]]\nname = "Good"\nurl = "https://example.com/good"\n\n'
        '[[item]]\nname = "Bad"\nurl = "https://example.com/bad"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(pw, "WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(pw, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr(pw, "notify", lambda *a, **k: None)
    monkeypatch.setattr(pw, "LMStudio", lambda: FakeLMStudio(replies=[{"price": 5.0, "currency": "$", "in_stock": True}]))

    def fetch(url):
        if "bad" in url:
            raise OSError("connection reset")
        return PAGE_HTML

    monkeypatch.setattr(pw, "fetch_page", fetch)

    pw.main()  # must not raise even though one item's fetch fails

    reports = sorted((tmp_path / "output").rglob("price-stock-watcher-*.html"))
    html = reports[-1].read_text(encoding="utf-8")
    assert "Good" in html and "Bad" in html
    assert "connection reset" in html
