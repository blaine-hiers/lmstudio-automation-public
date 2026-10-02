import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lmsa import Marker
from lmsa.store import Store
from lmsa.web import FetchError, Response
from tasks import osint_movements as om

FIXTURES = Path(__file__).parent / "fixtures" / "osint"
FEED_XML = (FIXTURES / "usni_feed.xml").read_bytes()
ARTICLE_HTML = (FIXTURES / "usni_article.html").read_bytes()
LATEST_ARTICLE_URL = "https://news.usni.org/2026/09/22/fleet-tracker-sept-22-2026"

NOW = datetime(2026, 9, 25, 6, 30, 0, tzinfo=timezone.utc)


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network."""

    def __init__(self, entries=None, chat_reply="A quiet day overall.", raise_on_extract=False):
        self.entries = [] if entries is None else entries
        self.chat_reply = chat_reply
        self.raise_on_extract = raise_on_extract
        self.extract_calls: list[str] = []
        self.chat_calls: list[str] = []

    def ensure_server(self) -> None:
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.extract_calls.append(prompt)
        assert system is not None and "untrusted" in system
        if self.raise_on_extract:
            raise RuntimeError("model unavailable")
        return {"entries": self.entries}

    def chat(self, prompt, *, system=None, max_tokens=None):
        self.chat_calls.append(prompt)
        assert system is not None
        return self.chat_reply


class FakeFetcher:
    """Stands in for lmsa.web.Fetcher: no live server, no network, no robots/
    cache/rate-limit plumbing -- just `.get(url) -> Response`, which is all
    `lmsa.feeds.fetch_feed` and this task's own `.get(link).text` call need."""

    def __init__(self, pages: dict[str, bytes] | None = None, raise_on: dict[str, Exception] | None = None):
        self.pages = pages or {}
        self.raise_on = raise_on or {}
        self.calls: list[str] = []

    def get(self, url: str) -> Response:
        self.calls.append(url)
        if url in self.raise_on:
            raise self.raise_on[url]
        if url not in self.pages:
            raise AssertionError(f"unexpected fetch {url}")
        return Response(200, url, {"content-type": "text/html; charset=utf-8"}, self.pages[url])


def make_fetcher(feed_xml: bytes = FEED_XML, article_html: bytes = ARTICLE_HTML, **kw) -> FakeFetcher:
    return FakeFetcher(pages={om.USNI_FEED_URL: feed_xml, LATEST_ARTICLE_URL: article_html}, **kw)


# -- config: type classes + regions --------------------------------------------


def test_load_config_reads_the_real_repo_file():
    type_class, regions = om.load_config(om.CONFIG_PATH)
    assert type_class["KC135"] == "tanker"
    assert type_class["C17"] == "transport"
    assert type_class["F35"] == "fighter"
    assert any(r.name == "Europe" for r in regions)


def test_load_config_missing_file_is_empty(tmp_path):
    type_class, regions = om.load_config(tmp_path / "nope.toml")
    assert type_class == {} and regions == []


def test_classify_type_unknown_is_unclassified():
    type_class = {"C17": "transport"}
    assert om.classify_type("C17", type_class) == "transport"
    assert om.classify_type("c17", type_class) == "transport"  # case-insensitive
    assert om.classify_type(None, type_class) == "unclassified"
    assert om.classify_type("CESSNA172", type_class) == "unclassified"


def test_region_for_matches_first_box_else_other():
    regions = [om.Region("Europe", 34, 72, -25, 45), om.Region("Middle East", 10, 42, 25, 63)]
    assert om.region_for(51, -1, regions) == "Europe"
    assert om.region_for(35, 40, regions) == "Europe"  # first match wins, even if Middle East also fits
    assert om.region_for(0, 0, regions) == "Other"
    assert om.region_for(None, None, regions) == "Other"


# -- daily aggregation + anomaly detection --------------------------------------


def _insert_sample(conn, sampled_at, hexid, actype, lat_bucket, lon_bucket):
    conn.execute(
        "INSERT INTO osint_air_samples (sampled_at, hex, actype, callsign, lat_bucket, lon_bucket, alt_baro) "
        "VALUES (?, ?, ?, 'X', ?, ?, 10000)",
        (sampled_at, hexid, actype, lat_bucket, lon_bucket),
    )


def test_positions_by_day_dedups_repeated_samples_of_the_same_aircraft(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        today = NOW.strftime("%Y-%m-%dT")
        # Same aircraft, same position, sampled three times this "day" (three dispatcher ticks).
        _insert_sample(store.conn, today + "01:00:00+00:00", "abc123", "C17", 51, -1)
        _insert_sample(store.conn, today + "02:00:00+00:00", "abc123", "C17", 51, -1)
        _insert_sample(store.conn, today + "03:00:00+00:00", "abc123", "C17", 51, -1)
        store.conn.commit()
        positions = om.positions_by_day(store.conn, NOW)
        today_key = om._day(NOW.isoformat())
        assert positions[today_key] == [("abc123", "C17", 51, -1)]


def test_class_region_counts_counts_distinct_aircraft_per_class_and_region():
    type_class = {"C17": "transport"}
    regions = [om.Region("Europe", 34, 72, -25, 45)]
    today = "2026-09-25"
    positions = {today: [("abc123", "C17", 51, -1)]}
    counts = om.class_region_counts_by_day(positions, type_class, regions)
    assert counts[today][("transport", "Europe")] == 1


def test_double_region_aircraft_counts_once_in_class_and_grand_total_not_twice():
    """The bug: one KC135 seen in North America and Europe the same day was
    counted twice toward the tanker/grand total (once per region) because
    the old code summed the (class, region) counter across regions. The
    fix: class/grand totals come from `class_totals_by_day`, which dedups
    by hex alone -- one aircraft, one class, counted once."""
    type_class = {"KC135": "tanker"}
    regions = [om.Region("North America", 5, 72, -170, -50), om.Region("Europe", 34, 72, -25, 45)]
    today = om._day(NOW.isoformat())
    positions = {today: [
        ("kc1", "KC135", 40, -100),  # North America
        ("kc1", "KC135", 51, -1),    # Europe -- same hex, same day
    ]}
    region_counts = om.class_region_counts_by_day(positions, type_class, regions)
    totals = om.class_totals_by_day(positions, type_class)

    today_counts, class_totals, region_totals, anomalies = om._aggregate(NOW, region_counts, totals)

    assert sum(class_totals.values()) == 1  # grand total: one aircraft, not two
    assert class_totals["tanker"] == 1
    assert region_totals["North America"] == 1  # each region still sees it once
    assert region_totals["Europe"] == 1


def test_detect_anomalies_flags_a_spike_above_baseline():
    # Some day-to-day noise (1-3) so the baseline has real variance, not a
    # constant that would make any jump a divide-by-zero.
    base_days = {
        f"2026-08-{d:02d}": Counter({("isr", "Europe"): 1 + (d % 3)}) for d in range(1, 26)
    }
    today = "2026-09-25"
    daily = dict(base_days)
    daily[today] = Counter({("isr", "Europe"): 12})  # far above the noisy baseline of ~2

    anomalies = om.detect_anomalies(daily, today)
    assert len(anomalies) == 1
    assert anomalies[0]["class"] == "isr"
    assert anomalies[0]["region"] == "Europe"
    assert anomalies[0]["today"] == 12


def test_detect_anomalies_needs_minimum_baseline_history():
    daily = {"2026-09-24": Counter({("isr", "Europe"): 1}), "2026-09-25": Counter({("isr", "Europe"): 9})}
    assert om.detect_anomalies(daily, "2026-09-25") == []  # only 1 baseline day < MIN_BASELINE_DAYS


def test_detect_anomalies_zero_variance_baseline_never_flags():
    daily = {f"2026-08-{d:02d}": Counter() for d in range(1, 10)}  # every baseline day is 0
    daily["2026-09-25"] = Counter({("tanker", "Other"): 1})
    assert om.detect_anomalies(daily, "2026-09-25") == []  # stdev 0 -> skipped, not a div-by-zero


def test_detect_anomalies_ordinary_day_is_quiet():
    daily = {f"2026-08-{d:02d}": Counter({("fighter", "Middle East"): 3}) for d in range(1, 26)}
    daily["2026-09-25"] = Counter({("fighter", "Middle East"): 3})
    assert om.detect_anomalies(daily, "2026-09-25") == []


# -- density markers -------------------------------------------------------------


def test_density_markers_sizes_by_count_and_caps():
    positions = [("h1", "C17", 51, -1), ("h2", "F16", 51, -1), ("h3", "KC135", 35, 40)]
    markers, truncated = om.density_markers(positions, cap=10)
    assert not truncated
    assert len(markers) == 2  # two occupied cells
    biggest = max(markers, key=lambda m: m.size)
    assert biggest.lat == 51.5 and biggest.lon == -0.5


def test_density_markers_truncates_and_reports_it():
    positions = [(f"h{i}", "C17", i, i) for i in range(5)]  # 5 distinct cells
    markers, truncated = om.density_markers(positions, cap=2)
    assert len(markers) == 2
    assert truncated is True


# -- USNI feed parsing (via lmsa.feeds) ------------------------------------------


def test_latest_feed_entry_picks_the_first_usable_item():
    fetcher = make_fetcher()
    item = om.latest_feed_entry(fetcher)
    assert item.url == LATEST_ARTICLE_URL
    assert "Sept. 22, 2026" in item.title
    assert item.published.year == 2026 and item.published.month == 9 and item.published.day == 22
    assert fetcher.calls == [om.USNI_FEED_URL]


def test_latest_feed_entry_no_entries_is_none():
    fetcher = FakeFetcher(pages={om.USNI_FEED_URL: b"<rss><channel></channel></rss>"})
    assert om.latest_feed_entry(fetcher) is None


# -- sync_usni --------------------------------------------------------------------


def test_sync_usni_not_due_yet_is_skipped(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        om._record_usni_check(store.conn, NOW, article_url=LATEST_ARTICLE_URL)
        lm = FakeLMStudio()
        fetcher = make_fetcher()
        result = om.sync_usni(store.conn, lm, NOW + timedelta(hours=1), fetcher=fetcher)
        assert result == {"checked": False}
        assert lm.extract_calls == []
        assert fetcher.calls == []


def test_sync_usni_new_article_extracts_and_stores(tmp_path):
    entries = [
        {"region": "Mediterranean Sea", "group": "Gerald R. Ford CSG", "note": "underway"},
        {"region": "South China Sea", "group": "Wasp ARG", "note": "routine operations"},
    ]
    with Store(tmp_path / "lmsa.sqlite") as store:
        lm = FakeLMStudio(entries=entries)
        result = om.sync_usni(store.conn, lm, NOW, fetcher=make_fetcher())
        assert result["checked"] is True
        assert result["new_article"] is True
        assert result["article_url"] == LATEST_ARTICLE_URL
        assert result["positions_stored"] == 2
        assert lm.extract_calls  # article text was sent to the model

        rows, url, title = om.latest_ship_positions(store.conn)
        assert url == LATEST_ARTICLE_URL
        assert len(rows) == 2
        assert ("Mediterranean Sea", "Gerald R. Ford CSG", "underway") in rows


def test_sync_usni_same_article_again_is_not_reextracted(tmp_path):
    entries = [{"region": "Mediterranean Sea", "group": "Gerald R. Ford CSG", "note": "underway"}]
    with Store(tmp_path / "lmsa.sqlite") as store:
        lm = FakeLMStudio(entries=entries)
        om.sync_usni(store.conn, lm, NOW, fetcher=make_fetcher())
        later = NOW + timedelta(days=om.USNI_MIN_INTERVAL_DAYS + 1)
        result = om.sync_usni(store.conn, lm, later, fetcher=make_fetcher())
        assert result["new_article"] is False
        assert len(lm.extract_calls) == 1  # not called a second time for the same article


def test_sync_usni_zero_entry_article_is_not_reextracted_on_next_due_check(tmp_path):
    """Regression: an article that extracts to zero *valid* entries never
    gets a row in osint_ship_positions, so checking "already ingested"
    against that table (instead of the kv checkpoint) made it look
    unprocessed forever -- re-extracted on every future due-check."""
    with Store(tmp_path / "lmsa.sqlite") as store:
        lm = FakeLMStudio(entries=[])  # nothing usable in the article
        first = om.sync_usni(store.conn, lm, NOW, fetcher=make_fetcher())
        assert first["new_article"] is True
        assert first["positions_stored"] == 0
        rows, url, _title = om.latest_ship_positions(store.conn)
        assert rows == [] and url is None  # nothing landed in the positions table

        later = NOW + timedelta(days=om.USNI_MIN_INTERVAL_DAYS + 1)
        second = om.sync_usni(store.conn, lm, later, fetcher=make_fetcher())
        assert second["new_article"] is False  # correctly recognized as already checked
        assert len(lm.extract_calls) == 1  # extract() never called a second time


def test_sync_usni_extract_failure_leaves_it_due_for_retry(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        lm = FakeLMStudio(raise_on_extract=True)
        result = om.sync_usni(store.conn, lm, NOW, fetcher=make_fetcher())
        assert "error" in result
        assert om.usni_due(store.conn, NOW + timedelta(minutes=1)) is True  # not marked checked


def test_sync_usni_feed_fetch_failure_is_isolated(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        lm = FakeLMStudio()
        fetcher = FakeFetcher(raise_on={om.USNI_FEED_URL: FetchError("dns failure")})
        result = om.sync_usni(store.conn, lm, NOW, fetcher=fetcher)
        assert result["checked"] is True
        assert "error" in result
        assert lm.extract_calls == []


# -- build_report smoke test: aggregate only, never a per-aircraft list --------


def test_build_report_never_shows_raw_aircraft_data_and_links_the_source(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    class_totals = Counter({"transport": 2, "tanker": 1})
    region_totals = Counter({"Europe": 3})
    class_region_counts = Counter({("transport", "Europe"): 2, ("tanker", "Europe"): 1})
    anomalies = [{"class": "isr", "region": "Europe", "today": 9, "baseline_mean": 2.0, "z": 3.1}]
    markers = [Marker(51.5, -0.5, "2 aircraft", tone="hot")]
    ship_rows = [("Mediterranean Sea", "Gerald R. Ford CSG", "underway")]

    path = om.build_report(
        NOW, class_totals, region_totals, class_region_counts, anomalies,
        markers, False, "A quiet day in aggregate.", ship_rows, LATEST_ARTICLE_URL, "USNI Sept 22",
    )
    text = Path(path).read_text(encoding="utf-8")
    assert "Gerald R. Ford CSG" in text  # aggregate ship-group name is fine
    assert "transport" in text and "Europe" in text
    assert f'href="{LATEST_ARTICLE_URL}"' in text  # source article rendered as a real link
    # never a per-aircraft hex/tail/callsign identifier
    assert "ae1460" not in text
    assert "REACH123" not in text


def test_build_report_with_no_article_url_renders_plain_text_not_a_broken_link(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    ship_rows = [("Mediterranean Sea", "Gerald R. Ford CSG", "underway")]
    path = om.build_report(
        NOW, Counter(), Counter(), Counter(), [], [], False, "quiet",
        ship_rows, None, None,
    )
    text = Path(path).read_text(encoding="utf-8")
    assert "Gerald R. Ford CSG" in text
    assert "href=\"None\"" not in text


# -- main() end to end ------------------------------------------------------------


def test_main_writes_report_and_run_row(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "_now", lambda: NOW)
    monkeypatch.setattr(om, "LMStudio", lambda: FakeLMStudio(entries=[]))
    monkeypatch.setattr(om, "default_fetcher", lambda: make_fetcher())
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    om.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        row = store.conn.execute("SELECT task, status FROM runs").fetchone()
        assert row == ("tasks.osint_movements", "ok")
