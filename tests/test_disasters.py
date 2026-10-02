"""tasks.disasters: GDACS/USGS/EONET/FIRMS adapters, cross-source dedupe, the
hazard metric and "new since last run" -- no network, canned fixtures only."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from lmsa import Store, geo, geo_metrics
from lmsa.digest import RunRow
from lmsa.web import FetchError, HTTPStatusError, InvalidJSON, Response, TooLarge, TransportResponse
from lmsa.web import Fetcher as RealFetcher
from tasks import disasters as ds
from tasks import risk_map as rm

FIX = Path(__file__).parent / "fixtures" / "disasters"
GDACS_XML = (FIX / "gdacs.xml").read_bytes()
USGS_JSON = json.loads((FIX / "usgs.geojson").read_text())
EONET_JSON = json.loads((FIX / "eonet.json").read_text())
FIRMS_CSV = (FIX / "firms.csv").read_text()

GDACS_URL = "https://www.gdacs.org/xml/rss.xml"
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson"
EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&days=3"
FIRMS_KEY = "SUPERSECRETMAPKEY"
FIRMS_URL = ds.firms_url(FIRMS_KEY, "VIIRS_SNPP_NRT", "world", 1)

NOW = datetime(2026, 9, 24, 6, 0, tzinfo=timezone.utc)


class FakeFetcher:
    """Stands in for lmsa.web.Fetcher: `.get(url) -> Response` / `.get_json`,
    no network, no robots/cache/rate-limit plumbing."""

    def __init__(self, pages: dict | None = None, raise_on: dict | None = None):
        self.pages = pages or {}
        self.raise_on = raise_on or {}
        self.calls: list[str] = []

    def get(self, url: str) -> Response:
        self.calls.append(url)
        if url in self.raise_on:
            raise self.raise_on[url]
        if url not in self.pages:
            raise AssertionError(f"unexpected fetch {url}")
        body = self.pages[url]
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        return Response(200, url, {"content-type": "application/octet-stream"}, body)

    def get_json(self, url: str):
        return self.get(url).json()


def make_fetcher(**overrides) -> FakeFetcher:
    pages = {GDACS_URL: GDACS_XML, USGS_URL: USGS_JSON, EONET_URL: EONET_JSON, FIRMS_URL: FIRMS_CSV}
    pages.update(overrides.pop("pages", {}))
    return FakeFetcher(pages=pages, **overrides)


def cfg(**kw) -> ds.Config:
    c = ds.Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


@pytest.fixture(autouse=True)
def _no_real_fetcher(monkeypatch):
    """No test may build a real, networked Fetcher by accident (e.g. the
    dedicated FIRMS one); tests that need one inject a transport."""
    def refuse(*a, **kw):
        raise AssertionError("a real lmsa.web.Fetcher was constructed in a test")
    monkeypatch.setattr(ds, "Fetcher", refuse)


@pytest.fixture
def conn(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        yield store.conn


# -- GDACS ----------------------------------------------------------------

def test_parse_gdacs():
    events = ds.parse_gdacs(GDACS_XML)
    assert len(events) == 3
    by_id = {e.id: e for e in events}
    eq = by_id["gdacs:1111111"]
    assert eq.type == "earthquake" and eq.severity == "Green" and eq.weight == 1.0
    assert eq.country == "ID" and eq.countries == ("ID",)
    assert eq.lat == pytest.approx(-7.70) and eq.lon == pytest.approx(115.20)
    assert eq.started == datetime(2026, 9, 24, 5, 0, tzinfo=timezone.utc)
    assert eq.merged_ids == (eq.id,)

    fl = by_id["gdacs:2222222"]
    assert fl.type == "flood" and fl.severity == "Red" and fl.weight == 10.0 and fl.country == "PH"

    # Armenia's alpha-3 (ARM) is denylisted for prose extraction (normalize_country
    # would reject it), but this is a structured gdacs:iso3 code, so it must resolve.
    am = by_id["gdacs:3333333"]
    assert am.country == "AM" and am.countries == ("AM",) and am.weight == 4.0 and am.severity == "Orange"


def test_parse_gdacs_malformed_xml_raises():
    import xml.etree.ElementTree as ET

    with pytest.raises(ET.ParseError):
        ds.parse_gdacs(b"<not-xml")


def test_parse_gdacs_multi_country_event_credits_every_listed_country():
    xml = b"""<?xml version="1.0"?>
    <rss xmlns:gdacs="http://www.gdacs.org" xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#">
    <channel><item>
    <title>Orange tropical cyclone alert</title>
    <link>https://www.gdacs.org/report.aspx?eventid=4444444&amp;eventtype=TC</link>
    <gdacs:eventid>4444444</gdacs:eventid>
    <gdacs:eventtype>TC</gdacs:eventtype>
    <gdacs:alertlevel>Orange</gdacs:alertlevel>
    <gdacs:country>Philippines; Vietnam</gdacs:country>
    <gdacs:iso3>PHL VNM</gdacs:iso3>
    <gdacs:fromdate>2026-09-25T00:00:00</gdacs:fromdate>
    <geo:Point><geo:lat>15.0</geo:lat><geo:long>115.0</geo:long></geo:Point>
    </item></channel></rss>"""
    [ev] = ds.parse_gdacs(xml)
    assert ev.countries == ("PH", "VN")
    assert ev.country == "PH"  # display/primary is the first listed

    totals = ds.hazard_today([ev], datetime(2026, 9, 25, 6, tzinfo=timezone.utc))
    # Both countries get the *full* weight -- not split between them.
    assert totals["PH"] == 4.0
    assert totals["VN"] == 4.0


# -- USGS -------------------------------------------------------------------

def test_parse_usgs():
    events = ds.parse_usgs(USGS_JSON)
    assert len(events) == 3
    by_id = {e.id: e for e in events}

    quake = by_id["usgs:us1111111"]
    assert quake.weight == 2.0 and quake.severity == "M5.6" and quake.country == "ID"

    japan = by_id["usgs:us2222222"]
    assert japan.weight == 4.0 and japan.country == "JP"

    # A US postal code after the last comma must resolve to "US", never be
    # left blank or (worse) mis-mapped to a foreign country of the same code.
    domestic = by_id["usgs:us3333333"]
    assert domestic.weight == 1.0 and domestic.country == "US" and domestic.countries == ("US",)


def test_parse_usgs_skips_bad_features_independently():
    data = {"features": [
        {"properties": {"mag": 5.0, "place": "x", "time": 1000}, "geometry": {"coordinates": [1.0, 2.0]}, "id": "ok"},
        {"properties": {"mag": 5.0, "place": "x"}, "geometry": {"coordinates": [1.0, 2.0]}, "id": "no-time"},
        {"properties": {"mag": 5.0, "place": "x", "time": 1000}, "geometry": {"coordinates": []}, "id": "no-coords"},
        "not-a-dict",
    ]}
    events = ds.parse_usgs(data)
    assert [e.id for e in events] == ["usgs:ok"]


# -- USGS place -> country: US states/DC/territories before any country name ----

@pytest.mark.parametrize("place, expected", [
    ("9km WSW of Enid, OK", "US"),
    ("10km NNW of Petrolia, CA", "US"),        # CA collides with Canada's ISO code
    ("5km N of Erie, PA", "US"),                # PA collides with Panama's ISO code
    ("3km S of Albany, GA", "US"),               # GA collides with Gabon's ISO code
    ("50km NE of Anchorage, Alaska", "US"),       # full state name
    ("2km E of San Juan, Puerto Rico", "PR"),      # territory keeps its own code
    ("2km E of San Juan, PR", "PR"),
    ("80km S of Hagatna, Guam", "GU"),               # territory full name
    ("80km S of Guam", None),                          # no comma at all: no guess
    ("100km SW of Mexico City, Mexico", "MX"),          # a real foreign country still resolves
    ("20km NW of Deming, New Mexico", "US"),             # must not be confused with "Mexico"
])
def test_usgs_place_country_states_before_countries(place, expected):
    assert ds._usgs_place_country(place) == expected


# -- EONET --------------------------------------------------------------------

def test_parse_eonet():
    events = ds.parse_eonet(EONET_JSON)
    assert len(events) == 3
    by_id = {e.id: e for e in events}

    wildfire = by_id["eonet:EONET_00001"]
    assert wildfire.type == "wildfire" and wildfire.weight == ds.EONET_WEIGHT
    assert wildfire.country is None and wildfire.countries == ()
    # Uses the *last* (most recent) geometry entry, not the first.
    assert wildfire.lat == pytest.approx(36.6) and wildfire.started == datetime(2026, 9, 24, tzinfo=timezone.utc)

    storm = by_id["eonet:EONET_00002"]
    assert storm.type == "cyclone"

    other = by_id["eonet:EONET_00003"]
    assert other.type == "other"  # unmapped category -> "other", never dropped


# -- FIRMS --------------------------------------------------------------------

def test_parse_firms_clusters_and_drops_low_confidence():
    events = ds.parse_firms(FIRMS_CSV)
    assert len(events) == 2
    by_place = {round(e.lat, 1): e for e in events}

    california = by_place[36.5]
    assert california.type == "wildfire" and california.country is None
    assert "3 detections" in california.severity  # the "l" (low) confidence row is dropped
    assert california.weight == pytest.approx(0.6)

    sydney = by_place[-33.9]
    assert sydney.weight == pytest.approx(0.4)


def test_firms_weight_is_capped():
    header = "latitude,longitude,acq_date,confidence"
    rows = ["10.0,20.0,2026-09-24,n" for _ in range(100)]
    csv_text = "\n".join([header] + rows)
    events = ds.parse_firms(csv_text)
    assert len(events) == 1
    assert events[0].weight == ds.FIRMS_WEIGHT_CAP


# -- dedupe -------------------------------------------------------------------

def test_dedupe_merges_same_type_nearby_events_across_sources():
    gdacs = ds.parse_gdacs(GDACS_XML)
    usgs = ds.parse_usgs(USGS_JSON)
    eonet = ds.parse_eonet(EONET_JSON)
    merged = ds.dedupe(gdacs + usgs + eonet)
    assert len(merged) == len(gdacs) + len(usgs) + len(eonet) - 1  # the Indonesia quake merges

    quake = next(e for e in merged if e.country == "ID")
    assert set(quake.sources) == {"gdacs", "usgs"}
    assert quake.weight == 2.0  # the higher-severity (USGS) record wins the merged fields
    assert set(quake.merged_ids) == {"gdacs:1111111", "usgs:us1111111"}

    # Unrelated events (different type, or too far apart) are untouched.
    assert sum(1 for e in merged if e.country == "PH") == 1
    assert sum(1 for e in merged if e.country == "AM") == 1


def test_dedupe_needs_coordinates_on_both_sides():
    a = ds.Event(id="a:1", source="a", sources=("a",), merged_ids=("a:1",), type="wildfire", severity="",
                 weight=1.0, country=None, countries=(), place="x", lat=None, lon=None, started=NOW, url=None)
    b = ds.Event(id="b:1", source="b", sources=("b",), merged_ids=("b:1",), type="wildfire", severity="",
                 weight=1.0, country=None, countries=(), place="x", lat=1.0, lon=1.0, started=NOW, url=None)
    assert len(ds.dedupe([a, b])) == 2


def test_dedupe_merges_countries_from_both_sides():
    a = ds.Event(id="a:1", source="a", sources=("a",), merged_ids=("a:1",), type="wildfire", severity="",
                 weight=1.0, country="US", countries=("US",), place="p", lat=10.0, lon=10.0, started=NOW, url=None)
    b = ds.Event(id="b:1", source="b", sources=("b",), merged_ids=("b:1",), type="wildfire", severity="",
                 weight=2.0, country="MX", countries=("MX",), place="p", lat=10.01, lon=10.01, started=NOW, url=None)
    [merged] = ds.dedupe([a, b])
    assert set(merged.countries) == {"US", "MX"}
    assert merged.country in merged.countries
    assert set(merged.merged_ids) == {"a:1", "b:1"}


# -- collect_events: fetch isolation + FIRMS gating ----------------------------

def test_collect_events_isolates_one_bad_source(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    fetcher = make_fetcher(raise_on={GDACS_URL: FetchError("boom")})
    events, stats = ds.collect_events(fetcher, NOW, cfg())
    assert stats.errors[0][0] == "gdacs"
    assert "boom" in stats.errors[0][1]
    assert stats.counts.get("usgs") == 3 and stats.counts.get("eonet") == 3
    assert all(e.source != "gdacs" for e in events)


def test_collect_events_skips_firms_when_no_key(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    events, stats = ds.collect_events(make_fetcher(), NOW, cfg())
    assert stats.firms_skipped == "LMSA_FIRMS_MAP_KEY not set"
    assert "firms" not in stats.counts


def test_collect_events_skips_firms_when_disabled(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    events, stats = ds.collect_events(make_fetcher(), NOW, cfg(firms_enabled=False))
    assert stats.firms_skipped == "disabled in config/disasters.toml"


def test_collect_events_includes_firms_and_merges_with_eonet_wildfire(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    fetcher = make_fetcher()
    events, stats = ds.collect_events(fetcher, NOW, cfg(), firms_fetcher=fetcher)
    assert stats.firms_skipped is None and stats.counts.get("firms") == 2
    wildfire = next(e for e in events if e.type == "wildfire" and "eonet:EONET_00001" in e.merged_ids)
    assert set(wildfire.sources) == {"eonet", "firms"}
    # The unrelated Sydney FIRMS cluster stays its own event.
    assert any(e.source == "firms" and e.country is None and e.weight == pytest.approx(0.4) for e in events)


def test_collect_is_the_plain_adapter_shape(monkeypatch):
    """collect(fetcher, now) -> list[Event]: the shape #52's engine will register."""
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    monkeypatch.setattr(ds, "load_config", lambda path=ds.CONFIG: cfg())
    events = ds.collect(make_fetcher(), NOW)
    assert isinstance(events, list) and all(isinstance(e, ds.Event) for e in events)


# -- per-source counts + "unrecognized response" vs a genuinely empty feed -----

def test_empty_feed_is_not_unrecognized(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    fetcher = make_fetcher(pages={EONET_URL: {"events": []}})
    events, stats = ds.collect_events(fetcher, NOW, cfg())
    assert stats.considered["eonet"] == 0 and stats.counts["eonet"] == 0
    assert "eonet" not in stats.unrecognized


def test_non_empty_body_that_parses_to_zero_is_flagged_unrecognized(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    # One EONET event with no usable Point geometry: the payload is not
    # empty (one entry), but nothing survives parsing.
    bad_eonet = {"events": [{"id": "EONET_X", "title": "x", "categories": [], "geometry": []}]}
    fetcher = make_fetcher(pages={EONET_URL: bad_eonet})
    events, stats = ds.collect_events(fetcher, NOW, cfg())
    assert stats.considered["eonet"] == 1 and stats.counts["eonet"] == 0
    assert "eonet" in stats.unrecognized

    report = ds.build_report(events, NOW.date(), set(), 0, [], stats, cfg())
    html = report.render()
    assert "unrecognized response" in html
    assert "Sources" in html


def test_report_shows_per_source_counts(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    events, stats = ds.collect_events(make_fetcher(), NOW, cfg())
    report = ds.build_report(events, NOW.date(), set(), 0, [], stats, cfg())
    html = report.render()
    for name in ("gdacs", "usgs", "eonet"):
        assert name in html
        assert str(stats.counts[name]) in html


# -- security: the FIRMS MAP_KEY must never leak through an error -------------

@pytest.mark.parametrize("exc_factory", [
    lambda url: HTTPStatusError(503, url),
    lambda url: TooLarge(f"body exceeds max_bytes for {url}"),
    lambda url: InvalidJSON(f"get_json got a non-JSON body from {url}"),
])
def test_firms_fetch_error_never_leaks_the_map_key(monkeypatch, exc_factory):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    fetcher = make_fetcher(raise_on={FIRMS_URL: exc_factory(FIRMS_URL)})
    events, stats = ds.collect_events(fetcher, NOW, cfg(), firms_fetcher=fetcher)
    assert stats.firms_skipped is None
    assert stats.errors and stats.errors[0][0] == "firms"
    message = stats.errors[0][1]
    assert FIRMS_KEY not in message
    assert "***" in message
    # Nothing else in stats carries the raw URL/key either.
    assert FIRMS_KEY not in json.dumps(stats.counts) and FIRMS_KEY not in json.dumps(stats.considered)


def test_firms_error_never_leaks_key_into_report_or_summary(conn, monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    fetcher = make_fetcher(raise_on={FIRMS_URL: HTTPStatusError(503, FIRMS_URL)})
    report, summary = ds.run_once(conn, cfg(), fetcher, NOW, firms_fetcher=fetcher)
    assert FIRMS_KEY not in report.render()
    assert FIRMS_KEY not in json.dumps(summary, default=str)


def test_main_never_prints_the_firms_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["disasters"])  # parser.parse_args() reads real argv otherwise
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))  # ds.main() writes a report; keep it out of repo output/
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    monkeypatch.setattr(ds, "load_config", lambda path=ds.CONFIG: cfg())
    fetcher = make_fetcher(raise_on={FIRMS_URL: HTTPStatusError(503, FIRMS_URL)})
    monkeypatch.setattr(ds, "Fetcher", lambda *a, **kw: fetcher)
    monkeypatch.setattr(ds, "Store", lambda *a, **kw: Store(tmp_path / "db.sqlite"))
    ds.main()
    assert FIRMS_KEY not in capsys.readouterr().out


# -- hazard metric --------------------------------------------------------------

def test_hazard_today_sums_active_events_and_credits_every_country():
    events = ds.dedupe(ds.parse_gdacs(GDACS_XML) + ds.parse_usgs(USGS_JSON))
    totals = ds.hazard_today(events, NOW)
    assert totals["ID"] == pytest.approx(2.0)    # merged GDACS+USGS quake, 1 h ago
    assert totals["PH"] == pytest.approx(10.0)   # ongoing flood (no todate)
    assert totals["AM"] == pytest.approx(4.0)    # quake 21 h ago
    assert totals["JP"] == pytest.approx(4.0)    # quake 20 h ago
    # The Enid, OK quake resolves to "US" but is ~2 days old: not active today.
    assert "US" not in totals
    assert all(c != "OK" for c in totals)


def test_write_hazard_metrics_is_idempotent(conn):
    events = ds.dedupe(ds.parse_gdacs(GDACS_XML) + ds.parse_usgs(USGS_JSON))
    written, errors = ds.write_hazard_metrics(conn, events, NOW, zero_fill=True)
    assert written == len(geo.countries()) and errors == []
    written_again, errors_again = ds.write_hazard_metrics(conn, events, NOW, zero_fill=True)
    assert written_again == written and errors_again == []
    assert geo_metrics.series(conn, "ID", "hazard") == [(date(2026, 9, 24), 2.0)]
    count = conn.execute("SELECT COUNT(*) FROM geo_metrics WHERE metric='hazard'").fetchone()[0]
    assert count == len(geo.countries())


# -- "new since last run" survives a dedupe winner swap ------------------------

def test_new_event_ids_survives_a_dedupe_winner_swap():
    seen_before = {"gdacs:1"}
    # A rerun where USGS now also reports the same physical event and wins
    # the merge (higher weight): the surviving event's own `.id` is "usgs:2",
    # but "gdacs:1" (already seen) is still one of its merged_ids.
    swapped = ds.Event(id="usgs:2", source="usgs", sources=("gdacs", "usgs"), merged_ids=("gdacs:1", "usgs:2"),
                       type="earthquake", severity="M5.6", weight=2.0, country="ID", countries=("ID",),
                       place="x", lat=1.0, lon=1.0, started=NOW, url=None)
    assert ds._new_event_ids([swapped], seen_before) == set()  # not new

    brand_new = ds.Event(id="eonet:9", source="eonet", sources=("eonet",), merged_ids=("eonet:9",),
                         type="wildfire", severity="", weight=2.0, country=None, countries=(),
                         place="x", lat=None, lon=None, started=NOW, url=None)
    assert ds._new_event_ids([brand_new], seen_before) == {"eonet:9"}


def test_run_once_new_since_last_run_survives_a_dedupe_winner_swap(conn, monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    c = cfg()
    empty_usgs, empty_eonet = {"features": []}, {"events": []}

    # Run 1: only GDACS sees the Indonesia quake.
    fetcher1 = make_fetcher(pages={USGS_URL: empty_usgs, EONET_URL: empty_eonet})
    _, summary1 = ds.run_once(conn, c, fetcher1, NOW)
    assert summary1["new"] == 3  # the 3 GDACS events, all new

    # Run 2: USGS now reports the same quake with a higher severity weight,
    # so USGS wins the merge -- the surviving event's own id changes.
    fetcher2 = make_fetcher(pages={EONET_URL: empty_eonet})
    events2, _ = ds.collect_events(fetcher2, NOW, c)
    quake2 = next(e for e in events2 if e.type == "earthquake" and "ID" in e.countries)
    assert quake2.id == "usgs:us1111111"  # confirms the winner really did swap

    _, summary2 = ds.run_once(conn, c, fetcher2, NOW + timedelta(minutes=5))
    # Only the two brand-new USGS quakes (Japan, Enid) are new -- the
    # Indonesia quake must NOT be recounted just because a different source
    # now wins the dedupe.
    assert summary2["new"] == 2


# -- end to end: run_once, report rendering ------------------------------------

def test_run_once_and_report(conn, monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    c = cfg()
    report1, summary1 = ds.run_once(conn, c, make_fetcher(), NOW)
    assert summary1["events"] == 8  # 3 gdacs + 3 usgs + 3 eonet - 1 merge, firms off
    assert summary1["new"] == 8
    assert summary1["hazard_zero_filled"] is True
    assert summary1["hazard_rows"] == len(geo.countries())  # 4 active countries + a 0 for every other
    html = report1.render()
    assert "Disaster fusion report" in html
    assert "Indonesia" in html or "gdacs+usgs" in html

    report2, summary2 = ds.run_once(conn, c, make_fetcher(), NOW)
    assert summary2["new"] == 0  # same fixtures, same ids: nothing new the second run


def test_build_report_notes_firms_skip(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    events, stats = ds.collect_events(make_fetcher(), NOW, cfg())
    report = ds.build_report(events, NOW.date(), set(), 0, [], stats, cfg())
    assert "FIRMS" in report.render()


# == coordinator fix round (#59 opus re-review) ===============================

def _quake(i: int, country: str, when: datetime, weight: float = 1.0, lon: float = 140.0) -> ds.Event:
    eid = f"usgs:q{i}"
    return ds.Event(id=eid, source="usgs", sources=("usgs",), merged_ids=(eid,), type="earthquake",
                    severity="M4.6", weight=weight, country=country, countries=(country,), place="x",
                    lat=36.0, lon=lon, started=when, url=None, ended=when)


def _alert(i: int, country: str, started: datetime, weight: float, ended: datetime | None = None) -> ds.Event:
    eid = f"gdacs:{i}"
    return ds.Event(id=eid, source="gdacs", sources=("gdacs",), merged_ids=(eid,), type="flood",
                    severity="Red", weight=weight, country=country, countries=(country,), place="x",
                    lat=12.0, lon=122.0, started=started, url=None, ended=ended)


# -- 1. explicit zeros give risk_map a baseline --------------------------------

def test_zero_filled_hazard_baseline_lets_risk_map_score_a_quake(conn):
    rcfg = rm.load_config()
    start = datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)
    for k in range(40):  # 40 daily clean runs; Japan has a small quake every 3rd day
        now = start + timedelta(days=k)
        events = [_quake(k, "JP", now - timedelta(hours=3))] if k % 3 == 0 else []
        ds.write_hazard_metrics(conn, events, now, zero_fill=True)
    today = start + timedelta(days=40)
    ds.write_hazard_metrics(conn, [_quake(999, "JP", today - timedelta(hours=2), weight=8.0)], today, zero_fill=True)

    jp = dict(geo_metrics.series(conn, "JP", "hazard"))
    assert len(jp) == 41 and jp[today.date()] == 8.0
    m = rm.metric_at(jp, today.date(), rcfg, "hazard")
    assert m.status == "ok" and m.z > 3  # scored, not "gap"
    rows = rm.load_rows(conn, today.date() - timedelta(days=60))
    score = rm.score_country("JP", rows["JP"], today.date(), rcfg)
    assert score.subscores["hazard"] is not None and score.subscores["hazard"] > 3
    # A country with no events at all has a flat 0 baseline, scored as calm.
    fr = rm.metric_at(dict(geo_metrics.series(conn, "FR", "hazard")), today.date(), rcfg, "hazard")
    assert fr.status == "ok" and fr.z == 0


def test_no_zero_rows_when_a_country_source_failed(conn, monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    fetcher = make_fetcher(raise_on={USGS_URL: HTTPStatusError(503, USGS_URL)})
    report, summary = ds.run_once(conn, cfg(), fetcher, NOW)
    assert summary["hazard_zero_filled"] is False
    assert geo_metrics.series(conn, "FR", "hazard") == []  # quiet country: no row, not a fake 0
    assert geo_metrics.series(conn, "PH", "hazard") == [(NOW.date(), 10.0)]
    assert "Quiet countries got no hazard row today" in report.render()


def test_partial_run_never_lowers_todays_row(conn):
    ds.write_hazard_metrics(conn, [_alert(1, "PH", NOW, 10.0), _alert(2, "PH", NOW, 4.0)], NOW, zero_fill=True)
    ds.write_hazard_metrics(conn, [_alert(1, "PH", NOW, 10.0)], NOW + timedelta(hours=1), zero_fill=False)
    assert geo_metrics.series(conn, "PH", "hazard") == [(NOW.date(), 14.0)]


# -- 2. FIRMS never touches the disk cache or the logs --------------------------

ROBOTS = b"User-agent: *\nAllow: /\n"


def _transport(firms):
    def transport(req):
        if req.url.endswith("/robots.txt"):
            return TransportResponse(200, {}, ROBOTS)
        if "gdacs.org" in req.url:
            return TransportResponse(200, {}, GDACS_XML)
        if "earthquake.usgs.gov" in req.url:
            return TransportResponse(200, {}, json.dumps(USGS_JSON).encode())
        if "eonet" in req.url:
            return TransportResponse(200, {}, json.dumps(EONET_JSON).encode())
        if "firms" in req.url:
            return firms(req)
        return TransportResponse(404, {}, b"")
    return transport


def _patch_real_fetcher(monkeypatch, transport) -> list[dict]:
    """ds.Fetcher -> a real lmsa.web.Fetcher on a fake transport; records kwargs."""
    built: list[dict] = []

    def make(*a, **kw):
        built.append(dict(kw))
        return RealFetcher(*a, **{**kw, "transport": transport, "resolver": lambda h: ["93.184.216.34"],
                                  "sleep": lambda s: None})
    monkeypatch.setattr(ds, "Fetcher", make)
    return built


@pytest.mark.parametrize("firms_status", [200, 503])
def test_firms_key_never_reaches_disk_cache_or_logs(tmp_path, monkeypatch, caplog, firms_status):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    transport = _transport(lambda req: TransportResponse(firms_status, {}, FIRMS_CSV.encode()))
    built = _patch_real_fetcher(monkeypatch, transport)
    cache = tmp_path / "web"
    shared = RealFetcher("t", per_host_delay=0, cache_dir=cache, transport=transport,
                         resolver=lambda h: ["93.184.216.34"], sleep=lambda s: None)
    caplog.set_level(logging.DEBUG)
    events, stats = ds.collect_events(shared, NOW, cfg())

    assert built and built[0]["cache_dir"] is None  # FIRMS got its own uncached Fetcher
    assert built[0]["max_bytes"] == ds.Config().firms_max_bytes
    if firms_status == 200:
        assert stats.counts["firms"] == 2
    else:
        assert stats.errors and stats.errors[0][0] == "firms"
    assert any(cache.rglob("*.json"))  # the shared fetcher did cache the other sources
    for f in cache.rglob("*"):
        if f.is_file():
            assert FIRMS_KEY.lower() not in f.read_text(errors="ignore").lower(), f
    for rec in caplog.records:
        assert FIRMS_KEY.lower() not in rec.getMessage().lower()
    assert FIRMS_KEY not in json.dumps(stats.errors)


# -- 3. key validation + pattern redaction --------------------------------------

@pytest.mark.parametrize("raw", [FIRMS_KEY + "\r", FIRMS_KEY + " ", " " + FIRMS_KEY + "\r\n"])
def test_firms_key_is_stripped(monkeypatch, raw):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: raw)
    fetcher = make_fetcher()
    events, stats = ds.collect_events(fetcher, NOW, cfg(), firms_fetcher=fetcher)
    assert FIRMS_URL in fetcher.calls and stats.counts["firms"] == 2


@pytest.mark.parametrize("raw", ["ab+c/d?e", "key with space", "abc%41", "abc\x00"])
def test_invalid_firms_key_is_refused_without_echoing_it(monkeypatch, raw):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: raw)
    fetcher = make_fetcher()
    events, stats = ds.collect_events(fetcher, NOW, cfg(), firms_fetcher=fetcher)
    assert not any("firms" in u for u in fetcher.calls)
    [(src, msg)] = [e for e in stats.errors if e[0] == "firms"]
    assert "invalid FIRMS key format" in msg and raw.strip() not in msg


def _pct(text: str) -> str:
    return "".join(f"%{ord(ch):02X}" for ch in text)


@pytest.mark.parametrize("location", [
    "https://firms.modaps.eosdis.gov/moved/" + FIRMS_KEY.upper() + "/x",       # upper-cased key
    "https://firms.modaps.eosdis.gov/moved/" + FIRMS_KEY.lower() + "/x",       # lower-cased key
    "https://firms.modaps.eosdis.gov/moved/" + _pct(FIRMS_KEY) + "/x",         # percent-encoded key
    "https://firms.modaps.eosdis.gov/moved/" + _pct(FIRMS_KEY).lower() + "/x",  # lower-case hex
    "/api/area/csv/" + FIRMS_KEY.upper() + "/VIIRS/world/1",                     # relative, key path segment
])
def test_redirected_key_is_redacted_in_every_form(monkeypatch, caplog, location):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY + "\r")

    def firms(req):
        if req.url == FIRMS_URL:
            return TransportResponse(302, {"Location": location}, b"")
        return TransportResponse(500, {}, b"")
    _patch_real_fetcher(monkeypatch, _transport(firms))
    caplog.set_level(logging.DEBUG)
    events, stats = ds.collect_events(make_fetcher(), NOW, cfg())
    [(src, msg)] = [e for e in stats.errors if e[0] == "firms"]
    assert "***" in msg
    blob = (msg + " ".join(r.getMessage() for r in caplog.records)).lower()
    for form in (FIRMS_KEY, _pct(FIRMS_KEY)):
        assert form.lower() not in blob


def test_redact_patterns():
    assert ds._redact("GET https://x/api/area/csv/anything-here/VIIRS/world/1", None) == \
        "GET https://x/api/area/csv/***/VIIRS/world/1"
    assert ds._redact("a " + FIRMS_KEY.lower() + " b", FIRMS_KEY) == "a *** b"
    mixed = "S%55PERSECRETMAPKE%79"  # partly encoded
    assert ds._redact(mixed, FIRMS_KEY) == "***"


# -- 4. HTTP-200 error bodies are "unrecognized", not empty feeds ----------------

def test_firms_error_body_is_unrecognized_with_redacted_snippet(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    body = "Invalid MAP_KEY " + FIRMS_KEY + ". " + "x" * 500
    fetcher = make_fetcher(pages={FIRMS_URL: body})
    events, stats = ds.collect_events(fetcher, NOW, cfg(), firms_fetcher=fetcher)
    assert "firms" in stats.unrecognized and stats.counts["firms"] == 0
    detail = stats.unrecognized_detail["firms"]
    assert FIRMS_KEY not in detail and "***" in detail and "Invalid MAP_KEY" in detail
    snippet = detail.split(": ", 1)[1].strip("'\"")
    assert len(snippet) <= ds.FIRMS_SNIPPET_CHARS
    html = ds.build_report(events, NOW.date(), set(), 0, [], stats, cfg()).render()
    assert "unrecognized response" in html and FIRMS_KEY not in html


@pytest.mark.parametrize("url, body, source", [
    (USGS_URL, {"error": "rate limited"}, "usgs"),
    (USGS_URL, ["not", "an", "object"], "usgs"),
    (EONET_URL, {"message": "service unavailable"}, "eonet"),
])
def test_usgs_eonet_error_bodies_are_unrecognized(monkeypatch, url, body, source):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    events, stats = ds.collect_events(make_fetcher(pages={url: body}), NOW, cfg())
    assert source in stats.unrecognized and not stats.clean(source)


def test_all_low_confidence_bbox_is_zero_kept_not_unrecognized(monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    csv_text = "latitude,longitude,acq_date,confidence\n" + "\n".join("10.0,20.0,2026-09-24,l" for _ in range(5))
    fetcher = make_fetcher(pages={FIRMS_URL: csv_text})
    events, stats = ds.collect_events(fetcher, NOW, cfg(), firms_fetcher=fetcher)
    assert stats.counts["firms"] == 0 and stats.considered["firms"] == 5
    assert "firms" not in stats.unrecognized and stats.clean("firms")


# -- 5. ongoing events credit today; past days are never rewritten ---------------

def test_ongoing_event_is_credited_to_today_not_its_start_date(conn):
    started = NOW - timedelta(days=5)
    events = [
        _alert(1, "PH", started, 10.0),                              # ongoing, no todate
        _alert(2, "VN", started, 4.0, ended=NOW + timedelta(days=2)),  # todate in the future
        _alert(3, "TH", started, 4.0, ended=NOW - timedelta(days=2)),  # ended two days ago
    ]
    ds.write_hazard_metrics(conn, events, NOW)
    assert geo_metrics.series(conn, "PH", "hazard") == [(NOW.date(), 10.0)]
    assert geo_metrics.series(conn, "VN", "hazard") == [(NOW.date(), 4.0)]
    assert geo_metrics.series(conn, "TH", "hazard") == []


def test_gdacs_todate_parsed_and_earthquake_is_instantaneous():
    xml = b"""<rss xmlns:gdacs="http://www.gdacs.org"><channel>
    <item><title>t</title><gdacs:eventid>1</gdacs:eventid><gdacs:eventtype>DR</gdacs:eventtype>
    <gdacs:alertlevel>Red</gdacs:alertlevel><gdacs:iso3>ETH</gdacs:iso3>
    <gdacs:fromdate>2026-05-01T00:00:00</gdacs:fromdate><gdacs:todate>2026-09-24T00:00:00</gdacs:todate></item>
    <item><title>q</title><gdacs:eventid>2</gdacs:eventid><gdacs:eventtype>EQ</gdacs:eventtype>
    <gdacs:alertlevel>Green</gdacs:alertlevel><gdacs:iso3>JPN</gdacs:iso3>
    <gdacs:fromdate>2026-09-20T00:00:00</gdacs:fromdate></item>
    </channel></rss>"""
    by_id = {e.id: e for e in ds.parse_gdacs(xml)}
    assert by_id["gdacs:1"].ended == datetime(2026, 9, 24, tzinfo=timezone.utc)
    assert by_id["gdacs:2"].ended == by_id["gdacs:2"].started
    assert ds.hazard_today(list(by_id.values()), NOW) == {"ET": 10.0}  # the 4-day-old quake isn't active


def test_each_run_writes_only_todays_row(conn):
    a, b = _alert(1, "PH", NOW - timedelta(days=3), 10.0), _alert(2, "PH", NOW - timedelta(days=3), 4.0)
    ds.write_hazard_metrics(conn, [a, b], NOW, zero_fill=True)
    ds.write_hazard_metrics(conn, [a], NOW + timedelta(days=1), zero_fill=True)  # b aged out of the feed
    assert geo_metrics.series(conn, "PH", "hazard") == [
        (NOW.date(), 14.0), (NOW.date() + timedelta(days=1), 10.0),
    ]


# -- 6. a source outage doesn't re-flag its events as new -------------------------

def test_source_outage_does_not_reflag_events_as_new(conn, monkeypatch):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    c = cfg()
    _, s1 = ds.run_once(conn, c, make_fetcher(), NOW)
    _, s2 = ds.run_once(conn, c, make_fetcher(raise_on={GDACS_URL: HTTPStatusError(503, GDACS_URL)}),
                        NOW + timedelta(hours=1))
    _, s3 = ds.run_once(conn, c, make_fetcher(), NOW + timedelta(hours=2))
    assert (s1["new"], s2["new"], s3["new"]) == (8, 0, 0)
    # An unrecognized body is an outage too.
    _, s4 = ds.run_once(conn, c, make_fetcher(pages={USGS_URL: {"error": "x"}}), NOW + timedelta(hours=3))
    _, s5 = ds.run_once(conn, c, make_fetcher(), NOW + timedelta(hours=4))
    assert (s4["new"], s5["new"]) == (0, 0)
    seen = json.loads(conn.execute("SELECT value_json FROM kv WHERE ns='disasters'").fetchone()[0])
    assert len(seen) == 9  # bounded: exactly one good fetch's worth of ids


# -- 7. "..., Georgia": the US state or the country, by longitude -----------------

@pytest.mark.parametrize("place, lon, expected", [
    ("12 km N of Telavi, Georgia", 45.5, "GE"),
    ("Kakheti, Georgia", 42.0, "GE"),
    ("5 km N of Macon, Georgia", -83.6, "US"),
    ("somewhere, Georgia", None, None),
    ("somewhere, Georgia", 0.0, None),
    ("3km S of Albany, GA", 45.0, "US"),  # the postal code is always the state
])
def test_georgia_is_disambiguated_by_longitude(place, lon, expected):
    assert ds._usgs_place_country(place, lon) == expected


def test_parse_usgs_georgia_uses_the_quake_longitude():
    data = {"features": [{"id": "ge1", "properties": {"mag": 4.9, "place": "10 km E of Tbilisi, Georgia", "time": 1000},
                          "geometry": {"coordinates": [44.9, 41.7]}}]}
    [ev] = ds.parse_usgs(data)
    assert ev.countries == ("GE",)


# -- 8. FIRMS size and row caps ---------------------------------------------------

def test_firms_caps_are_shipped_and_row_cap_is_disclosed(monkeypatch):
    shipped = ds.load_config()
    assert shipped.firms_max_bytes >= 50_000_000 and shipped.firms_max_rows > 0
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: FIRMS_KEY)
    fetcher = make_fetcher()
    events, stats = ds.collect_events(fetcher, NOW, cfg(firms_max_rows=3), firms_fetcher=fetcher)
    assert stats.considered["firms"] == 3
    assert stats.notes and "3" in stats.notes[0]
    html = ds.build_report(events, NOW.date(), set(), 0, [], stats, cfg(firms_max_rows=3)).render()
    assert "only the first 3 detection rows" in html


@pytest.mark.parametrize("body", ["<error>quota exceeded</error>", "<html><body>Service down</body></html>"])
def test_gdacs_non_rss_body_is_unrecognized_and_blocks_zero_fill(monkeypatch, body):
    monkeypatch.setattr(ds.lmsa_secrets, "resolve", lambda name: None)
    events, stats = ds.collect_events(make_fetcher(pages={GDACS_URL: body}), NOW, cfg())
    assert "gdacs" in stats.unrecognized and not stats.clean("gdacs")


# == digest() (#104: master report hook) ======================================

def _ev(eid: str, *, type_: str, severity: str, weight: float, place: str, country: str,
        url: str, started: datetime) -> ds.Event:
    return ds.Event(id=eid, source="gdacs", sources=("gdacs",), merged_ids=(eid,), type=type_,
                    severity=severity, weight=weight, country=country, countries=(country,),
                    place=place, lat=1.0, lon=1.0, started=started, url=url, ended=None)


RED = _ev("gdacs:1", type_="flood", severity="Red", weight=10.0, place="Manila", country="PH",
          url="https://www.gdacs.org/report.aspx?eventid=1", started=NOW)
QUAKE = _ev("usgs:1", type_="earthquake", severity="M6.1", weight=4.0, place="Tokyo", country="JP",
            url="https://earthquake.usgs.gov/x", started=NOW - timedelta(hours=1))
GREEN = _ev("gdacs:2", type_="flood", severity="Green", weight=1.0, place="Paris", country="FR",
            url="https://www.gdacs.org/report.aspx?eventid=2", started=NOW - timedelta(hours=2))


def _run_once_for_digest(conn, monkeypatch, events, *, counts=None):
    stats = ds.FetchStats(counts=counts or {"gdacs": 2, "usgs": 1, "eonet": 0})
    monkeypatch.setattr(
        ds, "collect_events",
        lambda fetcher, now, cfg=None, firms_fetcher=None: (events, stats),
    )
    _report, summary = ds.run_once(conn, cfg(), object(), NOW)
    return summary


def test_run_once_writes_top_events_and_red_count(conn, monkeypatch):
    summary = _run_once_for_digest(conn, monkeypatch, [RED, QUAKE, GREEN])

    assert summary["red_count"] == 1
    assert [e["type"] for e in summary["top_events"]] == ["flood", "earthquake", "flood"]  # weight-ranked
    assert summary["top_events"][0]["severity"] == "Red"
    assert summary["top_events"][0]["url"] == "https://www.gdacs.org/report.aspx?eventid=1"
    assert summary["top_events"][0]["place"] == "Manila"
    assert summary["top_events"][0]["country"] == "Philippines"


def test_run_once_caps_top_events(conn, monkeypatch):
    many = [
        _ev(f"gdacs:{i}", type_="flood", severity="Green", weight=float(i), place="x", country="FR",
            url="https://www.gdacs.org/e", started=NOW)
        for i in range(ds.DIGEST_EVENTS_CAP + 20)
    ]
    summary = _run_once_for_digest(conn, monkeypatch, many)
    assert len(summary["top_events"]) == ds.DIGEST_EVENTS_CAP


def _digest_run(summary: dict) -> RunRow:
    return RunRow(id=1, task="tasks.disasters", started=NOW.isoformat(), status="ok",
                  summary_json=json.dumps(summary))


def test_digest_tiles_items_and_red_callout(conn, monkeypatch):
    summary = _run_once_for_digest(conn, monkeypatch, [RED, QUAKE, GREEN])
    run = _digest_run(summary)

    d = ds.digest(run, max_items=5)

    assert {t.label: t.value for t in d.tiles} == {
        "Active events": 3, "Red-level": 1, "New since last run": 3, "Fetch errors": 0,
    }
    assert [i.title for i in d.items] == ["flood Red", "earthquake M6.1", "flood Green"]
    assert d.items[0].blurb == "Manila, Philippines"
    assert d.items[0].url == "https://www.gdacs.org/report.aspx?eventid=1"
    assert d.items[0].tone == "hot"
    assert d.items[1].tone is None
    assert d.callouts == ["1 red-level event(s) active"]


def test_digest_respects_max_items(conn, monkeypatch):
    summary = _run_once_for_digest(conn, monkeypatch, [RED, QUAKE, GREEN])
    run = _digest_run(summary)

    d = ds.digest(run, max_items=2)

    assert [i.title for i in d.items] == ["flood Red", "earthquake M6.1"]


def test_digest_no_red_events_has_no_callout(conn, monkeypatch):
    summary = _run_once_for_digest(conn, monkeypatch, [QUAKE, GREEN])
    run = _digest_run(summary)

    d = ds.digest(run, max_items=5)

    assert d.callouts == []
    assert all(i.tone is None for i in d.items)


def test_digest_never_touches_store_or_network(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("digest() must not open the store or the network")

    monkeypatch.setattr(ds, "Store", refuse)
    monkeypatch.setattr(ds, "Fetcher", refuse)
    run = _digest_run({"events": 1, "new": 0, "fetch_errors": 0, "red_count": 0, "top_events": []})

    d = ds.digest(run, max_items=5)
    assert d.tiles and not d.items


def test_digest_tolerates_malformed_summary_json():
    """Malformed/missing summary_json never raises -- it degrades to
    zero-value tiles and no items/callouts (`_summary_dict` swallows the
    parse error; the same fallback build_section() uses for any hook that
    fails, so this hook failing outright is no worse than not being run)."""
    run = RunRow(id=1, task="tasks.disasters", started=NOW.isoformat(), status="ok",
                 summary_json="not json")
    d = ds.digest(run, max_items=5)
    assert [t.value for t in d.tiles] == [0, 0, 0, 0]
    assert d.items == []
    assert d.callouts == []
