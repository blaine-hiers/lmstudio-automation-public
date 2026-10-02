"""tasks.risk_map + lmsa.geo_metrics: z-scores, missing sub-scores, baseline
building, and a GDELT backfill against a fake transport -- no network."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from lmsa import Store, geo_metrics, markets as mkt
from lmsa.web import Fetcher, TransportResponse
from tasks import risk_map as rm

FIX = Path(__file__).parent / "fixtures" / "risk_map"
NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
AS_OF = date(2026, 9, 24)


@pytest.fixture
def conn(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        yield store.conn


def cfg(**kw):
    c = rm.Config(countries=kw.pop("countries", ["UA", "DE"]), themes=["ARMEDCONFLICT", "TERROR"])
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def fill(conn, country, metric, values, end=AS_OF):
    """values oldest -> newest, the last one on `end`."""
    n = len(values)
    geo_metrics.record_many(conn, [
        (end - timedelta(days=n - 1 - i), country, metric, v) for i, v in enumerate(values)
    ])


def alternating(n, lo=1.0, hi=3.0):
    return [lo if i % 2 else hi for i in range(n)]  # mean 2, population std 1


# -- geo_metrics ---------------------------------------------------------------

def test_zscore_math():
    assert geo_metrics.zscore(4.0, [1.0, 3.0, 1.0, 3.0]) == pytest.approx(2.0)
    assert geo_metrics.zscore(0.0, [1.0, 3.0]) == pytest.approx(-2.0)
    assert geo_metrics.zscore(5.0, []) is None
    assert geo_metrics.zscore(2.0, [2.0, 2.0]) == 0.0   # flat and equal: no change
    assert geo_metrics.zscore(3.0, [2.0, 2.0]) is None  # flat and different: undefined, not huge
    assert geo_metrics.zscore(float("nan"), [1.0, 3.0]) is None


def test_record_is_an_idempotent_upsert_and_validates(conn):
    geo_metrics.record(conn, "2026-09-24", "ua", "hazard", 1.0)
    geo_metrics.record(conn, date(2026, 9, 24), "UA", "hazard", 2.5)
    assert geo_metrics.series(conn, "UA", "hazard") == [(date(2026, 9, 24), 2.5)]
    assert conn.execute("SELECT COUNT(*) FROM geo_metrics").fetchone()[0] == 1
    assert geo_metrics.latest_date(conn, "hazard") == date(2026, 9, 24)
    for bad in [("2026-09-24", "Ukraine", "hazard", 1.0), ("2026-13-01", "UA", "hazard", 1.0),
                ("2026-09-24", "UA", "hazard", math.inf), ("2026-09-24", "UA", "", 1.0)]:
        with pytest.raises(ValueError):
            geo_metrics.record(conn, *bad)
    with pytest.raises(ValueError):  # one bad row -> nothing from the batch is written
        geo_metrics.record_many(conn, [("2026-09-23", "UA", "hazard", 1.0), ("2026-09-23", "ZZ", "hazard", 1.0)])
    assert geo_metrics.series(conn, "UA", "hazard", start="2026-09-23", end="2026-09-23") == []


# -- scoring -------------------------------------------------------------------

def test_metric_z_needs_a_full_baseline_and_says_how_far_along():
    values = {AS_OF - timedelta(days=i): 2.0 + (i % 2) for i in range(1, 12)}
    values[AS_OF] = 5.0
    m = rm.metric_z(values, AS_OF, cfg())
    assert m.status == "building" and m.n == 11 and m.z is None
    assert rm.metric_z({}, AS_OF, cfg()).status == "missing"


def test_missing_subscores_drop_out_instead_of_counting_as_zero():
    w = {"conflict": 0.5, "hazard": 0.3, "markets": 0.2}
    assert rm.blend({"conflict": 2.0, "hazard": None, "markets": None}, w) == pytest.approx(2.0)
    assert rm.blend({"conflict": 2.0, "hazard": 1.0, "markets": None}, w) == pytest.approx((1.0 + 0.3) / 0.8)
    assert rm.blend({"conflict": None, "hazard": None, "markets": None}, w) is None
    assert rm.blend({"conflict": 2.0, "hazard": 4.0, "markets": None}, {**w, "hazard": 0}) == pytest.approx(2.0)


def test_score_country_combines_volume_and_inverted_tone(conn):
    fill(conn, "UA", "conflict_volume", alternating(30) + [4.0])   # z = +2
    fill(conn, "UA", "conflict_tone", alternating(30) + [1.0])     # z = -1 -> +1 stress
    fill(conn, "UA", "hazard", alternating(30) + [2.0])            # z = 0
    rows = rm.load_rows(conn, AS_OF - timedelta(days=60))
    s = rm.score_country("UA", rows["UA"], AS_OF, cfg())
    assert s.subscores["conflict"] == pytest.approx(1.5)
    assert s.subscores["hazard"] == pytest.approx(0.0)
    assert s.subscores["markets"] is None
    assert s.composite == pytest.approx((0.5 * 1.5 + 0.3 * 0.0) / 0.8)
    assert rm.parts_note(s, cfg(), AS_OF) == "conflict +1.50 · hazard +0.00 · markets —"


def test_stale_metric_counts_as_missing(conn):
    fill(conn, "UA", "hazard", alternating(31), end=AS_OF - timedelta(days=10))
    rows = rm.load_rows(conn, AS_OF - timedelta(days=90))
    s = rm.score_country("UA", rows["UA"], AS_OF, cfg(max_age_days=7))
    assert s.subscores["hazard"] is None and s.composite is None


def test_report_shows_baseline_building_until_30_days(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fill(conn, "UA", "conflict_volume", [1.0, 2.0] * 6)  # 12 days: 11 baseline days
    report, summary = rm.run_once(conn, cfg(), {"UA": "UP", "DE": "GM"}, fetcher=None)
    html = report.render()
    assert "baseline building (11/30 days)" in html
    assert "--backfill 31" in html
    assert summary["scored"] == 0 and summary["building"] == 1


def test_report_never_shows_a_composite_without_its_parts(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fill(conn, "UA", "conflict_volume", alternating(59) + [4.0])
    fill(conn, "DE", "conflict_volume", alternating(59) + [1.0])
    fill(conn, "DE", "markets", alternating(59) + [3.0])
    report, summary = rm.run_once(conn, cfg(), {"UA": "UP", "DE": "GM"}, fetcher=None)
    html = report.render()
    assert summary["scored"] == 2
    assert "geo-svg" in html and html.count('class="geo-svg"') == 1  # one map per report
    assert "conflict +2.00 · hazard — · markets —" in html
    assert "conflict -1.00 · hazard — · markets +1.00" in html
    for header in ("Composite", "Conflict", "Hazard", "Markets", "Volume z", "Tone z", "Parts"):
        assert f">{header}<" in html
    assert "By region" in html and "sparkline" in html


# -- GDELT fetch ---------------------------------------------------------------

def test_parse_timeline_averages_subdaily_points_and_skips_junk():
    data = json.loads((FIX / "timelinetone.json").read_text(encoding="utf-8"))
    assert rm.parse_timeline(data) == {
        date(2026, 9, 20): -4.5, date(2026, 9, 21): -4.25, date(2026, 9, 23): -4.0,
    }
    assert rm.parse_timeline({}) == {}
    with pytest.raises(ValueError):
        rm.parse_timeline(["nope"])


def test_query_uses_fips_codes_not_iso():
    fips = rm.load_fips()
    assert fips["UA"] == "UP" and fips["DE"] == "GM" and fips["AU"] == "AS" and fips["AT"] == "AU"
    assert len(set(fips.values())) == len(fips)
    assert rm.gdelt_query(["ARMEDCONFLICT", "TERROR"], fips["UA"]) == \
        "(theme:ARMEDCONFLICT OR theme:TERROR) sourcecountry:UP"
    assert rm.gdelt_query(["KILL"], "GM") == "theme:KILL sourcecountry:GM"


class FakeGdelt:
    """Answers DOC API timeline calls with a canned daily series per country:
    volume alternates 1/3 (tone -1/-3) with a spike on the last day."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    def __call__(self, req):
        parts = urlsplit(req.url)
        if parts.path == "/robots.txt":
            return TransportResponse(404, {}, b"")
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.calls.append(q)
        fips = q["query"].rsplit("sourcecountry:", 1)[1]
        if fips in self.fail:
            return TransportResponse(200, {"content-type": "text/plain"}, b"Please limit requests")
        start = datetime.strptime(q["startdatetime"], "%Y%m%d%H%M%S").date()
        end = datetime.strptime(q["enddatetime"], "%Y%m%d%H%M%S").date()
        pts, d = [], start
        while d <= end:
            base = 1.0 if (end - d).days % 2 else 3.0
            v = 6.0 if d == end else base
            pts.append({"date": f"{d:%Y%m%d}T000000Z", "value": -v if q["mode"] == "timelinetone" else v})
            d += timedelta(days=1)
        body = {"timeline": [{"series": "x", "data": pts}]}
        return TransportResponse(200, {"content-type": "application/json"}, json.dumps(body).encode())


def fetcher(transport):
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    f = Fetcher(rm.USER_AGENT, per_host_delay=5.0, cache_dir=None, transport=transport,
                resolver=lambda host: ["93.184.216.34"], clock=lambda: t[0], sleep=sleep)
    return f, slept


def test_backfill_with_canned_gdelt_data_scores_and_isolates_failures(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fake = FakeGdelt(fail={"GM"})
    f, slept = fetcher(fake)
    c = cfg(countries=["UA", "DE", "RU"])
    report, summary = rm.run_once(conn, c, rm.load_fips(), fetcher=f, backfill=40)

    assert summary["fetched"] == 2 and summary["fetch_errors"] == 1
    assert {q["query"].rsplit(":", 1)[1] for q in fake.calls} == {"UP", "GM", "RS"}  # FIPS, not ISO
    ua = [q for q in fake.calls if q["query"].endswith("UP")]
    assert ua[0]["startdatetime"] == "20260816000000" and ua[0]["enddatetime"] == "20260924235959"
    assert all(s >= 4.99 for s in slept[1:])  # 1 request / 5 s to the one host
    assert len(geo_metrics.series(conn, "UA", "conflict_volume")) == 40
    assert summary["scored"] == 2
    html = report.render()
    assert "Germany (DE): InvalidJSON" in html  # the failure is disclosed
    # volume spikes to 6 (z=+4) and tone to -6 (z=-4 -> +4 stress): conflict +4
    assert "conflict +4.00 · hazard — · markets —" in html

    # rerun the same day: upserts, no duplicate rows
    before = conn.execute("SELECT COUNT(*) FROM geo_metrics").fetchone()[0]
    rm.run_once(conn, c, rm.load_fips(), fetcher=fetcher(FakeGdelt())[0])
    assert conn.execute("SELECT COUNT(*) FROM geo_metrics").fetchone()[0] == before + 16  # DE's 8-day window


def test_rotation_caps_countries_per_run_stalest_first(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    c = cfg(countries=["UA", "DE", "RU"], max_countries_per_run=2)
    fake = FakeGdelt()
    stats = rm.collect(conn, fetcher(fake)[0], c, rm.load_fips(), AS_OF)
    assert stats.attempted == ["DE", "RU"]
    monkeypatch.setattr(rm, "_now", lambda: NOW + timedelta(days=1))
    stats = rm.collect(conn, fetcher(fake)[0], c, rm.load_fips(), AS_OF + timedelta(days=1))
    assert stats.attempted[0] == "UA"


def test_window_refills_rotation_gaps_within_limits(conn):
    fips_end = AS_OF
    assert rm.window_start(conn, "UA", fips_end) == AS_OF - timedelta(days=7)
    assert rm.window_start(conn, "UA", fips_end, backfill=500) == AS_OF - timedelta(days=89)
    geo_metrics.record(conn, AS_OF - timedelta(days=20), "UA", "conflict_volume", 1.0)
    assert rm.window_start(conn, "UA", fips_end) == AS_OF - timedelta(days=20)


def test_shipped_config_loads_cleanly():
    c = rm.load_config()
    assert not c.warnings
    assert len(c.countries) <= c.max_countries_per_run  # default list refreshes fully every day
    assert c.per_host_delay >= 5.0
    assert set(c.weights) == set(rm.SUBSCORES)


def test_rate_limit_stops_the_run_and_keeps_rotation_slots(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    inner = FakeGdelt()

    def transport(req):
        if "sourcecountry%3ARS" in req.url or "sourcecountry:RS" in req.url:
            return TransportResponse(429, {}, b"slow down")
        return inner(req)

    c = cfg(countries=["DE", "RU", "UA"])
    stats = rm.collect(conn, fetcher(transport)[0], c, rm.load_fips(), AS_OF)
    assert stats.attempted == ["DE", "RU"] and stats.ok == ["DE"] and stats.rate_limited
    assert rm.pick_countries(conn, c)[:2] == ["RU", "UA"]  # not marked fetched


# -- review round 2 --------------------------------------------------------------

def test_zscore_std_floor_keeps_a_flat_baseline_finite():
    assert geo_metrics.zscore(5.0, [0.0] * 30, min_std=0.5) == pytest.approx(10.0)
    assert geo_metrics.zscore(12.0, [10.0] * 30, min_rel_std=0.1) == pytest.approx(2.0)
    assert geo_metrics.zscore(4.0, [1.0, 3.0], min_std=0.5) == pytest.approx(2.0)  # real std wins


def test_spike_after_a_quiet_month_is_the_biggest_mover(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fill(conn, "ML", "conflict_volume", [0.0] * 30 + [5.0])       # flat, then a spike
    fill(conn, "UA", "conflict_volume", alternating(30) + [3.0])  # z = +1
    rows = rm.load_rows(conn, AS_OF - timedelta(days=60))
    m = rm.score_country("ML", rows["ML"], AS_OF, cfg())
    assert m.metrics["conflict_volume"].capped and m.composite == pytest.approx(rm.Z_CAP)
    report, summary = rm.run_once(conn, cfg(countries=["ML", "UA"]), {"ML": "ML", "UA": "UP"}, fetcher=None)
    html = report.render()
    assert summary["scored"] == 2
    movers = html[html.index("Top movers"):]
    assert movers.index("Mali") < movers.index("Ukraine")
    assert "conflict +10.00 (capped)" in html


def test_a_few_missing_baseline_days_still_score_but_many_are_a_gap(conn):
    values = {AS_OF - timedelta(days=i): (1.0 if i % 2 else 3.0) for i in range(0, 31)}
    values[AS_OF] = 4.0
    del values[AS_OF - timedelta(days=15)]
    assert rm.metric_z(values, AS_OF, cfg(), "conflict_volume").status == "ok"
    for i in range(2, 9):
        values.pop(AS_OF - timedelta(days=i))
    m = rm.metric_z(values, AS_OF, cfg(max_missing_days=5), "conflict_volume")
    assert m.status == "gap" and m.n == 22


def test_every_country_with_data_appears_in_the_report(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fill(conn, "UA", "conflict_volume", alternating(30) + [4.0])                  # scored
    fill(conn, "DE", "conflict_volume", [1.0] * 10)                               # building
    gap = alternating(31)
    fill(conn, "FR", "conflict_volume", gap)
    for i in range(3, 13):
        conn.execute("DELETE FROM geo_metrics WHERE country='FR' AND date=?",
                     ((AS_OF - timedelta(days=i)).isoformat(),))
    fill(conn, "JP", "hazard", alternating(31), end=AS_OF - timedelta(days=20))   # stale
    report, summary = rm.run_once(conn, cfg(countries=["UA"]), {"UA": "UP"}, fetcher=None)
    html = report.render()
    assert summary["scored"] == 1 and summary["unscored"] == 3
    assert "Not scored" in html
    assert "conflict_volume baseline building 9/30" in html
    assert "conflict_volume gap: 10 of 30 baseline days missing" in html
    assert "hazard stale (last 2026-09-04)" in html


def test_unrecognized_gdelt_body_is_an_error_not_a_quiet_day():
    assert rm.parse_timeline({}) == {}
    assert rm.parse_timeline({"timeline": [{"data": []}]}) == {}
    assert rm.parse_timeline({"timeline": [{"data": [{"date": "2026-09-01", "value": 2}]}]}) ==         {date(2026, 9, 1): 2.0}
    for bad in ({"error": "Your query was too short"}, {"timeline": None}, {"timeline": [{"data": None}]},
                {"timeline": [{"data": [{"date": "x", "value": "nan"}, {"date": "20260901", "value": True}]}]}):
        with pytest.raises(rm.UnrecognizedResponse):
            rm.parse_timeline(bad)


def test_odd_200_response_is_a_recorded_error_and_not_stamped(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    inner = FakeGdelt()

    def transport(req):
        if "sourcecountry%3AGM" in req.url:
            return TransportResponse(200, {"content-type": "application/json"}, b'{"status": "maintenance"}')
        return inner(req)

    c = cfg(countries=["DE", "UA"], max_countries_per_run=1)
    stats = rm.collect(conn, fetcher(transport)[0], c, rm.load_fips(), AS_OF)
    assert stats.attempted == ["DE"] and stats.ok == [] and "unrecognized GDELT response shape" in stats.errors[0][1]
    assert rm.pick_countries(conn, c) == ["DE"]  # a failure keeps its place at the front
    report = rm.build_report([], {}, AS_OF, c, stats)
    assert "Germany (DE): UnrecognizedResponse" in report.render()


def test_stale_subscores_show_their_age(conn):
    fill(conn, "UA", "conflict_volume", alternating(30) + [4.0], end=AS_OF - timedelta(days=6))
    fill(conn, "UA", "hazard", alternating(30) + [3.0])
    rows = rm.load_rows(conn, AS_OF - timedelta(days=90))
    s = rm.score_country("UA", rows["UA"], AS_OF, cfg())
    assert rm.parts_note(s, cfg(), AS_OF) == "conflict +2.00 (6d old) · hazard +1.00 · markets —"


def test_region_tiles_show_mean_subscores(conn):
    fill(conn, "UA", "conflict_volume", alternating(30) + [4.0])
    fill(conn, "DE", "conflict_volume", alternating(30) + [2.0])
    fill(conn, "DE", "hazard", alternating(30) + [3.0])
    rows = rm.load_rows(conn, AS_OF - timedelta(days=90))
    scores = [rm.score_country(c, rows[c], AS_OF, cfg()) for c in ("UA", "DE")]
    (tile,) = rm.region_tiles(scores, rows, AS_OF, cfg())
    assert tile.label == "Europe"
    assert tile.note == "conflict +1.00 (n=2) · hazard +1.00 (n=1) · markets — · 2 countries"


def test_method_section_explains_press_and_palestine_nuances(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    html = rm.run_once(conn, cfg(), {"UA": "UP", "DE": "GM"}, fetcher=None)[0].render()
    assert "foreign war" in html and "West Bank" in html and "Gaza" in html


# -- review round 3 --------------------------------------------------------------

def _series(base, today, day=AS_OF):
    """base oldest -> newest over the 30 days before `day`, then `today` on it."""
    v = {day - timedelta(days=len(base) - i): x for i, x in enumerate(base)}
    v[day] = today
    return v


@pytest.mark.parametrize("mean", [0.0003, 0.003, 0.03])
def test_volume_floor_keeps_realistic_gdelt_magnitudes_meaningful(mean):
    """TimelineVol is a share-of-coverage percentage, often 0.0003-0.01: the
    std floor must not flatten a real spike or inflate ordinary noise."""
    import random
    c = cfg()
    noisy = [mean * (0.7 if i % 2 else 1.3) for i in range(30)]           # +-30%, std = 0.3 x mean
    rng = random.Random(1)
    gauss = [max(0.0, rng.gauss(mean, mean * 0.3)) for _ in range(30)]
    for base in (noisy, gauss):
        for mult in (3, 10):
            m = rm.metric_z(_series(base, mean * mult), AS_OF, c, "conflict_volume")
            assert m.status == "ok" and (m.capped or m.z >= 3), (mean, mult, m)
        for today in (mean * 1.3, mean * 0.7, mean):
            m = rm.metric_z(_series(noisy, today), AS_OF, c, "conflict_volume")
            assert abs(m.z) < 2, (mean, today, m)
    shipped = rm.load_config()  # the shipped floor, not just the default
    m = rm.metric_z(_series(noisy, mean * 3), AS_OF, shipped, "conflict_volume")
    assert m.z >= 3


def test_all_zero_month_then_realistic_coverage_scores_high():
    m = rm.metric_z(_series([0.0] * 30, 0.003), AS_OF, cfg(), "conflict_volume")
    assert m.status == "ok" and m.capped and m.z == rm.Z_CAP
    m = rm.metric_z(_series([0.0] * 30, 0.0), AS_OF, cfg(), "conflict_volume")
    assert m.z == 0.0


class Scripted:
    """get_json stand-in: per-FIPS canned answers; `raise` = a fetch error."""

    def __init__(self, fips, answers, default=None):
        self.by_fips = {fips[c]: a for c, a in answers.items()}
        self.default = default
        self.calls = []

    def get_json(self, url, **kw):
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        code = q["query"].rsplit("sourcecountry:", 1)[1]
        self.calls.append(code)
        a = self.by_fips.get(code, self.default)
        if a == "raise":
            raise rm.UnrecognizedResponse("unrecognized GDELT response shape (test)")
        if a == "series":
            end = datetime.strptime(q["enddatetime"], "%Y%m%d%H%M%S").date()
            return {"timeline": [{"data": [{"date": f"{end:%Y%m%d}", "value": 0.002}]}]}
        return a


def test_permanent_failures_do_not_starve_the_rotation(conn, monkeypatch):
    fips = rm.load_fips()
    countries = sorted(c for c in fips if rm.geo.country(c))[:30]
    failing = countries[:12]                # sort first alphabetically: worst case
    c = cfg(countries=countries, max_countries_per_run=10)
    fake = Scripted(fips, {k: "raise" for k in failing}, default="series")
    ok_runs: dict[str, list[int]] = {}
    for run in range(12):
        day = AS_OF + timedelta(days=run)
        monkeypatch.setattr(rm, "_now", lambda day=day: datetime(day.year, day.month, day.day, 6, tzinfo=timezone.utc))
        stats = rm.collect(conn, fake, c, fips, day)
        for k in stats.ok:
            ok_runs.setdefault(k, []).append(run)
    healthy = [k for k in countries if k not in failing]
    assert set(ok_runs) == set(healthy)                          # every healthy one got fetched ...
    assert max(runs[0] for runs in ok_runs.values()) <= 6        # ... within a bounded number of runs
    assert all(len(runs) >= 2 for runs in ok_runs.values())      # ... and keeps rotating
    st = rm.fetch_state(conn)[failing[0]]
    assert st["fails"] >= rm.FAIL_BACKOFF_AFTER and st["fail_since"] == AS_OF.isoformat() and st["ok"] is None


def test_first_failures_still_retry_first_and_legacy_state_reads(conn, monkeypatch):
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    conn.execute("INSERT INTO kv(ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
                 (rm.KV_NS, "UA", json.dumps("2026-09-20"), NOW.isoformat()))  # older format
    assert rm.fetch_state(conn)["UA"]["ok"] == "2026-09-20"
    fips = rm.load_fips()
    c = cfg(countries=["DE", "UA"], max_countries_per_run=1)
    rm.collect(conn, Scripted(fips, {"DE": "raise"}), c, fips, AS_OF)
    assert rm.pick_countries(conn, c) == ["DE"]  # 1 failure < FAIL_BACKOFF_AFTER: retried first


def test_empty_timeline_is_no_coverage_not_an_error(conn, monkeypatch):
    assert rm.parse_timeline({"timeline": []}) == {}
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fips = rm.load_fips()
    c = cfg(countries=["DE", "UA", "FR"])
    fake = Scripted(fips, {"DE": {"timeline": []}, "FR": {}, "UA": "raise"})
    stats = rm.collect(conn, fake, c, fips, AS_OF)
    assert stats.no_coverage == ["DE", "FR"] and [e[0] for e in stats.errors] == ["UA"]
    assert set(stats.ok) == {"DE", "FR"}
    assert rm.pick_countries(conn, c)[0] == "UA"   # the empty answers stamped the rotation
    html = rm.build_report([], {}, AS_OF, c, stats, state=rm.fetch_state(conn)).render()
    fetch_errors = html[html.index("fetch errors"):]
    assert "Ukraine (UA): UnrecognizedResponse" in html
    assert "no coverage returned" in html and "Germany (DE), France (FR)" in html
    assert "Germany (DE)" not in fetch_errors[:fetch_errors.index("no coverage returned")]


def test_not_scored_lists_configured_countries_without_any_data(conn, monkeypatch):
    fips = rm.load_fips()
    monkeypatch.setattr(rm, "_now", lambda: NOW - timedelta(days=2))
    rm.collect(conn, Scripted(fips, {"FR": "raise", "DE": {"timeline": []}}), cfg(countries=["FR", "DE"]),
               fips, AS_OF - timedelta(days=2))
    monkeypatch.setattr(rm, "_now", lambda: NOW)
    fill(conn, "UA", "conflict_volume", alternating(30) + [4.0])
    c = cfg(countries=["UA", "FR", "DE", "JP"])
    report, summary = rm.run_once(conn, c, fips, fetcher=None)
    html = report.render()
    assert summary["scored"] == 1 and summary["no_data"] == 3
    table = html[html.index("Not scored</h"):]
    assert "Japan" in table and "not fetched yet" in table
    assert "France" in table and "fetch failing since 2026-09-22" in table
    assert "Germany" in table and "no coverage returned" in table


# ---------------------------------------------------------------------------
# Biggest movers (prediction markets, #60)
# ---------------------------------------------------------------------------

def test_market_movers_section_says_no_data_yet_with_no_conn():
    html = rm.build_report([], {}, AS_OF, cfg(), None).render()
    assert "Biggest movers (prediction markets)" in html
    assert "No market data yet" in html


def test_market_movers_section_says_no_data_yet_when_conn_is_empty(conn):
    html = rm.build_report([], {}, AS_OF, cfg(), None, conn=conn).render()
    assert "Biggest movers (prediction markets)" in html
    assert "No market data yet" in html


def test_market_movers_section_ranks_by_abs_24h_move_and_discloses(conn):
    def snap(day, market_id, question, prob, url=None):
        mkt.record_snapshots(
            conn, [mkt.Market(source="polymarket", market_id=market_id, question=question,
                               url=url, probability=prob)],
            day, fetched_at="t",
        )

    snap("2026-09-24", "m1", "Will Ukraine hold elections?", 0.40)
    snap("2026-09-25", "m1", "Will Ukraine hold elections?", 0.55, url="https://polymarket.com/m1")
    snap("2026-09-24", "m2", "Will France do X?", 0.50)
    snap("2026-09-25", "m2", "Will France do X?", 0.52, url="https://polymarket.com/m2")

    html = rm.build_report([], {}, AS_OF, cfg(), None, conn=conn).render()
    assert "Biggest movers (prediction markets)" in html
    assert "not a forecast by this tool" in html
    assert 'href="https://polymarket.com/m1"' in html
    idx1 = html.index("Will Ukraine hold elections")
    idx2 = html.index("Will France do X")
    assert idx1 < idx2  # 0.15 move ranked above 0.02
