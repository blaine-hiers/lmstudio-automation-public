"""Geopolitical risk map: a per-country composite stress index against each
country's own 30-day baseline, always shown next to its sub-scores.

Sub-scores (each a z-score vs the trailing baseline):
  conflict -- GDELT conflict-theme coverage: volume (up = worse) and tone
              (down = worse), averaged
  hazard   -- `geo_metrics` rows with metric "hazard" (disaster fusion)
  markets  -- `geo_metrics` rows with metric "markets" (prediction markets)
The composite is their weighted mean over the ones that exist (weights in
config/risk_map.toml); a missing sub-score is shown as missing, never as 0.

Source: the GDELT DOC 2.0 API (free, no key), modes TimelineVol and
TimelineTone, query `(theme:A OR theme:B ...) sourcecountry:<FIPS>` -- i.e.
conflict-themed coverage by the country's own press. Two small JSON requests
per country at 1 request / 5 s; at most `max_countries_per_run` countries a
run (stalest first, so a longer list rotates). Writes only data/ and output/.

    python -m tasks.risk_map                 # fetch yesterday's data + report
    python -m tasks.risk_map --backfill 30   # fill the baseline from GDELT
    python -m tasks.risk_map --no-fetch      # report from stored data only
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from lmsa import Report, Store, Tile, geo, geo_metrics
from lmsa.report import Link
from lmsa.web import Fetcher, HTTPStatusError

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/risk_map.toml"]}

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "risk_map.toml"
FIPS_TABLE = ROOT / "lmsa" / "geo_data" / "fips10_4.json"
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "lmsa-risk-map/0.1 (personal research; daily, 1 req/5s)"
KV_NS = "risk_map.fetched"

MIN_WINDOW_DAYS = 8    # GDELT returns daily resolution only for spans over a week
MAX_WINDOW_DAYS = 90   # DOC API start dates must be within the last 3 months
SUBSCORES = ("conflict", "hazard", "markets")
METRICS = ("conflict_volume", "conflict_tone", "hazard", "markets")
Z_CAP = 10.0          # |z| is clipped here so one flat-baseline spike can't wash out the map
# Absolute std floor per metric (in the metric's own units). It only has to
# keep a flat baseline finite, so it sits well below the metric's real
# day-to-day noise: GDELT TimelineVol is a share-of-coverage percentage,
# often 0.0003-0.01 for a country, so its floor is 1e-5 -- an all-zero month
# followed by any real coverage (>= 1e-4) scores at the cap. Larger baselines
# are floored relatively (Config.min_rel_std x |mean|) instead.
DEFAULT_MIN_STD = {"conflict_volume": 1e-5, "conflict_tone": 0.25, "hazard": 0.1, "markets": 0.01}
# A country whose fetch has failed this many runs in a row stops jumping the
# rotation queue: it is retried once per rotation instead, behind the stalest
# successful country, so permanent failures can't starve the rest.
FAIL_BACKOFF_AFTER = 3
TOP_MOVERS = 20
TOP_SPARKS = 10
SPARK_DAYS = 30
DASH = "—"


def _now() -> datetime:
    """Current UTC time. Tests monkeypatch this."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    baseline_days: int = 30
    max_age_days: int = 7
    max_missing_days: int = 5   # baseline days a metric may miss and still score
    min_std: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIN_STD))
    min_rel_std: float = 0.05   # std floor as a fraction of |baseline mean|
    weights: dict[str, float] = field(default_factory=lambda: {"conflict": 0.5, "hazard": 0.3, "markets": 0.2})
    per_host_delay: float = 5.0
    max_countries_per_run: int = 60
    themes: list[str] = field(default_factory=lambda: ["ARMEDCONFLICT"])
    countries: list[str] = field(default_factory=list)  # alpha-2 with a FIPS code
    warnings: list[str] = field(default_factory=list)


def load_fips(path: Path = FIPS_TABLE) -> dict[str, str]:
    """ISO alpha-2 -> FIPS 10-4 (GDELT's country codes)."""
    return dict(json.loads(path.read_text(encoding="utf-8"))["codes"])


def load_config(path: Path = CONFIG, fips: dict[str, str] | None = None) -> Config:
    fips = fips if fips is not None else load_fips()
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    cfg = Config()
    base = data.get("baseline", {})
    cfg.baseline_days = max(2, int(base.get("days", cfg.baseline_days)))
    cfg.max_age_days = max(0, int(base.get("max_age_days", cfg.max_age_days)))
    cfg.max_missing_days = min(max(0, int(base.get("max_missing_days", cfg.max_missing_days))),
                               cfg.baseline_days - 1)
    cfg.min_rel_std = max(0.0, float(base.get("min_rel_std", cfg.min_rel_std)))
    for metric, v in (base.get("min_std") or {}).items():
        cfg.min_std[str(metric)] = max(0.0, float(v))
    weights = data.get("weights", {})
    for name in weights:
        if name not in SUBSCORES:
            cfg.warnings.append(f"unknown weight {name!r} ignored")
    cfg.weights = {s: max(0.0, float(weights.get(s, cfg.weights[s]))) for s in SUBSCORES}
    g = data.get("gdelt", {})
    cfg.per_host_delay = max(5.0, float(g.get("per_host_delay", cfg.per_host_delay)))
    cfg.max_countries_per_run = max(0, int(g.get("max_countries_per_run", cfg.max_countries_per_run)))
    themes = [str(t).strip().upper() for t in g.get("themes", cfg.themes) if str(t).strip()]
    cfg.themes = [t for t in themes if t.replace("_", "").isalnum()] or ["ARMEDCONFLICT"]
    wanted = g.get("countries") or sorted(fips)
    seen: set[str] = set()
    for code in wanted:
        c = geo.country(str(code))
        if c is None:
            cfg.warnings.append(f"unknown country code {code!r} ignored")
        elif c.alpha2 not in fips:
            cfg.warnings.append(f"{c.alpha2} has no FIPS 10-4 code; GDELT can't be queried for it")
        elif c.alpha2 not in seen:
            seen.add(c.alpha2)
            cfg.countries.append(c.alpha2)
    return cfg


# ---------------------------------------------------------------------------
# GDELT fetch
# ---------------------------------------------------------------------------

def gdelt_query(themes: list[str], fips_code: str) -> str:
    terms = [f"theme:{t}" for t in themes]
    clause = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    return f"{clause} sourcecountry:{fips_code}"


def gdelt_url(query: str, mode: str, start: date, end: date) -> str:
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": f"{start:%Y%m%d}000000",
        "enddatetime": f"{end:%Y%m%d}235959",
    }
    return f"{GDELT_DOC}?{urlencode(params)}"


class UnrecognizedResponse(ValueError):
    """A GDELT answer that is not a timeline we understand."""


def _point_day(raw: object) -> date:
    text = str(raw).strip().replace("-", "")
    return datetime.strptime(text[:8], "%Y%m%d").date()


def parse_timeline(data: object) -> dict[date, float]:
    """First series of a DOC API timeline -> {UTC day: mean value}.

    Sub-daily points are averaged per day and malformed points skipped.
    `{}` (GDELT's "no matches"), `{"timeline": []}` and series whose data
    lists are empty give {} -- a successful fetch with no coverage.
    Anything else that yields no usable point -- a missing/odd `timeline`,
    points that are all junk -- raises UnrecognizedResponse, so a changed or
    broken response is reported instead of recorded as a quiet day."""
    if not isinstance(data, dict):
        raise UnrecognizedResponse("unrecognized GDELT response shape (not a JSON object)")
    if not data:
        return {}
    timeline = data.get("timeline")
    if timeline == []:
        return {}
    series = [t for t in timeline if isinstance(t, dict)] if isinstance(timeline, list) else []
    if not series or not isinstance(series[0].get("data"), list):
        raise UnrecognizedResponse("unrecognized GDELT response shape (no timeline series)")
    points = series[0]["data"]
    sums: dict[date, list[float]] = {}
    for p in points:
        if not isinstance(p, dict) or isinstance(p.get("value"), bool):
            continue
        try:
            day = _point_day(p.get("date", ""))
            value = float(p.get("value"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            sums.setdefault(day, []).append(value)
    if points and not sums:
        raise UnrecognizedResponse(f"unrecognized GDELT response shape ({len(points)} points, none usable)")
    return {d: sum(v) / len(v) for d, v in sums.items()}


def fetch_country(fetcher: Fetcher, themes: list[str], fips_code: str,
                  start: date, end: date) -> dict[str, dict[date, float]]:
    """Two DOC API calls: conflict_volume (% of all monitored coverage) and
    conflict_tone (average tone). Raises on any fetch/parse failure."""
    query = gdelt_query(themes, fips_code)
    out: dict[str, dict[date, float]] = {}
    for metric, mode in (("conflict_volume", "timelinevol"), ("conflict_tone", "timelinetone")):
        series = parse_timeline(fetcher.get_json(gdelt_url(query, mode, start, end)))
        out[metric] = {d: v for d, v in series.items() if start <= d <= end}
    return out


def fetch_state(conn: sqlite3.Connection) -> dict[str, dict]:
    """Country -> rotation record: {"ok": last successful fetch day or None,
    "tried": last attempt day, "fails": consecutive failures, "fail_since":
    first day of the current failure streak, "empty": the last successful
    fetch returned no coverage}. Days are ISO strings. A bare date string
    (the older format) is read as a success on that day."""
    out: dict[str, dict] = {}
    for key, raw in conn.execute("SELECT key, value_json FROM kv WHERE ns=?", (KV_NS,)):
        try:
            v = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(v, str):
            v = {"ok": v, "tried": v}
        if isinstance(v, dict):
            try:
                fails = max(0, int(v.get("fails") or 0))
            except (TypeError, ValueError):
                fails = 0
            out[key] = {"ok": v.get("ok"), "tried": v.get("tried") or v.get("ok"),
                        "fails": fails, "fail_since": v.get("fail_since"),
                        "empty": bool(v.get("empty"))}
    return out


def _rotation_key(state: dict | None, country: str) -> tuple[str, str, str]:
    if state is None:  # never attempted: first of all
        return ("", "", country)
    tried = str(state["tried"] or "")
    if state["fails"] >= FAIL_BACKOFF_AFTER:  # backed off: queue by the last attempt
        return (tried, tried, country)
    return (str(state["ok"] or ""), "", country)  # retried soon, like never-fetched


def pick_countries(conn: sqlite3.Connection, cfg: Config) -> list[str]:
    """Up to `max_countries_per_run` configured countries: never-attempted
    ones first, then least recently fetched *successfully* (a failing country
    counts from its last success, so it is retried soon) -- until it has
    failed FAIL_BACKOFF_AFTER runs in a row, after which it queues by its last
    attempt, i.e. behind every country that succeeded before that."""
    state = fetch_state(conn)
    order = sorted(cfg.countries, key=lambda c: _rotation_key(state.get(c), c))
    return order[: cfg.max_countries_per_run]


def _save_state(conn: sqlite3.Connection, country: str, state: dict) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO kv(ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
            (KV_NS, country, json.dumps(state), _now().isoformat()),
        )


def window_start(conn: sqlite3.Connection, country: str, end: date, backfill: int = 0) -> date:
    """Refetch from the day after the newest stored value (so rotation gaps
    fill in), at least MIN_WINDOW_DAYS and at most MAX_WINDOW_DAYS back."""
    newest = geo_metrics.latest_date(conn, "conflict_volume", country)
    days = (end - newest).days + 1 if newest else MIN_WINDOW_DAYS
    days = min(max(days, MIN_WINDOW_DAYS, backfill), MAX_WINDOW_DAYS)
    return end - timedelta(days=days - 1)


@dataclass
class FetchStats:
    configured: int = 0
    attempted: list[str] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    no_coverage: list[str] = field(default_factory=list)  # fetched fine, GDELT had no points
    rows: int = 0
    rate_limited: bool = False


def collect(conn: sqlite3.Connection, fetcher: Fetcher, cfg: Config, fips: dict[str, str],
            end: date, *, backfill: int = 0) -> FetchStats:
    """Fetch and store GDELT metrics for this run's countries. One country's
    failure (network, robots, bad JSON, anything) is recorded and skipped."""
    stats = FetchStats(configured=len(cfg.countries))
    state = fetch_state(conn)
    today = end.isoformat()
    for country in pick_countries(conn, cfg):
        stats.attempted.append(country)
        prev = state.get(country) or {"ok": None, "tried": None, "fails": 0, "fail_since": None, "empty": False}
        try:
            start = window_start(conn, country, end, backfill)
            data = fetch_country(fetcher, cfg.themes, fips[country], start, end)
            rows = [(d, country, metric, v) for metric, s in data.items() for d, v in s.items()]
            stats.rows += geo_metrics.record_many(conn, rows)
            stats.ok.append(country)
            if not rows:
                stats.no_coverage.append(country)
            # a success (with or without coverage) moves it to the back of the rotation
            _save_state(conn, country, {"ok": today, "tried": today, "fails": 0,
                                        "fail_since": None, "empty": not rows})
        except Exception as exc:  # isolate: one country never kills the run
            stats.errors.append((country, f"{type(exc).__name__}: {exc}"[:300]))
            if isinstance(exc, HTTPStatusError) and exc.status == 429:
                # GDELT is throttling us: stop for this run instead of hammering it.
                # This country and the rest keep their rotation slot for next time.
                stats.rate_limited = True
                break
            try:  # record the attempt so repeated failures back off (see pick_countries)
                _save_state(conn, country, {**prev, "tried": today, "fails": prev["fails"] + 1,
                                            "fail_since": prev["fail_since"] or today})
            except Exception:  # the fetch error above is what gets reported
                pass
    return stats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class MetricZ:
    status: str            # ok | building | gap | stale | missing
    z: float | None = None
    value: float | None = None
    day: date | None = None
    n: int = 0             # baseline days with a value
    capped: bool = False   # |z| was clipped to Z_CAP


@dataclass
class Score:
    country: str
    metrics: dict[str, MetricZ]
    subscores: dict[str, float | None]
    composite: float | None
    building: int | None = None  # best baseline progress while nothing scores
    sub_days: dict[str, date | None] = field(default_factory=dict)

    @property
    def as_of(self) -> date | None:
        days = [m.day for m in self.metrics.values() if m.status == "ok" and m.day]
        return max(days) if days else None

    @property
    def has_data(self) -> bool:
        return any(m.status != "missing" for m in self.metrics.values())


Rows = dict[str, dict[str, dict[date, float]]]  # country -> metric -> day -> value


def load_rows(conn: sqlite3.Connection, since: date) -> Rows:
    rows: Rows = {}
    for d, c, m, v in conn.execute(
        "SELECT date, country, metric, value FROM geo_metrics WHERE date >= ?", (since.isoformat(),)
    ):
        if v is None:
            continue
        try:
            rows.setdefault(c, {}).setdefault(m, {})[date.fromisoformat(d)] = float(v)
        except ValueError:
            continue
    return rows


def metric_z(values: dict[date, float], day: date, cfg: Config, metric: str = "") -> MetricZ:
    """z-score of the value on `day` against the `baseline_days` before it.

    building -- the metric's history is shorter than the baseline window
    gap      -- history is long enough but more than `max_missing_days` of
                the window have no value
    ok       -- scored; the std is floored (DEFAULT_MIN_STD / min_rel_std)
                so a flat baseline still gives a finite score, clipped to
                +-Z_CAP."""
    if day not in values:
        return MetricZ("missing")
    value = values[day]
    window = [day - timedelta(days=i) for i in range(1, cfg.baseline_days + 1)]
    base = [values[d] for d in window if d in values]
    if min(values) > window[-1]:
        return MetricZ("building", value=value, day=day, n=len(base))
    if cfg.baseline_days - len(base) > cfg.max_missing_days:
        return MetricZ("gap", value=value, day=day, n=len(base))
    z = geo_metrics.zscore(value, base, min_std=cfg.min_std.get(metric, 0.01), min_rel_std=cfg.min_rel_std)
    if z is None:  # only with an empty baseline, which the checks above rule out
        return MetricZ("gap", value=value, day=day, n=len(base))
    capped = abs(z) > Z_CAP
    return MetricZ("ok", z=max(-Z_CAP, min(Z_CAP, z)), value=value, day=day, n=len(base), capped=capped)


def metric_at(values: dict[date, float], as_of: date, cfg: Config, metric: str = "",
              *, exact: bool = False) -> MetricZ:
    """Score the newest value no older than `max_age_days` (or exactly as_of).
    Older data is `stale` (day = its newest date), never silently blended."""
    for age in range(0, 1 if exact else cfg.max_age_days + 1):
        day = as_of - timedelta(days=age)
        if day in values:
            return metric_z(values, day, cfg, metric)
    older = [d for d in values if d <= as_of]
    if older and not exact:
        return MetricZ("stale", day=max(older))
    return MetricZ("missing")


def score_country(country: str, rows: dict[str, dict[date, float]], as_of: date, cfg: Config,
                  *, exact: bool = False) -> Score:
    ms = {m: metric_at(rows.get(m, {}), as_of, cfg, m, exact=exact) for m in METRICS}
    parts, part_days = [], []
    if ms["conflict_volume"].status == "ok":
        parts.append(ms["conflict_volume"].z)
        part_days.append(ms["conflict_volume"].day)
    if ms["conflict_tone"].status == "ok":
        parts.append(-ms["conflict_tone"].z)  # more negative tone = more stress
        part_days.append(ms["conflict_tone"].day)
    subs: dict[str, float | None] = {
        "conflict": sum(parts) / len(parts) if parts else None,
        "hazard": ms["hazard"].z if ms["hazard"].status == "ok" else None,
        "markets": ms["markets"].z if ms["markets"].status == "ok" else None,
    }
    sub_days = {
        "conflict": min(part_days) if part_days else None,  # the oldest part sets the age
        "hazard": ms["hazard"].day if subs["hazard"] is not None else None,
        "markets": ms["markets"].day if subs["markets"] is not None else None,
    }
    composite = blend(subs, cfg.weights)
    building = None
    if composite is None:
        pending = [m.n for m in ms.values() if m.status == "building"]
        building = max(pending) if pending else None
    return Score(country, ms, subs, composite, building, sub_days)


def blend(subs: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Weighted mean over the sub-scores that exist and have weight > 0.
    A missing sub-score drops out (its weight too) -- it is not a zero."""
    used = [(weights.get(k, 0.0), v) for k, v in subs.items() if v is not None and weights.get(k, 0.0) > 0]
    total = sum(w for w, _ in used)
    return sum(w * v for w, v in used) / total if total else None


def unscored_reason(score: Score, cfg: Config) -> str:
    """Why a country with data has no composite, metric by metric."""
    out = []
    for name, m in score.metrics.items():
        if m.status == "building":
            out.append(f"{name} baseline building {m.n}/{cfg.baseline_days}")
        elif m.status == "gap":
            out.append(f"{name} gap: {cfg.baseline_days - m.n} of {cfg.baseline_days} baseline days missing")
        elif m.status == "stale":
            out.append(f"{name} stale (last {m.day:%Y-%m-%d})")
        elif m.status == "ok":
            out.append(f"{name} has no weight")
    return "; ".join(out) or "no signal"


def no_data_reason(country: str, state: dict | None) -> str:
    """Why a configured country has no data at all, from its rotation record."""
    if state is None:
        return "not fetched yet"
    if state["fails"]:
        return f"fetch failing since {state['fail_since']} ({state['fails']} attempts)"
    if state["empty"]:
        return f"no coverage returned (last checked {state['tried']})"
    return f"no recent data (last fetched {state['ok']})"


def composite_series(rows: dict[str, dict[date, float]], country: str, as_of: date, cfg: Config,
                     days: int = SPARK_DAYS) -> list[tuple[date, float]]:
    out = []
    for i in range(days - 1, -1, -1):
        d = as_of - timedelta(days=i)
        s = score_country(country, rows, d, cfg, exact=True)
        if s.composite is not None:
            out.append((d, s.composite))
    return out


def region_series(rows: Rows, countries: list[str], as_of: date, cfg: Config,
                  days: int = SPARK_DAYS) -> dict[str, list[tuple[date, float, int]]]:
    """Region -> [(day, mean composite, n countries)] over the last `days`."""
    by_day: dict[str, dict[date, list[float]]] = {}
    for c in countries:
        region = geo.region_of(c) or "Other"
        for d, v in composite_series(rows.get(c, {}), c, as_of, cfg, days):
            by_day.setdefault(region, {}).setdefault(d, []).append(v)
    return {
        r: [(d, sum(vs) / len(vs), len(vs)) for d, vs in sorted(days_.items())]
        for r, days_ in sorted(by_day.items())
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _z(v: float | None) -> str:
    return f"{v:+.2f}" if v is not None else DASH


TOP_MARKET_MOVERS = 15


def _pct_pt(v: float | None) -> str:
    return f"{v * 100:+.0f}pt" if v is not None else DASH


def market_movers_section(r: Report, conn: sqlite3.Connection | None) -> None:
    """"Biggest movers (prediction markets)" (#60): every snapshotted market
    with a computable 24h move, on its own most recent snapshot day, ranked
    by |24h move|. `conn` is None for callers with no db (tests building a
    report from bare scores) -- then, like an empty table, the section says
    there's no data yet rather than being skipped outright."""
    from lmsa.markets import ODDS_DISCLAIMER, latest_snapshot_day, movement, snapshots_on

    sec = r.section("Biggest movers (prediction markets)", tone="warn")
    day = None if conn is None else latest_snapshot_day(conn)
    if day is None:
        sec.callout("No market data yet: `python -m tasks.markets` hasn't stored a snapshot.",
                    label="no data", tone="muted")
        return
    rows = []
    for snap in snapshots_on(conn, day):
        if snap.probability is None:
            continue
        mv = movement(conn, snap.source, snap.market_id, day)
        if mv.pct_24h is None:
            continue
        rows.append((abs(mv.pct_24h), snap, mv))
    if not rows:
        sec.callout(f"No market data yet with a computable 24h move as of {day}.",
                    label="no data", tone="muted")
        return
    rows.sort(key=lambda x: x[0], reverse=True)
    sec.text(ODDS_DISCLAIMER)
    sec.table(
        ["Market", "Country", "Odds", "24h", "7d"],
        [
            (
                Link(snap.question[:80], snap.url),
                geo.country_name(snap.country) or snap.country if snap.country else DASH,
                f"{mv.probability * 100:.0f}%",
                _pct_pt(mv.pct_24h),
                _pct_pt(mv.pct_7d),
            )
            for _, snap, mv in rows[:TOP_MARKET_MOVERS]
        ],
    )


_RELATED = {"conflict": ("conflict_volume", "conflict_tone"), "hazard": ("hazard",), "markets": ("markets",)}


def _sub_cell(score: Score, name: str, cfg: Config, as_of: date) -> str:
    if cfg.weights.get(name, 0) <= 0:
        return "off"
    related = [score.metrics[m] for m in _RELATED[name]]
    v = score.subscores[name]
    if v is not None:
        text = _z(v)
        if any(m.capped for m in related if m.status == "ok"):
            text += " (capped)"
        day = score.sub_days.get(name)
        if day and day < as_of:
            text += f" ({(as_of - day).days}d old)"
        return text
    statuses = {m.status for m in related}
    if "building" in statuses:
        return f"building {max(m.n for m in related if m.status == 'building')}/{cfg.baseline_days}"
    if "gap" in statuses:
        return "gap"
    if "stale" in statuses:
        return "stale"
    return DASH


def parts_note(score: Score, cfg: Config, as_of: date) -> str:
    return " · ".join(f"{name} {_sub_cell(score, name, cfg, as_of)}" for name in SUBSCORES)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def region_tiles(scored: list[Score], rows: Rows, as_of: date, cfg: Config) -> list[Tile]:
    """One tile per region: mean composite, the mean of each sub-score over the
    countries that have it, and a 30-day sparkline of the mean composite."""
    groups: dict[str, list[Score]] = {}
    for s in scored:
        groups.setdefault(geo.region_of(s.country) or "Other", []).append(s)
    series = region_series(rows, [s.country for s in scored], as_of, cfg)
    tiles = []
    for region in sorted(groups):
        members = groups[region]
        value = _mean([s.composite for s in members])
        parts = []
        for name in SUBSCORES:
            if cfg.weights.get(name, 0) <= 0:
                parts.append(f"{name} off")
                continue
            vals = [s.subscores[name] for s in members if s.subscores[name] is not None]
            parts.append(f"{name} {_z(_mean(vals))}" + (f" (n={len(vals)})" if vals else ""))
        tiles.append(Tile(
            region, _z(value), tone="hot" if value and value > 0 else "info",
            note=" · ".join(parts) + f" · {len(members)} countries",
            spark=[v for _, v, _ in series.get(region, [])],
        ))
    return tiles


def build_report(scores: list[Score], rows: Rows, as_of: date, cfg: Config,
                 stats: FetchStats | None, backfill: int = 0,
                 state: dict[str, dict] | None = None,
                 conn: sqlite3.Connection | None = None) -> Report:
    state = state or {}
    scored = [s for s in scores if s.composite is not None]
    unscored = [s for s in scores if s.composite is None and s.has_data]
    configured = set(cfg.countries)
    no_data = sorted(s.country for s in scores if not s.has_data and s.country in configured)
    building = [s for s in unscored if s.building is not None]
    progress = max((s.building for s in building), default=0)
    days = cfg.baseline_days
    ranked = sorted(scored, key=lambda s: (-abs(s.composite), s.country))

    r = Report(
        "Geopolitical risk map",
        kicker="Daily · GDELT + geo_metrics",
        subtitle=f"Composite z-score per country vs its trailing {days}-day baseline, as of {as_of:%Y-%m-%d} (UTC).",
        sticker=f"{len(scored)} countries scored" if scored else f"baseline building ({progress}/{days} days)",
        sticker_tone="acc" if scored else "warn",
        source="GDELT DOC 2.0 API (TimelineVol, TimelineTone)",
    )
    top = ranked[0] if ranked else None
    r.tiles([
        Tile("Countries scored", len(scored), tone="acc" if scored else "muted"),
        Tile("Not scored", len(unscored) + len(no_data), tone="warn" if unscored or no_data else "muted",
             note=f"{len(building)} building, best {progress}/{days} days" if building else None),
        Tile("Biggest mover", geo.country_name(top.country) if top else DASH,
             delta=_z(top.composite) if top else None, tone="hot" if top and top.composite > 0 else "info",
             note=parts_note(top, cfg, as_of) if top else None),
        Tile("Fetched this run", f"{len(stats.ok)}/{stats.configured}" if stats else "skipped",
             tone="info", note=(", ".join(
                 ([f"{len(stats.errors)} errors"] if stats.errors else [])
                 + ([f"{len(stats.no_coverage)} no coverage"] if stats.no_coverage else [])
             ) or None) if stats else None),
    ])

    if not scored:
        r.section("Baseline building", tone="warn").callout(
            f"baseline building ({progress}/{days} days). Every sub-score is a z-score against the "
            f"country's previous {days} days, so nothing is scored until {days} days of data exist. "
            f"Run `python -m tasks.risk_map --backfill {days + 1}` to fill it from GDELT's timeline mode.",
            label="baseline", tone="warn",
        )
    else:
        weights = ", ".join(f"{k} {w:g}" for k, w in cfg.weights.items())
        r.section("Composite vs baseline", tone="hot").world_map(
            {s.country: s.composite for s in scored}, scale="diverging",
            legend=f"composite z vs {days}-day baseline", label="composite risk delta by country",
        ).text(
            f"Composite = weighted mean of the sub-scores a country has (weights: {weights}); "
            "a missing sub-score is left out, not counted as zero. Positive = more stress than "
            f"the country's own recent normal. Scores are clipped at +-{Z_CAP:g}. Unshaded "
            "countries have no score yet (see Not scored)."
        )
        r.section("Top movers").table(
            ["Country", "Region", "Composite", "Conflict", "Hazard", "Markets",
             "Volume z", "Tone z", "Parts"],
            [
                [
                    geo.country_name(s.country) or s.country,
                    geo.region_of(s.country) or DASH,
                    round(s.composite, 2),
                    *(_sub_cell(s, name, cfg, as_of) for name in SUBSCORES),
                    _z(s.metrics["conflict_volume"].z), _z(s.metrics["conflict_tone"].z),
                    f"{sum(v is not None for v in s.subscores.values())}/{sum(w > 0 for w in cfg.weights.values())}",
                ]
                for s in ranked[:TOP_MOVERS]
            ],
        )
        r.section(f"Top {TOP_SPARKS}: last {SPARK_DAYS} days").tiles(
            Tile(
                geo.country_name(s.country) or s.country, _z(s.composite),
                tone="hot" if s.composite > 0 else "info", note=parts_note(s, cfg, as_of),
                spark=[v for _, v in composite_series(rows.get(s.country, {}), s.country, as_of, cfg)],
            )
            for s in ranked[:TOP_SPARKS]
        )
        r.section("By region: mean composite and sub-scores", tone="info").tiles(
            region_tiles(scored, rows, as_of, cfg)
        )
    if unscored or no_data:
        r.section("Not scored", tone="warn").table(
            ["Country", "Why"],
            [[geo.country_name(s.country) or s.country, unscored_reason(s, cfg)]
             for s in sorted(unscored, key=lambda s: (-(s.building or 0), s.country))]
            + [[geo.country_name(c) or c, no_data_reason(c, state.get(c))] for c in no_data],
        )

    market_movers_section(r, conn)

    method = r.section("Coverage and method", tone="muted")
    lines = [
        "Conflict sub-score: GDELT DOC 2.0 API, `(" + " OR ".join(f"theme:{t}" for t in cfg.themes)
        + ") sourcecountry:<FIPS>` -- conflict-themed coverage by the country's own press. Volume is the "
        "share of all monitored coverage; tone is GDELT's average tone (lower = more negative). "
        "conflict = mean(z volume, -z tone).",
        "Read it as \"what this country's press is saying\", not \"what is happening there\": a "
        "country's outlets covering a foreign war count for that country. Palestine (PS) is "
        "GDELT's West Bank (WE) press only; Gaza (GZ) outlets are not included.",
        f"A metric scores once it has {days} days of history with at most {cfg.max_missing_days} "
        f"of them missing (else \"gap\"); data older than {cfg.max_age_days} days is \"stale\" and "
        "left out; a sub-score older than the report day shows its age. The baseline std is "
        f"floored at the larger of a tiny per-metric minimum (volume: "
        f"{cfg.min_std.get('conflict_volume', 0):g}) and {cfg.min_rel_std:g} x |mean|, so it never "
        "flattens real day-to-day noise, and an all-zero month followed by real coverage scores at "
        "the cap instead of blank.",
        "Hazard and markets sub-scores come from other tasks' geo_metrics rows and appear once they exist.",
    ]
    if stats is not None:
        cycle = math.ceil(stats.configured / cfg.max_countries_per_run) if cfg.max_countries_per_run else 0
        lines.append(
            f"This run fetched {len(stats.ok)} of {stats.configured} configured countries "
            f"(cap {cfg.max_countries_per_run} per run, stalest first; a failed country is retried "
            f"first until it has failed {FAIL_BACKOFF_AFTER} runs in a row, then once per rotation"
            + (f"; a full rotation takes {cycle} runs" if cycle > 1 else "") + ")."
            + (f" Backfill: up to {min(backfill, MAX_WINDOW_DAYS)} days." if backfill else "")
            + (" GDELT answered 429 (rate limited), so the run stopped fetching early; the "
               "skipped countries go first next run." if stats.rate_limited else "")
        )
    else:
        lines.append("No fetch this run (--no-fetch): scored from stored data only.")
    method.text("\n\n".join(lines))
    if stats and stats.errors:
        method.callout(
            "\n\n".join(f"{geo.country_name(c) or c} ({c}): {e}" for c, e in stats.errors),
            label="fetch errors", tone="hot",
        )
    if stats and stats.no_coverage:
        method.callout(
            "GDELT answered with no matching coverage (not an error) for: "
            + ", ".join(f"{geo.country_name(c) or c} ({c})" for c in stats.no_coverage),
            label="no coverage returned", tone="warn",
        )
    if cfg.warnings:
        method.callout("\n\n".join(cfg.warnings), label="config", tone="warn")
    return r


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_once(conn: sqlite3.Connection, cfg: Config, fips: dict[str, str], *,
             fetcher: Fetcher | None, backfill: int = 0) -> tuple[Report, dict]:
    as_of = _now().date() - timedelta(days=1)  # last complete UTC day
    stats = collect(conn, fetcher, cfg, fips, as_of, backfill=backfill) if fetcher is not None else None
    since = as_of - timedelta(days=SPARK_DAYS + cfg.baseline_days + cfg.max_age_days + 1)
    rows = load_rows(conn, since)
    countries = sorted(set(cfg.countries) | set(rows))
    scores = [score_country(c, rows.get(c, {}), as_of, cfg) for c in countries]
    state = fetch_state(conn)
    report = build_report(scores, rows, as_of, cfg, stats, backfill, state, conn=conn)
    scored = [s for s in scores if s.composite is not None]
    summary = {
        "as_of": as_of.isoformat(),
        "scored": len(scored),
        "building": sum(s.composite is None and s.building is not None for s in scores),
        "unscored": sum(s.composite is None and s.has_data for s in scores),
        "no_data": sum(not s.has_data and s.country in set(cfg.countries) for s in scores),
        "fetched": len(stats.ok) if stats else 0,
        "fetch_errors": len(stats.errors) if stats else 0,
        "no_coverage": len(stats.no_coverage) if stats else 0,
        "rows": stats.rows if stats else 0,
        "rate_limited": bool(stats and stats.rate_limited),
    }
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Geopolitical risk map: per-country composite vs 30-day baseline.")
    parser.add_argument("--backfill", type=int, default=0, metavar="N",
                        help="fetch up to N past days (max 90) from GDELT to fill the baseline")
    parser.add_argument("--no-fetch", action="store_true",
                        help="don't call GDELT; report from stored data only")
    args = parser.parse_args()

    # No model calls here, so no lm.ensure_server(): the task is pure data.
    fips = load_fips()
    cfg = load_config(CONFIG, fips)
    fetcher = None if args.no_fetch else Fetcher(USER_AGENT, per_host_delay=cfg.per_host_delay)
    with Store() as store, store.record_run("tasks.risk_map") as run:
        report, run.summary = run_once(store.conn, cfg, fips, fetcher=fetcher, backfill=max(0, args.backfill))
        run.report_path = report.write()
    print(run.report_path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
