"""Daily OSINT movements report: military air traffic + USNI Fleet Tracker (#58).

Aggregates the samples `tasks.osint_air_sample` has been collecting from
adsb.lol's public `/v2/mil` feed into counts by aircraft type class (tanker,
ISR, transport, bomber, fighter -- `config/osint_air.toml`) and region (a
bounding-box table, same file), flags any (class, region) whose count today
is far above its 30-day baseline, and plots activity density on the world
map. Separately, once every ~6 days, it checks USNI News's Fleet and Marine
Tracker RSS for a new article and extracts carrier/ARG/ship positions by
region from it with the model.

The report is aggregate-only by design: it never lists individual
aircraft, tail numbers, or a track. The model's one-paragraph narrative is
written over the aggregate counts, never raw per-aircraft data.

    python -m tasks.osint_movements
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import statistics
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lmsa import LMStudio, Link, Marker, Report, Store, Tile
from lmsa.feeds import FeedItem, fetch_feed
from lmsa.prompts import load as load_prompt
from lmsa.web import Fetcher

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/osint_air.toml"], "model": "chat"}

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "osint_air.toml"

USER_AGENT = "lmsa-osint-movements/1.0 (+public aggregate use only)"
FETCH_TIMEOUT = 15
MAX_RESPONSE_BYTES = 5_000_000
MAX_ARTICLE_CHARS = 6000

USNI_FEED_URL = "https://news.usni.org/category/fleet-tracker/feed"
USNI_MIN_INTERVAL_DAYS = 6  # weekly-ish; see NOTES in the PR for why this is a soft check, not a cadence.

WINDOW_HOURS = 24        # "today" window for the type/region counts
BASELINE_DAYS = 30       # trailing days used for the anomaly baseline
MIN_BASELINE_DAYS = 5    # don't flag anomalies until there's this much history
Z_THRESHOLD = 2.0
MARKER_CAP = 60          # bound the map's SVG size on a very busy day

NARRATIVE_SYSTEM = load_prompt("osint_movements/narrative_system")
USNI_EXTRACT_SYSTEM = load_prompt("osint_movements/usni_extract_system")

USNI_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "group": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["region", "group", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entries"],
    "additionalProperties": False,
}

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def default_fetcher() -> Fetcher:
    """A fresh `lmsa.web.Fetcher` for this task. Tests pass in a fake
    fetcher instead of calling this."""
    return Fetcher(USER_AGENT, timeout=FETCH_TIMEOUT, max_bytes=MAX_RESPONSE_BYTES)


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


def strip_to_text(page_html: str) -> str:
    no_script = _SCRIPT_STYLE.sub(" ", page_html)
    no_tags = _TAG.sub(" ", no_script)
    return _WS.sub(" ", html.unescape(no_tags)).strip()


def truncate(text: str, limit: int = MAX_ARTICLE_CHARS) -> tuple[str, bool]:
    return text[:limit], len(text) > limit


# -- config: type classes + regions -------------------------------------------


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, str], list[Region]]:
    """`{}, []` if the file is missing -- everything then classifies as
    "unclassified" / "Other" rather than the task failing outright."""
    if not path.exists():
        return {}, []
    with path.open("rb") as f:
        data = tomllib.load(f)
    type_class: dict[str, str] = {}
    for cls, codes in (data.get("type_class") or {}).items():
        for code in codes:
            type_class[str(code).strip().upper()] = cls
    regions = [
        Region(
            name=r["name"],
            lat_min=float(r["lat_min"]), lat_max=float(r["lat_max"]),
            lon_min=float(r["lon_min"]), lon_max=float(r["lon_max"]),
        )
        for r in data.get("region", [])
    ]
    return type_class, regions


def classify_type(actype: str | None, type_class: dict[str, str]) -> str:
    if not actype:
        return "unclassified"
    return type_class.get(str(actype).strip().upper(), "unclassified")


def region_for(lat_bucket: int | None, lon_bucket: int | None, regions: list[Region]) -> str:
    if lat_bucket is None or lon_bucket is None:
        return "Other"
    for r in regions:
        if r.lat_min <= lat_bucket <= r.lat_max and r.lon_min <= lon_bucket <= r.lon_max:
            return r.name
    return "Other"


# -- air sample aggregation -----------------------------------------------------


def _day(iso_ts: str) -> str:
    return iso_ts[:10]


def positions_by_day(
    conn: sqlite3.Connection, now: datetime, days: int = BASELINE_DAYS + 1,
) -> dict[str, list[tuple[str, str | None, int | None, int | None]]]:
    """calendar date -> distinct (hex, actype, lat_bucket, lon_bucket) rows
    seen that UTC day. Distinct, not row counts, because the sampler
    records the same aircraft again every tick it's still airborne. An
    aircraft that crossed a region boundary the same day appears more than
    once here *on purpose* (once per distinct position bucket) -- callers
    that need a true per-aircraft total (not per-region) dedup by hex
    themselves; see `class_totals_by_day`."""
    since = (now - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT sampled_at, hex, actype, lat_bucket, lon_bucket FROM osint_air_samples "
        "WHERE sampled_at >= ? ORDER BY sampled_at",
        (since,),
    ).fetchall()
    seen_by_day: dict[str, set[tuple[str, str | None, int | None, int | None]]] = defaultdict(set)
    for sampled_at, hexid, actype, lat_bucket, lon_bucket in rows:
        seen_by_day[_day(sampled_at)].add((hexid, actype, lat_bucket, lon_bucket))
    return {day: list(positions) for day, positions in seen_by_day.items()}


def class_region_counts_by_day(
    positions: dict[str, list[tuple[str, str | None, int | None, int | None]]],
    type_class: dict[str, str], regions: list[Region],
) -> dict[str, Counter]:
    """calendar date -> Counter[(class, region)] of distinct aircraft. Used
    for the type x region matrix and the anomaly baseline; **not** for a
    class or grand total -- summing this across regions for one class would
    double-count an aircraft seen in two regions the same day."""
    counts: dict[str, Counter] = {}
    for day, day_positions in positions.items():
        c: Counter = Counter()
        seen: set[tuple[str, str, str]] = set()
        for hexid, actype, lat_bucket, lon_bucket in day_positions:
            key = (hexid, classify_type(actype, type_class), region_for(lat_bucket, lon_bucket, regions))
            if key in seen:
                continue
            seen.add(key)
            c[key[1:]] += 1
        counts[day] = c
    return counts


def class_totals_by_day(
    positions: dict[str, list[tuple[str, str | None, int | None, int | None]]], type_class: dict[str, str],
) -> dict[str, Counter]:
    """calendar date -> Counter[class] of distinct aircraft (by hex alone,
    deduped *across* regions) -- one airframe seen over two regions the
    same day counts once, toward one class, not once per region."""
    totals: dict[str, Counter] = {}
    for day, day_positions in positions.items():
        by_hex_class: dict[str, str] = {}
        for hexid, actype, _lat_bucket, _lon_bucket in day_positions:
            by_hex_class.setdefault(hexid, classify_type(actype, type_class))
        c: Counter = Counter()
        for cls in by_hex_class.values():
            c[cls] += 1
        totals[day] = c
    return totals


def detect_anomalies(
    daily: dict[str, Counter], today: str, z_threshold: float = Z_THRESHOLD,
    min_baseline_days: int = MIN_BASELINE_DAYS,
) -> list[dict]:
    """(class, region) pairs whose today count is z_threshold-sigma (or
    more) above the mean of every *other* day in `daily` -- population
    stdev, so a single wild day doesn't get compared against itself.
    Skipped when there's under `min_baseline_days` of history, or when the
    baseline has zero variance (a jump from a constant 0 would otherwise
    divide by zero, or "always 3" flag every single non-3 count)."""
    today_counts = daily.get(today, Counter())
    baseline_days = [d for d in daily if d != today]
    keys = set(today_counts) | {k for d in baseline_days for k in daily[d]}
    anomalies = []
    for key in keys:
        values = [daily[d].get(key, 0) for d in baseline_days]
        if len(values) < min_baseline_days:
            continue
        today_v = today_counts.get(key, 0)
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values)
        if stdev <= 0:
            continue
        z = (today_v - mean) / stdev
        if z >= z_threshold:
            cls, reg = key
            anomalies.append({
                "class": cls, "region": reg, "today": today_v,
                "baseline_mean": round(mean, 1), "z": round(z, 2),
            })
    anomalies.sort(key=lambda a: -a["z"])
    return anomalies


def today_positions(conn: sqlite3.Connection, now: datetime) -> list[tuple[str, str | None, int, int]]:
    """Distinct (hex, actype, lat_bucket, lon_bucket) seen in the last
    WINDOW_HOURS -- the raw rows behind both the counts and the map, never
    surfaced to the report as a list."""
    since = (now - timedelta(hours=WINDOW_HOURS)).isoformat()
    rows = conn.execute(
        "SELECT DISTINCT hex, actype, lat_bucket, lon_bucket FROM osint_air_samples WHERE sampled_at >= ?",
        (since,),
    ).fetchall()
    return [tuple(r) for r in rows]


def density_markers(positions: list[tuple[str, str | None, int, int]], cap: int = MARKER_CAP) -> tuple[list[Marker], bool]:
    """One marker per occupied 1-degree cell, sized by how many distinct
    aircraft passed through it today. Returns (markers, truncated)."""
    clusters: Counter = Counter()
    for _hexid, _actype, lat_bucket, lon_bucket in positions:
        clusters[(lat_bucket, lon_bucket)] += 1
    ranked = sorted(clusters.items(), key=lambda kv: -kv[1])
    truncated = len(ranked) > cap
    top = ranked[:cap]
    max_c = max((c for _, c in top), default=1)
    markers = [
        Marker(
            lat=lat_bucket + 0.5, lon=lon_bucket + 0.5,
            label=f"{count} aircraft", tone="hot",
            size=0.5 + 2.5 * (count / max_c),
        )
        for (lat_bucket, lon_bucket), count in top
    ]
    return markers, truncated


# -- USNI Fleet Tracker ---------------------------------------------------------


def latest_feed_entry(fetcher: Fetcher, url: str = USNI_FEED_URL, limit: int = 5) -> FeedItem | None:
    """The first usable entry (title + a real http(s) link) in an RSS/Atom
    feed -- USNI's feed is newest-first, same as every other WordPress feed.
    `lmsa.feeds.parse_feed` already skips titleless entries; `limit` bounds
    how many we look through for one with a usable link. None if none
    qualifies. Raises `FetchError` (network/robots/size) or
    `xml.etree.ElementTree.ParseError` (malformed feed) -- caller isolates."""
    for item in fetch_feed(fetcher, url, source="usni", limit=limit):
        if item.url:
            return item
    return None


def usni_due(conn: sqlite3.Connection, now: datetime, min_interval_days: int = USNI_MIN_INTERVAL_DAYS) -> bool:
    row = conn.execute(
        "SELECT value_json FROM kv WHERE ns='osint_movements' AND key='usni_last_check'"
    ).fetchone()
    if row is None:
        return True
    last_checked = datetime.fromisoformat(json.loads(row[0])["checked_at"])
    return (now - last_checked) >= timedelta(days=min_interval_days)


def _record_usni_check(conn: sqlite3.Connection, now: datetime, article_url: str | None) -> None:
    conn.execute(
        "INSERT INTO kv (ns, key, value_json, updated) VALUES ('osint_movements', 'usni_last_check', ?, ?) "
        "ON CONFLICT(ns, key) DO UPDATE SET value_json=excluded.value_json, updated=excluded.updated",
        (json.dumps({"checked_at": now.isoformat(), "article_url": article_url}), now.isoformat()),
    )
    conn.commit()


def _last_checked_article_url(conn: sqlite3.Connection) -> str | None:
    """The article URL from the kv checkpoint (`_record_usni_check`), not
    from `osint_ship_positions` -- an article that extracted zero valid
    entries still gets a checkpoint (see `sync_usni`) but never gets a row
    in that table, so reading "already ingested" from the positions table
    would make a zero-entry article due for re-extraction forever."""
    row = conn.execute(
        "SELECT value_json FROM kv WHERE ns='osint_movements' AND key='usni_last_check'"
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0]).get("article_url")


def store_ship_positions(
    conn: sqlite3.Connection, now: datetime, article_url: str, article_title: str, entries: list[dict],
) -> int:
    """Insert one row per valid entry; a malformed entry (no region or no
    group name) is skipped rather than failing the whole extraction."""
    rows = [
        (now.isoformat(), article_url, article_title, str(e["region"]).strip(), str(e["group"]).strip(), str(e.get("note") or "").strip())
        for e in entries
        if isinstance(e, dict) and str(e.get("region") or "").strip() and str(e.get("group") or "").strip()
    ]
    if rows:
        conn.executemany(
            "INSERT INTO osint_ship_positions (fetched_at, article_url, article_title, region, group_name, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    return len(rows)


def latest_ship_positions(conn: sqlite3.Connection) -> tuple[list[tuple], str | None, str | None]:
    """Every row from the most-recently-fetched article, plus (url, title)
    -- what the report shows every day, whether or not USNI was actually
    checked this run (the article only changes weekly)."""
    row = conn.execute(
        "SELECT article_url, article_title FROM osint_ship_positions ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return [], None, None
    article_url, article_title = row
    rows = conn.execute(
        "SELECT region, group_name, note FROM osint_ship_positions WHERE article_url = ? ORDER BY region, group_name",
        (article_url,),
    ).fetchall()
    return [tuple(r) for r in rows], article_url, article_title


def sync_usni(conn: sqlite3.Connection, lm: LMStudio, now: datetime, fetcher: Fetcher) -> dict:
    """Check (at most every USNI_MIN_INTERVAL_DAYS) for a new Fleet Tracker
    article and extract its positions. Every failure mode (not due, fetch
    error, unparseable feed, already-checked article, extraction error) is
    isolated here and reported, never raised -- the air-traffic half of the
    report must still render on a bad USNI day.

    "Already checked" compares against the kv checkpoint (`_last_checked_
    article_url`), not `osint_ship_positions`: an article that extracts to
    zero valid entries still gets checkpointed below, so it's correctly
    treated as done rather than re-extracted on every future due-check."""
    if not usni_due(conn, now):
        return {"checked": False}
    try:
        entry = latest_feed_entry(fetcher)
    except Exception as exc:
        return {"checked": True, "error": f"feed fetch failed: {type(exc).__name__}: {exc}"}
    if entry is None:
        _record_usni_check(conn, now, article_url=None)
        return {"checked": True, "error": "feed had no usable entry"}
    link, title = entry.url, entry.title
    if link == _last_checked_article_url(conn):
        _record_usni_check(conn, now, article_url=link)
        return {"checked": True, "new_article": False, "article_url": link}
    try:
        text, truncated = truncate(strip_to_text(fetcher.get(link).text))
        result = lm.extract(text, USNI_SCHEMA, system=USNI_EXTRACT_SYSTEM, name="ship_positions")
    except Exception as exc:
        # Don't record the check on an extraction failure: leave it due so
        # the next run retries this same new article instead of skipping it.
        return {"checked": True, "new_article": True, "article_url": link, "error": f"extract failed: {type(exc).__name__}: {exc}"}
    stored = store_ship_positions(conn, now, link, title, result.get("entries") or [])
    _record_usni_check(conn, now, article_url=link)
    return {
        "checked": True, "new_article": True, "article_url": link, "article_title": title,
        "positions_stored": stored, "truncated": truncated,
    }


# -- narrative --------------------------------------------------------------


def summarize(lm: LMStudio, class_totals: Counter, region_totals: Counter, anomalies: list[dict], ship_rows: list[tuple]) -> str:
    lines = ["Today's OSINT movements aggregates:"]
    lines += [f"- aircraft by type: {cls} {n}" for cls, n in sorted(class_totals.items(), key=lambda kv: -kv[1])]
    lines += [f"- aircraft by region: {reg} {n}" for reg, n in sorted(region_totals.items(), key=lambda kv: -kv[1])]
    for a in anomalies:
        lines.append(f"- anomaly: {a['class']} over {a['region']}: {a['today']} today vs baseline mean {a['baseline_mean']} (z={a['z']})")
    if ship_rows:
        lines.append(f"- USNI Fleet Tracker: {len(ship_rows)} group(s) reported across {len({r[0] for r in ship_rows})} region(s)")
    return lm.chat("\n".join(lines), system=NARRATIVE_SYSTEM, max_tokens=300)


# -- report -------------------------------------------------------------------


def build_report(
    now: datetime, class_totals: Counter, region_totals: Counter, class_region_counts: Counter,
    anomalies: list[dict], markers: list[Marker], markers_truncated: bool, narrative: str,
    ship_rows: list[tuple], usni_article_url: str | None, usni_article_title: str | None,
) -> Path:
    total_aircraft = sum(class_totals.values())
    r = Report(
        "OSINT movements",
        kicker="lmsa · adsb.lol + USNI Fleet Tracker",
        subtitle=f"Military aircraft tracked in the last {WINDOW_HOURS}h, and the latest USNI Fleet Tracker positions.",
        sticker=f"{len(anomalies)} anomaly(ies)" if anomalies else "no anomalies",
        sticker_tone="hot" if anomalies else "acc",
        source="adsb.lol /v2/mil, USNI News Fleet and Marine Tracker (public data, aggregated)",
    )
    r.tiles([
        Tile("Aircraft tracked (24h)", total_aircraft, tone="info"),
        Tile("Regions active", sum(1 for v in region_totals.values() if v), tone="acc"),
        Tile("Anomalies vs 30d baseline", len(anomalies), tone="hot" if anomalies else "muted"),
        Tile("USNI groups (latest article)", len(ship_rows), tone="violet" if ship_rows else "muted"),
    ])
    r.section("Narrative").text(narrative)

    if anomalies:
        lines = "\n".join(
            f"{a['class']} over {a['region']}: {a['today']} today vs baseline {a['baseline_mean']} (z={a['z']})"
            for a in anomalies
        )
        r.section("Notable patterns", tone="hot").callout(lines, label="2 sigma or more above 30-day baseline", tone="hot")

    by_type = r.section("Aircraft by type")
    by_type.bars(sorted(class_totals.items(), key=lambda kv: -kv[1]))

    by_region = r.section("Aircraft by region")
    by_region.bars(sorted(region_totals.items(), key=lambda kv: -kv[1]))

    matrix = r.section("Type x region")
    classes = sorted(class_totals)
    regions_sorted = sorted(region_totals)
    matrix.table(
        ["Type"] + regions_sorted,
        [[cls] + [class_region_counts.get((cls, reg), 0) for reg in regions_sorted] for cls in classes],
    )

    geo_section = r.section("Activity map")
    if markers:
        geo_section.world_map(legend=False, markers=markers, label="military air activity, last 24h")
        if markers_truncated:
            geo_section.text(f"Showing the top {MARKER_CAP} busiest cells; some lower-activity cells are omitted.")
    else:
        geo_section.text("No positioned samples in the last 24h.")

    usni = r.section("USNI Fleet Tracker", tone="violet")
    if ship_rows:
        source_label = usni_article_title or "latest Fleet Tracker article"
        usni.table(["Source"], [[Link(text=source_label, href=usni_article_url)]])
        usni.table(["Region", "Group", "Note"], [[region, group, note] for region, group, note in ship_rows])
    else:
        usni.text("No USNI Fleet Tracker positions stored yet.")

    return r.write()


def _aggregate(now: datetime, region_counts: dict[str, Counter], class_totals: dict[str, Counter]):
    """today's (class, region) counts, class totals, region totals and
    anomalies. `class_totals` (distinct hex per class, deduped across
    regions) is the source of truth for per-class and grand totals;
    `region_totals` is summed from `region_counts` instead, which is
    correct there -- for one fixed region, a hex appears under at most one
    (class, region) key, so no dedup issue arises the way it does when
    summing across regions for one class."""
    today = _day(now.isoformat())
    today_counts = region_counts.get(today, Counter())
    today_class_totals = class_totals.get(today, Counter())
    region_totals: Counter = Counter()
    for (_cls, reg), n in today_counts.items():
        region_totals[reg] += n
    anomalies = detect_anomalies(region_counts, today)
    return today_counts, today_class_totals, region_totals, anomalies


def main() -> None:
    lm = LMStudio()
    lm.ensure_server()
    now = _now()
    type_class, regions = load_config(CONFIG_PATH)

    with Store() as store, store.record_run("tasks.osint_movements") as run:
        positions = positions_by_day(store.conn, now)
        region_counts = class_region_counts_by_day(positions, type_class, regions)
        totals_by_day = class_totals_by_day(positions, type_class)
        class_region_counts, class_totals, region_totals, anomalies = _aggregate(now, region_counts, totals_by_day)

        today_pos = today_positions(store.conn, now)
        markers, markers_truncated = density_markers(today_pos)

        usni_result = sync_usni(store.conn, lm, now, fetcher=default_fetcher())
        ship_rows, usni_article_url, usni_article_title = latest_ship_positions(store.conn)

        narrative = summarize(lm, class_totals, region_totals, anomalies, ship_rows)

        path = build_report(
            now, class_totals, region_totals, class_region_counts, anomalies,
            markers, markers_truncated, narrative, ship_rows, usni_article_url, usni_article_title,
        )
        run.report_path = path
        run.summary = {
            "aircraft_today": sum(class_totals.values()),
            "anomalies": len(anomalies),
            "usni": usni_result,
        }
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
