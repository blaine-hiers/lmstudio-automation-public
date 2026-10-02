"""Disaster fusion report: GDACS + USGS earthquakes + NASA EONET (+ optional
NASA FIRMS fire detections), deduped across sources into one daily view (#59).

Standalone today: `collect(fetcher, now) -> list[Event]` is written as a
clean, DB-free function so the future briefing engine (#52) can register it
as an adapter without change. `tasks.disasters.main()` is the actual
scheduled job -- it calls `collect_events` (same fetch, plus per-source
stats) and writes a per-country `hazard` metric into `geo_metrics` for the
geopolitical risk map (`tasks.risk_map`) to pick up.

Sources (all free, no key except FIRMS):
    GDACS   RSS   https://www.gdacs.org/xml/rss.xml
    USGS    GeoJSON summary feed, e.g. .../summary/significant_week.geojson
    EONET   NASA EONET v3 events API
    FIRMS   optional; needs LMSA_FIRMS_MAP_KEY (`lmsa.secrets`). Absent or
            disabled -> skipped with a note, never an error.

Hazard metric (written to `geo_metrics` as "hazard", per country, for the
UTC day of the run only -- past days are never rewritten): the sum of the
severity `weight` of every deduped event with a resolved country that is
*active* at run time -- an ongoing event (no end date, e.g. a GDACS flood or
cyclone alert still open) or one that ended within the last 24 h (a quake).
An ongoing event is therefore credited to every day it stays active, not
only to its start date. When every country-attributing source (GDACS, USGS)
fetched cleanly, every other country gets an explicit 0.0 row for the day,
so `tasks.risk_map` has a daily baseline; otherwise quiet countries get no
row and the report says so. A coordinate-only event
(a quake at sea, an unattributed fire cluster) never invents a country and
so never contributes to the metric, though it still appears on the map and
in the event list. See `build_report`'s "Coverage and method" section for
the exact weight table.

    python -m tasks.disasters
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import islice
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from lmsa import Link, Marker, Report, Store, Tile, geo, geo_metrics
from lmsa import secrets as lmsa_secrets
from lmsa.digest import Digest, DigestItem, RunRow
from lmsa.feeds import is_http_url, parse_date
from lmsa.geo import normalize_country
from lmsa.web import Fetcher

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/disasters.toml"], "secrets": ["firms"]}

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "disasters.toml"

USER_AGENT = "lmsa-disasters/0.1 (personal research; daily; public hazard feeds)"
KV_NS = "disasters"
DASH = "—"

# -- severity weight tables (see build_report's "Coverage and method" for the
# same numbers spelled out in the report itself) --------------------------
GDACS_TYPE = {"EQ": "earthquake", "TC": "cyclone", "FL": "flood", "VO": "volcano", "DR": "drought", "WF": "wildfire"}
GDACS_ALERT_WEIGHT = {"green": 1.0, "orange": 4.0, "red": 10.0}
EONET_TYPE = {
    "wildfires": "wildfire", "severeStorms": "cyclone", "volcanoes": "volcano",
    "floods": "flood", "drought": "drought", "earthquakes": "earthquake",
}
EONET_WEIGHT = 2.0  # flat: EONET doesn't grade severity uniformly across categories
FIRMS_GRID_DEG = 0.5           # cluster same-day detections into ~50 km grid cells
FIRMS_WEIGHT_PER_DETECTION = 0.2
FIRMS_WEIGHT_CAP = 5.0

TYPE_TONE = {
    "earthquake": "hot", "cyclone": "warn", "flood": "info",
    "volcano": "violet", "drought": "muted", "wildfire": "hot", "other": "muted",
}
MARKER_CAP = 150
TABLE_CAP = 150
DIGEST_EVENTS_CAP = 50  # top_events kept in summary_json for the digest() hook; SectionConfig.max_items tops out at 50


def _now() -> datetime:
    """Current UTC time. Tests monkeypatch this."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    id: str                     # this record's own dedupe key: "<source>:<source id>"
    source: str                  # the source whose fields this record uses: "gdacs" | "usgs" | "eonet" | "firms"
    sources: tuple[str, ...]     # every source that reported this event, after dedupe
    merged_ids: tuple[str, ...]  # every raw event id folded into this one by dedupe (incl. its own `id`) --
                                  # "new since last run" checks these, not `id`, so a rerun where a
                                  # different source wins the merge doesn't re-flag the same event as new
    type: str                    # "earthquake" | "cyclone" | "flood" | "volcano" | "drought" | "wildfire" | "other"
    severity: str                 # human label: GDACS alert level, "M5.6", "N detections", or ""
    weight: float                  # severity weight (hazard metric input + marker sizing)
    country: str | None            # primary/display country: countries[0] if any, else None -- never guessed
    countries: tuple[str, ...]     # every country this event is attributed to; the hazard metric credits each in full
    place: str                     # short human place description
    lat: float | None
    lon: float | None
    started: datetime               # tz-aware UTC
    url: str | None
    ended: datetime | None = None   # None = ongoing; == started for an instantaneous event (a quake)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    gdacs_url: str = "https://www.gdacs.org/xml/rss.xml"
    usgs_feed: str = "significant_week"
    eonet_base_url: str = "https://eonet.gsfc.nasa.gov/api/v3/events"
    eonet_status: str = "open"
    eonet_days: int = 3
    firms_enabled: bool = True
    firms_source: str = "VIIRS_SNPP_NRT"
    firms_area: str = "world"
    firms_day_range: int = 1
    firms_max_bytes: int = 50_000_000   # FIRMS's worldwide daily CSV is far over lmsa.web's 5 MB default
    firms_max_rows: int = 300_000       # detection rows parsed per run; the rest are dropped and disclosed
    dedupe_radius_km: float = 100.0
    dedupe_window_hours: float = 24.0
    per_host_delay: float = 2.0
    warnings: list[str] = field(default_factory=list)

    @property
    def usgs_url(self) -> str:
        return f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{self.usgs_feed}.geojson"

    @property
    def eonet_url(self) -> str:
        return f"{self.eonet_base_url}?status={self.eonet_status}&days={self.eonet_days}"


def load_config(path: Path = CONFIG) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    cfg = Config()
    g = data.get("gdacs", {})
    cfg.gdacs_url = str(g.get("rss_url", cfg.gdacs_url))
    u = data.get("usgs", {})
    cfg.usgs_feed = str(u.get("feed", cfg.usgs_feed))
    e = data.get("eonet", {})
    cfg.eonet_base_url = str(e.get("url", cfg.eonet_base_url))
    cfg.eonet_status = str(e.get("status", cfg.eonet_status))
    cfg.eonet_days = max(1, int(e.get("days", cfg.eonet_days)))
    f = data.get("firms", {})
    cfg.firms_enabled = bool(f.get("enabled", cfg.firms_enabled))
    cfg.firms_source = str(f.get("source", cfg.firms_source))
    cfg.firms_area = str(f.get("area", cfg.firms_area))
    cfg.firms_day_range = max(1, int(f.get("day_range", cfg.firms_day_range)))
    cfg.firms_max_bytes = max(1, int(f.get("max_bytes", cfg.firms_max_bytes)))
    cfg.firms_max_rows = max(1, int(f.get("max_rows", cfg.firms_max_rows)))
    d = data.get("dedupe", {})
    cfg.dedupe_radius_km = max(0.0, float(d.get("radius_km", cfg.dedupe_radius_km)))
    cfg.dedupe_window_hours = max(0.0, float(d.get("window_hours", cfg.dedupe_window_hours)))
    return cfg


# ---------------------------------------------------------------------------
# Country resolution helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _iso3_table() -> dict[str, str]:
    """ISO alpha-3 -> alpha-2, from `lmsa.geo`'s own country table. Structured
    feeds (GDACS `gdacs:iso3`) give alpha-3, but `geo.country()` only indexes
    alpha-2 and `normalize_country`'s CODE_DENYLIST would wrongly reject real
    alpha-3 codes that collide with English words (ARM, GEO, FIN, PAN, ...),
    so a plain code table is used instead -- never normalize_country for a
    code we already know is a country code."""
    return {c.alpha3: c.alpha2 for c in geo.countries()}


def _iso3_to_alpha2(code: str) -> str | None:
    return _iso3_table().get(code.strip().upper()) if code else None


# USGS `place` text ends in ", <US state>" for a domestic quake or
# ", <country>" for a foreign one. US state postal codes and full names
# collide with real ISO country codes (CA=Canada, PA=Panama, GA=Gabon,
# DE=Germany, ...), so states/DC/territories are matched explicitly *first*;
# only a tail that isn't one of these falls through to `normalize_country`.
US_STATE_ABBR = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
})
US_STATE_NAMES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia",
})
# Unlike the 50 states, these US territories carry their own ISO alpha-2 code.
US_TERRITORY_CODE = {"PR": "PR", "GU": "GU", "VI": "VI"}
US_TERRITORY_NAMES = {"puerto rico": "PR", "guam": "GU", "virgin islands": "VI", "us virgin islands": "VI"}


# "Georgia" is both a US state and a country: USGS writes "..., Georgia" for
# either. Georgia the country spans roughly 40-47 E; the US state roughly
# 80.8-85.6 W. Decide by longitude (with a margin), never by name alone.
GEORGIA_COUNTRY_LON = (38.0, 49.0)
GEORGIA_STATE_LON = (-88.0, -78.0)


def _usgs_place_country(place: str, lon: float | None = None) -> str | None:
    """The part of a USGS `place` string after the last comma, resolved to a
    country. US states/DC are mapped to "US" explicitly before anything else
    is tried, because their postal codes and some full names collide with
    real ISO country codes (CA/Canada, PA/Panama, GA/Gabon, ...) -- see
    contracts.md's #59 review lesson. Puerto Rico, Guam and the US Virgin
    Islands keep their own ISO code rather than folding into "US". "Georgia"
    is resolved by the quake's longitude (US state vs the country), and is
    left unattributed when the longitude fits neither. Anything else falls
    through to `normalize_country` for a foreign country name."""
    if "," not in place:
        return None
    tail = place.rsplit(",", 1)[-1].strip()
    if not tail:
        return None
    upper = tail.upper()
    if upper in US_TERRITORY_CODE:
        return US_TERRITORY_CODE[upper]
    if upper in US_STATE_ABBR:
        return "US"
    lower = re.sub(r"\s+", " ", re.sub(r"[.’']", "", tail)).strip().lower()
    if lower in US_TERRITORY_NAMES:
        return US_TERRITORY_NAMES[lower]
    if lower == "georgia":
        if lon is not None and GEORGIA_COUNTRY_LON[0] <= lon <= GEORGIA_COUNTRY_LON[1]:
            return "GE"
        if lon is not None and GEORGIA_STATE_LON[0] <= lon <= GEORGIA_STATE_LON[1]:
            return "US"
        return None
    if lower in US_STATE_NAMES:
        return "US"
    return normalize_country(tail)


FIRMS_KEY_RE = re.compile(r"[A-Za-z0-9]+")
_FIRMS_PATH_KEY_RE = re.compile(r"(/api/area/csv/)[^/?#\s'\"]*", re.IGNORECASE)


def _redact(text: str, secret: str | None) -> str:
    """Scrub the FIRMS MAP_KEY from `text`. The key rides in the URL path,
    and every `lmsa.web.FetchError` includes the URL in `str(exc)`
    (contracts.md's #59 review lesson), so any FIRMS failure must pass
    through here before it reaches stats, the run summary, logs or the
    report. Redacts by pattern, not just the exact raw key: the key segment
    of any `/api/area/csv/<segment>/` path, and -- case-insensitively -- the
    key itself with any of its characters percent-encoded (a redirect
    Location may upper-case or encode it)."""
    text = _FIRMS_PATH_KEY_RE.sub(r"\1***", text)
    if secret:
        pattern = "".join(
            "(?:" + "|".join([re.escape(ch)] + [f"%{ord(v):02x}" for v in {ch.lower(), ch.upper()}]) + ")"
            for ch in secret
        )
        text = re.sub(pattern, "***", text, flags=re.IGNORECASE)
    return text


def _clean_firms_key(raw: str | None) -> tuple[str | None, str | None]:
    """(key, error). Strips surrounding whitespace (a key set through piped
    stdin on Windows often carries a trailing CR) and accepts only
    `[A-Za-z0-9]+`; anything else is refused with a note that never echoes
    the value."""
    key = (raw or "").strip()
    if not key:
        return None, None
    if not FIRMS_KEY_RE.fullmatch(key):
        return None, "invalid FIRMS key format (expected letters and digits only; value not shown)"
    return key, None


class UnrecognizedResponse(ValueError):
    """A 200 response whose body isn't the expected shape (an error page or
    message served with HTTP 200) -- distinct from a genuinely empty feed."""


def _usgs_weight(mag: float | None) -> float:
    if mag is None:
        return 0.0
    if mag >= 8.0:
        return 16.0
    if mag >= 7.0:
        return 8.0
    if mag >= 6.0:
        return 4.0
    if mag >= 5.0:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# GDACS
# ---------------------------------------------------------------------------

def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _find_text(elem: ET.Element, *names: str) -> str:
    for child in elem.iter():
        if _local(child.tag) in names and (child.text or "").strip():
            return child.text.strip()
    return ""


def parse_gdacs(xml_bytes: bytes | str) -> list[Event]:
    """GDACS RSS -> Events. Tolerant of geo:Point/geo:lat+geo:long nesting or
    bare geo:lat/geo:long directly on the item; an unrecognized
    gdacs:eventtype becomes "other" rather than dropping the item. Raises
    `xml.etree.ElementTree.ParseError` on malformed XML (caller isolates)."""
    root = ET.fromstring(xml_bytes)
    events: list[Event] = []
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        eventid = _find_text(item, "eventid")
        eventtype_code = _find_text(item, "eventtype").strip().upper()
        alertlevel = _find_text(item, "alertlevel").strip().lower()
        iso3 = _find_text(item, "iso3")
        country_field = _find_text(item, "country")
        title = _find_text(item, "title")
        link = _find_text(item, "link")
        lat_text = _find_text(item, "lat")
        lon_text = _find_text(item, "long", "lon")
        fromdate = _find_text(item, "fromdate")
        todate = _find_text(item, "todate")
        pubdate = _find_text(item, "pubDate")
        try:
            lat = float(lat_text) if lat_text else None
            lon = float(lon_text) if lon_text else None
        except ValueError:
            lat = lon = None
        started = parse_date(fromdate) or parse_date(pubdate)
        if started is None:
            continue  # can't place it in time; skip rather than guess "now"
        # gdacs:iso3 can list more than one code for an event that crosses
        # borders (a cyclone, a flood) -- attribute it to every one, not just
        # the first, per #59.
        countries: list[str] = []
        for code in re.split(r"[;,\s]+", iso3.strip()) if iso3.strip() else []:
            a2 = _iso3_to_alpha2(code)
            if a2 and a2 not in countries:
                countries.append(a2)
        if not countries and country_field:
            for name in re.split(r"[;,]", country_field):
                a2 = normalize_country(name.strip())
                if a2 and a2 not in countries:
                    countries.append(a2)
        event_type = GDACS_TYPE.get(eventtype_code, "other")
        # No todate = still ongoing. An earthquake is instantaneous even when
        # GDACS leaves todate blank, so it never counts as ongoing forever.
        ended = parse_date(todate) if todate else None
        if ended is None and event_type == "earthquake":
            ended = started
        weight = GDACS_ALERT_WEIGHT.get(alertlevel, 1.0)
        eid = f"gdacs:{eventid or link or title}"
        events.append(Event(
            id=eid, source="gdacs", sources=("gdacs",), merged_ids=(eid,),
            type=event_type, severity=alertlevel.capitalize(),
            weight=weight, country=countries[0] if countries else None, countries=tuple(countries),
            place=country_field or title or "unknown location",
            lat=lat, lon=lon, started=started, url=link or None, ended=ended,
        ))
    return events


def _gdacs_item_count(xml_bytes: bytes | str) -> int:
    """Number of `<item>` elements in the raw feed, regardless of whether
    each one parsed into a usable Event -- lets `collect_events` tell an
    empty feed (0 items) apart from an unrecognized one (items present, 0
    parsed)."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return 0
    return sum(1 for el in root.iter() if _local(el.tag) == "item")


def _check_gdacs_shape(raw: bytes | str) -> None:
    """A GDACS feed is RSS: an <rss>/<channel> document. Anything else served
    with HTTP 200 (an <error/> document, an HTML page) is unrecognized, not an
    empty feed -- otherwise a quiet-country zero-fill would follow it."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return  # parse_gdacs raises the real ParseError; the caller isolates it
    if _local(root.tag) != "rss" and not any(_local(e.tag) == "channel" for e in root.iter()):
        raise UnrecognizedResponse(f"GDACS response is not an RSS feed (root <{_local(root.tag)}>)")


def fetch_gdacs(fetcher: Fetcher, url: str) -> tuple[list[Event], int]:
    raw = fetcher.get(url).content
    _check_gdacs_shape(raw)
    return parse_gdacs(raw), _gdacs_item_count(raw)


# ---------------------------------------------------------------------------
# USGS earthquakes
# ---------------------------------------------------------------------------

def parse_usgs(data: dict) -> list[Event]:
    """USGS earthquake GeoJSON summary feed -> Events. Each feature is parsed
    independently; a malformed one is skipped rather than dropping the rest."""
    out: list[Event] = []
    for feat in (data or {}).get("features") or []:
        try:
            if not isinstance(feat, dict):
                continue
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            time_ms = props.get("time")
            if time_ms is None:
                continue
            started = datetime.fromtimestamp(float(time_ms) / 1000.0, tz=timezone.utc)
            mag = props.get("mag")
            mag = float(mag) if isinstance(mag, (int, float)) and not isinstance(mag, bool) else None
            place = str(props.get("place") or "").strip()
            fid = str(feat.get("id") or place or started.isoformat())
            url = props.get("url")
            country = _usgs_place_country(place, lon)
            eid = f"usgs:{fid}"
            out.append(Event(
                id=eid, source="usgs", sources=("usgs",), merged_ids=(eid,),
                type="earthquake", severity=f"M{mag:.1f}" if mag is not None else "",
                weight=_usgs_weight(mag), country=country, countries=(country,) if country else (),
                place=place or "unknown location", lat=lat, lon=lon, started=started,
                url=str(url) if is_http_url(str(url or "")) else None, ended=started,
            ))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def fetch_usgs(fetcher: Fetcher, url: str) -> tuple[list[Event], int]:
    data = fetcher.get_json(url)
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise UnrecognizedResponse("USGS response has no 'features' list")
    return parse_usgs(data), len(data["features"])


# ---------------------------------------------------------------------------
# NASA EONET
# ---------------------------------------------------------------------------

def parse_eonet(data: dict) -> list[Event]:
    """NASA EONET v3 events -> Events, one per event using its most recent
    Point geometry. Coordinate-only by nature (EONET gives no country), so
    `country` is always None here, per #59."""
    out: list[Event] = []
    for ev in (data or {}).get("events") or []:
        try:
            geoms = ev.get("geometry") or []
            if not geoms:
                continue
            g = geoms[-1]
            if g.get("type") != "Point":
                continue
            coords = g.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            when = parse_date(str(g.get("date") or ""))
            if when is None:
                continue
            cats = ev.get("categories") or []
            cat_id = str(cats[0].get("id")) if cats and isinstance(cats[0], dict) else ""
            title = str(ev.get("title") or "").strip()
            eid = str(ev.get("id") or title or when.isoformat())
            url = None
            for s in ev.get("sources") or []:
                if isinstance(s, dict) and is_http_url(str(s.get("url") or "")):
                    url = s["url"]
                    break
            full_id = f"eonet:{eid}"
            closed = parse_date(str(ev.get("closed") or "")) if ev.get("closed") else None
            out.append(Event(
                id=full_id, source="eonet", sources=("eonet",), merged_ids=(full_id,),
                type=EONET_TYPE.get(cat_id, "other"), severity="", weight=EONET_WEIGHT,
                country=None, countries=(), place=title or "unknown location",
                lat=lat, lon=lon, started=when, url=url, ended=closed,
            ))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return out


def fetch_eonet(fetcher: Fetcher, url: str) -> tuple[list[Event], int]:
    data = fetcher.get_json(url)
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise UnrecognizedResponse("EONET response has no 'events' list")
    return parse_eonet(data), len(data["events"])


# ---------------------------------------------------------------------------
# NASA FIRMS (optional)
# ---------------------------------------------------------------------------

FIRMS_BASE = "https://firms.modaps.eosdis.gov/api/area/csv"


def firms_url(map_key: str, source: str, area: str, day_range: int) -> str:
    return f"{FIRMS_BASE}/{map_key}/{source}/{area}/{day_range}"


def parse_firms(csv_text: str) -> list[Event]:
    """VIIRS/MODIS NRT area CSV -> clustered wildfire Events. FIRMS returns
    one row per detection pixel (often thousands a day), so detections are
    grouped into `FIRMS_GRID_DEG`-degree grid cells per UTC day rather than
    kept individually -- coordinate-only, like every fire in this report, so
    no country is invented for a cluster. A "low" confidence row (a common
    false-positive source) or a malformed row is dropped."""
    return _cluster_firms(csv.DictReader(io.StringIO(csv_text)))


def _cluster_firms(rows) -> list[Event]:
    buckets: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
    for row in rows:
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            acq_date = str(row.get("acq_date") or "").strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not acq_date:
            continue
        if str(row.get("confidence") or "").strip().lower() in ("l", "low"):
            continue
        cell = (acq_date, round(lat / FIRMS_GRID_DEG), round(lon / FIRMS_GRID_DEG))
        buckets.setdefault(cell, []).append((lat, lon))
    out: list[Event] = []
    for (acq_date, lat_i, lon_i), points in buckets.items():
        try:
            day = datetime.strptime(acq_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        lat = sum(p[0] for p in points) / len(points)
        lon = sum(p[1] for p in points) / len(points)
        weight = min(FIRMS_WEIGHT_CAP, FIRMS_WEIGHT_PER_DETECTION * len(points))
        eid = f"firms:{acq_date}:{lat_i}:{lon_i}"
        out.append(Event(
            id=eid, source="firms", sources=("firms",), merged_ids=(eid,),
            type="wildfire", severity=f"{len(points)} detections", weight=weight,
            country=None, countries=(), place=f"fire cluster near {lat:.2f},{lon:.2f}",
            lat=lat, lon=lon, started=day, url=None, ended=day,
        ))
    return out


FIRMS_REQUIRED_COLUMNS = frozenset({"latitude", "longitude", "acq_date"})
FIRMS_SNIPPET_CHARS = 80


def parse_firms_response(text: str, map_key: str | None, max_rows: int) -> tuple[list[Event], int, bool]:
    """(events, rows considered, truncated). FIRMS answers a bad key, an
    exceeded quota or a bad area with a short text message and HTTP 200, so
    the CSV header must name latitude/longitude/acq_date or the body is an
    `UnrecognizedResponse` -- whose message shows only the first
    `FIRMS_SNIPPET_CHARS` characters of the body, key redacted first. With a
    valid header, zero kept clusters (every detection low-confidence) is a
    quiet day, not unrecognized. At most `max_rows` rows are parsed."""
    reader = csv.DictReader(io.StringIO(text))
    columns = {(c or "").strip().lower() for c in (reader.fieldnames or [])}
    if not FIRMS_REQUIRED_COLUMNS <= columns:
        snippet = re.sub(r"\s+", " ", _redact(text, map_key)).strip()[:FIRMS_SNIPPET_CHARS]
        raise UnrecognizedResponse(f"FIRMS body is not a detection CSV: {snippet!r}")
    rows = list(islice(reader, max_rows + 1))
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return _cluster_firms(rows), len(rows), truncated


def make_firms_fetcher(cfg: Config) -> Fetcher:
    """A Fetcher used only for FIRMS: no disk cache (`cache_dir=None`),
    because the HTTP cache writes the full URL -- MAP_KEY included -- to
    data/web/*.json and its I/O warnings log it (contracts.md #59), and a
    larger byte cap, because the worldwide daily CSV is well over 5 MB."""
    return Fetcher(USER_AGENT, per_host_delay=cfg.per_host_delay, max_bytes=cfg.firms_max_bytes, cache_dir=None)


def fetch_firms(fetcher: Fetcher, map_key: str, cfg: Config) -> tuple[list[Event], int, bool]:
    url = firms_url(map_key, cfg.firms_source, cfg.firms_area, cfg.firms_day_range)
    text = fetcher.get(url).text
    return parse_firms_response(text, map_key, cfg.firms_max_rows)


# ---------------------------------------------------------------------------
# Cross-source dedupe
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
    return 2 * r * asin(sqrt(a))


def dedupe(events: list[Event], *, radius_km: float = 100.0, window_hours: float = 24.0) -> list[Event]:
    """Merge events of the same normalized `type` that different sources
    reported for what is almost certainly the same physical event -- within
    `radius_km` and `window_hours` of an already-kept event (a GDACS
    earthquake bulletin and the USGS feed for the same quake; a GDACS/EONET/
    FIRMS wildfire). Events without both coordinates are never merged. The
    surviving record is whichever has the higher severity weight; its
    `sources` gains the other's, so the report can show "gdacs+usgs"."""
    kept: list[Event] = []
    for ev in sorted(events, key=lambda e: e.started):
        match_i = None
        for i, k in enumerate(kept):
            if k.type != ev.type:
                continue
            if None in (k.lat, k.lon, ev.lat, ev.lon):
                continue
            if abs((k.started - ev.started).total_seconds()) > window_hours * 3600:
                continue
            if _haversine_km(k.lat, k.lon, ev.lat, ev.lon) > radius_km:
                continue
            match_i = i
            break
        if match_i is None:
            kept.append(ev)
            continue
        k = kept[match_i]
        merged_sources = tuple(dict.fromkeys(k.sources + ev.sources))
        merged_countries = tuple(dict.fromkeys(k.countries + ev.countries))
        # Every raw id folded into this event, from both sides -- so "new
        # since last run" survives a later rerun where the other side's
        # record wins (a different `.id`), per #59.
        merged_ids = tuple(dict.fromkeys(k.merged_ids + ev.merged_ids))
        winner = ev if ev.weight > k.weight else k
        # Still ongoing if either side is; otherwise the later end.
        ended = None if k.ended is None or ev.ended is None else max(k.ended, ev.ended)
        kept[match_i] = replace(
            winner, sources=merged_sources, countries=merged_countries,
            country=merged_countries[0] if merged_countries else None, merged_ids=merged_ids,
            ended=ended,
        )
    return sorted(kept, key=lambda e: (-e.weight, e.started))


# ---------------------------------------------------------------------------
# Collect (engine-adapter shape for #52) + per-run stats
# ---------------------------------------------------------------------------

@dataclass
class FetchStats:
    errors: list[tuple[str, str]] = field(default_factory=list)   # (source, message)
    counts: dict[str, int] = field(default_factory=dict)           # source -> events kept (pre-dedupe)
    considered: dict[str, int] = field(default_factory=dict)       # source -> raw items seen (pre-filter)
    unrecognized: list[str] = field(default_factory=list)          # sources whose body wasn't the expected shape
    unrecognized_detail: dict[str, str] = field(default_factory=dict)  # source -> why (FIRMS: redacted 80-char snippet)
    notes: list[str] = field(default_factory=list)                 # caps/truncation disclosures
    firms_skipped: str | None = None

    def clean(self, source: str) -> bool:
        """True when `source` fetched and parsed as expected this run."""
        return (source in self.counts and source not in self.unrecognized
                and all(s != source for s, _ in self.errors))


def collect_events(fetcher: Fetcher, now: datetime, cfg: Config | None = None, *,
                   firms_fetcher: Fetcher | None = None) -> tuple[list[Event], FetchStats]:
    """Fetch every source, isolating one source's failure from the rest, then
    dedupe. Used by `main()`, which needs the stats to report fetch errors
    and the FIRMS skip reason; `collect()` below is the plain adapter shape.
    FIRMS is fetched with `firms_fetcher`, default `make_firms_fetcher(cfg)`
    (uncached) -- never with the shared, disk-caching `fetcher`."""
    cfg = cfg or load_config()
    stats = FetchStats()
    raw: list[Event] = []
    for name, fn in (
        ("gdacs", lambda: fetch_gdacs(fetcher, cfg.gdacs_url)),
        ("usgs", lambda: fetch_usgs(fetcher, cfg.usgs_url)),
        ("eonet", lambda: fetch_eonet(fetcher, cfg.eonet_url)),
    ):
        try:
            got, considered = fn()
            raw.extend(got)
            stats.counts[name] = len(got)
            stats.considered[name] = considered
            # A non-empty body that parses to zero events is a schema
            # mismatch, not a quiet day -- disclose it separately from a
            # genuinely empty feed/array (considered == 0).
            if considered > 0 and not got:
                stats.unrecognized.append(name)
        except UnrecognizedResponse as exc:  # an error body served with HTTP 200
            stats.counts[name] = 0
            stats.unrecognized.append(name)
            stats.unrecognized_detail[name] = str(exc)[:300]
        except Exception as exc:  # one source never blocks the others
            stats.errors.append((name, f"{type(exc).__name__}: {exc}"[:300]))

    if not cfg.firms_enabled:
        stats.firms_skipped = "disabled in config/disasters.toml"
    else:
        map_key, key_error = _clean_firms_key(lmsa_secrets.resolve("firms"))
        if key_error:
            stats.errors.append(("firms", key_error))
        elif not map_key:
            stats.firms_skipped = "LMSA_FIRMS_MAP_KEY not set"
        else:
            try:
                got, considered, truncated = fetch_firms(firms_fetcher or make_firms_fetcher(cfg), map_key, cfg)
                raw.extend(got)
                stats.counts["firms"] = len(got)
                stats.considered["firms"] = considered
                if truncated:
                    stats.notes.append(
                        f"FIRMS: only the first {cfg.firms_max_rows:,} detection rows were parsed "
                        "(firms.max_rows); the rest were dropped -- narrow firms.area to a bbox for full coverage."
                    )
            except UnrecognizedResponse as exc:
                stats.counts["firms"] = 0
                stats.unrecognized.append("firms")
                stats.unrecognized_detail["firms"] = _redact(str(exc), map_key)[:300]
            except Exception as exc:
                # The MAP_KEY rides in the FIRMS URL, and FetchError.__str__
                # includes the URL -- redact it before it reaches stats, the
                # report or the run summary (contracts.md #59 review lesson).
                msg = _redact(f"{type(exc).__name__}: {exc}", map_key)[:300]
                stats.errors.append(("firms", msg))

    events = dedupe(raw, radius_km=cfg.dedupe_radius_km, window_hours=cfg.dedupe_window_hours)
    return events, stats


def collect(fetcher: Fetcher, now: datetime) -> list[Event]:
    """Engine-adapter shape for the future briefing engine (#52): events only,
    fetched with `config/disasters.toml` defaults, no DB access. `main()`
    uses `collect_events` instead so it can surface fetch errors and the
    FIRMS skip reason in the report."""
    events, _stats = collect_events(fetcher, now)
    return events


# ---------------------------------------------------------------------------
# Hazard metric
# ---------------------------------------------------------------------------

# Sources that attribute events to countries: a quiet country's 0.0 row is
# only trustworthy when every one of them fetched cleanly this run.
COUNTRY_SOURCES = ("gdacs", "usgs")
ACTIVE_WINDOW = timedelta(hours=24)


def is_active(ev: Event, now: datetime) -> bool:
    """Active at `now`: started, and either ongoing (no end) or ended within
    `ACTIVE_WINDOW` -- so a daily run credits an ongoing alert on every day
    it stays open and a quake on the run after it happened."""
    if ev.started > now:
        return False
    return ev.ended is None or ev.ended >= now - ACTIVE_WINDOW


def hazard_today(events: list[Event], now: datetime) -> dict[str, float]:
    """country -> summed severity weight of the events active at `now` (see
    `is_active`), for events with at least one resolved country. An event
    naming more than one country (a GDACS cyclone/flood crossing a border)
    credits its *full* weight to each -- not split -- since the hazard is
    real for every country it touched."""
    out: dict[str, float] = {}
    for ev in events:
        if not ev.countries or not is_active(ev, now):
            continue
        for country in ev.countries:
            out[country] = out.get(country, 0.0) + ev.weight
    return out


def write_hazard_metrics(conn, events: list[Event], now: datetime, *,
                         zero_fill: bool = False) -> tuple[int, list[str]]:
    """Upsert the `hazard` geo_metrics row for the run's UTC day only --
    past days are never rewritten. With `zero_fill` (every COUNTRY_SOURCES
    source fetched cleanly) every other country gets an explicit 0.0 so the
    risk map has a daily baseline. Without it, quiet countries get nothing,
    and a country already holding a higher value for today (from an earlier,
    complete run) is never lowered by this partial one. One bad row is
    skipped and reported, never failing the whole write -- contracts.md
    #56: `record` raises per row, caught here."""
    day = now.astimezone(timezone.utc).date()
    totals = hazard_today(events, now)
    if zero_fill:
        for c in geo.countries():
            totals.setdefault(c.alpha2, 0.0)
    written = 0
    errors: list[str] = []
    for country, value in sorted(totals.items()):
        try:
            if not zero_fill:
                prior = geo_metrics.series(conn, country, "hazard", start=day, end=day)
                if prior and prior[0][1] >= value:
                    continue
            geo_metrics.record(conn, day, country, "hazard", value, commit=False)
            written += 1
        except ValueError as exc:
            errors.append(f"{country} {day}: {exc}")
    conn.commit()
    return written, errors


# ---------------------------------------------------------------------------
# "New since last run"
# ---------------------------------------------------------------------------

def _load_seen(conn) -> set[str]:
    row = conn.execute("SELECT value_json FROM kv WHERE ns=? AND key='seen_ids'", (KV_NS,)).fetchone()
    if not row:
        return set()
    try:
        return set(json.loads(row[0]))
    except (TypeError, ValueError):
        return set()


def _save_seen(conn, ids: set[str], now: datetime) -> None:
    conn.execute(
        "INSERT INTO kv (ns, key, value_json, updated) VALUES (?, 'seen_ids', ?, ?) "
        "ON CONFLICT(ns, key) DO UPDATE SET value_json=excluded.value_json, updated=excluded.updated",
        (KV_NS, json.dumps(sorted(ids)), now.isoformat()),
    )
    conn.commit()


def _next_seen(events: list[Event], seen_before: set[str], failed_sources: set[str]) -> set[str]:
    """What to persist as "seen": this run's contributing ids, plus the
    previous ids of every source that failed (error or unrecognized body)
    this run -- so an outage doesn't forget that source's events and re-flag
    them all as new when it comes back. Bounded: carried ids are a subset of
    the previous set, so the set never grows past one good fetch per source."""
    carried = {i for i in seen_before if i.split(":", 1)[0] in failed_sources}
    return _contributing_ids(events) | carried


def _contributing_ids(events: list[Event]) -> set[str]:
    """Every raw source-event id behind this run's (possibly merged)
    events -- what gets persisted as "seen", so a later rerun where a
    different source wins a merge still recognizes the event."""
    ids: set[str] = set()
    for e in events:
        ids.update(e.merged_ids)
    return ids


def _new_event_ids(events: list[Event], seen_before: set[str]) -> set[str]:
    """Ids of events that are new since last run: an event counts as new only
    if *none* of the raw ids dedupe folded into it were already seen. Keying
    on an event's own `.id` instead would re-flag the same physical event as
    new every time a different source happens to win the merge (#59)."""
    return {e.id for e in events if not (set(e.merged_ids) & seen_before)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

ALL_SOURCES = ("gdacs", "usgs", "eonet", "firms")


def _countries_cell(ev: Event) -> str:
    if not ev.countries:
        return DASH
    return ", ".join(geo.country_name(c) or c for c in ev.countries)


def _marker_for(ev: Event, max_weight: float) -> Marker | None:
    if ev.lat is None or ev.lon is None:
        return None
    size = 0.5 + 2.5 * (ev.weight / max_weight) if max_weight else 1.0
    label = f"{ev.type} {ev.severity}".strip()
    if ev.place:
        label = f"{label} · {ev.place}" if label else ev.place
    return Marker(lat=ev.lat, lon=ev.lon, label=label, tone=TYPE_TONE.get(ev.type, "muted"), size=size, href=ev.url)


def build_report(
    events: list[Event], as_of: date, new_ids: set[str],
    hazard_written: int, hazard_errors: list[str], stats: FetchStats, cfg: Config,
    now: datetime | None = None, zero_filled: bool | None = None,
) -> Report:
    with_coords = [e for e in events if e.lat is not None and e.lon is not None]
    max_weight = max((e.weight for e in events), default=0.0) or 1.0
    ranked_coords = sorted(with_coords, key=lambda e: -e.weight)
    markers = [m for e in ranked_coords[:MARKER_CAP] if (m := _marker_for(e, max_weight)) is not None]
    truncated_markers = len(with_coords) > MARKER_CAP

    by_type = Counter(e.type for e in events)
    new_events = [e for e in events if e.id in new_ids]
    now = now or datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=timezone.utc)
    country_hazard_today = {c: v for c, v in hazard_today(events, now).items() if v > 0}
    has_firms = "firms" in stats.counts

    r = Report(
        "Disaster fusion report",
        kicker="Daily · GDACS + USGS + NASA EONET" + (" + FIRMS" if has_firms else ""),
        subtitle=(
            f"Active/recent hazard events fused from GDACS, USGS earthquakes and NASA EONET"
            + (", NASA FIRMS" if has_firms else "")
            + f", deduped across sources, as of {as_of:%Y-%m-%d} (UTC)."
        ),
        sticker=f"{len(new_events)} new since last run" if new_events else "no new events",
        sticker_tone="hot" if new_events else "acc",
        source="GDACS RSS, USGS earthquake GeoJSON, NASA EONET v3" + (", NASA FIRMS" if has_firms else ""),
    )
    r.tiles([
        Tile("Events tracked", len(events), tone="info"),
        Tile("New since last run", len(new_events), tone="hot" if new_events else "muted"),
        Tile("Countries with hazard today", len(country_hazard_today), tone="acc"),
        Tile("Sources with errors", len(stats.errors), tone="hot" if stats.errors else "muted",
             note=", ".join(s for s, _ in stats.errors) or None),
    ])

    if by_type:
        r.section("By type").bars(sorted(by_type.items(), key=lambda kv: -kv[1]))

    geo_section = r.section("Map", tone="hot")
    if markers:
        geo_section.world_map(legend=False, markers=markers, label="disaster events by type and severity")
        if truncated_markers:
            geo_section.text(
                f"Showing the {MARKER_CAP} highest-severity events on the map; "
                f"{len(with_coords) - MARKER_CAP} lower-severity ones are omitted (all appear in the list below)."
            )
    else:
        geo_section.text("No events with coordinates to plot.")

    ranked = sorted(events, key=lambda e: (-e.weight, -e.started.timestamp()))
    listing = r.section("Events by severity")
    listing.table(
        ["Type", "Severity", "Country", "Place", "Started (UTC)", "Sources", "Link"],
        [
            [
                e.type, e.severity or DASH, _countries_cell(e),
                e.place or DASH, e.started.strftime("%Y-%m-%d %H:%M"), "+".join(e.sources),
                Link(text="details", href=e.url) if e.url else DASH,
            ]
            for e in ranked[:TABLE_CAP]
        ],
    )
    if len(ranked) > TABLE_CAP:
        listing.text(f"Showing the {TABLE_CAP} highest-severity of {len(ranked)} events.")

    if new_events:
        r.section("New since last run", tone="hot").table(
            ["Type", "Severity", "Country", "Place", "Started (UTC)"],
            [
                [e.type, e.severity or DASH, _countries_cell(e),
                 e.place or DASH, e.started.strftime("%Y-%m-%d %H:%M")]
                for e in sorted(new_events, key=lambda e: -e.weight)[:TABLE_CAP]
            ],
        )

    sources_section = r.section("Sources", tone="info")
    err_names = {s for s, _ in stats.errors}
    source_rows = []
    for name in ALL_SOURCES:
        if name == "firms" and stats.firms_skipped:
            source_rows.append([name, DASH, DASH, f"skipped: {stats.firms_skipped}"])
            continue
        if name in err_names:
            source_rows.append([name, stats.considered.get(name, DASH), 0, "fetch error (see below)"])
            continue
        if name not in stats.counts and name not in stats.considered:
            continue
        if name in stats.unrecognized:
            detail = stats.unrecognized_detail.get(name)
            status = f"unrecognized response ({detail})" if detail else "unrecognized response (non-empty, 0 parsed)"
        else:
            status = "ok"
        source_rows.append([name, stats.considered.get(name, DASH), stats.counts.get(name, 0), status])
    sources_section.table(["Source", "Raw items", "Events kept", "Status"], source_rows)

    method = r.section("Coverage and method", tone="muted")
    lines = [
        f"Sources: GDACS RSS (every active alert), USGS earthquake GeoJSON ({cfg.usgs_feed}), "
        f"NASA EONET v3 (status={cfg.eonet_status}, last {cfg.eonet_days}d)"
        + (", NASA FIRMS fire detections" if has_firms else "") + ".",
        f"Cross-source dedupe: events of the same normalized type within {cfg.dedupe_radius_km:g} km and "
        f"{cfg.dedupe_window_hours:g} h of each other are merged into one, keeping the higher-severity "
        "record's fields and the union of reporting sources (shown as e.g. \"gdacs+usgs\").",
        "Hazard metric (`geo_metrics`, name \"hazard\", per country, for today's UTC row only -- past "
        "days are never rewritten): the sum of the severity weights of events with a resolved country "
        f"that are active now -- ongoing (no end date) or ended within the last "
        f"{ACTIVE_WINDOW.total_seconds() / 3600:g} h -- so an open alert counts on every day it stays "
        "open. When GDACS and USGS both fetched cleanly, every other country gets an explicit 0 for "
        "the day, giving the risk map a daily baseline. Weights -- GDACS "
        "alert level: green=1, orange=4, red=10. USGS magnitude: <5=1, 5-5.9=2, 6-6.9=4, 7-7.9=8, "
        f">=8=16. EONET: a flat {EONET_WEIGHT:g} per open event (EONET doesn't grade severity uniformly "
        f"across categories). FIRMS: {FIRMS_WEIGHT_PER_DETECTION:g} per detection in a "
        f"{FIRMS_GRID_DEG:g}-degree/day cluster, capped at {FIRMS_WEIGHT_CAP:g}. A coordinate-only event "
        "(a quake at sea, an unattributed fire cluster) never invents a country and so never "
        "contributes to the metric, though it still appears on the map and in the list above.",
        "Country comes from a structured code when a source gives one (GDACS `iso3`, resolved through "
        "the ISO alpha-3 table, not `normalize_country`, which would wrongly reject real codes like "
        "ARM or GEO -- a GDACS event listing more than one iso3 code, e.g. a cyclone crossing a "
        "border, is attributed to every one, each crediting the full weight), or from the part of the "
        "USGS `place` text after the last comma (US states/DC map to \"US\" -- and Puerto Rico/Guam/the "
        "US Virgin Islands to their own code -- before a country name is even tried, since state codes "
        "like CA/PA/GA collide with real ISO country codes). EONET and FIRMS give coordinates only, so "
        "their events are never attributed to a country. \"..., Georgia\" is decided by longitude "
        "(the US state vs the country).",
        f"FIRMS is fetched with its own uncached fetcher (the key is in the URL, so nothing is written "
        f"to the web cache), capped at {cfg.firms_max_bytes / 1e6:g} MB and {cfg.firms_max_rows:,} "
        f"detection rows per run (area: {cfg.firms_area}).",
    ]
    if hazard_errors:
        lines.append(f"{len(hazard_errors)} hazard row(s) failed to write: " + "; ".join(hazard_errors[:10]))
    else:
        lines.append(f"{hazard_written} hazard row(s) written this run.")
    if zero_filled is False:
        lines.append(
            "Quiet countries got no hazard row today: " + ", ".join(s for s in COUNTRY_SOURCES if not stats.clean(s))
            + " did not fetch cleanly, so a 0 could hide a real event. Rows already written today were "
            "only ever raised by this run, never lowered."
        )
    method.text("\n\n".join(lines))
    if stats.errors:
        method.callout("\n\n".join(f"{s}: {e}" for s, e in stats.errors), label="fetch errors", tone="hot")
    if stats.firms_skipped:
        method.callout(f"FIRMS skipped: {stats.firms_skipped}.", label="FIRMS", tone="info")
    if stats.notes:
        method.callout("\n\n".join(stats.notes), label="caps", tone="warn")
    if cfg.warnings:
        method.callout("\n\n".join(cfg.warnings), label="config", tone="warn")
    return r


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _event_brief(e: Event) -> dict:
    """One `top_events` entry in `summary_json` -- just enough for the
    digest() hook to build an item without re-reading anything (#104):
    title parts, a source link, and whether it's GDACS red-level."""
    country = ", ".join(geo.country_name(c) or c for c in e.countries) if e.countries else None
    return {
        "type": e.type,
        "severity": e.severity,
        "place": e.place or None,
        "country": country,
        "started": e.started.isoformat(),
        "url": e.url,
    }


def run_once(conn, cfg: Config, fetcher: Fetcher, now: datetime, *,
             firms_fetcher: Fetcher | None = None) -> tuple[Report, dict]:
    events, stats = collect_events(fetcher, now, cfg, firms_fetcher=firms_fetcher)
    as_of = now.astimezone(timezone.utc).date()
    seen_before = _load_seen(conn)
    new_ids = _new_event_ids(events, seen_before)
    failed = {s for s, _ in stats.errors} | set(stats.unrecognized)
    _save_seen(conn, _next_seen(events, seen_before, failed), now)
    zero_fill = all(stats.clean(s) for s in COUNTRY_SOURCES)
    written, hazard_errors = write_hazard_metrics(conn, events, now, zero_fill=zero_fill)
    report = build_report(events, as_of, new_ids, written, hazard_errors, stats, cfg,
                          now=now, zero_filled=zero_fill)
    # Most severe first, newest breaking ties -- same ranking build_report's
    # "Events by severity" table uses. Kept in summary_json (bounded) so the
    # digest() hook below can list events without a store table of its own.
    ranked = sorted(events, key=lambda e: (-e.weight, -e.started.timestamp()))
    summary = {
        "events": len(events),
        "new": len(new_ids),
        "hazard_rows": written,
        "hazard_zero_filled": zero_fill,
        "hazard_errors": len(hazard_errors),
        "fetch_errors": len(stats.errors),
        "firms_skipped": stats.firms_skipped,
        "red_count": sum(1 for e in events if e.severity.lower() == "red"),
        "top_events": [_event_brief(e) for e in ranked[:DIGEST_EVENTS_CAP]],
    }
    return report, summary


def _summary_dict(summary_json: str | None) -> dict:
    try:
        data = json.loads(summary_json) if summary_json else {}
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def digest(run: RunRow, max_items: int) -> Digest:
    """Master report hook (#104): tiles and items built entirely from this
    run's own `summary_json` (aggregate counts plus the `top_events` list
    `run_once()` writes there) -- no store access, no network. Items are
    the most severe (then newest) events, already in that order; a callout
    flags any GDACS red-level event."""
    summary = _summary_dict(run.summary_json)
    events_n, new_n = summary.get("events", 0), summary.get("new", 0)
    red_n, errors_n = summary.get("red_count", 0), summary.get("fetch_errors", 0)
    tiles = [
        Tile("Active events", events_n, tone="info"),
        Tile("Red-level", red_n, tone="hot" if red_n else "muted"),
        Tile("New since last run", new_n, tone="hot" if new_n else "muted"),
        Tile("Fetch errors", errors_n, tone="hot" if errors_n else "muted"),
    ]

    items = []
    for e in summary.get("top_events") or []:
        if not isinstance(e, dict) or len(items) >= max_items:
            break
        label = f"{e.get('type') or 'event'} {e.get('severity') or ''}".strip()
        items.append(DigestItem(
            title=label,
            url=e.get("url") if isinstance(e.get("url"), str) else None,
            blurb=", ".join(p for p in (e.get("place"), e.get("country")) if p),
            tone="hot" if str(e.get("severity") or "").lower() == "red" else None,
        ))

    callouts = [f"{red_n} red-level event(s) active"] if red_n else []
    return Digest(tiles=tiles, items=items, callouts=callouts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disaster fusion report: GDACS + USGS earthquakes + NASA EONET (+ optional FIRMS)."
    )
    parser.parse_args()

    # No model calls: this task is pure data fusion, like tasks.risk_map.
    cfg = load_config()
    fetcher = Fetcher(USER_AGENT, per_host_delay=cfg.per_host_delay)
    with Store() as store, store.record_run("tasks.disasters") as run:
        report, run.summary = run_once(store.conn, cfg, fetcher, _now())
        run.report_path = report.write()
    print(run.report_path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
