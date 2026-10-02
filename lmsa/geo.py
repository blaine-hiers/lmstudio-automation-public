"""Country tagging and the world-map projection, fully offline.

    >>> normalize_country("Britain"), normalize_country("Ukrainian"), normalize_country("DPRK")
    ('GB', 'UA', 'KP')
    >>> region_of("GB"), subregion_of("GB")
    ('Europe', 'Northern Europe')

Everything here reads package data under `lmsa/geo_data/` -- no network:

- `countries.json` -- ISO-3166 alpha-2/alpha-3, name, UN region/subregion,
  label point and alternate names, generated from Natural Earth admin-0
  (public domain) by `scripts/build_geo_data.py`.
- `aliases.json` -- hand-maintained aliases and demonyms ("UK", "Britain",
  "Ukrainian", "DPRK") plus a few ISO codes Natural Earth folds into a
  sovereign. Edit this one by hand; it needs no regeneration.
- `world_110m.json` -- Natural Earth 1:110m country outlines, pre-projected
  with `project()` (Equal Earth) and simplified to SVG path data keyed by
  alpha-2. `Section.world_map` in `lmsa.report` draws it.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "geo_data"

# Equal Earth projection (Savric, Patterson & Jenny 2018) mapped onto an SVG
# viewBox MAP_W wide. The shipped path data was projected with exactly these
# numbers, so markers placed with project() line up with the outlines.
_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796
_M = math.sqrt(3) / 2
_X_MAX = 2 * math.sqrt(3) * math.pi / (3 * _A1)          # x at lon=180, lat=0
_Y_MAX = _A4 * (math.pi / 3) ** 9 + _A3 * (math.pi / 3) ** 7 + _A2 * (math.pi / 3) ** 3 + _A1 * (math.pi / 3)
MAP_W = 1000.0
_SCALE = (MAP_W / 2) / _X_MAX
_HALF_H = _Y_MAX * _SCALE
MAP_H = round(2 * _HALF_H, 1)


def project(lat: float, lon: float) -> tuple[float, float]:
    """Project a WGS84 point to world-map SVG coordinates (Equal Earth).

    Returns (x, y) inside the `0 0 MAP_W MAP_H` viewBox: (0, 0) is top-left,
    lon=0/lat=0 is the centre. Out-of-range input is clamped.
    """
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    theta = math.asin(_M * math.sin(math.radians(lat)))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    x = (2 * math.sqrt(3) * math.radians(lon) * math.cos(theta)
         / (3 * (9 * _A4 * t6 * t2 + 7 * _A3 * t6 + 3 * _A2 * t2 + _A1)))
    y = theta * (_A4 * t6 * t2 + _A3 * t6 + _A2 * t2 + _A1)
    return MAP_W / 2 + x * _SCALE, _HALF_H - y * _SCALE


@dataclass(frozen=True)
class Country:
    alpha2: str
    alpha3: str
    name: str
    region: str
    subregion: str
    lat: float | None = None  # label point, used when the map has no outline
    lon: float | None = None


def _key(text: str) -> str:
    """Fold a country mention to a lookup key: case, accents, punctuation,
    a leading "the", "St." vs "Saint" and "&" vs "and" all compare equal."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = text.replace("&", " and ").replace("'", "").replace("’", "").replace(".", "")
    words = re.sub(r"[^0-9a-z]+", " ", text).split()
    if words and words[0] == "the":
        words = words[1:]
    return " ".join("saint" if w == "st" else w for w in words)


# Uppercase ISO codes that are far more often an acronym or an English word
# than a country ("IT" department, "AI", "TV", "ID", "CAN", "AND"...). A bare
# code on this list is not treated as a country; use country() for codes.
CODE_DENYLIST = frozenset({
    "IT", "AI", "TV", "ID", "IN", "IS", "TO", "NO", "ME", "AM", "BE", "DO", "AS", "AT",
    "BY", "MY", "SO", "PR", "PS", "CV", "CD", "MS", "MD", "PM", "FM", "LA",
    "CAN", "AND", "ARM", "PER", "COL", "FIN", "MAR", "PAN", "BEN", "GEO", "MAC", "COM",
    "LIE", "GIN", "TON", "VAT", "SUR",
})
# A plural is tried only for curated demonyms: aliases (never primary country
# names) ending in these, or the listed words. "Americas", "Koreas",
# "Congos", "Sudans" stay unknown because they are names, not demonyms.
_PLURAL_ENDINGS = ("an", "i", "er")
_PLURAL_WORDS = frozenset({"swede", "dane", "brit", "turk", "filipino", "finn", "serb", "croat",
                           "kazakh", "uzbek", "tajik", "slovak", "slovene", "kosovar", "emirati",
                           "czech", "greek", "spaniard", "argentine", "scot"})
_ARTICLE_RE = re.compile(r"^the\s+", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"['’]s?$")


@lru_cache(maxsize=1)
def _tables() -> tuple[dict[str, Country], dict[str, str], dict[str, str], frozenset[str]]:
    """(countries by alpha-2, alpha-3 -> alpha-2, name key -> alpha-2,
    demonym keys that may take a plural "s")."""
    raw = json.loads((DATA / "countries.json").read_text(encoding="utf-8"))
    curated = json.loads((DATA / "aliases.json").read_text(encoding="utf-8"))
    rows = list(raw["countries"]) + list(curated.get("extra", []))
    countries: dict[str, Country] = {}
    alpha3: dict[str, str] = {}
    names: dict[str, str] = {}
    later: list[tuple[str, str]] = []
    for row in rows:
        a2 = row["alpha2"]
        if a2 in countries:
            continue
        countries[a2] = Country(a2, row["alpha3"], row["name"], row["region"], row["subregion"],
                                row.get("lat"), row.get("lon"))
        alpha3[row["alpha3"]] = a2
        names.setdefault(_key(row["name"]), a2)  # primary names win over everything
        later.extend((n, a2) for n in row.get("names", ()))
    primary = set(names)
    demonyms: set[str] = set()
    for a2, aliases in curated["aliases"].items():  # curated aliases next
        for alias in aliases:
            k = _key(alias)
            names.setdefault(k, a2)
            last = k.rsplit(" ", 1)[-1]
            if k not in primary and (last.endswith(_PLURAL_ENDINGS) or last in _PLURAL_WORDS):
                demonyms.add(k)
    for name, a2 in later:  # Natural Earth's long/formal/alt names last
        names.setdefault(_key(name), a2)
    names.pop("", None)
    return countries, alpha3, names, frozenset(demonyms)


def normalize_country(text: str | None) -> str | None:
    """Map a country mention -- name, alias, demonym, or an UPPERCASE ISO
    alpha-2/alpha-3 code -- to its ISO 3166-1 alpha-2 code, or None.

    Pass an extracted country mention (or an uppercase code), not arbitrary
    words: "UK", "Britain", "British" -> "GB"; "Ukrainian(s)" -> "UA";
    "DPRK" -> "KP"; "USA"/"US" -> "US"; "Côte d'Ivoire" -> "CI".

    - Bare codes, and any alias of 3 letters or fewer ("UK", "U.S."), only
      match when the input has no lowercase letters, so "in", "it", "us",
      "Us", "can", "and" are not countries.
    - Codes in `CODE_DENYLIST` ("IT", "AI", "TV", "ID", "CAN", "AND", ...)
      never match as bare codes; resolve a known code with `country()`.
    - A leading "the" and a possessive "'s" are ignored ("the UK",
      "Russia's"), including for the case rule above.
    - A plural is only tried for demonyms ("Ukrainians", "North Koreans",
      "New Zealanders"), so "the Americas", "Koreas", "Congos" -> None.
    - Ambiguous words resolve to the most common reading ("Korea"/"Korean"
      -> "KR", "Congo" -> "CG", "Dominican" -> "DO").
    """
    if not text or not str(text).strip():
        return None
    countries, alpha3, names, demonyms = _tables()
    raw = str(text).strip()
    # An article and a possessive are not part of the mention: "the UK",
    # "Russia's", "U.S.'s" are judged (case rule included) without them.
    raw = _POSSESSIVE_RE.sub("", _ARTICLE_RE.sub("", raw)).strip()
    if not raw:
        return None
    shouting = not any(c.islower() for c in raw)
    if raw.isascii() and raw.isalpha() and shouting and raw not in CODE_DENYLIST:
        if len(raw) == 2 and raw in countries:
            return raw
        if len(raw) == 3 and raw in alpha3:
            return alpha3[raw]
    key = _key(raw)
    if len(key.replace(" ", "")) <= 3 and not shouting:
        return None
    if key in names:
        return names[key]
    if key.endswith("s") and key[:-1] in demonyms:
        return names[key[:-1]]
    return None


def country(alpha2: str) -> Country | None:
    """The table row for an alpha-2 code (case-insensitive), or None."""
    return _tables()[0].get(str(alpha2 or "").upper())


def country_name(alpha2: str) -> str | None:
    c = country(alpha2)
    return c.name if c else None


def region_of(alpha2: str) -> str | None:
    """UN region ("Africa", "Americas", "Asia", "Europe", "Oceania",
    "Antarctica") for an alpha-2 code, or None if unknown."""
    c = country(alpha2)
    return c.region if c else None


def subregion_of(alpha2: str) -> str | None:
    """UN subregion ("Northern Europe", "South-Eastern Asia", ...) or None."""
    c = country(alpha2)
    return c.subregion if c else None


def countries() -> list[Country]:
    """Every known country, sorted by alpha-2."""
    return [c for _, c in sorted(_tables()[0].items())]


@lru_cache(maxsize=1)
def world_paths() -> dict[str, str]:
    """alpha-2 -> SVG path data for the 1:110m outlines (projected)."""
    return json.loads((DATA / "world_110m.json").read_text(encoding="utf-8"))["paths"]


# -- reverse geocoding (nearest_country) ---------------------------------------
#
# `world_paths()` stores each country's 1:110m outline as SVG path data,
# already projected into the same Equal Earth space `project()` produces,
# written as one or more closed rings ("M<x> <y>l<dx> <dy>...z" -- see
# `scripts/build_geo_data.py:_ring_path`, absolute move + relative linetos).
# Reversing a lat/lon to a country reuses that same data (no separate
# boundary dataset to ship or keep in sync): project the point the same way
# the outlines were built, then test it against the parsed rings.

_PATH_NUM_RE = re.compile(r"-?\d*\.?\d+")


def _parse_path_rings(d: str) -> list[list[tuple[float, float]]]:
    """Each closed ring in an SVG path built by `_ring_path` (`M` absolute
    start + relative `l` lineto pairs, one or more rings concatenated with no
    separator) as a list of absolute `(x, y)` points, ring-closing point not
    repeated. Malformed/too-short rings (fewer than 3 points) are skipped."""
    rings = []
    for ring_text in re.findall(r"M[^M]*?z", d):
        nums = [float(n) for n in _PATH_NUM_RE.findall(ring_text)]
        if len(nums) < 6 or len(nums) % 2:  # need >= 3 points (x,y pairs)
            continue
        x, y = nums[0], nums[1]
        pts = [(x, y)]
        for i in range(2, len(nums), 2):
            x, y = x + nums[i], y + nums[i + 1]
            pts.append((x, y))
        if len(pts) >= 3:
            rings.append(pts)
    return rings


@lru_cache(maxsize=1)
def _country_polygons() -> dict[str, tuple[tuple[float, float, float, float], list[list[tuple[float, float]]]]]:
    """alpha-2 -> (bbox, rings), parsed once from `world_paths()` and cached
    for the process lifetime. `bbox` is `(min_x, min_y, max_x, max_y)` over
    every ring, used to cheaply reject a country before testing its edges."""
    result = {}
    for a2, d in world_paths().items():
        rings = _parse_path_rings(d)
        if not rings:
            continue
        xs = [x for ring in rings for x, _ in ring]
        ys = [y for ring in rings for _, y in ring]
        result[a2] = ((min(xs), min(ys), max(xs), max(ys)), rings)
    return result


def _point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    """Standard even-odd ray-casting test, one ring."""
    inside = False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _point_in_rings(x: float, y: float, rings: list[list[tuple[float, float]]]) -> bool:
    """Even-odd across every ring of a country: correctly handles a
    multipolygon (an archipelago's islands are independent rings) and a
    polygon-with-holes (a hole ring flips the result back to outside) with
    the same toggle."""
    inside = False
    for ring in rings:
        if _point_in_ring(x, y, ring):
            inside = not inside
    return inside


def _dist_point_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _units_per_km(lat: float) -> float:
    """Approx projected units per km near latitude `lat`.

    Equal Earth is area-preserving, not conformal, so its local linear scale
    varies by latitude and direction -- rather than deriving that in closed
    form, this samples `project()` a small step away in each direction and
    takes the larger (longitude degrees shrink at higher latitudes, so a
    "how far in projected units is 50 km" question needs the more generous
    of the two to avoid under-snapping near the poles).
    """
    eps = 0.05  # degrees
    x0, y0 = project(lat, 0.0)
    x_lon, y_lon = project(lat, eps)
    x_lat, y_lat = project(min(lat + eps, 90.0), 0.0)
    per_deg_lon = math.hypot(x_lon - x0, y_lon - y0) / eps
    per_deg_lat = math.hypot(x_lat - x0, y_lat - y0) / eps
    per_degree = max(per_deg_lon, per_deg_lat, 1e-9)
    return per_degree / 111.32  # ~km per degree of latitude (~= of longitude at the equator)


def nearest_country(lat: float, lon: float, *, max_snap_km: float = 50.0) -> str | None:
    """Cheap offline reverse-geocode: the alpha-2 of the country whose
    1:110m outline (`world_paths()`) contains `(lat, lon)`, or None.

    Point-in-polygon (even-odd, multipolygon/hole aware -- see
    `_point_in_rings`), tested in the same projected space the outlines are
    stored in: `(lat, lon)` is run through the same `project()` used to
    build them. A point that lands in no polygon at all -- open ocean, or a
    coastal point the 1:110m simplification trimmed just outside the
    coastline -- snaps to the nearest polygon edge within `max_snap_km`
    (~a country's width at this map's resolution); beyond that, None rather
    than guessing a distant country.

    Polygons are parsed from `world_paths()` once per process and cached
    (`_country_polygons`); each lookup bbox-rejects most countries before
    testing any edges, so this stays fast enough for a photo library's worth
    of calls. Not authoritative -- 1:110m outlines are simplified, and a
    point right on a real border can land on either side.
    """
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return None
    x, y = project(lat_f, lon_f)
    polys = _country_polygons()

    for a2, (bbox, rings) in polys.items():
        min_x, min_y, max_x, max_y = bbox
        if min_x <= x <= max_x and min_y <= y <= max_y and _point_in_rings(x, y, rings):
            return a2

    max_snap_units = max_snap_km * _units_per_km(lat_f)
    best_a2, best_d = None, math.inf
    for a2, (bbox, rings) in polys.items():
        min_x, min_y, max_x, max_y = bbox
        if x < min_x - max_snap_units or x > max_x + max_snap_units:
            continue
        if y < min_y - max_snap_units or y > max_y + max_snap_units:
            continue
        for ring in rings:
            n = len(ring)
            for i in range(n):
                x1, y1 = ring[i]
                x2, y2 = ring[(i + 1) % n]
                d = _dist_point_to_segment(x, y, x1, y1, x2, y2)
                if d < best_d:
                    best_d, best_a2 = d, a2
    return best_a2 if best_a2 is not None and best_d <= max_snap_units else None
