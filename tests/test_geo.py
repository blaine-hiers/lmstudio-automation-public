"""lmsa.geo (country tagging, Equal Earth projection) and Section.world_map."""

from __future__ import annotations

import math
import re

import pytest

from lmsa import Marker, Report, geo


@pytest.mark.parametrize("text, code", [
    ("UK", "GB"), ("U.K.", "GB"), ("Britain", "GB"), ("British", "GB"), ("United Kingdom", "GB"),
    ("Ukrainian", "UA"), ("Ukrainians", "UA"), ("Ukraine", "UA"), ("the Ukraine", "UA"),
    ("DPRK", "KP"), ("North Korea", "KP"), ("South Korea", "KR"),
    ("USA", "US"), ("US", "US"), ("U.S.", "US"), ("United States", "US"), ("American", "US"),
    ("Americans", "US"), ("Swedes", "SE"), ("GB", "GB"),
    ("DEU", "DE"), ("germany", "DE"), ("Côte d’Ivoire", "CI"), ("Ivory Coast", "CI"),
    ("St. Kitts and Nevis", "KN"), ("Czech Republic", "CZ"), ("Kosovo", "XK"),
    ("Réunion", "RE"), ("Burma", "MM"), ("Türkiye", "TR"),
])
def test_normalize_country(text, code):
    assert geo.normalize_country(text) == code


@pytest.mark.parametrize("text", [
    None, "", "   ", "Atlantis", "ZZ", "QQQ", "Europe",
    # English words are not codes: bare codes need uppercase input
    "in", "it", "and", "can", "us", "It", "Us", "uk", "Lao",
    # acronyms that collide with codes are denied even in uppercase
    "IT", "AI", "TV", "ID", "CAN", "AND", "ARM", "PER", "COL", "FIN", "MAR", "PAN",
    # plurals of a region / of several countries
    "Americas", "the Americas", "Koreas", "Congos", "Sudans", "Guineas",
])
def test_normalize_country_unknown_is_none(text):
    assert geo.normalize_country(text) is None


def test_region_and_table():
    assert geo.region_of("GB") == "Europe" and geo.subregion_of("gb") == "Northern Europe"
    assert geo.region_of("KP") == "Asia" and geo.region_of("BR") == "Americas"
    assert geo.region_of("ZZ") is None and geo.subregion_of("") is None
    codes = {c.alpha2 for c in geo.countries()}
    assert len(codes) >= 245 and {"US", "FR", "NO", "XK", "SG", "GF"} <= codes
    assert geo.country("fr").alpha3 == "FRA" and geo.country_name("US") == "United States of America"


def test_projection_of_known_points():
    cx, cy = geo.MAP_W / 2, geo.MAP_H / 2
    assert geo.project(0, 0) == pytest.approx((cx, cy), abs=0.05)
    assert geo.project(0, 180) == pytest.approx((geo.MAP_W, cy), abs=0.05)
    assert geo.project(0, -180) == pytest.approx((0, cy), abs=0.05)
    assert geo.project(90, 0) == pytest.approx((cx, 0), abs=0.05)
    assert geo.project(-90, 0) == pytest.approx((cx, geo.MAP_H), abs=0.05)
    # Equal Earth is equal-area: a 10x10 degree cell near the pole covers the
    # same share of the map as of the sphere, relative to one at the equator.
    def cell_area(lat0: float) -> float:
        ring = ([(lat0, 10 * i / 20) for i in range(21)] + [(lat0 + 10 * i / 20, 10) for i in range(21)]
                + [(lat0 + 10, 10 - 10 * i / 20) for i in range(21)] + [(lat0 + 10 - 10 * i / 20, 0) for i in range(21)])
        pts = [geo.project(la, lo) for la, lo in ring]
        return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2

    sphere = (math.sin(math.radians(70)) - math.sin(math.radians(60))) / math.sin(math.radians(10))
    assert cell_area(60) / cell_area(0) == pytest.approx(sphere, rel=2e-3)
    # symmetric, and out-of-range input is clamped
    x1, y1 = geo.project(30, 60)
    x2, y2 = geo.project(-30, -60)
    assert (x1 + x2, y1 + y2) == pytest.approx((geo.MAP_W, geo.MAP_H), abs=0.05)
    assert geo.project(120, 400) == geo.project(90, 180)


def _bbox(d: str) -> tuple[float, float, float, float]:
    """Bounding box of the shipped path data (M abs, l relative pairs, z)."""
    xs, ys = [], []
    for sub in re.findall(r"M([^z]*)z", d):
        start, _, rel = sub.partition("l")
        x, y = map(float, re.findall(r"-?(?:\d+\.?\d*|\.\d+)", start))
        xs.append(x), ys.append(y)
        nums = list(map(float, re.findall(r"-?(?:\d+\.?\d*|\.\d+)", rel)))
        for dx, dy in zip(nums[::2], nums[1::2]):
            x, y = x + dx, y + dy
            xs.append(x), ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


@pytest.mark.parametrize("cc, lat, lon", [
    ("FR", 48.86, 2.35), ("US", 39.1, -94.6), ("AU", -25.3, 133.8), ("BR", -15.8, -47.9), ("JP", 35.7, 139.7),
])
def test_projected_points_land_in_their_country(cc, lat, lon):
    x0, y0, x1, y1 = _bbox(geo.world_paths()[cc])
    x, y = geo.project(lat, lon)
    assert x0 <= x <= x1 and y0 <= y <= y1


def test_package_data_is_compact_and_keyed_by_known_codes():
    size = (geo.DATA / "world_110m.json").stat().st_size
    assert size < 250_000
    paths = geo.world_paths()
    assert len(paths) >= 170 and all(geo.country(cc) for cc in paths)
    assert all(re.fullmatch(r"[0-9Mlz .\-]+", d) for d in paths.values())


def _map_html(**kw) -> str:
    html = Report("r").section("s").world_map(**kw)._parts[-1]
    assert html.startswith('<figure class="geo-map">')
    return html


def test_world_map_shades_links_and_escapes():
    html = _map_html(
        values={"us": 10, "GB": 5, "SG": 7},
        links={"GB": "#country-gb", "US": "javascript:alert(1)"},
        markers=[Marker(51.5, -0.1, "<script>alert(1)</script>", href="javascript:x")],
        legend="stories <b>",
    )
    assert 'class="geo-c geo-s4" data-cc="US"' in html
    assert '<a href="#country-gb"><path class="geo-c geo-s0" data-cc="GB"' in html
    assert "javascript:" not in html
    assert "<script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "stories &lt;b&gt;" in html
    assert "<title>United States of America: 10</title>" in html
    assert '<circle class="geo-c geo-dot geo-s2" data-cc="SG"' in html  # too small for 1:110m
    assert 'class="geo-marker t-hot"' in html
    assert "http" not in html and "url(" not in html  # renders with no network
    assert "geo-note" not in html


def test_world_map_ignores_unknown_codes_with_a_count():
    html = _map_html(values={"ZZ": 1, "Atlantis": 2, "FR": 3, "DE": float("nan")},
                     links={"QQ": "#x"}, markers=[Marker(float("nan"), 0)])
    assert "ignored: 3 unknown country codes, 1 non-numeric value, 1 marker without coordinates" in html
    assert 'data-cc="ZZ"' not in html and 'class="geo-c geo-s4" data-cc="FR"' in html


def test_world_map_diverging_and_empty():
    html = _map_html(values={"US": -4, "BR": 4, "IN": 0.1, "JP": -1.5}, scale="diverging", legend=False)
    assert 'geo-dn2" data-cc="US"' in html and 'geo-dp2" data-cc="BR"' in html
    assert 'geo-d0" data-cc="IN"' in html and 'geo-dn1" data-cc="JP"' in html
    assert "geo-legend" not in html
    empty = _map_html()
    assert "geo-legend" not in empty and 'geo-none" data-cc="US"' in empty
    with pytest.raises(ValueError):
        _map_html(scale="rainbow")


def test_world_map_survives_overflow_inf_and_odd_marker_sizes():
    html = _map_html(values={"US": 10 ** 400, "FR": float("inf"), "DE": 2},
                     markers=[Marker(0, 0, "zero", size=0), Marker(10, 10, "huge", size=float("inf")),
                              Marker(20, 20, "nan", size=float("nan")), Marker(30, 30, "big", size=10 ** 400)])
    assert "ignored: 2 non-numeric values" in html and "without coordinates" not in html
    assert 'geo-s4" data-cc="DE"' in html
    radii = re.findall(r'class="geo-marker t-hot" cx="[^"]+" cy="[^"]+" r="([^"]+)"', html)
    assert radii == ["1.1", "17.5", "3.5", "17.5"]  # 0 -> min, inf -> max, NaN -> default


# (city, lat, lon, expected alpha-2). Includes every case the nearest-centroid
# implementation this replaced got wrong (NYC, LA, Miami, Seattle, Honolulu,
# Sydney, Istanbul, London, Barcelona, Naples, Toronto, Stockholm, Helsinki,
# Boston, Anchorage -- a centroid can be closer to a small, distant country
# than to the true, oddly-shaped one the point is actually inside), plus
# further coastal and interior points across every populated continent.
_CITIES = [
    ("NYC", 40.7128, -74.0060, "US"), ("LA", 34.0522, -118.2437, "US"),
    ("Miami", 25.7617, -80.1918, "US"), ("Seattle", 47.6062, -122.3321, "US"),
    ("Honolulu", 21.3069, -157.8583, "US"), ("Boston", 42.3601, -71.0589, "US"),
    ("Anchorage", 61.2181, -149.9003, "US"),
    ("Sydney", -33.8688, 151.2093, "AU"), ("Istanbul", 41.0082, 28.9784, "TR"),
    ("London", 51.5072, -0.1276, "GB"), ("Barcelona", 41.3874, 2.1686, "ES"),
    ("Naples", 40.8518, 14.2681, "IT"), ("Toronto", 43.6532, -79.3832, "CA"),
    ("Stockholm", 59.3293, 18.0686, "SE"), ("Helsinki", 60.1699, 24.9384, "FI"),
    ("Paris", 48.8566, 2.3522, "FR"), ("Berlin", 52.5200, 13.4050, "DE"),
    ("Madrid", 40.4168, -3.7038, "ES"), ("Rome", 41.9028, 12.4964, "IT"),
    ("Tokyo", 35.6762, 139.6503, "JP"), ("Beijing", 39.9042, 116.4074, "CN"),
    ("Moscow", 55.7558, 37.6173, "RU"), ("Cairo", 30.0444, 31.2357, "EG"),
    ("Lagos", 6.5244, 3.3792, "NG"), ("Nairobi", -1.2921, 36.8219, "KE"),
    ("Mumbai", 19.0760, 72.8777, "IN"), ("Delhi", 28.7041, 77.1025, "IN"),
    ("Bangkok", 13.7563, 100.5018, "TH"), ("Jakarta", -6.2088, 106.8456, "ID"),
    ("Manila", 14.5995, 120.9842, "PH"), ("Seoul", 37.5665, 126.9780, "KR"),
    ("Mexico City", 19.4326, -99.1332, "MX"), ("Rio", -22.9068, -43.1729, "BR"),
    ("Buenos Aires", -34.6037, -58.3816, "AR"), ("Vancouver", 49.2827, -123.1207, "CA"),
    ("Chicago (interior)", 41.8781, -87.6298, "US"), ("Denver (interior)", 39.7392, -104.9903, "US"),
]


@pytest.mark.parametrize("city, lat, lon, expected", _CITIES, ids=[c[0] for c in _CITIES])
def test_nearest_country_real_cities(city, lat, lon, expected):
    assert geo.nearest_country(lat, lon) == expected


def test_nearest_country_is_point_in_polygon_not_nearest_centroid():
    """The bug this replaced: a point can be much closer to a small, distant
    country's *label point* than to the true (oddly-shaped) country's own
    label point, while still sitting well inside the true country's actual
    outline. Naples (Italy) was nearest-centroid-closer to Vatican City."""
    assert geo.nearest_country(40.8518, 14.2681) == "IT"  # Naples, not VA


def test_parse_path_rings_recovers_absolute_points_from_relative_lineto():
    # M<abs x><abs y>l<dx><dy><dx><dy>...z -- exactly what scripts/build_geo_data.py emits.
    d = "M10 20l5 0 0 5-5 0z"
    assert geo._parse_path_rings(d) == [[(10.0, 20.0), (15.0, 20.0), (15.0, 25.0), (10.0, 25.0)]]


def test_parse_path_rings_handles_multiple_concatenated_rings():
    d = "M0 0l1 0 0 1-1 0zM10 10l1 0 0 1-1 0z"
    assert len(geo._parse_path_rings(d)) == 2


def test_parse_path_rings_skips_a_degenerate_ring():
    assert geo._parse_path_rings("M0 0l1 0z") == []  # only 2 points -- not a polygon


def test_point_in_rings_even_odd_handles_a_hole():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    assert geo._point_in_rings(2, 2, [outer, hole]) is True  # inside the outer ring, outside the hole
    assert geo._point_in_rings(5, 5, [outer, hole]) is False  # inside the hole -> not inside the country


def test_dist_point_to_segment_endpoints_and_perpendicular():
    assert geo._dist_point_to_segment(0, 5, 0, 0, 0, 10) == pytest.approx(0.0)  # on the segment
    assert geo._dist_point_to_segment(3, 5, 0, 0, 0, 10) == pytest.approx(3.0)  # perpendicular
    assert geo._dist_point_to_segment(0, -5, 0, 0, 0, 10) == pytest.approx(5.0)  # past an endpoint


def test_nearest_country_snaps_a_coastal_point_to_the_nearest_edge_within_range(monkeypatch):
    """A point that falls just outside a (simplified) polygon -- e.g. a photo
    taken right at a coastline the 1:110m data smoothed away -- still
    resolves via the nearest-edge fallback rather than coming back None."""
    lat0, lon0 = 10.0, 10.0
    x0, y0 = geo.project(lat0, lon0)
    offset = 5.0 * geo._units_per_km(lat0)  # ~5 km away: within the 50 km default
    square = [(x0 + offset, y0 - 1), (x0 + offset + 2, y0 - 1), (x0 + offset + 2, y0 + 1), (x0 + offset, y0 + 1)]
    bbox = (x0 + offset, y0 - 1, x0 + offset + 2, y0 + 1)
    monkeypatch.setattr(geo, "_country_polygons", lambda: {"ZZ": (bbox, [square])})

    assert geo.nearest_country(lat0, lon0) == "ZZ"


def test_nearest_country_none_when_too_far_from_any_edge(monkeypatch):
    lat0, lon0 = 10.0, 10.0
    x0, y0 = geo.project(lat0, lon0)
    offset = 500.0 * geo._units_per_km(lat0)  # ~500 km away: past the 50 km default
    square = [(x0 + offset, y0 - 1), (x0 + offset + 2, y0 - 1), (x0 + offset + 2, y0 + 1), (x0 + offset, y0 + 1)]
    bbox = (x0 + offset, y0 - 1, x0 + offset + 2, y0 + 1)
    monkeypatch.setattr(geo, "_country_polygons", lambda: {"ZZ": (bbox, [square])})

    assert geo.nearest_country(lat0, lon0) is None


def test_nearest_country_micro_state_without_its_own_outline_resolves_to_neighbor():
    """Singapore (and Vatican/Monaco/...) is too small to have its own
    1:110m outline (see Section.world_map's docstring); a point there
    resolves to whatever real polygon contains/borders it instead of to
    itself -- a documented limit of an offline, outline-based lookup."""
    assert geo.nearest_country(1.3521, 103.8198) != "SG"
    assert geo.nearest_country(1.3521, 103.8198) is not None  # still lands on a real neighbor


def test_nearest_country_none_for_bad_input():
    assert geo.nearest_country(float("nan"), 0) is None
    assert geo.nearest_country(None, None) is None


def test_nearest_country_none_far_out_at_sea():
    assert geo.nearest_country(0.0, -140.0) is None  # mid-Pacific, nowhere near land


def test_nearest_country_results_are_cached_and_fast(monkeypatch):
    """`_country_polygons()` parses `world_paths()` once; a lookup must stay
    fast enough for a photo library's worth of calls."""
    import time

    geo.nearest_country(0, 0)  # warm the cache
    t0 = time.perf_counter()
    for _ in range(500):
        geo.nearest_country(40.7128, -74.0060)  # a coastal point (exercises the edge-snap fallback)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"500 lookups took {elapsed:.2f}s -- polygons aren't being cached"


def test_anchor_is_escaped():
    s = Report("r").section("s").anchor('a"b')
    assert s._parts[-1] == '<span class="anchor" id="a&quot;b"></span>'


def test_style_preview_shows_three_maps(tmp_path, monkeypatch):
    import tasks.style_preview as style_preview

    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    style_preview.main()
    page = (tmp_path / "style-preview.html").read_text(encoding="utf-8")
    assert page.count('<figure class="geo-map">') == 3
    assert "geo-dp2" in page and "geo-marker" in page and 'id="geo-ukraine"' in page


@pytest.mark.parametrize("text,code", [
    ("the UK", "GB"), ("The U.S.", "US"), ("the USA", "US"), ("the UAE", "AE"), ("the DRC", "CD"),
    ("Russia's", "RU"), ("Russia’s", "RU"), ("China's", "CN"), ("Belarus's", "BY"), ("UK's", "GB"),
    ("U.S.'s", "US"), ("North Koreans", "KP"), ("South Africans", "ZA"), ("New Zealanders", "NZ"),
    ("Sri Lankans", "LK"), ("Czechs", "CZ"), ("Greeks", "GR"), ("Spaniards", "ES"), ("Argentines", "AR"),
    ("Scots", "GB"), ("Gaza Strip", "PS"), ("State of Palestine", "PS"),
    ("Democratic Republic of Congo", "CD"), ("Republic of Ireland", "IE"), ("North Macedonian", "MK"),
])
def test_articles_possessives_and_demonym_plurals_resolve(text, code):
    assert geo.normalize_country(text) == code


@pytest.mark.parametrize("text", ["the Americas", "the it", "it's", "us's", "the", "the Koreas"])
def test_article_and_possessive_stripping_adds_no_false_positives(text):
    assert geo.normalize_country(text) is None
