"""Regenerate lmsa/geo_data/{countries,world_110m}.json from Natural Earth.

Natural Earth admin-0 countries are public domain (naturalearthdata.com).
This is a one-off maintenance script -- the runtime and the tests only read
the generated JSON and never touch the network.

    python scripts/build_geo_data.py --download
    python scripts/build_geo_data.py --src-110m ne_110m.geojson --src-50m ne_50m.geojson

- countries.json: the country table (alpha-2/3, name = ADMIN, UN region/subregion,
  label point, alternate names) from the 1:50m layer, which lists more small
  countries than 1:110m. `ISO_A2_EH`/`ISO_A3_EH` are used because the plain
  ISO fields are -99 for France, Norway and Kosovo.
- world_110m.json: 1:110m outlines projected with `lmsa.geo.project` (Equal
  Earth), simplified (Douglas-Peucker, --tolerance viewBox units), rounded
  to one decimal and written as relative SVG path data keyed by alpha-2.
  Features without an ISO code are folded into their sovereign (N. Cyprus ->
  CY, Somaliland -> SO).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmsa.geo import DATA, MAP_H, MAP_W, project  # noqa: E402

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
       "geojson/ne_{scale}_admin_0_countries.geojson")
FOLD = {"N. Cyprus": "CY", "Somaliland": "SO"}
A3_FALLBACK = {"XK": "XKX"}  # Kosovo: user-assigned code, the one most data sources use
NAME_FIELDS = ("NAME", "NAME_EN", "NAME_LONG", "BRK_NAME", "GEOUNIT", "SUBUNIT",
               "FORMAL_EN", "NAME_SORT", "NAME_ALT", "NAME_CIAWF")


def _code(props: dict) -> str | None:
    a2 = props.get("ISO_A2_EH")
    if a2 and a2 != "-99":
        return a2
    return FOLD.get(props.get("NAME"))


def build_countries(features: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for f in features:
        p = f["properties"]
        a2, a3 = p.get("ISO_A2_EH"), p.get("ISO_A3_EH")
        if a3 == "-99":
            a3 = A3_FALLBACK.get(a2, a3)
        if not a2 or a2 == "-99" or not a3 or a3 == "-99":
            continue
        if a2 in rows and not p.get("HOMEPART") == 1:
            continue  # e.g. Australia's Indian Ocean territories: keep the homepart row
        names: list[str] = []
        for field in NAME_FIELDS:
            v = (p.get(field) or "").strip()
            if v and v != p["ADMIN"] and v not in names:
                names.append(v)
        rows[a2] = {
            "alpha2": a2, "alpha3": a3, "name": p["ADMIN"],
            "region": p.get("REGION_UN") or "", "subregion": p.get("SUBREGION") or "",
            "lat": round(p["LABEL_Y"], 2), "lon": round(p["LABEL_X"], 2),
            "names": names,
        }
    return [rows[k] for k in sorted(rows)]


def _perp(p, a, b) -> float:
    (x, y), (x1, y1), (x2, y2) = p, a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / (dx * dx + dy * dy) ** 0.5


def _simplify(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        best, idx = 0.0, None
        for k in range(i + 1, j):
            d = _perp(pts[k], pts[i], pts[j])
            if d > best:
                best, idx = d, k
        if idx is not None and best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(pts, keep) if k]


def _ring_path(ring: list[list[float]], tol: float) -> str:
    pts = [project(lat, lon) for lon, lat in ring]
    pts = _simplify(pts, tol)
    q = []
    for x, y in pts:
        p = (round(x * 10), round(y * 10))  # tenths of a viewBox unit
        if not q or q[-1] != p:
            q.append(p)
    if len(q) > 1 and q[0] == q[-1]:
        q.pop()
    if len(q) < 3:
        return ""

    def num(v: int) -> str:
        s = f"{v / 10:.1f}".rstrip("0").rstrip(".")
        return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s

    out = [f"M{num(q[0][0])} {num(q[0][1])}l"]
    prev = q[0]
    for p in q[1:]:
        dx, dy = p[0] - prev[0], p[1] - prev[1]
        sx, sy = num(dx), num(dy)
        out.append((sx if sx.startswith("-") or len(out) == 1 else " " + sx)
                   + (sy if sy.startswith("-") else " " + sy))
        prev = p
    return "".join(out) + "z"


def build_paths(features: list[dict], tol: float) -> dict[str, str]:
    paths: dict[str, str] = {}
    for f in features:
        a2 = _code(f["properties"])
        if not a2:
            continue
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        d = "".join(_ring_path(ring, tol) for poly in polys for ring in poly)
        if d:
            paths[a2] = paths.get(a2, "") + d
    return dict(sorted(paths.items()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="fetch both layers from the Natural Earth GitHub repo")
    ap.add_argument("--src-110m", type=Path)
    ap.add_argument("--src-50m", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.2)
    args = ap.parse_args(argv)
    if args.download:
        tmp = Path(tempfile.mkdtemp())
        for scale in ("110m", "50m"):
            dest = tmp / f"ne_{scale}.geojson"
            urllib.request.urlretrieve(URL.format(scale=scale), dest)  # noqa: S310 - fixed https URL
            setattr(args, f"src_{scale}", dest)
    if not (args.src_110m and args.src_50m):
        ap.error("give --download or both --src-110m and --src-50m")
    f110 = json.loads(args.src_110m.read_text(encoding="utf-8"))["features"]
    f50 = json.loads(args.src_50m.read_text(encoding="utf-8"))["features"]

    source = "Natural Earth admin-0 countries (public domain), naturalearthdata.com"
    countries = {"source": source + " 1:50m", "countries": build_countries(f50)}
    world = {"source": source + " 1:110m; Equal Earth; generated by scripts/build_geo_data.py",
             "viewBox": [0, 0, MAP_W, MAP_H], "paths": build_paths(f110, args.tolerance)}
    for name, obj in (("countries.json", countries), ("world_110m.json", world)):
        path = DATA / name
        path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
