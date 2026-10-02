"""Per-country daily metrics: the `geo_metrics` table and its baseline math.

One row per (UTC date, ISO alpha-2 country, metric name). Tasks that measure
something per country per day write here; `tasks.risk_map` reads every metric
back and z-scores it against its own trailing baseline.

    from lmsa import Store, geo_metrics
    with Store() as store:
        geo_metrics.record(store.conn, "2026-09-24", "UA", "hazard", 3.5)
        geo_metrics.series(store.conn, "UA", "hazard", start="2026-08-25")

Metric names in use: `conflict_volume`, `conflict_tone` (risk_map, from GDELT),
`hazard` (disasters, #59), `markets` (prediction markets, #60). A metric is a
raw daily level -- "higher = more stress" -- not a z-score; the baseline is
computed at read time, so a writer never needs history of its own.

Writes are upserts: re-recording the same (date, country, metric) replaces
the value, so reruns never duplicate rows.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date
from typing import Iterable, Sequence

from . import geo

__all__ = ["record", "record_many", "series", "latest_date", "zscore"]


def _day(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)).isoformat()  # validates, raises ValueError


def _row(day: date | str, country: str, metric: str, value: float) -> tuple[str, str, str, float]:
    c = geo.country(str(country))
    if c is None:
        raise ValueError(f"unknown country code {country!r} (ISO alpha-2 expected)")
    if not metric or not isinstance(metric, str):
        raise ValueError("metric must be a non-empty string")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value for {c.alpha2}/{metric}: {value!r}")
    return _day(day), c.alpha2, metric, v


def record(conn: sqlite3.Connection, day: date | str, country: str, metric: str, value: float,
           *, commit: bool = True) -> None:
    """Upsert one value. `country` is an ISO alpha-2 code (any case; resolved
    with `geo.country`, never `normalize_country`); `day` a date or
    YYYY-MM-DD. Raises ValueError for an unknown country, bad date or a
    non-finite value."""
    conn.execute(
        "INSERT OR REPLACE INTO geo_metrics(date, country, metric, value) VALUES (?, ?, ?, ?)",
        _row(day, country, metric, value),
    )
    if commit:
        conn.commit()


def record_many(conn: sqlite3.Connection, rows: Iterable[tuple[date | str, str, str, float]]) -> int:
    """Upsert (day, country, metric, value) rows in one transaction; returns
    how many were written. Validation is per row: a bad row raises before
    anything is written."""
    checked = [_row(*r) for r in rows]
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO geo_metrics(date, country, metric, value) VALUES (?, ?, ?, ?)",
            checked,
        )
    return len(checked)


def series(conn: sqlite3.Connection, country: str, metric: str, *,
           start: date | str | None = None, end: date | str | None = None) -> list[tuple[date, float]]:
    """(date, value) pairs for one country and metric, oldest first,
    optionally limited to start <= date <= end."""
    c = geo.country(str(country))
    if c is None:
        return []
    sql = "SELECT date, value FROM geo_metrics WHERE country=? AND metric=?"
    args: list[object] = [c.alpha2, metric]
    if start is not None:
        sql += " AND date >= ?"
        args.append(_day(start))
    if end is not None:
        sql += " AND date <= ?"
        args.append(_day(end))
    rows = conn.execute(sql + " ORDER BY date", args).fetchall()
    return [(date.fromisoformat(d), float(v)) for d, v in rows if v is not None]


def latest_date(conn: sqlite3.Connection, metric: str, country: str | None = None) -> date | None:
    """Newest date stored for `metric` (for one country, or any)."""
    if country is None:
        row = conn.execute("SELECT MAX(date) FROM geo_metrics WHERE metric=?", (metric,)).fetchone()
    else:
        c = geo.country(str(country))
        if c is None:
            return None
        row = conn.execute(
            "SELECT MAX(date) FROM geo_metrics WHERE metric=? AND country=?", (metric, c.alpha2)
        ).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def zscore(value: float, baseline: Sequence[float], *, min_std: float = 0.0,
           min_rel_std: float = 0.0) -> float | None:
    """(value - mean) / std of `baseline` (population std), with the std
    floored at max(min_std, min_rel_std * |mean|).

    The floor is what keeps a quiet month meaningful: 30 days of 0 then a
    spike would otherwise divide by zero. With a floor the score is finite
    and large. Without one (both floors 0) a flat baseline gives 0.0 when
    `value` equals it and None when it differs. None also for an empty
    baseline or a non-finite value."""
    xs = [float(x) for x in baseline if x is not None and math.isfinite(float(x))]
    if not xs or value is None or not math.isfinite(float(value)):
        return None
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    std = max(std, float(min_std), float(min_rel_std) * abs(mean))
    if std < 1e-12:
        return 0.0 if abs(float(value) - mean) < 1e-12 else None
    return (float(value) - mean) / std
