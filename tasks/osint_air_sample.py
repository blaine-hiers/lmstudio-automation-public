"""Sample adsb.lol's public military-flagged aircraft feed (#58).

Model "none", cheap, meant to run every dispatcher tick: one bounded GET of
the free `/v2/mil` endpoint (military-flagged aircraft currently tracked
worldwide, no auth), storing a compact, privacy-bucketed sample per
aircraft: hex, ICAO type code, callsign, lat/lon bucketed to 1 degree, and
barometric altitude. Samples older than RETENTION_DAYS are pruned every run.

This is data collection only -- it never renders a report or a per-aircraft
track list. `tasks.osint_movements` aggregates these rows into counts by
type class and region for the daily report.

    python -m tasks.osint_air_sample
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from lmsa import Store
from lmsa.web import Fetcher

# No SETUP block: this task calls no model and needs no config/secrets/command
# (same as tasks/server_logs.py, also model "none" with nothing to declare).

ADSB_MIL_URL = "https://api.adsb.lol/v2/mil"
USER_AGENT = "lmsa-osint-air-sample/1.0 (+public aggregate use only)"
# adsb.lol's /v2/mil is normally a few hundred KB; this is a hard backstop
# against a misbehaving or compromised endpoint, not a realistic size.
MAX_RESPONSE_BYTES = 5_000_000
RETENTION_DAYS = 90


def default_fetcher() -> Fetcher:
    """A fresh `lmsa.web.Fetcher` for this task. No disk cache: this is a
    live tracker, and a 304-served stale sample would defeat the point of
    sampling. Tests pass in a fake fetcher instead of calling this."""
    return Fetcher(USER_AGENT, max_bytes=MAX_RESPONSE_BYTES, cache_dir=None)


def fetch_mil(fetcher: Fetcher) -> dict:
    return fetcher.get_json(ADSB_MIL_URL)


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


def _bucket(value: object) -> int | None:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(v):
        return None
    return math.floor(v)


def _alt(value: object) -> int | None:
    """adsb.lol reports `alt_baro` as a number of feet, or the string
    "ground" when parked/taxiing -- treat "ground" as 0, anything else
    unparseable as unknown rather than a fetch error."""
    if isinstance(value, str):
        return 0 if value.strip().lower() == "ground" else None
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(v) if math.isfinite(v) else None


@dataclass
class Sample:
    sampled_at: str
    hex: str
    actype: str | None
    callsign: str | None
    lat_bucket: int | None
    lon_bucket: int | None
    alt_baro: int | None


def sample_from_aircraft(ac: dict, now: datetime) -> Sample | None:
    """One aircraft record -> a bucketed `Sample`, or None if it can't even
    be identified/positioned. Never raises: a malformed record (wrong
    types, missing keys) just yields None so one bad entry in a large feed
    doesn't take the rest of the sample down."""
    try:
        hexid = str(ac.get("hex") or "").strip().lower()
        if not hexid:
            return None
        lat_bucket = _bucket(ac.get("lat"))
        lon_bucket = _bucket(ac.get("lon"))
        if lat_bucket is None or lon_bucket is None:
            return None
        actype = str(ac.get("t") or ac.get("type") or "").strip().upper() or None
        callsign = str(ac.get("flight") or "").strip() or None
        return Sample(
            sampled_at=now.isoformat(),
            hex=hexid,
            actype=actype,
            callsign=callsign,
            lat_bucket=lat_bucket,
            lon_bucket=lon_bucket,
            alt_baro=_alt(ac.get("alt_baro")),
        )
    except Exception:
        return None


def store_samples(conn: sqlite3.Connection, samples: list[Sample]) -> None:
    conn.executemany(
        "INSERT INTO osint_air_samples "
        "(sampled_at, hex, actype, callsign, lat_bucket, lon_bucket, alt_baro) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(s.sampled_at, s.hex, s.actype, s.callsign, s.lat_bucket, s.lon_bucket, s.alt_baro) for s in samples],
    )
    conn.commit()


def prune_old_samples(conn: sqlite3.Connection, now: datetime, retention_days: int = RETENTION_DAYS) -> int:
    cutoff = (now - timedelta(days=retention_days)).isoformat()
    cur = conn.execute("DELETE FROM osint_air_samples WHERE sampled_at < ?", (cutoff,))
    conn.commit()
    return cur.rowcount


def run_sample(conn: sqlite3.Connection, now: datetime, fetcher: Fetcher) -> dict:
    """Fetch, bucket, store, and prune in one call; returns a summary dict.

    A fetch failure (`FetchError` -- network, robots, size cap, bad JSON --
    or any other problem reading the response) is caught and reported as
    zero stored rather than raised -- this runs every dispatcher tick and
    must never abort the tick over one flaky HTTP call. Storing and pruning
    still happen even on a zero-aircraft response, so retention keeps
    working on a quiet feed.
    """
    try:
        raw = fetch_mil(fetcher)
        aircraft = raw.get("ac") if isinstance(raw, dict) else None
        if not isinstance(aircraft, list):
            aircraft = []
    except Exception as exc:
        pruned = prune_old_samples(conn, now)
        return {"aircraft_seen": 0, "stored": 0, "skipped": 0, "pruned": pruned, "error": f"{type(exc).__name__}: {exc}"}

    samples: list[Sample] = []
    skipped = 0
    for ac in aircraft:
        s = sample_from_aircraft(ac, now) if isinstance(ac, dict) else None
        if s is None:
            skipped += 1
        else:
            samples.append(s)

    store_samples(conn, samples)
    pruned = prune_old_samples(conn, now)
    return {
        "aircraft_seen": len(aircraft),
        "stored": len(samples),
        "skipped": skipped,
        "pruned": pruned,
        "error": None,
    }


def main() -> None:
    now = _now()
    with Store() as store, store.record_run("tasks.osint_air_sample") as run:
        summary = run_sample(store.conn, now, fetcher=default_fetcher())
        run.summary = summary
        if summary["error"]:
            print(f"osint_air_sample: fetch failed ({summary['error']}); pruned {summary['pruned']} old row(s)")
        else:
            print(
                f"osint_air_sample: stored {summary['stored']} of {summary['aircraft_seen']} "
                f"(skipped {summary['skipped']}, pruned {summary['pruned']})"
            )


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
