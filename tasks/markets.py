"""Prediction markets snapshot collector (#60): Polymarket, Manifold and
Metaculus odds for open geopolitics/world markets, stored one row per
market per UTC day (`lmsa.markets`, `config/markets.toml`).

Model "none" -- this task only fetches and stores raw odds. Matching a
market to a briefing story (by embedding similarity) happens inside
`tasks.briefing` itself, in `lmsa/briefing/adapters/markets.py`. This task
also writes `lmsa.geo_metrics` "markets" rows that `tasks.risk_map` folds
into its composite (already weighted in `config/risk_map.toml`).

Data collection only, like `tasks.osint_air_sample`: no report is rendered
here. Every value collected is someone else's market odds, not a forecast
by this tool.

    python -m tasks.markets
"""

from __future__ import annotations

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from lmsa import Store, geo_metrics
from lmsa.markets import MarketsConfig, collect, country_metrics, record_snapshots
from lmsa.web import Fetcher

SETUP = {"config": ["config/markets.toml"]}

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "markets.toml"
USER_AGENT = "lmsa-markets/1.0 (personal research; daily snapshot)"


def _now() -> datetime:
    """Current UTC time. Tests monkeypatch this."""
    return datetime.now(timezone.utc)


def load_config(path: Path = CONFIG) -> MarketsConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    poly = data.get("polymarket") or {}
    manifold = data.get("manifold") or {}
    metaculus = data.get("metaculus") or {}
    poly_tags: list[str | None] = [str(t) for t in poly.get("tags", []) if str(t).strip()] or [None]
    manifold_queries = [str(q) for q in manifold.get("queries", []) if str(q).strip()]
    metaculus_queries: list[str | None] = (
        [str(q) for q in metaculus.get("queries", []) if str(q).strip()] or [None]
    )
    limit = max(1, int(data.get("limit_per_source", 100)))
    return MarketsConfig(
        polymarket_tags=poly_tags, manifold_queries=manifold_queries,
        metaculus_queries=metaculus_queries, limit_per_source=limit,
    )


def default_fetcher() -> Fetcher:
    """A fresh `lmsa.web.Fetcher` for this task -- no disk cache, this is a
    daily snapshot and a stale cached read would defeat the point."""
    return Fetcher(USER_AGENT, max_bytes=5_000_000, cache_dir=None)


def run_collection(conn, cfg: MarketsConfig, now: datetime, fetcher: Fetcher) -> dict:
    """Fetch every configured source, snapshot the day, and feed the
    `geo_metrics` "markets" metric. Never raises: source failures are
    per-source notes, storage failures are per-row errors."""
    markets, notes = collect(fetcher, cfg)
    day = now.date().isoformat()
    stored, store_errors = record_snapshots(conn, markets, day, fetched_at=now.isoformat())
    metric_rows = country_metrics(conn, day)
    scored = 0
    metric_errors: list[str] = []
    try:
        scored = geo_metrics.record_many(conn, metric_rows)
    except Exception as exc:  # a metric writer must never raise into the caller
        metric_errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "day": day,
        "fetched": len(markets),
        "stored": stored,
        "store_errors": store_errors,
        "countries_scored": scored,
        "metric_errors": metric_errors,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily prediction-market odds snapshot (Polymarket/Manifold/Metaculus)."
    )
    parser.parse_args()
    now = _now()
    cfg = load_config()
    with Store() as store, store.record_run("tasks.markets") as run:
        summary = run_collection(store.conn, cfg, now, default_fetcher())
        run.summary = summary
        bits = [f"fetched {summary['fetched']}", f"stored {summary['stored']}",
                f"{summary['countries_scored']} countr(ies) scored"]
        if summary["store_errors"]:
            bits.append(f"{len(summary['store_errors'])} store error(s)")
        if summary["notes"]:
            bits.append(f"notes: {'; '.join(summary['notes'])}")
        print("markets: " + ", ".join(bits))


if __name__ == "__main__":
    main()
