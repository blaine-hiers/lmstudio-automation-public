"""Prediction markets: daily odds snapshots from Polymarket, Manifold and
Metaculus (#60).

This module only fetches and stores. It never predicts anything itself --
every value here is **someone else's market odds, not a forecast by this
tool** (see `ODDS_DISCLAIMER`). `tasks.markets` is the daily collector
(model "none"); `lmsa/briefing/adapters/markets.py` matches snapshots to
briefing stories by embedding similarity at briefing time.

    from lmsa import Store
    from lmsa.markets import collect, record_snapshots, movement, country_metrics
    from lmsa.web import Fetcher

    with Store() as store:
        markets, notes = collect(Fetcher("lmsa-markets/1.0"), cfg)
        written, errors = record_snapshots(store.conn, markets, "2026-09-25",
                                            fetched_at="2026-09-25T04:00:00+00:00")
        movement(store.conn, "polymarket", "12345", "2026-09-25")  # -> MarketMovement
        country_metrics(store.conn, "2026-09-25")  # -> geo_metrics "markets" rows

Storage: table `market_snapshots(source, market_id, day, question, url,
probability, country, category, volume, closed, fetched_at)`, PK
`(source, market_id, day)` -- reruns the same day replace, never duplicate.
`probability` is the source's own 0..1 probability of its first/primary
outcome; nothing here is normalized across sources beyond that.

Country attribution (`attribute_country`) is deliberately conservative and
prefers structured data: an exactly-one-distinct-hit among structured
`tags` wins outright; only with no tag hit does the question text get
scanned, and then a US-state collision ("New Mexico", "New Jersey") is
excluded outright, a small ambiguous-word list (country names that double as
a common given name, noun, or single-word US state -- "Turkey", "Chad",
"Jordan", "Georgia", ...) needs a geopolitical cue word in the same
question, and a match right next to what looks like a person's name
("Jordan Peterson") is dropped. Zero hits or a tie between countries either
way means "no country", never a guess.

The `markets` `geo_metrics` metric (see `country_metrics`) is, per country
per UTC day: the largest absolute 24h probability move (a 0..1 fraction,
e.g. 0.07 = a 7-point swing) across that country's attributed, currently
open markets. It is a raw daily level like `hazard` -- `tasks.risk_map`
z-scores it against its own trailing baseline (`[baseline.min_std] markets`
in `config/risk_map.toml`, already set to 0.01 -- i.e. a quiet country's
noise floor is about one probability point, matching this metric's 0..1
scale). A country with no computable 24h move that day gets no row at all
("no data", never a silent 0).

Parsing every source is defensive on purpose: none of these three public
APIs' exact response shapes could be verified live while writing this
module, so each `fetch_*` accepts whatever plausible shape it recognizes
and raises `UnrecognizedResponse` only when a non-empty response contains
not one item it can even identify as a market (id + question/title) --
never for an ordinarily-empty result set, and never for one bad item among
many (those are silently skipped, not counted as failures).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence
from urllib.parse import urlencode

from . import geo
from .web import FetchError

__all__ = [
    "Market", "MarketMovement", "Snapshot", "MarketError", "UnrecognizedResponse",
    "ODDS_DISCLAIMER", "fetch_polymarket", "fetch_manifold", "fetch_metaculus",
    "collect", "record_snapshots", "snapshots_on", "latest_snapshot_day",
    "movement", "country_metrics", "attribute_country",
]

ODDS_DISCLAIMER = "Market odds, not a forecast by this tool."

POLYMARKET_URL = "https://gamma-api.polymarket.com/markets"
MANIFOLD_URL = "https://api.manifold.markets/v0/search-markets"
METACULUS_URL = "https://www.metaculus.com/api2/questions/"

METRIC = "markets"


class MarketError(Exception):
    """Base for this module's own errors (never raised for a routine empty
    result -- see module docstring)."""


class UnrecognizedResponse(MarketError):
    """A non-empty 200 whose body parsed as JSON but contained not one item
    shaped like a market. Distinguishes "the API changed / we're wrong
    about the shape" from "there just aren't any markets right now"."""


@dataclass
class Market:
    """One market as fetched, before country attribution and storage."""
    source: str
    market_id: str
    question: str
    url: str | None
    probability: float | None
    tags: tuple[str, ...] = ()
    volume: float | None = None
    closed: bool = False
    country: str | None = None  # set by `attribute_country` if not already known


@dataclass
class Snapshot:
    """One stored row, as read back for matching / reporting."""
    source: str
    market_id: str
    question: str
    url: str | None
    probability: float | None
    country: str | None


@dataclass
class MarketMovement:
    """`probability` is the value on the snapshot day itself; `pct_24h` /
    `pct_7d` are signed 0..1-fraction deltas vs the nearest prior snapshot
    within a small tolerance window, or None when there isn't one close
    enough to mean "24h ago" / "7d ago" (a gap, not a zero move)."""
    probability: float | None
    pct_24h: float | None
    pct_7d: float | None


# ---------------------------------------------------------------------------
# Parsing helpers (shared, defensive)
# ---------------------------------------------------------------------------

def _to_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _string_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw.strip() else []
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                label = x.get("label") or x.get("name") or x.get("slug")
                if isinstance(label, str) and label.strip():
                    out.append(label.strip())
    return out


def _first_price(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if isinstance(raw, (list, tuple)) and raw:
        return _to_float(raw[0])
    return None


# ---------------------------------------------------------------------------
# Polymarket (Gamma API)
# ---------------------------------------------------------------------------

def _parse_polymarket_item(item: object) -> Market | None:
    if not isinstance(item, dict):
        return None
    market_id = item.get("id") or item.get("conditionId") or item.get("slug")
    question = item.get("question") or item.get("title")
    if not market_id or not question:
        return None
    slug = item.get("slug")
    tags = _string_list(item.get("tags"))
    if item.get("category"):
        tags = tags + [str(item["category"])]
    return Market(
        source="polymarket",
        market_id=str(market_id),
        question=str(question)[:500],
        url=f"https://polymarket.com/event/{slug}" if slug else None,
        probability=_first_price(item.get("outcomePrices")),
        tags=tuple(tags),
        volume=_to_float(item.get("volume") or item.get("volumeNum")),
        closed=bool(item.get("closed", False)),
    )


def fetch_polymarket(fetcher, *, tag: str | None = None, limit: int = 100) -> list[Market]:
    """`GET gamma-api.polymarket.com/markets` for open markets, optionally
    filtered by `tag`. Raises `UnrecognizedResponse` if the body has items
    but none look like a market at all."""
    params = {"active": "true", "closed": "false", "limit": str(limit)}
    if tag:
        params["tag"] = tag
    data = fetcher.get_json(f"{POLYMARKET_URL}?{urlencode(params)}")
    if isinstance(data, dict):
        items = data.get("data") or data.get("markets") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    markets = [m for m in (_parse_polymarket_item(it) for it in items) if m is not None
               and m.probability is not None]
    recognized = any(_parse_polymarket_item(it) is not None for it in items)
    if items and not recognized:
        raise UnrecognizedResponse(f"polymarket: {len(items)} item(s), none had id+question")
    return markets


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------

def _manifold_url(item: dict) -> str | None:
    url = item.get("url")
    if isinstance(url, str) and url:
        return url
    creator, slug = item.get("creatorUsername"), item.get("slug")
    if creator and slug:
        return f"https://manifold.markets/{creator}/{slug}"
    return None


def fetch_manifold(fetcher, *, query: str, limit: int = 50) -> list[Market]:
    """`GET api.manifold.markets/v0/search-markets?term=<query>`. Only
    binary markets carry a plain `probability`; other outcome types are
    skipped (not a parse failure). Raises `UnrecognizedResponse` if the
    body has items but none look like a market at all."""
    data = fetcher.get_json(f"{MANIFOLD_URL}?{urlencode({'term': query, 'limit': str(limit)})}")
    items = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
    items = items or []
    markets: list[Market] = []
    recognized = False
    for item in items:
        if not isinstance(item, dict):
            continue
        mid, question = item.get("id"), item.get("question")
        if not mid or not question:
            continue
        recognized = True
        prob = _to_float(item.get("probability"))
        if prob is None:
            continue
        tags = _string_list(item.get("groupSlugs")) + _string_list(item.get("tags"))
        markets.append(Market(
            source="manifold", market_id=str(mid), question=str(question)[:500],
            url=_manifold_url(item), probability=prob, tags=tuple(tags),
            volume=_to_float(item.get("volume")),
            closed=bool(item.get("isResolved") or item.get("closed")),
        ))
    if items and not recognized:
        raise UnrecognizedResponse(f"manifold: {len(items)} item(s), none had id+question")
    return markets


# ---------------------------------------------------------------------------
# Metaculus
# ---------------------------------------------------------------------------

def _metaculus_probability(item: dict) -> float | None:
    cp = item.get("community_prediction")
    if isinstance(cp, dict):
        full = cp.get("full")
        if isinstance(full, dict):
            v = _to_float(full.get("q2"))
            if v is not None:
                return v
    q = item.get("question")
    if isinstance(q, dict):
        agg = q.get("aggregations")
        rw = agg.get("recency_weighted") if isinstance(agg, dict) else None
        latest = rw.get("latest") if isinstance(rw, dict) else None
        centers = latest.get("centers") if isinstance(latest, dict) else None
        if isinstance(centers, list) and centers:
            v = _to_float(centers[0])
            if v is not None:
                return v
    return _to_float(item.get("prediction"))


def _metaculus_title(item: dict) -> str | None:
    title = item.get("title")
    if isinstance(title, str) and title:
        return title
    q = item.get("question")
    if isinstance(q, dict) and isinstance(q.get("title"), str):
        return q["title"]
    return None


def fetch_metaculus(fetcher, *, query: str | None = None, limit: int = 50) -> list[Market]:
    """`GET metaculus.com/api2/questions/` (verify the current public
    endpoint/shape before relying on this live -- it was not reachable
    while writing this). Non-binary questions (no single probability) are
    skipped, not a parse failure. Raises `UnrecognizedResponse` if the
    body has items but none look like a market at all."""
    params = {"limit": str(limit)}
    if query:
        params["search"] = query
    data = fetcher.get_json(f"{METACULUS_URL}?{urlencode(params)}")
    if isinstance(data, dict):
        items = data.get("results")
        if items is None:
            items = data if isinstance(data, list) else []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    items = items or []
    markets: list[Market] = []
    recognized = False
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        title = _metaculus_title(item)
        if not mid or not title:
            continue
        recognized = True
        prob = _metaculus_probability(item)
        if prob is None:
            continue
        page_url = item.get("page_url") or item.get("url")
        if isinstance(page_url, str) and page_url.startswith("/"):
            url = f"https://www.metaculus.com{page_url}"
        elif isinstance(page_url, str) and page_url:
            url = page_url
        else:
            url = f"https://www.metaculus.com/questions/{mid}/"
        tags = _string_list(item.get("categories")) + _string_list(item.get("tags"))
        markets.append(Market(
            source="metaculus", market_id=str(mid), question=str(title)[:500], url=url,
            probability=prob, tags=tuple(tags),
            closed=bool(item.get("resolved") or item.get("closed")),
        ))
    if items and not recognized:
        raise UnrecognizedResponse(f"metaculus: {len(items)} item(s), none had id+title")
    return markets


# ---------------------------------------------------------------------------
# Country attribution
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-zÀ-ſ'’-]+")

# Words that are a real country per `geo.normalize_country` but collide often
# enough with a common English given name, a common noun, or a US state that
# a bare match is not trustworthy on its own (#60 review). One of these only
# counts when a geopolitical cue word (below) also appears in the same
# question, and never when it's immediately next to what looks like a
# person's given name/surname ("Jordan Peterson").
AMBIGUOUS_COUNTRY_WORDS = frozenset({
    "turkey", "chad", "jordan", "georgia", "niger", "guinea", "jersey", "china",
})

# Substring-stemmed (via `str.startswith`) so plurals/derived forms count:
# "elections" -> "election", "sanctions" -> "sanction", "parliamentary" ->
# "parliament", "presidential" -> "president".
CUE_WORDS = (
    "war", "election", "president", "government", "invade", "invasion",
    "sanction", "military", "strike", "pm", "minister", "parliament",
    "coup", "border",
)

# Two-word US state names that contain (or sit next to) an unrelated country
# name -- "New Mexico", "New Jersey" -- so their words are pulled out of
# country-candidate consideration entirely, not just gated by a cue word.
# Single-word collisions (e.g. "Georgia") are handled by
# `AMBIGUOUS_COUNTRY_WORDS` instead, so an explicit "Georgia (country)
# election" can still resolve -- a blanket veto here would also kill that.
US_STATE_MULTIWORD = frozenset({
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "rhode island",
    "south carolina", "south dakota", "west virginia",
})

# Words too common (or too likely to be sentence-initial capitalization,
# "Will Chad ...?") to count as evidence of an adjacent person's name.
_NAME_GUARD_STOPWORDS = frozenset({
    "will", "is", "are", "was", "were", "did", "does", "do", "can", "could",
    "should", "would", "the", "a", "an", "in", "on", "at", "of", "to", "for",
    "and", "or", "with", "after", "before", "by", "as", "from", "this",
    "that", "its", "new", "world", "how", "why", "what", "when", "who",
})


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _state_span_indices(words: Sequence[str]) -> set[int]:
    lower = [w.lower() for w in words]
    excluded: set[int] = set()
    for i in range(len(lower) - 1):
        if f"{lower[i]} {lower[i + 1]}" in US_STATE_MULTIWORD:
            excluded.add(i)
            excluded.add(i + 1)
    return excluded


def _looks_like_name(word: str) -> bool:
    """A Title-Case word that isn't a cue word or a common function word --
    weak evidence it's a person's given name or surname, not a country."""
    return (
        len(word) > 1 and word[0].isupper() and word[1:].islower()
        and word.lower() not in _NAME_GUARD_STOPWORDS
        and not any(word.lower().startswith(c) for c in CUE_WORDS)
    )


def _adjacent_to_a_name(words: Sequence[str], i: int) -> bool:
    if i + 1 < len(words) and _looks_like_name(words[i + 1]):
        return True
    # Position 0 is skipped: it's almost always just sentence-initial
    # capitalization ("Will Chad hold elections?"), not a name.
    if i - 1 >= 1 and _looks_like_name(words[i - 1]):
        return True
    return False


def _title_hits(text: str) -> set[str]:
    words = _words(text)
    if not words:
        return set()
    lower_words = [w.lower() for w in words]
    has_cue = any(lw.startswith(c) for lw in lower_words for c in CUE_WORDS)
    state_span = _state_span_indices(words)
    hits: set[str] = set()
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            if any(j in state_span for j in range(i, i + size)):
                continue
            a2 = geo.normalize_country(" ".join(words[i:i + size]))
            if not a2:
                continue
            if size == 1:
                bare = re.sub(r"['’]s$", "", lower_words[i])  # "Turkey's" -> "turkey"
                if bare in AMBIGUOUS_COUNTRY_WORDS and (not has_cue or _adjacent_to_a_name(words, i)):
                    continue
            hits.add(a2)
    return hits


def attribute_country(question: str, tags: Iterable[str] = ()) -> str | None:
    """Structured `tags` are trusted first: exactly one distinct
    ISO alpha-2 hit among them wins outright (title text is not even
    consulted); more than one distinct hit among the tags is itself
    ambiguous and stops there -- it never falls through to the title.

    With no usable tag hit, a mention extracted from `question` is used
    instead, subject to `_title_hits`'s US-state and ambiguous-word/cue-word/
    name-adjacency guards. Zero or multiple distinct hits -> None either
    way. Never guesses."""
    tag_hits = {a2 for a2 in (geo.normalize_country(str(t)) for t in tags) if a2}
    if tag_hits:
        return next(iter(tag_hits)) if len(tag_hits) == 1 else None
    hits = _title_hits(question)
    return next(iter(hits)) if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# Collection (per-source isolation)
# ---------------------------------------------------------------------------

@dataclass
class MarketsConfig:
    polymarket_tags: Sequence[str | None] = field(default_factory=lambda: [None])
    manifold_queries: Sequence[str] = field(default_factory=list)
    metaculus_queries: Sequence[str | None] = field(default_factory=lambda: [None])
    limit_per_source: int = 100


def collect(fetcher, cfg: MarketsConfig) -> tuple[list[Market], list[str]]:
    """Fetch every configured source/query. One bad source or query is
    recorded as a note and skipped -- never aborts the others."""
    markets: list[Market] = []
    notes: list[str] = []

    def _run(label: str, fn) -> None:
        try:
            markets.extend(fn())
        except UnrecognizedResponse as exc:
            notes.append(f"{label}: unrecognized response -- {exc}")
        except FetchError as exc:
            notes.append(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
        except Exception as exc:  # a source we don't fully control must never kill the run
            notes.append(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")

    for tag in (cfg.polymarket_tags or [None]):
        _run(f"polymarket ({tag or 'all'})",
             lambda tag=tag: fetch_polymarket(fetcher, tag=tag, limit=cfg.limit_per_source))
    for query in (cfg.manifold_queries or []):
        _run(f"manifold ({query!r})",
             lambda query=query: fetch_manifold(fetcher, query=query, limit=cfg.limit_per_source))
    for query in (cfg.metaculus_queries or [None]):
        _run(f"metaculus ({query or 'all'})",
             lambda query=query: fetch_metaculus(fetcher, query=query, limit=cfg.limit_per_source))
    return markets, notes


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _to_day(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)).isoformat()  # validates, raises ValueError


def record_snapshots(conn, markets: Iterable[Market], day, *, fetched_at: str | None = None,
                      commit: bool = True) -> tuple[int, list[str]]:
    """Upsert one row per market for `day` (same day rerun replaces, never
    duplicates). Country attribution runs here if `Market.country` isn't
    already set. Per-row isolation: one bad market is recorded in the
    returned error list and skipped, never aborts the batch."""
    d = _to_day(day)
    ts = fetched_at or datetime.now(timezone.utc).isoformat()
    written = 0
    errors: list[str] = []
    for m in markets:
        try:
            country = m.country if m.country else attribute_country(m.question, m.tags)
            conn.execute(
                "INSERT OR REPLACE INTO market_snapshots"
                " (source, market_id, day, question, url, probability, country, category, volume, closed, fetched_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (m.source, str(m.market_id), d, m.question, m.url, m.probability, country,
                 (",".join(m.tags)[:300] or None) if m.tags else None, m.volume,
                 int(bool(m.closed)), ts),
            )
            written += 1
        except Exception as exc:
            errors.append(f"{m.source}:{m.market_id}: {type(exc).__name__}: {exc}")
    if commit:
        conn.commit()
    return written, errors


def snapshots_on(conn, day) -> list[Snapshot]:
    """Every snapshot stored for exactly `day`."""
    d = _to_day(day)
    rows = conn.execute(
        "SELECT source, market_id, question, url, probability, country"
        " FROM market_snapshots WHERE day=?", (d,),
    ).fetchall()
    return [Snapshot(*r) for r in rows]


def latest_snapshot_day(conn) -> str | None:
    """The newest day any snapshot was recorded for, or None. Lets a
    briefing run match against yesterday's snapshot if today's collector
    hasn't run yet."""
    row = conn.execute("SELECT MAX(day) FROM market_snapshots").fetchone()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def _prob_near(conn, source: str, market_id: str, on_or_before: str, earliest: str) -> float | None:
    row = conn.execute(
        "SELECT probability FROM market_snapshots"
        " WHERE source=? AND market_id=? AND day<=? AND day>=? AND probability IS NOT NULL"
        " ORDER BY day DESC LIMIT 1",
        (source, market_id, on_or_before, earliest),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def movement(conn, source: str, market_id: str, as_of) -> MarketMovement:
    """`pct_24h`/`pct_7d` compare `as_of`'s stored probability against the
    nearest prior snapshot within a small tolerance window (a day or two of
    slack for 24h, three for 7d, so one missed collector run doesn't erase
    the comparison) -- None when nothing is close enough, not 0.0."""
    d = _to_day(as_of)
    row = conn.execute(
        "SELECT probability FROM market_snapshots WHERE source=? AND market_id=? AND day=?",
        (source, str(market_id), d),
    ).fetchone()
    cur = float(row[0]) if row and row[0] is not None else None
    if cur is None:
        return MarketMovement(None, None, None)
    dd = date.fromisoformat(d)
    prev24 = _prob_near(conn, source, market_id,
                         (dd - timedelta(days=1)).isoformat(), (dd - timedelta(days=3)).isoformat())
    prev7 = _prob_near(conn, source, market_id,
                        (dd - timedelta(days=7)).isoformat(), (dd - timedelta(days=10)).isoformat())
    return MarketMovement(
        probability=cur,
        pct_24h=(cur - prev24) if prev24 is not None else None,
        pct_7d=(cur - prev7) if prev7 is not None else None,
    )


# ---------------------------------------------------------------------------
# geo_metrics feed
# ---------------------------------------------------------------------------

def country_metrics(conn, day) -> list[tuple[str, str, str, float]]:
    """`(day, country, "markets", value)` rows for `lmsa.geo_metrics.record_many`:
    per country, the largest absolute 24h probability move across that
    country's attributed, open markets snapshotted on `day`. A country with
    an attributed market but no computable 24h move is left out entirely --
    never written as 0."""
    d = _to_day(day)
    rows = conn.execute(
        "SELECT source, market_id, country FROM market_snapshots"
        " WHERE day=? AND country IS NOT NULL AND closed=0",
        (d,),
    ).fetchall()
    best: dict[str, float] = {}
    for source, market_id, country in rows:
        mv = movement(conn, source, market_id, d)
        if mv.pct_24h is None:
            continue
        val = abs(mv.pct_24h)
        if country not in best or val > best[country]:
            best[country] = val
    return [(d, country, METRIC, value) for country, value in best.items()]
