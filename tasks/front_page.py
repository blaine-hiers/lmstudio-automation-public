"""Personal front page: HN + subreddits + RSS/Atom feeds ranked by interest.

Pulls candidate items from Hacker News (official Firebase API), a list of
subreddits (JSON feeds), and a list of RSS/Atom feeds -- all configured in
`config/front_page.toml` -- then ranks them by cosine similarity between
each item's title+snippet embedding and a set of interest descriptions,
tempered by a recency decay. The report shows the top N as cards and a
"wildcards" section of high-popularity items that didn't make the
interest-ranked cut.

    python -m tasks.front_page
"""

from __future__ import annotations

import json
import math
import os
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lmsa import LMStudio, Report, Store, Tile
from lmsa import feeds
from lmsa.feeds import is_http_url as _is_http_url
from lmsa.feeds import strip_html
from lmsa.report import Card

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/front_page.toml"], "model": "embed"}

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.getenv("LMSA_FRONT_PAGE_CONFIG", str(ROOT / "config" / "front_page.toml")))
FETCH_TIMEOUT = 15
USER_AGENT = "lmsa-front-page/1.0 (personal front page)"
HN_BASE = "https://hacker-news.firebaseio.com/v0"
TEXT_TRUNC_DEFAULT = 4000
SOURCE_LABEL = {"hn": "HN", "reddit": "Reddit", "rss": "RSS"}


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


@dataclass
class Item:
    source: str  # "hn" | "reddit" | "rss"
    tag: str  # short label shown on the card, e.g. "HN", "r/homelab", feed name
    title: str
    url: str
    snippet: str = ""
    published: datetime | None = None
    popularity: float | None = None  # source-native score/upvotes; None when the source has no signal
    score: float = 0.0  # filled in by rank()
    interest: str | None = None  # best-matching interest name, filled in by rank()


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def normalize_url(url: str) -> str:
    """Scheme/host lowercased, trailing slash stripped, utm_* params and the
    fragment dropped -- enough to catch the same story linked from HN, an
    RSS mirror of HN, or a subreddit crosspost. Returns "" for anything that
    doesn't parse as an absolute URL (never used for de-duping those)."""
    if not url:
        return ""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    query = urllib.parse.urlencode(
        [
            (k, v)
            for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_")
        ]
    )
    path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---- Hacker News ------------------------------------------------------------

def fetch_hn(list_name: str, limit: int) -> tuple[list[Item], list[str]]:
    """Top/best story ids, then per-item details. A bad id is skipped, not fatal."""
    try:
        ids = _get_json(f"{HN_BASE}/{list_name}.json")
    except Exception as exc:
        return [], [f"HN {list_name} list: {exc}"]

    items: list[Item] = []
    errors: list[str] = []
    for item_id in list(ids)[:limit]:
        try:
            data = _get_json(f"{HN_BASE}/item/{item_id}.json")
        except Exception as exc:
            errors.append(f"HN item {item_id}: {exc}")
            continue
        if not data or data.get("type") != "story" or data.get("dead") or data.get("deleted"):
            continue
        title = data.get("title") or ""
        if not title:
            continue
        raw_url = data.get("url") or ""
        discussion_url = f"https://news.ycombinator.com/item?id={item_id}"
        url = raw_url if _is_http_url(raw_url) else discussion_url
        published = datetime.fromtimestamp(data["time"], tz=timezone.utc) if data.get("time") else None
        items.append(Item(
            source="hn", tag="HN", title=title, url=url,
            snippet=strip_html(data.get("text") or ""),
            published=published,
            popularity=float(data.get("score") or 0),
        ))
    return items, errors


# ---- Reddit -------------------------------------------------------------

def fetch_reddit(subreddits: list[dict], limit: int) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    errors: list[str] = []
    for sub in subreddits:
        name = sub["name"]
        try:
            data = _get_json(f"https://www.reddit.com/r/{name}/hot.json?limit={limit}&raw_json=1")
        except Exception as exc:
            errors.append(f"r/{name}: {exc}")
            continue
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            if post.get("stickied"):
                continue
            title = post.get("title") or ""
            if not title:
                continue
            permalink = post.get("permalink") or ""
            # The permalink is always ours to construct (fixed https://reddit.com
            # prefix), so it's inherently safe; only the raw `url` fallback
            # (used when a post has no permalink) needs the scheme check.
            link = f"https://www.reddit.com{permalink}" if permalink else (post.get("url") or "")
            published = (
                datetime.fromtimestamp(post["created_utc"], tz=timezone.utc)
                if post.get("created_utc") else None
            )
            items.append(Item(
                source="reddit", tag=f"r/{name}", title=title,
                url=link if _is_http_url(link) else "",
                snippet=strip_html(post.get("selftext") or ""),
                published=published,
                popularity=float(post.get("score") or 0),
            ))
    return items, errors


# ---- RSS / Atom -----------------------------------------------------------

def parse_feed(xml_text: str, tag: str, limit: int) -> list[Item]:
    """Feed entries as front-page Items; parsing lives in `lmsa.feeds`."""
    return [
        Item(
            source="rss", tag=tag, title=fi.title, url=fi.url,
            snippet=fi.summary, published=fi.published, popularity=None,
        )
        for fi in feeds.parse_feed(xml_text, source=tag, limit=limit)
    ]


def fetch_feeds(feeds: list[dict], limit: int) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    errors: list[str] = []
    for feed in feeds:
        name, url = feed["name"], feed["url"]
        try:
            items.extend(parse_feed(_get_text(url), name, limit))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return items, errors


def dedupe(items: list[Item]) -> list[Item]:
    """Collapse items whose normalized URL matches (e.g. the same story from
    the HN API and an RSS mirror of HN), keeping the first-seen item and
    merging the others' source labels onto its tag (e.g. "HN · RSS") instead
    of showing the same story twice. Items with no usable URL never merge --
    each is kept as its own key."""
    kept: dict[str, Item] = {}
    labels: dict[str, list[str]] = {}
    order: list[str] = []
    for item in items:
        norm = normalize_url(item.url)
        key = norm or f"__no_url__:{id(item)}"
        label = SOURCE_LABEL.get(item.source, item.source)
        if key in kept:
            existing = kept[key]
            if label not in labels[key]:
                labels[key].append(label)
            if existing.popularity is None and item.popularity is not None:
                existing.popularity = item.popularity
            if not existing.snippet and item.snippet:
                existing.snippet = item.snippet
            if existing.published is None and item.published is not None:
                existing.published = item.published
            continue
        kept[key] = item
        labels[key] = [label]
        order.append(key)

    for key, item in kept.items():
        if len(labels[key]) > 1:
            item.tag = " · ".join(labels[key])
    return [kept[key] for key in order]


def gather(cfg: dict) -> tuple[list[Item], list[str]]:
    """Each source's failures are isolated (see fetch_*); one bad feed or
    subreddit never drops the others. Results are deduped by URL across
    sources before returning."""
    fetch_cfg = cfg.get("fetch", {})
    source_cfg = cfg.get("source", {})
    items: list[Item] = []
    errors: list[str] = []

    hn_items, hn_errors = fetch_hn(fetch_cfg.get("hn_list", "topstories"), fetch_cfg.get("hn_limit", 40))
    items += hn_items
    errors += hn_errors

    reddit_items, reddit_errors = fetch_reddit(
        source_cfg.get("subreddits", []), fetch_cfg.get("subreddit_limit", 20)
    )
    items += reddit_items
    errors += reddit_errors

    feed_items, feed_errors = fetch_feeds(source_cfg.get("feeds", []), fetch_cfg.get("feed_limit", 20))
    items += feed_items
    errors += feed_errors

    return dedupe(items), errors


# ---- Ranking ----------------------------------------------------------------

def item_text(item: Item, max_chars: int) -> str:
    """Title + snippet, bounded before it goes to the embedding model --
    item text is untrusted (scraped from HN/Reddit/RSS)."""
    return f"{item.title}\n\n{item.snippet}".strip()[:max_chars]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    value = dot / (norm_a * norm_b)
    return value if math.isfinite(value) else 0.0


def recency_decay(published: datetime | None, now: datetime, half_life_hours: float) -> float:
    """1.0 fresh, halving every half_life_hours; an unknown date (some feeds
    omit one) gets no penalty rather than being scored as infinitely old."""
    if published is None or half_life_hours <= 0:
        return 1.0
    age_hours = max((now - published).total_seconds() / 3600, 0.0)
    return 0.5 ** (age_hours / half_life_hours)


def normalize(values: list[float]) -> list[float]:
    """Min-max scale to [0, 1]; an empty, flat, or non-finite input scales to all 0.0."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 0:
        return [0.0 for _ in values]
    return [((v - lo) / span) if math.isfinite(v) else 0.0 for v in values]


def rank(
    items: list[Item],
    interests: list[dict],
    interest_vectors: list[list[float]],
    item_vectors: list[list[float]],
    now: datetime,
    half_life_hours: float,
) -> None:
    """Fills in each item's `.score` (interest match x recency) and `.interest`, in place."""
    for item, vector in zip(items, item_vectors):
        sims = [cosine(vector, iv) for iv in interest_vectors]
        best = max(sims) if sims else 0.0
        best_idx = sims.index(best) if sims else -1
        item.interest = interests[best_idx]["name"] if best_idx >= 0 else None
        item.score = max(best, 0.0) * recency_decay(item.published, now, half_life_hours)


def wildcard_scores(items: list[Item], now: datetime, half_life_hours: float) -> list[float]:
    """Popularity-only score (no interest match) for the wildcards section.

    Items with no source popularity signal (RSS entries carry none) score 0
    here by design -- wildcards surfaces what's already trending on HN/Reddit
    outside the owner's stated interests, not undifferentiated RSS items.
    """
    pops = normalize([i.popularity if i.popularity is not None else 0.0 for i in items])
    return [p * recency_decay(i.published, now, half_life_hours) for p, i in zip(pops, items)]


def _card(item: Item, *, wildcard: bool = False) -> Card:
    return Card(
        title=item.title,
        meta=f"{item.interest or 'general'} · score {item.score:.2f}" if not wildcard else item.tag,
        body=item.snippet[:280] if item.snippet else "",
        tags=[item.tag],
        href=item.url or None,
        tone="violet" if wildcard else "acc",
        ribbon="WILDCARD" if wildcard else None,
    )


def main() -> None:
    cfg = load_config()
    fetch_cfg = cfg.get("fetch", {})
    interests = cfg.get("interests", [])
    if not interests:
        raise RuntimeError(f"{CONFIG_PATH} must define at least one [[interests]] entry")

    top_n = fetch_cfg.get("top_n", 30)
    wildcard_n = fetch_cfg.get("wildcard_n", 5)
    half_life = float(fetch_cfg.get("recency_half_life_hours", 36))
    snippet_chars = fetch_cfg.get("snippet_chars", TEXT_TRUNC_DEFAULT)

    lm = LMStudio()
    lm.ensure_server()
    now = _now()

    with Store() as store, store.record_run("front_page") as run:
        items, errors = gather(cfg)

        if items:
            interest_vectors = lm.embed([i["description"] for i in interests])
            item_vectors = lm.embed([item_text(i, snippet_chars) for i in items])
            rank(items, interests, interest_vectors, item_vectors, now, half_life)
            w_scores = wildcard_scores(items, now, half_life)
        else:
            w_scores = []

        ranked = sorted(items, key=lambda i: i.score, reverse=True)
        top = ranked[:top_n]
        top_ids = {id(i) for i in top}

        remaining = sorted(
            ((item, s) for item, s in zip(items, w_scores) if id(item) not in top_ids),
            key=lambda pair: pair[1],
            reverse=True,
        )
        wildcards = [item for item, _ in remaining[:wildcard_n]]

        r = Report(
            "Front page",
            kicker="lmsa · personal digest",
            subtitle=f"{len(items)} candidates from HN, Reddit, and RSS, ranked against {len(interests)} interests.",
            sticker=f"top {len(top)}",
            sticker_tone="acc",
            source="HN + Reddit + RSS",
        )
        r.tiles([
            Tile("Candidates", len(items), tone="info"),
            Tile(
                "Shown", len(top),
                note=f"of {len(items)} (capped at {top_n})" if len(items) > top_n else None,
                tone="acc",
            ),
            Tile("Wildcards", len(wildcards), tone="violet"),
            Tile("Fetch errors", len(errors), tone="hot" if errors else "muted"),
        ])

        if errors:
            r.section("Fetch errors", tone="warn").callout(
                "\n".join(errors), label="skipped", tone="warn"
            )

        r.section("Top picks").cards([_card(item) for item in top])
        r.section("Wildcards", tone="violet").cards([_card(item, wildcard=True) for item in wildcards])

        path = r.write()
        run.report_path = path
        run.summary = {
            "candidates": len(items),
            "shown": len(top),
            "wildcards": len(wildcards),
            "fetch_errors": len(errors),
        }
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
