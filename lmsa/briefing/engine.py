"""The briefing pipeline (#52): collect -> keyword pre-filter -> follow links
-> embed -> cluster into stories -> rank -> per-story extract (memo'd, quotes
verified) -> what changed since last run -> persist -> report.

    result = run_briefing(cfg, lm=lm, conn=store.conn, run_id=run.id, fetcher=fetcher)
    report = build_report(result)

Everything a remote party controls is capped, and every cap that bites is
disclosed in the report's notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np

from .. import geo
from ..feeds import is_http_url
from ..memo import Memo, has_keys, memo_version, model_id
from ..prompts import load as load_prompt
from ..prompts import prompt_hash
from ..report import TONES, Report, StackedBar, Story, StoryExtra, StoryQuote, StorySource, Tile
from ..store import pack_vector, unpack_vector
from ..web import extract_article, normalize_url
from .config import BriefingConfig
from .quotes import verify_quote
from .sources import ADAPTERS, Doc, SourceContext, active_extensions

log = logging.getLogger(__name__)

TASK = "briefing"
MEMO_VERSION = "briefing-1"
VECTOR_NS = "briefings"
SYSTEM_PROMPT = "briefing/story_system"
USER_PROMPT = "briefing/story_user"

TITLE_CHARS = 300
MAX_DOC_CHARS = 20_000       # article text kept per doc
EMBED_CHARS = 2_000          # per doc, sent to the embedding model
STORY_DOC_CHARS = 2_500      # per doc, sent to extract()
STORY_PROMPT_CHARS = 12_000  # all docs of one story, sent to extract()
SHORT_TEXT = 400             # follow_links fetches docs with less text than this
STORED_TEXT_CHARS = 4_000    # briefing_articles.text
EMBED_BATCH = 64
READ_WPM = 230
MAX_SOURCES_SHOWN = 8

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "highlights": {"type": "array", "items": {"type": "string"}},
        "key_quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"quote": {"type": "string"}, "source_url": {"type": "string"}},
                "required": ["quote", "source_url"],
                "additionalProperties": False,
            },
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"date": {"type": "string"}, "event": {"type": "string"}},
                "required": ["date", "event"],
                "additionalProperties": False,
            },
        },
        "why_it_matters": {"type": "string"},
        "countries": {"type": "array", "items": {"type": "string"}},
        "actors": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "integer", "minimum": 0, "maximum": 3},
    },
    "required": ["headline", "summary", "highlights", "key_quotes", "timeline",
                 "why_it_matters", "countries", "actors", "severity"],
    "additionalProperties": False,
}

_STOPWORDS = frozenset("""
about above after again against also among around because been before being below between both
could doing during each every from further have having here into itself just more most much
other over same should some such than that their them then there these they this those through
under until very what when where which while with within without would your yours news latest
things stuff anything everything including related especially mostly topics topic updates
""".split())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class StoryResult:
    """One story of one run. `extract` is the sanitized model answer (None
    if extraction failed: see `error`); `quotes` are `{quote, source_url,
    verified}`; `status` is new | updated | ongoing; `extras` is filled by
    extensions (JSON-able values, persisted in data_json)."""
    story_id: str
    docs: list[Doc]
    centroid: list[float]
    score: float = 0.0
    interest: float = 0.0
    fingerprint: str = ""
    extract: dict | None = None
    quotes: list[dict] = field(default_factory=list)
    status: str = "new"
    countries: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    error: str | None = None
    trimmed: bool = False

    @property
    def anchor(self) -> str:
        return f"story-{self.story_id}"

    @property
    def headline(self) -> str:
        if self.extract and self.extract.get("headline"):
            return self.extract["headline"]
        return self.docs[0].title if self.docs else "(untitled)"

    @property
    def urls(self) -> list[str]:
        return [d.meta["norm_url"] for d in self.docs]

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(d.source for d in self.docs if d.source))

    @property
    def newest(self) -> datetime | None:
        dates = [d.published for d in self.docs if d.published]
        return max(dates) if dates else None


@dataclass
class BriefingResult:
    cfg: BriefingConfig
    run_id: int
    now: datetime
    stories: list[StoryResult] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    memo: Memo | None = None
    last_ok: str | None = None
    ctx: SourceContext | None = None

    def summary(self) -> dict:
        out = dict(self.counts)
        out["stories"] = len(self.stories)
        out["new"] = sum(1 for s in self.stories if s.status == "new")
        out["updated"] = sum(1 for s in self.stories if s.status == "updated")
        out["dropped"] = len(self.dropped)
        if self.memo is not None:
            out.update(self.memo.summary())
        return out


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def _aware(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def normalize_doc(doc: object, kind: str, entry: dict) -> Doc | None:
    """A clean copy of `doc`, or None if it can't be used (not a Doc, not an
    http(s) URL, no title). Titles/text are coerced to str and capped."""
    if not isinstance(doc, Doc):
        return None
    url = str(doc.url or "").strip()
    if not is_http_url(url):
        return None
    try:
        norm = normalize_url(url)
    except ValueError:
        return None
    title = " ".join(str(doc.title or "").split())[:TITLE_CHARS]
    text = str(doc.text or "")
    if not title:
        return None
    meta = dict(doc.meta) if isinstance(doc.meta, dict) else {}
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]
        meta["truncated"] = True
    meta["norm_url"] = norm
    meta.setdefault("kind", kind)
    meta.setdefault("entry", entry.get("name", ""))
    if entry.get("prefilter") is False:
        meta["prefilter_exempt"] = True  # title-only feeds: keywords can't match before follow_links (#128)
    for key in ("lean", "reliability", "group"):
        meta.setdefault(key, entry.get(key, "unknown" if key != "group" else ""))
    return Doc(url=url, title=title, text=text, published=_aware(doc.published),
               source=str(doc.source or entry.get("name") or "")[:120], meta=meta)


def collect(cfg: BriefingConfig, ctx: SourceContext) -> list[Doc]:
    """Run every source entry through its adapter. One failing entry (any
    exception) is a note, and the docs it yielded before failing are kept.
    Docs are deduped by normalized URL; `max_items` caps the total. Adapters
    get `ctx.limit` = cap + 1, so a cap that cut items is always noted."""
    docs: list[Doc] = []
    seen: set[str] = set()
    entries = cfg.entries()
    for n_entry, (kind, entry) in enumerate(entries):
        remaining = cfg.max_items - len(docs)
        if remaining <= 0:
            left = len(entries) - n_entry
            ctx.note(f"max_items ({cfg.max_items}) reached: {left} later source(s) were not collected")
            break
        adapter = entry["adapter"] if kind == "api" else kind
        spec = ADAPTERS.get(adapter)
        label = f"{kind}: {entry.get('name', adapter)}"
        if spec is None:
            ctx.note(f"{label}: not available (no '{adapter}' adapter is installed); skipped")
            ctx.entry_stats[n_entry] = {"taken": 0, "error": "not available"}
            continue
        entry_cap = int(entry.get("limit", cfg.max_items))
        limit = min(entry_cap, remaining)
        # One more than we keep: an extra item is how we know the cap cut something.
        ctx.limit = limit + 1
        taken = bad = 0
        error = None
        try:
            for raw in spec.fn(entry, ctx):
                try:
                    doc = normalize_doc(raw, kind, entry)
                except Exception:  # a malformed doc is skipped, not fatal
                    doc = None
                if doc is None:
                    bad += 1
                    continue
                if doc.meta["norm_url"] in seen:
                    continue
                if taken >= limit:
                    why = (f"its limit ({entry_cap})" if "limit" in entry and entry_cap <= remaining
                           else f"max_items ({cfg.max_items})")
                    ctx.note(f"{label}: cut at {limit} items by {why}; more were available")
                    break
                seen.add(doc.meta["norm_url"])
                docs.append(doc)
                taken += 1
        except Exception as exc:
            log.warning("briefing source %s failed", label, exc_info=True)
            ctx.note(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        ctx.entry_stats[n_entry] = {"taken": taken, "error": error}
        if bad:
            ctx.note(f"{label}: {bad} item(s) without a usable http(s) URL or title skipped")
    return docs


# ---------------------------------------------------------------------------
# Keyword pre-filter
# ---------------------------------------------------------------------------

def prefilter_terms(cfg: BriefingConfig) -> list[str]:
    """`keywords` if given, else significant words (5+ letters, not stop
    words) from the interests; plus the names of the `regions` countries."""
    terms: list[str] = [k.lower() for k in cfg.keywords]
    if not terms:
        for interest in cfg.interests:
            for w in re.findall(r"[A-Za-z][A-Za-z'-]{4,}", interest):
                w = w.lower().strip("'-")
                if w not in _STOPWORDS:
                    terms.append(w)
    for code in cfg.regions:
        name = geo.country_name(code)
        if name:
            terms.append(name.lower())
    return list(dict.fromkeys(t for t in terms if t))


def prefilter_pattern(terms: list[str]) -> re.Pattern | None:
    if not terms:
        return None
    alts = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alts})(?:s|es)?(?!\w)", re.IGNORECASE)


def prefilter(cfg: BriefingConfig, docs: list[Doc]) -> tuple[list[Doc], int]:
    """(kept, dropped count). Off when `prefilter = false` or no terms. Docs
    from a source entry with `prefilter = false` are always kept."""
    pattern = prefilter_pattern(prefilter_terms(cfg)) if cfg.prefilter else None
    if pattern is None:
        return docs, 0
    kept = [d for d in docs if d.meta.get("prefilter_exempt") or pattern.search(f"{d.title}\n{d.text[:EMBED_CHARS * 2]}")]
    return kept, len(docs) - len(kept)


# ---------------------------------------------------------------------------
# Follow links
# ---------------------------------------------------------------------------

def follow_links(cfg: BriefingConfig, docs: list[Doc], ctx: SourceContext) -> tuple[int, int]:
    """Fetch the article for docs with little text, up to `max_fetch`.
    Returns (fetched, failed); a failure keeps the doc's own text."""
    if not cfg.follow_links or cfg.max_fetch <= 0:
        return 0, 0
    todo = [d for d in docs if len(d.text) < SHORT_TEXT]
    if len(todo) > cfg.max_fetch:
        ctx.note(f"follow_links: fetched {cfg.max_fetch} of {len(todo)} short items (max_fetch)")
        todo = todo[:cfg.max_fetch]
    fetched = failed = 0
    for d in todo:
        try:
            resp = ctx.fetcher.get(d.url)
            art = extract_article(resp.content, resp.url)
            text = str(art.text or "")
            if len(text) > len(d.text):
                if len(text) > MAX_DOC_CHARS:
                    text = text[:MAX_DOC_CHARS]
                    d.meta["truncated"] = True
                d.text = text
                d.meta["fetched"] = True
            if d.published is None:
                d.published = _aware(art.published)
            fetched += 1
        except Exception as exc:  # FetchError, parse errors, anything: keep the snippet
            failed += 1
            log.info("follow link failed for %s: %s", d.url, exc)
    if failed:
        ctx.note(f"follow_links: {failed} article fetch(es) failed; their feed/search text was used")
    return fetched, failed


# ---------------------------------------------------------------------------
# Embed, cluster, rank
# ---------------------------------------------------------------------------

def embed_texts(lm: Any, texts: list[str], model: str | None) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(lm.embed(texts[i:i + EMBED_BATCH], model=model))
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise RuntimeError(f"embedding model returned {arr.shape} for {len(texts)} texts")
    return arr


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(norms > 0, m / norms, 0.0)
    return np.nan_to_num(u)


def cluster(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy single pass: each vector joins the cluster whose centroid is
    most similar (cosine >= threshold), else starts a new one. Deterministic
    for a given input order."""
    units = _unit(vectors)
    members: list[list[int]] = []
    sums: list[np.ndarray] = []
    for i, v in enumerate(units):
        best, best_sim = -1, threshold
        if sums:
            cents = _unit(np.stack(sums))
            sims = cents @ v
            j = int(np.argmax(sims))
            if sims[j] >= best_sim:
                best = j
        if best < 0:
            members.append([i])
            sums.append(v.copy())
        else:
            members[best].append(i)
            sums[best] += v
    return members


def recency_decay(newest: datetime | None, now: datetime, half_life_hours: float) -> float:
    if newest is None:
        return 1.0
    age_h = max((now - newest).total_seconds() / 3600, 0.0)
    return 0.5 ** (age_h / half_life_hours)


def story_score(interest: float, n_docs: int, n_sources: int, decay: float) -> float:
    """interest similarity x recency decay x (cluster size + source diversity)."""
    size = 1 + 0.25 * math.log2(max(n_docs, 1)) + 0.5 * math.log2(max(n_sources, 1))
    return max(interest, 0.0) * decay * size


# ---------------------------------------------------------------------------
# Previous run: match, status, dropped
# ---------------------------------------------------------------------------

def _prev_run_id(conn, name: str, run_id: int) -> int | None:
    row = conn.execute(
        "SELECT MAX(run_id) FROM briefing_runs WHERE briefing=? AND run_id<>? AND status='ok'", (name, run_id)
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def _candidates(conn, name: str, since: datetime, embed_model: str) -> list[dict]:
    rows = conn.execute(
        "SELECT story_id, item_id, fingerprint, summary, urls_json, last_run_id, headline, last_seen "
        "FROM briefing_stories WHERE briefing=? AND last_seen>=?", (name, since.isoformat())
    ).fetchall()
    out = []
    for sid, item_id, fp, summary, urls_json, last_run, headline, last_seen in rows:
        try:
            urls = set(json.loads(urls_json))
        except ValueError:
            urls = set()
        vec_row = conn.execute(
            "SELECT vector FROM embeddings WHERE ns=? AND item_id=? AND model=?",
            (VECTOR_NS, item_id, embed_model),
        ).fetchone()
        out.append({"story_id": sid, "fingerprint": fp, "summary": summary, "urls": urls,
                    "last_run_id": last_run, "headline": headline, "last_seen": last_seen,
                    "vector": np.asarray(unpack_vector(vec_row[0]), dtype=np.float32) if vec_row else None})
    return out


def match_previous(stories: list[StoryResult], prev: list[dict], threshold: float) -> dict[int, dict]:
    """Map story index -> previous story row: most shared URLs first, then
    centroid cosine >= threshold. Each previous story matches at most once."""
    taken: set[str] = set()
    out: dict[int, dict] = {}
    for i, s in enumerate(stories):
        urls = set(s.urls)
        best, best_overlap = None, 0
        for p in prev:
            if p["story_id"] in taken:
                continue
            overlap = len(urls & p["urls"])
            if overlap > best_overlap:
                best, best_overlap = p, overlap
        if best is None:
            cen = _unit(np.asarray(s.centroid, dtype=np.float32))
            best_sim = threshold
            for p in prev:
                if p["story_id"] in taken or p["vector"] is None or p["vector"].shape != cen.shape:
                    continue
                sim = float(_unit(p["vector"]) @ cen)
                if sim >= best_sim:
                    best, best_sim = p, sim
        if best is not None:
            taken.add(best["story_id"])
            out[i] = best
    return out


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def _docs_block(docs: list[Doc]) -> tuple[str, bool]:
    parts, total, trimmed = [], 0, False
    for n, d in enumerate(docs, 1):
        when = d.published.date().isoformat() if d.published else "undated"
        text = d.text
        if len(text) > STORY_DOC_CHARS:
            text, trimmed = text[:STORY_DOC_CHARS], True
        block = f"[{n}] {d.source} | {when} | {d.url}\nTITLE: {d.title}\n{text}"
        if total + len(block) > STORY_PROMPT_CHARS:
            block, trimmed = block[:max(STORY_PROMPT_CHARS - total, 0)], True
        if block:
            parts.append(block)
            total += len(block)
        if total >= STORY_PROMPT_CHARS:
            trimmed = trimmed or n < len(docs)
            break
    return "\n\n".join(parts), trimmed


def fingerprint(cfg: BriefingConfig, docs: list[Doc]) -> str:
    """Changes whenever the model's input would: the docs (URL + text), and
    the briefing's interests (they shape why_it_matters)."""
    h = hashlib.sha256()
    h.update(json.dumps(cfg.interests, ensure_ascii=False).encode("utf-8"))
    for d in sorted(docs, key=lambda d: d.meta["norm_url"]):
        h.update(d.meta["norm_url"].encode("utf-8"))
        h.update(hashlib.sha1(f"{d.title}\n{d.text[:STORY_DOC_CHARS]}".encode("utf-8")).digest())
    return h.hexdigest()[:24]


def _strs(value: object, limit: int, chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip()[:chars] for v in value if isinstance(v, str) and v.strip()][:limit]


def sanitize_extract(raw: dict) -> dict:
    """Coerce a model answer to the documented shape and caps."""
    sev = raw.get("severity")
    try:
        sev = min(max(int(sev), 0), 3)
    except (TypeError, ValueError):
        sev = 0
    quotes = []
    for q in raw.get("key_quotes") or []:
        if isinstance(q, dict) and isinstance(q.get("quote"), str) and q["quote"].strip():
            quotes.append({"quote": q["quote"].strip()[:500], "source_url": str(q.get("source_url") or "")[:2000]})
    timeline = []
    for t in raw.get("timeline") or []:
        if isinstance(t, dict) and isinstance(t.get("event"), str) and t["event"].strip():
            timeline.append({"date": str(t.get("date") or "")[:40], "event": t["event"].strip()[:300]})
    countries = []
    for c in raw.get("countries") or []:
        if isinstance(c, str) and re.fullmatch(r"[A-Za-z]{2}", c.strip()) and geo.country(c.strip()):
            countries.append(c.strip().upper())
    return {
        "headline": str(raw.get("headline") or "").strip()[:200],
        "summary": str(raw.get("summary") or "").strip()[:2000],
        "highlights": _strs(raw.get("highlights"), 6, 300),
        "key_quotes": quotes[:4],
        "timeline": timeline[:8],
        "why_it_matters": str(raw.get("why_it_matters") or "").strip()[:500],
        "countries": list(dict.fromkeys(countries))[:12],
        "actors": _strs(raw.get("actors"), 12, 120),
        "severity": sev,
    }


def verify_quotes(story: StoryResult) -> list[dict]:
    """Each key quote checked against its cited source's text; if it isn't
    there but another source of the story has it, the citation is corrected
    to that source. Everything else is `verified: False`."""
    by_url = {d.meta["norm_url"]: d for d in story.docs}
    out = []
    for q in (story.extract or {}).get("key_quotes", []):
        try:
            cited = by_url.get(normalize_url(q["source_url"])) if is_http_url(q["source_url"]) else None
        except ValueError:
            cited = None
        found = None
        if cited is not None and verify_quote(q["quote"], f"{cited.title}\n{cited.text}"):
            found = cited
        else:
            for d in story.docs:
                if d is not cited and verify_quote(q["quote"], f"{d.title}\n{d.text}"):
                    found = d
                    break
        if found is not None:
            out.append({"quote": q["quote"], "source_url": found.url, "verified": True})
        else:
            out.append({"quote": q["quote"], "source_url": cited.url if cited else "", "verified": False})
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_briefing(
    cfg: BriefingConfig,
    *,
    lm: Any,
    conn: Any,
    run_id: int,
    fetcher: Any,
    now: datetime | None = None,
    searxng_factory: Callable[[], Any] | None = None,
    fresh: bool = False,
) -> BriefingResult:
    now = now or _now()
    embed_model = model_id(lm, "embed_model")
    ctx = SourceContext(briefing=cfg, fetcher=fetcher, lm=lm, conn=conn, now=now, limit=cfg.max_items,
                        searxng_factory=searxng_factory, fresh=fresh)
    result = BriefingResult(cfg=cfg, run_id=run_id, now=now, ctx=ctx, notes=ctx.notes)
    row = conn.execute(
        "SELECT MAX(started) FROM briefing_runs WHERE briefing=? AND status='ok'", (cfg.name,)
    ).fetchone()
    result.last_ok = row[0] if row else None

    docs = collect(cfg, ctx)
    collected = len(docs)
    docs, skipped = prefilter(cfg, docs)
    fetched, _ = follow_links(cfg, docs, ctx)
    result.counts.update(collected=collected, prefiltered_out=skipped, fetched=fetched, candidates=len(docs))
    truncated = sum(1 for d in docs if d.meta.get("truncated"))
    if truncated:
        ctx.note(f"{truncated} article(s) longer than {MAX_DOC_CHARS:,} characters were truncated")

    version = memo_version(MEMO_VERSION, prompt_hash(SYSTEM_PROMPT), prompt_hash(USER_PROMPT), SCHEMA,
                           model_id(lm))
    memo = Memo(conn, TASK, version, fresh=fresh)
    result.memo = memo
    prev_run = _prev_run_id(conn, cfg.name, run_id)

    if docs:
        doc_vecs = embed_texts(lm, [f"{d.title}\n\n{d.text[:EMBED_CHARS]}" for d in docs], embed_model)
        interest_vecs = _unit(embed_texts(lm, cfg.interests, embed_model))
        groups = cluster(doc_vecs, cfg.cluster_threshold)
    else:
        doc_vecs, interest_vecs, groups = np.zeros((0, 0), dtype=np.float32), None, []
    result.counts["clusters"] = len(groups)

    stories: list[StoryResult] = []
    for g in groups:
        members = [docs[i] for i in g]
        centroid = doc_vecs[g].mean(axis=0)
        interest = float(np.max(interest_vecs @ _unit(centroid))) if interest_vecs is not None else 0.0
        s = StoryResult(story_id="", docs=members, centroid=centroid.tolist(), interest=interest)
        s.docs.sort(key=lambda d: d.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        s.score = story_score(interest, len(members), len(s.sources),
                              recency_decay(s.newest, now, cfg.half_life_hours))
        stories.append(s)
    stories.sort(key=lambda s: s.score, reverse=True)
    if len(stories) > cfg.max_stories:
        ctx.note(f"showing the top {cfg.max_stories} of {len(stories)} stories (max_stories)")
        stories = stories[:cfg.max_stories]

    # Identity + status against earlier runs.
    prev = _candidates(conn, cfg.name, now - timedelta(days=cfg.memory_days), embed_model)
    matches = match_previous(stories, prev, cfg.cluster_threshold)
    known_ids = {p["story_id"] for p in prev} | {
        r[0] for r in conn.execute("SELECT story_id FROM briefing_stories WHERE briefing=?", (cfg.name,))}
    for i, s in enumerate(stories):
        if i in matches:
            s.story_id = matches[i]["story_id"]
        else:
            base = hashlib.sha1(f"{cfg.name}|{s.docs[0].meta['norm_url']}".encode("utf-8")).hexdigest()[:12]
            sid, n = base, 1
            while sid in known_ids:
                sid, n = f"{base}-{n}", n + 1
            s.story_id = sid
        known_ids.add(s.story_id)

    # Extract, memo'd by story fingerprint.
    system = load_prompt(SYSTEM_PROMPT)
    interests_txt = "\n".join(f"- {t}" for t in cfg.interests)
    trimmed = failed = 0
    for i, s in enumerate(stories):
        used = s.docs[:cfg.max_docs_per_story]
        s.fingerprint = fingerprint(cfg, used)
        block, s.trimmed = _docs_block(used)
        trimmed += s.trimmed or len(s.docs) > len(used)
        prompt = load_prompt(USER_PROMPT, title=cfg.title, interests=interests_txt, count=len(used), docs=block)
        try:
            raw = memo.cached(f"{cfg.name}:{s.fingerprint}",
                              lambda: lm.extract(prompt, SCHEMA, system=system, name="story"),
                              valid=lambda r: has_keys(r, SCHEMA["required"]))
            if not isinstance(raw, dict):
                raise ValueError("model returned no JSON object")
            s.extract = sanitize_extract(raw)
            s.quotes = verify_quotes(s)
            s.countries = s.extract["countries"]
        except Exception as exc:  # one story's failure never kills the briefing
            failed += 1
            s.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            log.warning("briefing story extract failed", exc_info=True)
        p = matches.get(i)
        if p is None:
            s.status = "new"
        elif p["fingerprint"] == s.fingerprint:
            s.status = "ongoing"
        elif set(s.urls) - p["urls"] or (s.extract and s.extract["summary"] != p["summary"]):
            s.status = "updated"
        else:
            s.status = "ongoing"
    if trimmed:
        ctx.note(f"{trimmed} stor{'y' if trimmed == 1 else 'ies'}: the model saw at most "
                 f"{cfg.max_docs_per_story} sources, {STORY_DOC_CHARS:,} characters each "
                 f"({STORY_PROMPT_CHARS:,} per story)")
    if failed:
        ctx.note(f"{failed} stor{'y' if failed == 1 else 'ies'} could not be summarized (shown with source titles only)")

    if cfg.regions:
        wanted = set(cfg.regions)
        kept = [s for s in stories if not s.countries or set(s.countries) & wanted]
        if len(kept) < len(stories):
            ctx.note(f"{len(stories) - len(kept)} stor{'y' if len(stories) - len(kept) == 1 else 'ies'} "
                     f"outside regions {', '.join(cfg.regions)} left out")
        stories = kept

    for ext in active_extensions(cfg):
        if ext.enrich is None:
            continue
        try:
            ext.enrich(stories, ctx)
        except Exception as exc:
            ctx.note(f"extension {ext.name}: {type(exc).__name__}: {str(exc)[:200]}")

    matched_ids = {s.story_id for s in stories}
    result.dropped = [
        {"story_id": p["story_id"], "headline": p["headline"], "last_seen": p["last_seen"]}
        for p in prev if p["last_run_id"] == prev_run and p["story_id"] not in matched_ids
    ] if prev_run is not None else []
    result.stories = stories
    result.counts["extract_calls"] = memo.misses

    persist(conn, result, docs, doc_vecs, embed_model)
    return result


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def article_item_id(briefing: str, norm_url: str) -> str:
    return f"{briefing}:article:{hashlib.sha1(norm_url.encode('utf-8')).hexdigest()[:16]}"


def story_item_id(briefing: str, story_id: str) -> str:
    return f"{briefing}:story:{story_id}"


def persist(conn, result: BriefingResult, docs: list[Doc], doc_vecs: np.ndarray, embed_model: str) -> None:
    """Upsert stories, articles and their vectors; record the run. Keyed so
    a rerun never duplicates rows."""
    cfg, now = result.cfg, result.now.isoformat()
    story_of = {d.meta["norm_url"]: s.story_id for s in result.stories for d in s.docs}
    vectors: list[tuple[str, list[float]]] = []
    for i, d in enumerate(docs):
        url = d.meta["norm_url"]
        item_id = article_item_id(cfg.name, url)
        meta = {k: v for k, v in d.meta.items() if k not in ("norm_url",)}
        conn.execute(
            "INSERT INTO briefing_articles(briefing, url, item_id, story_id, title, source, published, lean, "
            "reliability, grp, text, text_hash, meta_json, first_seen, last_seen, last_run_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(briefing, url) DO UPDATE SET "
            "story_id=excluded.story_id, title=excluded.title, source=excluded.source, "
            "published=excluded.published, lean=excluded.lean, reliability=excluded.reliability, "
            "grp=excluded.grp, text=excluded.text, text_hash=excluded.text_hash, meta_json=excluded.meta_json, "
            "last_seen=excluded.last_seen, last_run_id=excluded.last_run_id",
            (cfg.name, url, item_id, story_of.get(url), d.title, d.source,
             d.published.isoformat() if d.published else None, str(d.meta.get("lean", "unknown")),
             str(d.meta.get("reliability", "unknown")), str(d.meta.get("group", "")),
             d.text[:STORED_TEXT_CHARS], hashlib.sha1(d.text.encode("utf-8")).hexdigest(),
             json.dumps(meta, default=str), now, now, result.run_id),
        )
        if doc_vecs.size:
            vectors.append((item_id, doc_vecs[i].tolist()))
    for s in result.stories:
        ex = s.extract or {}
        data = {"extract": s.extract, "quotes": s.quotes, "error": s.error, "interest": s.interest,
                "sources": s.sources, "extras": s.extras}
        item_id = story_item_id(cfg.name, s.story_id)
        conn.execute(
            "INSERT INTO briefing_stories(briefing, story_id, item_id, fingerprint, headline, summary, status, "
            "score, severity, countries_json, urls_json, data_json, first_seen, last_seen, first_run_id, "
            "last_run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(briefing, story_id) DO UPDATE SET "
            "item_id=excluded.item_id, "
            # a failed extraction (empty summary) keeps the last good fingerprint too, so
            # the next successful run compares against what was actually summarized
            "fingerprint=CASE WHEN excluded.summary='' THEN briefing_stories.fingerprint "
            "ELSE excluded.fingerprint END, "
            # a failed extraction (empty summary) keeps the last good headline/summary
            "headline=CASE WHEN excluded.summary='' THEN briefing_stories.headline ELSE excluded.headline END, "
            "summary=CASE WHEN excluded.summary='' THEN briefing_stories.summary ELSE excluded.summary END, status=excluded.status, score=excluded.score, severity=excluded.severity, "
            "countries_json=excluded.countries_json, urls_json=excluded.urls_json, data_json=excluded.data_json, "
            "last_seen=excluded.last_seen, last_run_id=excluded.last_run_id",
            (cfg.name, s.story_id, item_id, s.fingerprint, s.headline, ex.get("summary", ""), s.status,
             s.score, ex.get("severity"), json.dumps(s.countries), json.dumps(s.urls),
             json.dumps(data, default=str), now, now, result.run_id, result.run_id),
        )
        vectors.append((item_id, list(s.centroid)))
    if vectors:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings(ns, item_id, model, dim, vector) VALUES (?, ?, ?, ?, ?)",
            [(VECTOR_NS, item_id, embed_model, len(v), pack_vector(v)) for item_id, v in vectors],
        )
    conn.execute(
        "INSERT OR REPLACE INTO briefing_runs(briefing, run_id, started, status, summary_json) VALUES (?,?,?,?,?)",
        (cfg.name, result.run_id, now, "ok", json.dumps(result.summary(), default=str)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_BADGE = {"new": ("NEW", "hot"), "updated": ("UPDATED", "warn"), "ongoing": (None, "acc")}


def _age(newest: datetime | None, now: datetime) -> str:
    if newest is None:
        return "undated"
    h = max((now - newest).total_seconds() / 3600, 0)
    return f"{h:.0f} h ago" if h < 48 else f"{h / 24:.0f} d ago"


def story_card(s: StoryResult, now: datetime, ctx: SourceContext | None = None, extensions=()) -> Story:
    ex = s.extract or {}
    badge, tone = _BADGE.get(s.status, (None, "acc"))
    meta = [f"{len(s.sources)} source{'s' if len(s.sources) != 1 else ''}", _age(s.newest, now)]
    if ex:
        meta.append(f"severity {ex.get('severity', 0)}")
    if s.countries:
        meta.append(", ".join(geo.country_name(c) or c for c in s.countries[:6]))
    links: list = [StorySource(f"{d.source}: {d.title[:90]}" if d.source else d.title[:90], d.url)
                   for d in s.docs[:MAX_SOURCES_SHOWN]]
    if len(s.docs) > MAX_SOURCES_SHOWN:
        links.append(StorySource(f"+{len(s.docs) - MAX_SOURCES_SHOWN} more"))
    extras: list = []
    bars: list = []

    def failed(ext, hook: str, exc: Exception) -> None:
        if ctx is not None:
            ctx.note(f"extension {ext.name} ({hook}): {type(exc).__name__}: {str(exc)[:200]}")

    for ext in extensions:
        if ext.render_sources is not None:
            try:
                out = ext.render_sources(s, list(links), ctx)
                if out is not None:
                    links = [x for x in out if isinstance(x, StorySource)]
            except Exception as exc:
                failed(ext, "render_sources", exc)
        if ext.render is not None:
            try:
                for part in ext.render(s, ctx) or []:
                    if isinstance(part, StackedBar):
                        bars.append(part)
                    elif isinstance(part, StoryExtra):
                        extras.append(part)
            except Exception as exc:
                failed(ext, "render", exc)
    # An extension's bad tone must not break the whole report at render time.
    def tone_ok(t: object) -> str:
        return t if t in TONES else "muted"

    for x in links:
        x.tone = tone_ok(x.tone)
    for x in extras:
        x.tone = tone_ok(x.tone)
    clean_bars = []
    for b in bars:
        try:
            b.segments = [(str(lbl), v, tone_ok(t)) for lbl, v, t in b.segments]
            clean_bars.append(b)
        except (TypeError, ValueError):  # malformed segments: drop the bar
            pass
    bars = clean_bars
    summary = ex.get("summary", "") if ex else "Could not be summarized this run; see the sources below."
    return Story(
        title=s.headline, summary=summary, meta=" · ".join(meta), badge=badge, tone=tone,
        highlights=ex.get("highlights", []),
        quotes=[StoryQuote(q["quote"], q["source_url"] or None, q["verified"]) for q in s.quotes],
        timeline=[(t["date"], t["event"]) for t in ex.get("timeline", [])],
        why=ex.get("why_it_matters", ""), sources=links, tags=ex.get("actors", [])[:6],
        extras=extras, anchor=s.anchor, bars=bars,
    )


def read_minutes(stories: list[StoryResult]) -> int:
    words = 0
    for s in stories:
        ex = s.extract or {}
        parts = [s.headline, ex.get("summary", ""), ex.get("why_it_matters", ""), *ex.get("highlights", []),
                 *(q["quote"] for q in s.quotes), *(t["event"] for t in ex.get("timeline", []))]
        words += sum(len(str(p).split()) for p in parts)
    return max(1, math.ceil(words / READ_WPM)) if words else 0


def build_report(result: BriefingResult) -> Report:
    cfg, stories = result.cfg, result.stories
    extensions = active_extensions(cfg)
    new = sum(1 for s in stories if s.status == "new")
    updated = sum(1 for s in stories if s.status == "updated")
    sources = {src for s in stories for src in s.sources}
    minutes = read_minutes(stories)
    r = Report(cfg.title, kicker=f"lmsa · briefing · {cfg.name}", subtitle=cfg.description,
               sticker=f"{len(stories)} stories", sticker_tone="acc" if stories else "muted",
               source=f"config/briefings/{cfg.name}.toml")
    r.tiles([
        Tile("Stories", len(stories), tone="acc",
             note=f"{result.counts.get('candidates', 0)} articles, {result.counts.get('collected', 0)} collected"),
        Tile("Sources", len(sources), tone="info"),
        Tile("New since last run", new, delta=f"+{updated} updated" if updated else None,
             tone="hot" if new else "muted", note=None if result.last_ok else "first run"),
        Tile("Read time", f"~{minutes} min" if minutes else "0 min", tone="violet"),
    ])
    cards = [story_card(s, result.now, result.ctx, extensions) for s in stories]
    sec = r.section("Stories", tone="acc")
    if cards:
        sec.stories(cards)
    else:
        sec.callout("No stories this run: nothing collected matched the briefing's keywords and interests.",
                    label="empty", tone="info")
    for ext in extensions:
        if ext.section is None:
            continue
        try:
            ext.section(r, stories, result.ctx)
        except Exception as exc:
            result.notes.append(f"extension {ext.name} (section): {type(exc).__name__}: {str(exc)[:200]}")
    if result.dropped:
        r.section("Dropped since last run", tone="muted").table(
            ["Story", "Last seen"], [(d["headline"], d["last_seen"][:16].replace("T", " ")) for d in result.dropped])
    if result.notes:
        r.section("Notes", tone="warn").callout("\n\n".join(dict.fromkeys(result.notes)), label="caps & skips",
                                                tone="warn")
    r.section("You're caught up", tone="acc").callout(
        f"That's all {len(stories)} stor{'y' if len(stories) == 1 else 'ies'} "
        f"({new} new, {updated} updated).", label="end", tone="acc")
    return r
