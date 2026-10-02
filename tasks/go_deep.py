"""Go deep: research one briefing story (or any topic) further, with a cited report.

    python -m tasks.go_deep --story example:1a2b3c4d5e6f   # a briefing story (<briefing>:<story_id>)
    python -m tasks.go_deep --topic "Red Sea shipping insurance"

The model plans 3-6 sub-questions; each one gets a SearXNG search, a few
fetched pages (`lmsa.web`), and a section drafted from only the most
relevant passages (pages are chunked and embedded, top-k per question) with
inline `[n]` citations. A final pass writes an executive summary and open
questions. Every `[n]` must name a page that was fetched *and* shown to the
model for that section -- anything else is replaced with `[?]` and counted;
quotes are checked word for word against their page (`verify_quote`).

Hard caps (`--max-searches`, `--max-pages`, `--max-chars` sent to the
model, `--max-minutes` wall clock) stop the run cleanly: the report keeps
what was done and says which cap stopped it. Manual only -- launch it from a
story card's "go deep" link, which opens the dashboard with `--story`
pre-filled (nothing runs until you press run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lmsa import LMStudio, Store, layout
from lmsa.memo import Memo, add_footer, has_keys, memo_version, model_id
from lmsa.prompts import load as load_prompt
from lmsa.prompts import prompt_hash
from lmsa.report import Link, Report, Story, StoryQuote, StorySource, Tile

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"model": "chat", "searxng": True}

TASK = "go_deep"
USER_AGENT = "lmsa-go-deep/1.0 (personal research report)"
MEMO_VERSION = "go_deep-1"
PLAN_PROMPT = "go_deep/plan_system"
SECTION_PROMPT = "go_deep/section_system"
SYNTH_PROMPT = "go_deep/synth_system"

MIN_QUESTIONS, MAX_QUESTIONS = 3, 6
RESULTS_PER_SEARCH = 8        # search results considered per sub-question
PAGES_PER_QUESTION = 3        # new pages fetched per sub-question
SEED_PAGES = 2                # a story's own article links fetched first
PAGE_TEXT_CHARS = 20_000      # kept per page (chunked, and the quote-check source)
CHUNK_WORDS, CHUNK_OVERLAP = 160, 30
MAX_CHUNKS_PER_PAGE = 40
TOP_K = 6                     # passages per sub-question
PASSAGE_CHARS = 1_200         # one passage, trimmed
SECTION_PASSAGE_CHARS = 6_000  # all passages of one section prompt
SEED_CONTEXT_CHARS = 1_500
SYNTH_SECTION_CHARS = 1_500   # each section as the synthesis sees it
SYNTH_MIN_CHARS = 1_500       # below this much budget left, no synthesis
MAX_QUOTES = 3

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "query": {"type": "string"}},
        "required": ["question", "query"], "additionalProperties": False}}},
    "required": ["questions"], "additionalProperties": False,
}
SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "quotes": {"type": "array", "items": {
            "type": "object",
            "properties": {"quote": {"type": "string"}, "source": {"type": "integer"}},
            "required": ["quote", "source"], "additionalProperties": False}},
    },
    "required": ["text", "quotes"], "additionalProperties": False,
}
SYNTH_SCHEMA = {
    "type": "object",
    "properties": {"executive_summary": {"type": "string"},
                   "open_questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["executive_summary", "open_questions"], "additionalProperties": False,
}

CAP_FLAGS = {"searches": "--max-searches", "pages": "--max-pages", "chars": "--max-chars",
             "time": "--max-minutes"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Caps:
    searches: int = 6
    pages: int = 12
    chars: int = 60_000
    seconds: float = 20 * 60


class Budget:
    """Counts searches, pages and characters sent to the model against
    `Caps`, plus a wall-clock deadline. `allow(kind)` both checks and
    spends; the first refusal of each kind is recorded in `hit` (in order)."""

    def __init__(self, caps: Caps, clock: Callable[[], float] = time.monotonic):
        self.caps, self.clock = caps, clock
        self.start = clock()
        self.used = {"searches": 0, "pages": 0, "chars": 0}
        self.hit: list[str] = []

    def _stop(self, kind: str) -> bool:
        if kind not in self.hit:
            self.hit.append(kind)
        return False

    def expired(self) -> bool:
        if self.clock() - self.start >= self.caps.seconds:
            self._stop("time")
            return True
        return False

    def allow(self, kind: str, amount: int = 1, *, limit: int | None = None) -> bool:
        if self.expired():
            return False
        cap = limit if limit is not None else getattr(self.caps, kind)
        if self.used[kind] + amount > cap:
            return self._stop(kind)
        self.used[kind] += amount
        return True

    def spent(self, kind: str) -> bool:
        """True (and recorded as a hit) once a count cap is fully used, so no
        search is spent gathering pages that could never be fetched."""
        if kind in self.hit:
            return True
        if self.used[kind] >= getattr(self.caps, kind):
            return not self._stop(kind)
        return False

    @property
    def chars_left(self) -> int:
        return max(self.caps.chars - self.used["chars"], 0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Source:
    n: int
    url: str
    title: str
    text: str
    truncated: bool = False
    seed: bool = False


@dataclass
class Chunk:
    source: int
    text: str
    vec: list[float] | None = None


@dataclass
class Question:
    question: str
    query: str
    text: str = ""
    quotes: list[dict] = field(default_factory=list)
    shown: list[int] = field(default_factory=list)   # source numbers whose passages the model saw
    cited: list[int] = field(default_factory=list)
    dropped: int = 0                                 # [n] citations the model invented
    trimmed: bool = False
    skipped: str = ""                                # why no section was drafted
    error: str = ""


@dataclass
class Research:
    topic: str
    story: dict | None = None
    questions: list[Question] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    summary: str = ""
    open_questions: list[str] = field(default_factory=list)
    summary_dropped: int = 0
    notes: list[str] = field(default_factory=list)
    caps_hit: list[str] = field(default_factory=list)
    used: dict = field(default_factory=dict)
    caps: Caps = field(default_factory=Caps)
    memo: Any = None

    def note(self, text: str) -> None:
        self.notes.append(str(text)[:300])

    @property
    def citations_dropped(self) -> int:
        return self.summary_dropped + sum(q.dropped for q in self.questions)

    def summary_dict(self) -> dict:
        return {"topic": self.topic[:200], "questions": len(self.questions),
                "sections": sum(1 for q in self.questions if q.text), "pages": len(self.sources),
                "citations_dropped": self.citations_dropped, "caps_hit": self.caps_hit, **self.used}


# ---------------------------------------------------------------------------
# Story seed
# ---------------------------------------------------------------------------

_ADDRESS_RE = re.compile(r"^([a-z0-9][a-z0-9_-]{0,63}):([A-Za-z0-9_-]{1,64})$")


def load_story(conn, address: str) -> dict:
    """The briefing story `<briefing>:<story_id>` as a seed dict; ValueError
    when the address is malformed or no such story is stored."""
    m = _ADDRESS_RE.fullmatch(address.strip())
    if not m:
        raise ValueError(f"--story must look like <briefing>:<story_id>, got {address[:80]!r}")
    briefing, story_id = m.groups()
    row = conn.execute(
        "SELECT headline, summary, urls_json, data_json, last_run_id FROM briefing_stories "
        "WHERE briefing=? AND story_id=?", (briefing, story_id)).fetchone()
    if row is None:
        raise ValueError(f"no stored briefing story {briefing}:{story_id}")
    headline, summary, urls_json, data_json, last_run = row
    try:
        urls = [u for u in json.loads(urls_json or "[]") if isinstance(u, str)]
    except ValueError:
        urls = []
    try:
        extract = (json.loads(data_json or "{}") or {}).get("extract") or {}
    except ValueError:
        extract = {}
    report = conn.execute("SELECT report_path FROM briefing_runs WHERE briefing=? AND run_id=?",
                          (briefing, last_run)).fetchone()
    return {"address": f"{briefing}:{story_id}", "briefing": briefing, "story_id": story_id,
            "headline": headline, "summary": summary or "",
            "highlights": [h for h in extract.get("highlights", []) if isinstance(h, str)],
            "urls": urls, "report_path": report[0] if report and report[0] else None}


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def chunk_text(text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    toks = text.split()
    if not toks:
        return []
    step = max(words - overlap, 1)
    out = []
    for i in range(0, len(toks), step):
        out.append(" ".join(toks[i:i + words]))
        if i + words >= len(toks) or len(out) >= MAX_CHUNKS_PER_PAGE:
            break
    return out


def _cos(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


_WORD = re.compile(r"[^\W_]{3,}")


def _overlap(q: str, text: str) -> float:
    qw = set(_WORD.findall(q.lower()))
    if not qw:
        return 0.0
    return len(qw & set(_WORD.findall(text.lower()))) / len(qw)


def top_passages(question: str, qvec: list[float] | None, chunks: list[Chunk], k: int = TOP_K) -> list[Chunk]:
    """The k chunks closest to the question: cosine when both sides have a
    vector, word overlap otherwise (an embedding failure degrades, not aborts)."""
    def score(c: Chunk) -> float:
        if qvec is not None and c.vec is not None:
            return _cos(qvec, c.vec)
        return _overlap(question, c.text)
    return sorted(chunks, key=score, reverse=True)[:k]


_ITEM = r"\d+(?:\s*[-‐-―]\s*\d+)?"  # 3, or a range 1-3 / 1–3 (any dash)
_CITE = re.compile(rf"\[(\s*{_ITEM}(?:\s*[,;]\s*{_ITEM})*\s*)\]")
_RANGE_SPLIT = re.compile(r"\s*[-‐-―]\s*")
MAX_CITE_DIGITS = 4
MAX_RANGE = 20


def _resolve_item(item: str, allowed: set[int]) -> list[int] | None:
    """The source numbers one citation item names, or None unless every one
    of them is allowed (an item that doesn't fully resolve is dropped whole)."""
    parts = _RANGE_SPLIT.split(item.strip())
    if any(len(p) > MAX_CITE_DIGITS for p in parts):
        return None
    lo, hi = int(parts[0]), int(parts[-1])
    if lo > hi or hi - lo > MAX_RANGE:
        return None
    nums = list(range(lo, hi + 1))
    return nums if all(n in allowed for n in nums) else None


def check_citations(text: str, allowed: set[int]) -> tuple[str, list[int], int]:
    """Keep `[n]` citations (and ranges `[1-3]`, expanded to `[1][2][3]`)
    whose every number is in `allowed`; any other bracketed number or range
    becomes `[?]` and counts as dropped. Returns (text, cited numbers in
    order, dropped count)."""
    cited: list[int] = []
    dropped = 0

    def fix(m: re.Match) -> str:
        nonlocal dropped
        out = []
        for item in re.split(r"\s*[,;]\s*", m.group(1).strip()):
            nums = _resolve_item(item, allowed)
            if nums is None:
                dropped += 1
                out.append("[?]")
                continue
            for n in nums:
                out.append(f"[{n}]")
                if n not in cited:
                    cited.append(n)
        return "".join(out)

    return _CITE.sub(fix, text), cited, dropped


def check_quotes(raw: list, by_n: dict[int, Source], allowed: set[int]) -> list[dict]:
    """Each quote verified against its cited page (or another page the model
    was shown); an unverifiable one is kept flagged, never shown as checked."""
    from lmsa.briefing.quotes import verify_quote

    out = []
    for q in raw[:MAX_QUOTES]:
        if not isinstance(q, dict) or not isinstance(q.get("quote"), str) or not q["quote"].strip():
            continue
        text = q["quote"].strip()[:500]
        cited = q.get("source") if isinstance(q.get("source"), int) and q.get("source") in allowed else None
        found = None
        if cited is not None and verify_quote(text, by_n[cited].text):
            found = cited
        else:
            found = next((n for n in sorted(allowed) if n != cited and verify_quote(text, by_n[n].text)), None)
        out.append({"quote": text, "source": found if found is not None else cited, "verified": found is not None})
    return out


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def research(topic: str, *, lm, searx, fetcher, conn, caps: Caps = Caps(), story: dict | None = None,
             clock: Callable[[], float] = time.monotonic, fresh: bool = False) -> Research:
    """Plan -> (search -> fetch -> passages -> section) per sub-question ->
    synthesis, inside `caps`. Never raises for one bad search, page or
    model answer: those become notes; a cap stops the gathering or the
    drafting and is listed in `caps_hit`."""
    from lmsa.search import SearchError
    from lmsa.web import extract_article, normalize_url

    res = Research(topic=topic, story=story, caps=caps)
    budget = Budget(caps, clock)
    version = memo_version(MEMO_VERSION, prompt_hash(PLAN_PROMPT), prompt_hash(SECTION_PROMPT),
                           prompt_hash(SYNTH_PROMPT), PLAN_SCHEMA, SECTION_SCHEMA, SYNTH_SCHEMA, model_id(lm))
    memo = res.memo = Memo(conn, TASK, version, fresh=fresh)
    embed_model = model_id(lm, "embed_model")
    # Sections may spend the budget minus a reserve that keeps room for the synthesis.
    synth_reserve = min(10_000, caps.chars // 4)
    section_limit = caps.chars - synth_reserve
    chunks: list[Chunk] = []
    seen: set[str] = set()

    def ask(system_name: str, prompt: str, schema: dict, valid_keys: tuple[str, ...], *, limit: int | None = None):
        """One memo'd extract() call charged to the chars budget. None when a cap refuses it."""
        system = load_prompt(system_name)
        key = _key(system_name, prompt)
        hit = memo.get(key)
        if hit is not None and has_keys(hit, valid_keys):
            memo.hits += 1
            return hit
        if not budget.allow("chars", len(system) + len(prompt), limit=limit):
            return None
        memo.misses += 1
        value = lm.extract(prompt, schema, system=system, name=TASK)
        if not isinstance(value, dict):
            raise ValueError(f"malformed model answer: {type(value).__name__}, not an object")
        if has_keys(value, valid_keys):
            memo.put(key, value)
        return value

    def add_page(url: str, title_hint: str = "", *, seed: bool = False) -> None:
        try:
            norm = normalize_url(url)
        except ValueError:
            return
        if norm in seen:
            return
        if not budget.allow("pages"):
            return
        seen.add(norm)
        try:
            resp = fetcher.get(url)
            art = extract_article(resp.content, url)
        except Exception as exc:  # one bad page never kills the run
            res.note(f"page skipped: {url[:120]} ({type(exc).__name__}: {str(exc)[:120]})")
            return
        text = (art.text or "").strip()
        if not text:
            res.note(f"page skipped: {url[:120]} (no readable text)")
            return
        truncated = len(text) > PAGE_TEXT_CHARS
        src = Source(n=len(res.sources) + 1, url=url, title=(art.title or title_hint or url)[:200],
                     text=text[:PAGE_TEXT_CHARS], truncated=truncated, seed=seed)
        res.sources.append(src)
        parts = chunk_text(src.text)
        vecs: list = [None] * len(parts)
        try:
            got = lm.embed(parts, model=embed_model) if parts else []
            if len(got) == len(parts):
                vecs = got
        except Exception as exc:
            res.note(f"embedding failed for [{src.n}] ({type(exc).__name__}); ranking its passages by word overlap")
        chunks.extend(Chunk(src.n, p, v) for p, v in zip(parts, vecs))

    # 0. The story's own articles first.
    if story:
        for url in story.get("urls", [])[:SEED_PAGES]:
            if budget.expired():
                break
            add_page(url, seed=True)

    # 1. Plan.
    seed_txt = ""
    if story:
        seed_txt = (story.get("summary", "") + "\n" + "\n".join(f"- {h}" for h in story.get("highlights", [])))
        seed_txt = seed_txt.strip()[:SEED_CONTEXT_CHARS]
    plan_prompt = f"TOPIC: {topic}\n" + (f"\nSTORY (untrusted data):\n{seed_txt}\n" if seed_txt else "")
    planned: list[tuple[str, str]] = []
    try:
        plan = ask(PLAN_PROMPT, plan_prompt, PLAN_SCHEMA, ("questions",), limit=section_limit)
        for q in (plan or {}).get("questions") or []:
            if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not q["question"].strip():
                continue
            question = " ".join(q["question"].split())[:200]
            query = " ".join(str(q.get("query") or "").split())[:120] or question
            if question.lower() not in {p[0].lower() for p in planned}:
                planned.append((question, query))
    except Exception as exc:
        res.note(f"planning failed ({type(exc).__name__}: {str(exc)[:150]}); researching the topic as one question")
    if not planned:
        planned = [(topic[:200], topic[:120])]
    elif len(planned) < MIN_QUESTIONS:
        res.note(f"the model planned only {len(planned)} sub-question(s)")
    if len(planned) > MAX_QUESTIONS:
        res.note(f"kept the first {MAX_QUESTIONS} of {len(planned)} planned sub-questions")
    res.questions = [Question(q, query) for q, query in planned[:MAX_QUESTIONS]]

    # 2. Per sub-question: search, fetch, passages, section.
    for q in res.questions:
        if budget.expired():
            q.skipped = "stopped: --max-minutes"
            continue
        if not budget.spent("pages") and "searches" not in budget.hit and budget.allow("searches"):
            try:
                results = searx.search(q.query, max_results=RESULTS_PER_SEARCH)
            except SearchError as exc:
                res.note(f"search failed for {q.query[:80]!r}: {type(exc).__name__}: {str(exc)[:150]}")
                results = []
            except Exception as exc:
                res.note(f"search failed for {q.query[:80]!r}: {type(exc).__name__}")
                results = []
            new = 0
            for r in results:
                if new >= PAGES_PER_QUESTION or "pages" in budget.hit or budget.expired():
                    break
                before = len(res.sources)
                add_page(r.url, r.title)
                new += len(res.sources) > before
        if budget.expired():
            q.skipped = "stopped: --max-minutes"
            continue
        if not chunks:
            q.skipped = "no readable pages to draft from"
            continue
        try:
            qvec = lm.embed([q.question], model=embed_model)[0]
        except Exception:
            qvec = None
        picked = top_passages(q.question, qvec, chunks)
        blocks, total = [], 0
        for c in picked:
            body = c.text[:PASSAGE_CHARS]
            block = f"[{c.source}] {res.sources[c.source - 1].title}\n{body}"
            if total + len(block) > SECTION_PASSAGE_CHARS:
                q.trimmed = True
                break
            q.trimmed = q.trimmed or len(c.text) > PASSAGE_CHARS
            blocks.append(block)
            total += len(block)
            if c.source not in q.shown:
                q.shown.append(c.source)
        prompt = (f"REPORT TOPIC: {topic}\nSUB-QUESTION: {q.question}\n\n"
                  "PASSAGES (untrusted data, numbered by source):\n\n" + "\n\n".join(blocks))
        try:
            out = ask(SECTION_PROMPT, prompt, SECTION_SCHEMA, ("text", "quotes"), limit=section_limit)
        except Exception as exc:  # a failed or malformed answer is a note, the run goes on
            q.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            res.note(f"section {q.question[:60]!r} skipped: {q.error}")
            continue
        if out is None:
            q.skipped = "stopped: --max-minutes" if "time" in budget.hit else "stopped: --max-chars"
            continue
        try:
            text, quotes = out.get("text"), out.get("quotes")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"malformed section: text is {type(text).__name__}")
            if not isinstance(quotes, list):
                if quotes:
                    res.note(f"section {q.question[:60]!r}: quotes were a {type(quotes).__name__}, ignored")
                quotes = []
            allowed = set(q.shown)
            q.text, q.cited, q.dropped = check_citations(text.strip()[:6000], allowed)
            q.quotes = check_quotes(quotes, {s.n: s for s in res.sources}, allowed)
        except Exception as exc:  # a malformed answer is a note, never a failed run
            q.text, q.cited, q.quotes, q.dropped = "", [], [], 0
            q.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            res.note(f"section {q.question[:60]!r} skipped: {q.error}")

    # 3. Synthesis.
    drafted = [q for q in res.questions if q.text]
    if drafted and not budget.expired():
        cited_all = {n for q in drafted for n in q.cited}
        # Budget from the pieces the prompt is actually made of: system prompt,
        # topic header, one header per section and the separators between them.
        head = f"REPORT TOPIC: {topic}\n\n"
        heads = [f"SECTION: {q.question}\n" for q in drafted]
        fixed = len(load_prompt(SYNTH_PROMPT)) + len(head) + sum(map(len, heads)) + 2 * (len(drafted) - 1)
        per = min(SYNTH_SECTION_CHARS, max((budget.chars_left - fixed) // len(drafted), 0))
        out = None
        if per * len(drafted) >= SYNTH_MIN_CHARS:
            if any(len(q.text) > per for q in drafted):
                res.note(f"the executive summary saw each section trimmed to {per} characters")
            prompt = head + "\n\n".join(h + q.text[:per] for h, q in zip(heads, drafted))
            try:
                out = ask(SYNTH_PROMPT, prompt, SYNTH_SCHEMA, ("executive_summary", "open_questions"))
                if out is not None:
                    summary = out.get("executive_summary")
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError(f"malformed summary: executive_summary is {type(summary).__name__}")
                    res.summary, _, res.summary_dropped = check_citations(summary.strip()[:4000], cited_all)
                    oq = out.get("open_questions")
                    res.open_questions = [" ".join(x.split())[:300] for x in (oq if isinstance(oq, list) else [])
                                          if isinstance(x, str) and x.strip()][:5]
            except Exception as exc:  # malformed or failed: a note, the sections still stand
                res.note(f"executive summary failed: {type(exc).__name__}: {str(exc)[:150]}")
                out = False
        if out is None:
            if "time" in budget.hit:
                res.note("executive summary skipped: --max-minutes")
            else:
                budget._stop("chars")
                res.note("executive summary skipped: chars cap (--max-chars left too little room)")
    elif not drafted:
        res.note("no section could be drafted, so there is no executive summary")

    res.caps_hit = list(budget.hit)
    res.used = dict(budget.used, seconds=round(clock() - budget.start, 1))
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _cap_line(res: Research) -> str:
    c = res.caps
    return (f"caps: {c.searches} searches, {c.pages} pages, {c.chars:,} characters to the model, "
            f"{c.seconds / 60:g} minutes")


def build_report(res: Research, *, story_href: str | None = None) -> Report:
    by_n = {s.n: s for s in res.sources}
    drafted = sum(1 for q in res.questions if q.text)
    r = Report(f"Go deep: {res.topic[:120]}", kicker="lmsa · go deep",
               subtitle=f"A cited research report in {len(res.questions)} sub-questions.",
               sticker="partial" if res.caps_hit else f"{len(res.sources)} sources",
               sticker_tone="warn" if res.caps_hit else "acc", source="tasks/go_deep.py")
    r.tiles([
        Tile("Sections", f"{drafted}/{len(res.questions)}", tone="acc"),
        Tile("Pages read", len(res.sources), tone="info", note=f"{res.used.get('searches', 0)} searches"),
        Tile("Model input", f"{res.used.get('chars', 0):,}", tone="violet", note="characters"),
        Tile("Citations dropped", res.citations_dropped, tone="hot" if res.citations_dropped else "muted",
             note="invented [n] replaced with [?]"),
    ])
    if res.caps_hit:
        names = ", ".join(f"{CAP_FLAGS[k]} ({_cap_value(res, k)})" for k in res.caps_hit)
        r.section("Stopped at a cap", tone="warn").callout(
            f"This report is partial: the run stopped at {names}. Everything below was finished before "
            f"that point; raise the cap and run again to go further.", label="partial", tone="warn")
    if res.story:
        sec = r.section("Origin", tone="info")
        sec.table(["Briefing story", "From"], [(
            Link(res.story.get("headline") or res.story["address"], story_href),
            res.story["address"])])
    sec = r.section("Executive summary", tone="acc")
    if res.summary:
        sec.text(res.summary)
    else:
        sec.callout("No executive summary this run -- see the notes.", label="missing", tone="warn")
    if res.open_questions:
        sec.table(["Open questions"], [(q,) for q in res.open_questions])
    cards = []
    for q in res.questions:
        if q.text:
            summary = q.text
        elif q.error:
            summary = f"Could not be drafted: {q.error}"
        else:
            summary = f"Not drafted ({q.skipped or 'skipped'})."
        meta = [f"search: {q.query}", f"{len(q.shown)} page{'s' if len(q.shown) != 1 else ''} shown"]
        if q.trimmed:
            meta.append(f"passages trimmed to {SECTION_PASSAGE_CHARS:,} characters")
        if q.dropped:
            meta.append(f"{q.dropped} invented citation{'s' if q.dropped != 1 else ''} removed")
        cards.append(Story(
            title=q.question, summary=summary, meta=" · ".join(meta),
            badge=None if q.text else "SKIPPED", tone="acc" if q.text else "muted",
            quotes=[StoryQuote(x["quote"], by_n[x["source"]].url if x["source"] in by_n else None, x["verified"])
                    for x in q.quotes],
            sources=[StorySource(f"[{n}] {by_n[n].title[:90]}", by_n[n].url) for n in q.cited if n in by_n],
        ))
    if cards:
        r.section("Findings", tone="acc").stories(cards)
    cited = {n for q in res.questions for n in q.cited}
    if res.sources:
        r.section("Sources", tone="info").table(
            ["#", "Page", "Cited", "Note"],
            [(s.n, Link(s.title, s.url), "yes" if s.n in cited else "no",
              "; ".join(x for x in ("story link" if s.seed else "",
                                    f"text cut at {PAGE_TEXT_CHARS:,} chars" if s.truncated else "") if x))
             for s in res.sources])
    notes = list(dict.fromkeys(res.notes)) + [
        _cap_line(res),
        f"each section saw the top {TOP_K} passages ({CHUNK_WORDS}-word chunks, at most "
        f"{SECTION_PASSAGE_CHARS:,} characters); pages are cut at {PAGE_TEXT_CHARS:,} characters",
        "every [n] names a page the model was shown for that section; invented numbers are shown as [?]; "
        "quotes are checked word for word and flagged when they can't be found",
    ]
    r.section("Notes", tone="warn").callout("\n\n".join(notes), label="caps & skips", tone="warn")
    return r


def _cap_value(res: Research, kind: str) -> str:
    c = res.caps
    return {"searches": f"{c.searches}", "pages": f"{c.pages}", "chars": f"{c.chars:,}",
            "time": f"{c.seconds / 60:g} min"}[kind]


def story_href(story: dict | None, out_dir: Path) -> str | None:
    """Relative link to the story's card in its briefing report -- works both
    file-opened and served under the dashboard's /reports/ (same folder)."""
    if not story or not story.get("report_path"):
        return None
    try:
        Path(story["report_path"]).resolve().relative_to(out_dir.resolve())
    except (ValueError, OSError):
        return None
    return f"{layout.link(Path(story['report_path']), out_dir)}#story-{story['story_id']}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48].strip("-") or "topic"


def record(conn, run_id: int, res: Research, path: Path, now: datetime) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO go_deep_reports(run_id, story, topic, report_path, created, stopped_by, summary_json) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, (res.story or {}).get("address", ""), res.topic[:500], str(path.resolve()), now.isoformat(),
         ",".join(res.caps_hit) or None, json.dumps(res.summary_dict(), default=str)))
    conn.commit()


def make_fetcher():
    from lmsa.web import Fetcher

    return Fetcher(USER_AGENT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Research a briefing story or topic further; cited report.")
    parser.add_argument("--story", help="briefing story to research, as BRIEFING:STORY_ID")
    parser.add_argument("--topic", help="free-text topic (instead of --story)")
    parser.add_argument("--max-searches", type=int, default=6, help="stop after this many web searches")
    parser.add_argument("--max-pages", type=int, default=12, help="stop after fetching this many pages")
    parser.add_argument("--max-chars", type=int, default=60000, help="characters sent to the model, in total")
    parser.add_argument("--max-minutes", type=int, default=20, help="wall-clock limit")
    parser.add_argument("--fresh", action="store_true", help="ignore remembered model answers")
    args = parser.parse_args(argv)
    if bool(args.story) == bool(args.topic):
        parser.error("give exactly one of --story or --topic")
    for name in ("max_searches", "max_pages", "max_chars", "max_minutes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    caps = Caps(args.max_searches, args.max_pages, args.max_chars, args.max_minutes * 60)

    with Store() as store:
        story = load_story(store.conn, args.story) if args.story else None  # fail fast, before the model
    topic = story["headline"] if story else " ".join(args.topic.split())[:300]

    from lmsa.config import settings
    from lmsa.search import SearXNG

    lm = LMStudio()
    lm.ensure_server()
    fetcher = make_fetcher()
    now = _now()
    with Store() as store, store.record_run(TASK) as run:
        res = research(topic, lm=lm, searx=SearXNG(settings().searxng_url), fetcher=fetcher, conn=store.conn, caps=caps,
                       story=story, fresh=args.fresh)
        out_dir = settings().output_dir
        report = build_report(res, story_href=story_href(story, out_dir))
        add_footer(report, res.memo, store.last_successful_run(TASK))
        path = report.write(f"go-deep-{slug(topic)}-{report.created:%Y%m%d-%H%M}.html")
        record(store.conn, run.id, res, path, now)
        run.report_path = path
        run.summary = res.summary_dict()
        print(path)
        if res.caps_hit:
            print("stopped at: " + ", ".join(CAP_FLAGS[k] for k in res.caps_hit))


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
