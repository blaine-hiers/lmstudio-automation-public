"""Conflict theaters extension (#55): group a briefing's stories by theater,
rank theaters by activity, flag escalation, map them.

Opt in with `[briefing] extensions = ["theaters"]` (config/briefings/conflicts.toml).

- **Theater assignment.** A story's countries are the model's `countries`
  (the engine's extract), or, when it tagged none, the structured countries
  its sources carry (`meta["countries"]`, e.g. ReliefWeb). A story joins the
  multi-country theater in `THEATERS` whose *core* countries it names (most
  overlap wins; the theater's other countries add weight but never start a
  theater on their own, so a lone China trade story is not "Taiwan Strait").
  Otherwise it joins a theater that already has stories this run and lists
  one of its countries (a Russia-only story joins Russia–Ukraine), else it
  forms a one-country theater (or a two-country pair, folded into a
  one-country theater of either member that has stories), so one conflict
  is not split across several theaters. A story with no
  country is matched on the theater aliases in its headline, summary and
  actors ("Houthi" -> Yemen & Red Sea), else lands in "Unplaced".
- **Ranking.** activity = stories x distinct sources x (1 + max severity);
  "Unplaced" always ranks last and is never mapped.
- **Escalation flag.** One memo'd `extract()` per theater (top
  `MAX_ASSESSED`): trend escalating | de-escalating | steady | unclear with a
  one-line reason. Every theater gets a flag -- over the cap or on a model
  failure it is "unclear" with the reason said.
- **Figures.** Casualty/displacement figures are shown only when the model
  quotes them and the quote is found word for word in one of the theater's
  sources *and* contains the figure; everything else is dropped (and the
  count noted). No figure is ever computed or estimated.
- **Report.** A "Theaters by activity" section (one world map shaded by
  activity, countries linking to their theater; a ranking table), one
  section per top theater (flag, what changed in 24 h, quoted figures), a
  "Sources this run" table so feeds that failed or yielded nothing are
  visible, and a theater line on each story card.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ... import geo
from ...feeds import is_http_url
from ...memo import Memo, has_keys, memo_version, model_id
from ...prompts import load as load_prompt
from ...prompts import prompt_hash
from ...report import Link, StoryExtra
from ...web import normalize_url
from ..quotes import tokens, verify_quote
from ..sources import SourceContext, register_extension

log = logging.getLogger(__name__)

NAME = "theaters"
MEMO_TASK = "briefing.theaters"
MEMO_VERSION = "theaters-1"
SYSTEM_PROMPT = "briefing_conflicts/theater_system"
USER_PROMPT = "briefing_conflicts/theater_user"

MAX_ASSESSED = 12         # theaters that get a model call per run (ranked)
MAX_SECTIONS = 12         # theaters with their own report section
MAX_PROMPT_STORIES = 8    # stories per theater in the prompt
STORY_CHARS = 600         # per story summary in the prompt
MAX_PROMPT_DOCS = 8       # source excerpts per theater in the prompt
EXCERPT_CHARS = 1_500     # per excerpt
PROMPT_DOC_CHARS = 10_000 # all excerpts of one theater
MAX_FIGURES = 6
REASON_CHARS = 200
UNPLACED = "unplaced"
TRENDS = ("escalating", "de-escalating", "steady", "unclear")
TREND_TONE = {"escalating": "hot", "de-escalating": "acc", "steady": "info", "unclear": "muted"}
CHANGED_WINDOW = timedelta(hours=24)

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "trend": {"type": "string", "enum": list(TRENDS)},
        "reason": {"type": "string"},
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["casualties", "displacement"]},
                    "figure": {"type": "string"},
                    "quote": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["kind", "figure", "quote", "source_url"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["trend", "reason", "figures"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TheaterDef:
    slug: str
    name: str
    core: tuple[str, ...]
    other: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


# Multi-country theaters. Anything else becomes a one-country theater, so
# coverage stays global without a fixed list.
THEATERS: tuple[TheaterDef, ...] = (
    TheaterDef("russia-ukraine", "Russia–Ukraine war", ("UA",), ("RU", "BY"), ("Donbas", "Zaporizhzhia", "Kharkiv")),
    TheaterDef("israel-gaza-lebanon", "Israel, Gaza & Lebanon", ("PS", "IL", "LB"), ("SY",),
               ("Gaza", "West Bank", "Hamas", "Hezbollah")),
    TheaterDef("yemen-red-sea", "Yemen & Red Sea", ("YE",), (), ("Houthi", "Houthis", "Red Sea")),
    TheaterDef("central-sahel", "Central Sahel", ("ML", "BF", "NE"), (), ("Sahel",)),
    TheaterDef("great-lakes", "DR Congo & Great Lakes", ("CD",), ("RW", "BI", "UG"), ("M23",)),
    TheaterDef("korean-peninsula", "Korean Peninsula", ("KP",), ("KR",)),
    TheaterDef("taiwan-strait", "Taiwan Strait", ("TW",), ("CN",)),
    TheaterDef("armenia-azerbaijan", "Armenia–Azerbaijan", ("AM", "AZ"), (), ("Nagorno-Karabakh",)),
    TheaterDef("sudan", "Sudan", ("SD",), (), ("Darfur", "Rapid Support Forces")),
)


@dataclass
class Theater:
    slug: str
    name: str
    countries: list[str]
    stories: list = field(default_factory=list)
    trend: str = "unclear"
    reason: str = ""
    figures: list[dict] = field(default_factory=list)

    @property
    def anchor(self) -> str:
        return f"theater-{self.slug}"

    @property
    def sources(self) -> set[str]:
        return {d.source for s in self.stories for d in s.docs if d.source}

    @property
    def max_severity(self) -> int:
        return max(((s.extract or {}).get("severity", 0) or 0 for s in self.stories), default=0)

    @property
    def activity(self) -> float:
        return float(len(self.stories) * max(len(self.sources), 1) * (1 + self.max_severity))


# ---------------------------------------------------------------------------
# Assignment and ranking
# ---------------------------------------------------------------------------

# Outside powers that show up in many unrelated conflicts. When a story
# names them next to other countries, the theater is keyed on the others,
# so "US strikes in Somalia" and "US sanctions on Venezuela" don't pool into
# one "United States" theater.
EXTERNAL_ACTORS = frozenset({"US", "GB", "FR", "RU", "CN"})


def story_countries(story: Any) -> list[str]:
    """The model's countries, else the sources' structured countries --
    sorted and deduplicated, so nothing downstream depends on model order."""
    raw = story.countries or [c for d in story.docs for c in (d.meta.get("countries") or [])]
    return sorted({c.upper() for c in raw if isinstance(c, str) and geo.country(c)})


def _alias_re(alias: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)


def _mentions(story: Any, code: str) -> int:
    """How many of the story's docs are about `code`: structured countries or
    the country's name in the title/text."""
    name_re = _alias_re(geo.country_name(code) or code)
    return sum(1 for d in story.docs
               if code in (d.meta.get("countries") or []) or name_re.search(f"{d.title}\n{d.text[:4000]}"))


def _fallback(story: Any, countries: list[str]) -> tuple[str, str, list[str]]:
    """Order-independent theater for countries matching no THEATERS entry:
    external actors set aside when others are present; one country -> that
    country; two -> the pair; more -> the one most of the docs are about
    (ties by code)."""
    local = [c for c in countries if c not in EXTERNAL_ACTORS] or countries
    if len(local) == 2:
        a, b = sorted(local, key=lambda c: geo.country_name(c) or c)
        return (f"pair-{'-'.join(sorted(c.lower() for c in local))}",
                f"{geo.country_name(a) or a}–{geo.country_name(b) or b}", sorted(local))
    c = local[0] if len(local) == 1 else min(local, key=lambda c: (-_mentions(story, c), c))
    return f"country-{c.lower()}", geo.country_name(c) or c, [c]


def _members(t: TheaterDef, cset: set[str]) -> list[str]:
    return [c for c in (*t.core, *t.other) if c in cset or c in t.core]


def _primary(story: Any, countries: list[str]) -> tuple[str, str, list[str]] | None:
    """The THEATERS entry whose core countries the story names (most overlap
    wins), or -- for a story with no country -- whose alias its text names;
    None when the story needs a fallback theater."""
    cset = set(countries)
    best, best_score = None, 0
    for t in THEATERS:
        core = len(cset & set(t.core))
        if not core:
            continue
        score = 2 * core + len(cset & set(t.other))
        if score > best_score:
            best, best_score = t, score
    if best is not None:
        return best.slug, best.name, _members(best, cset)
    if countries:
        return None
    ex = story.extract or {}
    text = " ".join([story.headline, str(ex.get("summary", "")), *map(str, ex.get("actors", []))])
    for t in THEATERS:
        if any(_alias_re(a).search(text) for a in t.aliases):
            return t.slug, t.name, list(t.core)
    return UNPLACED, "Unplaced (no country)", []


def assign(story: Any) -> tuple[str, str, list[str]]:
    """(slug, name, countries) of the story's theater on its own (no other
    stories considered); independent of the order the model listed
    countries in. `group` additionally folds fallback theaters into the
    run's active ones."""
    countries = story_countries(story)
    return _primary(story, countries) or _fallback(story, countries)


def _fold_into_active(countries: list[str], active: set[str]) -> tuple[str, str, list[str]] | None:
    """An active THEATERS entry (it has stories this run) that one of the
    story's countries belongs to, as a core or other member -- external
    actors only count when the story names nothing else. Most core, then
    most member overlap wins; ties by THEATERS order."""
    local = set(c for c in countries if c not in EXTERNAL_ACTORS) or set(countries)
    best, best_key = None, (0, 0)
    for t in THEATERS:
        if t.slug not in active:
            continue
        key = (len(local & set(t.core)), len(local & (set(t.core) | set(t.other))))
        if key[1] and key > best_key:
            best, best_key = t, key
    return (best.slug, best.name, _members(best, local)) if best is not None else None


def assign_all(stories: list) -> list[tuple[str, str, list[str]]]:
    """Each story's theater, with one conflict kept in one theater: a story
    matching no THEATERS core joins an active THEATERS entry it has a member
    of ("Russia strikes" -> Russia–Ukraine when that has stories); else its
    fallback theater, where a pair folds into a one-country theater of either
    member that has stories this run (more stories wins, then code)."""
    out: list = [None] * len(stories)
    pending = []
    for n, s in enumerate(stories):
        countries = story_countries(s)
        out[n] = _primary(s, countries)
        if out[n] is None:
            pending.append((n, s, countries))
    active = {a[0] for a in out if a is not None}
    for n, s, countries in pending:
        out[n] = _fold_into_active(countries, active) or _fallback(s, countries)
    singles: dict[str, int] = {}
    for slug, _, _ in out:
        if slug.startswith("country-"):
            singles[slug] = singles.get(slug, 0) + 1
    for n, (slug, _, countries) in enumerate(out):
        if not slug.startswith("pair-"):
            continue
        hosts = [c for c in countries if f"country-{c.lower()}" in singles]
        if hosts:
            c = min(hosts, key=lambda c: (-singles[f"country-{c.lower()}"], c))
            out[n] = (f"country-{c.lower()}", geo.country_name(c) or c, [c])
    return out


def group(stories: list) -> list[Theater]:
    """Theaters ranked by activity (Unplaced last); ties by name."""
    by_slug: dict[str, Theater] = {}
    for s, (slug, name, countries) in zip(stories, assign_all(stories)):
        t = by_slug.setdefault(slug, Theater(slug, name, []))
        for c in countries:
            if c not in t.countries:
                t.countries.append(c)
        t.stories.append(s)
    return sorted(by_slug.values(), key=lambda t: (t.slug == UNPLACED, -t.activity, t.name))


# ---------------------------------------------------------------------------
# Escalation flag + quoted figures (model)
# ---------------------------------------------------------------------------

def _theater_docs(t: Theater) -> list:
    docs, seen = [], set()
    for s in t.stories:
        for d in s.docs:
            if d.meta.get("norm_url") not in seen:
                seen.add(d.meta.get("norm_url"))
                docs.append(d)
    return docs


def build_prompt(t: Theater) -> tuple[str, bool]:
    """The user prompt and whether anything was cut to fit the caps."""
    trimmed = len(t.stories) > MAX_PROMPT_STORIES
    lines = []
    for s in t.stories[:MAX_PROMPT_STORIES]:
        ex = s.extract or {}
        summary = str(ex.get("summary", ""))
        if len(summary) > STORY_CHARS:
            summary, trimmed = summary[:STORY_CHARS], True
        # No NEW/UPDATED status here: it changes between runs with the same
        # content, which would defeat the memo. Dates carry the recency.
        when = s.newest.date().isoformat() if s.newest else "undated"
        lines.append(f"- [{when}] {s.headline} (severity {ex.get('severity', 0)}): {summary}")
    docs = sorted(_theater_docs(t), key=lambda d: d.published.timestamp() if d.published else 0, reverse=True)
    trimmed = trimmed or len(docs) > MAX_PROMPT_DOCS
    parts, total = [], 0
    for n, d in enumerate(docs[:MAX_PROMPT_DOCS], 1):
        when = d.published.date().isoformat() if d.published else "undated"
        text = d.text
        if len(text) > EXCERPT_CHARS:
            text, trimmed = text[:EXCERPT_CHARS], True
        block = f"[{n}] {d.source} | {when} | {d.url}\nTITLE: {d.title}\n{text}"
        if total + len(block) > PROMPT_DOC_CHARS:
            block, trimmed = block[:max(PROMPT_DOC_CHARS - total, 0)], True
        if not block:
            break
        parts.append(block)
        total += len(block)
    names = ", ".join(geo.country_name(c) or c for c in t.countries) or "unknown"
    prompt = load_prompt(USER_PROMPT, theater=t.name, countries=names, count=len(lines),
                         stories="\n".join(lines), docs="\n\n".join(parts) or "(none)")
    return prompt, trimmed


_ELLIPSIS = re.compile(r"\.\s*\.\s*\.|…")
# Sentences split on . ! ? ; -- not ":" ("Killed: 40." is one statement).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WORD = re.compile(r"\d[\d,.]*\d|\d|[^\W\d_][\w'-]*")
# Words that make a number a casualty / displacement figure. The number and
# one of these must sit close together in one sentence of the quote.
KIND_WORDS = {
    "casualties": re.compile(r"kill\w*|dead|deaths?|died|dies|wound\w*|injur\w*|casualt\w*|fatalit\w*|"
                             r"massacr\w*|slain|lives|bodies", re.IGNORECASE),
    "displacement": re.compile(r"displac\w*|refugees?|fled|flee\w*|evacuat\w*|uproot\w*|homeless|asylum",
                               re.IGNORECASE),
}
_UNITS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * n for n, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
# Plural / vague magnitudes: a number, but with no exact value.
_VAGUE = frozenset("hundreds thousands millions billions dozen dozens tens scores".split())
_PLURAL_MAGNITUDES = frozenset({"hundreds", "thousands", "millions", "billions"})
NUMBER_WORDS = frozenset((set(_UNITS) - {"zero"}) | set(_TENS) | set(_SCALES) | {"hundred"} | _VAGUE)
FIGURE_WINDOW = 8  # max words between the number and its casualty/displacement word
# "The number of people killed in ... has risen to 40": the figure trails the
# casualty word by a whole clause, so the window is wider for this phrasing.
TOLL_WINDOW = 14
_TOLL = re.compile(r"\b(?:toll|number)\b.*\b(?:rose|risen|rises|reached|reaches|climbed|climbs)\s+to\b",
                   re.IGNORECASE)
# An intervening number inside one of these clauses is a subset of the figure
# ("40 civilians, including 12 children, were killed"), not a rival figure.
_SUBSET_MARKERS = (("including",), ("among", "them"), ("of", "whom"), ("among", "whom"))


def _norm_word(word: str) -> str:
    return word.replace(",", "").rstrip(".") if word[0].isdigit() else word.lower()


def _is_spelled(norm: str) -> bool:
    return bool(norm) and not norm[0].isdigit() and all(p in NUMBER_WORDS for p in norm.split("-"))


def _is_number(norm: str) -> bool:
    return norm[0].isdigit() or _is_spelled(norm)


def _spans(norm: list[str]) -> list[tuple[int, int]]:
    """(first, last) token index of each number in a sentence -- a number
    runs over its magnitude words: "2 million", "tens of thousands",
    "hundreds of thousands", "two hundred and fifty"."""
    spans, i = [], 0
    while i < len(norm):
        if not _is_number(norm[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(norm):
            cur, nxt = norm[j], norm[j + 1]
            after = norm[j + 2] if j + 2 < len(norm) else ""
            if _is_spelled(nxt) and (not cur[0].isdigit() or nxt in _SCALES or nxt in _PLURAL_MAGNITUDES
                                     or nxt == "hundred"):
                j += 1  # "two hundred", "2 million", "twenty-five thousand"
            elif nxt == "of" and cur in _VAGUE and after in _PLURAL_MAGNITUDES:
                j += 2  # "tens of thousands", "hundreds of thousands"
            elif nxt == "and" and cur in {"hundred", "thousand"} and _is_spelled(after) and after not in _VAGUE:
                j += 2  # "one hundred and fifty"
            else:
                break
        spans.append((i, j))
        i = j + 1
    return spans


def _value(parts: list[str]) -> float | None:
    """The exact value of a number span, or None when it is vague
    ("thousands", "tens of thousands") or not parseable."""
    total = current = 0.0
    seen = False
    for p in parts:
        if p[0].isdigit():
            if seen:
                return None
            try:
                current, seen = float(p), True
            except ValueError:
                return None
            continue
        for w in p.split("-"):
            if w in ("and", "of"):
                continue
            if w in _VAGUE:
                return None
            if w in _UNITS or w in _TENS:
                current += _UNITS.get(w, 0) + _TENS.get(w, 0)
            elif w == "hundred":
                current = (current or 1) * 100
            elif w in _SCALES:
                total += (current or 1) * _SCALES[w]
                current = 0.0
            else:
                return None
            seen = True
    return total + current if seen else None


def _figure_number(figure: str) -> tuple[frozenset[str], float | None] | None:
    """The model figure's first number: its words and exact value (None if vague)."""
    norm = [_norm_word(w) for w in _WORD.findall(figure)]
    spans = _spans(norm)
    if not spans:
        return None
    lo, hi = spans[0]
    parts = norm[lo:hi + 1]
    return frozenset(p for p in parts if p not in ("of", "and")), _value(parts)


def _in_subset_clause(norm: list[str], start: int, at: int) -> bool:
    """A subset marker ("including", "among them", "of whom") sits between
    token `start` and the intervening number at token `at`."""
    return any(tuple(norm[k:k + len(m)]) == m and k + len(m) <= at
               for k in range(start, at) for m in _SUBSET_MARKERS)


def figure_support(kind: str, figure: str, quote: str) -> str | None:
    """The number as the *quote* writes it, magnitude words included ("2
    million", "tens of thousands"), if the quote supports the model's figure
    as a `kind` figure, else None. Supported means: the quote has no ellipsis
    (so it is one contiguous span of the source); in one of its sentences a
    number equal to the figure (1,200 = 1200, "five" = 5, "2 million" =
    2,000,000) -- or, for a vague figure such as "thousands", one containing
    its words -- sits within FIGURE_WINDOW words (TOLL_WINDOW for "the toll /
    number of ... killed has risen to N") of a casualty/displacement word
    ("killed", "displaced", ...) with no other number in between, except a
    subset in an "including / among them / of whom" clause. The model's own
    wording is never returned."""
    pattern = KIND_WORDS.get(kind)
    if pattern is None or not figure or not quote or _ELLIPSIS.search(quote):
        return None
    fig = _figure_number(figure)
    if fig is None:
        return None
    fig_parts, fig_value = fig
    for sentence in _SENTENCE_SPLIT.split(quote):
        matches = list(_WORD.finditer(sentence))
        words = [m.group() for m in matches]
        norm = [_norm_word(w) for w in words]
        spans = _spans(norm)
        keys = [j for j, w in enumerate(words) if pattern.fullmatch(w)]
        window = TOLL_WINDOW if _TOLL.search(sentence) else FIGURE_WINDOW
        for lo, hi in spans:
            parts = norm[lo:hi + 1]
            value = _value(parts)
            if fig_value is not None and value is not None:
                if value != fig_value:
                    continue
            elif not fig_parts <= set(parts):
                continue
            for j in keys:
                a, b = (hi, j) if j > hi else (j, lo)
                if b - a > window:
                    continue
                rivals = [s for s in spans if a < s[0] < b]
                if any(not _in_subset_clause(norm, a + 1, s[0]) for s in rivals):
                    continue  # another number is closer to the word: it is not this figure's
                return sentence[matches[lo].start():matches[hi].end()].rstrip(".,")
    return None


def verify_figures(t: Theater, raw: object) -> tuple[list[dict], int]:
    """(kept figures, dropped count). Kept = the quote supports the
    figure (`figure_support`) and is found word for word in a theater source
    (the cited one first). `figure` is the number as the quote writes it."""
    docs = _theater_docs(t)
    by_url = {d.meta.get("norm_url"): d for d in docs}
    kept, dropped = [], 0
    for f in (raw if isinstance(raw, list) else [])[:MAX_FIGURES * 2]:
        if not isinstance(f, dict):
            dropped += 1
            continue
        kind, figure, quote = f.get("kind"), str(f.get("figure") or "").strip(), str(f.get("quote") or "").strip()
        number = figure_support(str(kind), figure, quote)
        if number is None:
            dropped += 1
            continue
        url = str(f.get("source_url") or "")
        cited = None
        if is_http_url(url):
            try:
                cited = by_url.get(normalize_url(url))
            except ValueError:
                cited = None
        found = cited if cited is not None and verify_quote(quote, f"{cited.title}\n{cited.text}") else None
        if found is None:
            found = next((d for d in docs if d is not cited and verify_quote(quote, f"{d.title}\n{d.text}")), None)
        if found is None:
            dropped += 1
            continue
        # Only what the source says: the number as the quote writes it, and the quote.
        kept.append({"kind": kind, "figure": number[:40], "quote": quote[:500], "source_url": found.url})
    return kept[:MAX_FIGURES], dropped + max(len(kept) - MAX_FIGURES, 0)


def assess(theaters: list[Theater], ctx: SourceContext) -> None:
    """Fill trend/reason/figures for the top MAX_ASSESSED theaters (memo'd
    by prompt); the rest are "unclear" with the reason why."""
    cfg = ctx.briefing
    lm = ctx.lm
    version = memo_version(MEMO_VERSION, prompt_hash(SYSTEM_PROMPT), prompt_hash(USER_PROMPT), SCHEMA,
                           model_id(lm))
    memo = Memo(ctx.conn, MEMO_TASK, version, fresh=ctx.fresh)
    system = load_prompt(SYSTEM_PROMPT)
    trimmed = failed = dropped = 0
    candidates = [t for t in theaters if t.slug != UNPLACED]
    for n, t in enumerate(candidates):
        if n >= MAX_ASSESSED:
            t.trend, t.reason = "unclear", f"not assessed: only the top {MAX_ASSESSED} theaters are assessed per run"
            continue
        prompt, cut = build_prompt(t)
        trimmed += cut
        key = f"{cfg.name}:{t.slug}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:24]}"
        try:
            raw = memo.cached(key, lambda: lm.extract(prompt, SCHEMA, system=system, name="theater"),
                              valid=lambda r: has_keys(r, SCHEMA["required"]))
            if not isinstance(raw, dict):
                raise ValueError("model returned no JSON object")
            trend = raw.get("trend")
            t.trend = trend if trend in TRENDS else "unclear"
            reason = " ".join(str(raw.get("reason") or "").split())[:REASON_CHARS]
            t.reason = reason or "no reason given"
            t.figures, d = verify_figures(t, raw.get("figures"))
            dropped += d
        except Exception as exc:  # one theater's failure never kills the briefing
            failed += 1
            t.trend, t.reason = "unclear", f"could not be assessed ({type(exc).__name__})"
            log.warning("theater assessment failed for %s", t.slug, exc_info=True)
    for t in theaters:
        if t.slug == UNPLACED:
            t.trend, t.reason = "unclear", "stories with no country are not assessed"
    if len(candidates) > MAX_ASSESSED:
        ctx.note(f"theaters: {len(candidates) - MAX_ASSESSED} lower-ranked theater(s) not assessed "
                 f"(at most {MAX_ASSESSED} per run)")
    if trimmed:
        ctx.note(f"theaters: for {trimmed} theater(s) the model saw at most {MAX_PROMPT_STORIES} stories and "
                 f"{MAX_PROMPT_DOCS} source excerpts of {EXCERPT_CHARS:,} characters ({PROMPT_DOC_CHARS:,} total)")
    if failed:
        ctx.note(f"theaters: {failed} theater(s) could not be assessed; flagged unclear")
    if dropped:
        ctx.note(f"theaters: {dropped} model-reported figure(s) dropped -- not found word for word in a source")
    setattr(ctx, "_theater_memo", memo)


# ---------------------------------------------------------------------------
# Extension hooks
# ---------------------------------------------------------------------------

def enrich(stories: list, ctx: SourceContext) -> None:
    theaters = group(stories)
    assess(theaters, ctx)
    for rank, t in enumerate(theaters, 1):
        for s in t.stories:
            s.extras[NAME] = {"slug": t.slug, "name": t.name, "rank": rank, "trend": t.trend,
                              "reason": t.reason, "href": target(t, rank)}
    setattr(ctx, "_theaters", theaters)


def render(story: Any, ctx: SourceContext | None) -> list:
    info = story.extras.get(NAME)
    if not isinstance(info, dict):
        return []
    trend = info.get("trend") if info.get("trend") in TRENDS else "unclear"
    href = info.get("href") if isinstance(info.get("href"), str) else "#theaters"
    return [StoryExtra("theater", f"{info.get('name', '')} · {trend}", href=href, tone=TREND_TONE[trend])]


def target(t: Theater, rank: int) -> str:
    """Where links to a theater go: its own section, or the ranking table
    for theaters past MAX_SECTIONS (they have no section)."""
    return f"#{t.anchor}" if rank <= MAX_SECTIONS else "#theaters"


def changed_recently(story: Any, now) -> bool:
    """NEW or UPDATED this run, with a source from the last 24 h (undated counts)."""
    if story.status not in ("new", "updated"):
        return False
    newest = story.newest
    return newest is None or now - newest <= CHANGED_WINDOW


def map_values(theaters: list[Theater]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    links: dict[str, str] = {}
    for rank, t in enumerate(theaters, 1):
        if t.slug == UNPLACED:
            continue
        for c in t.countries:
            if t.activity > values.get(c, -1):
                values[c] = t.activity
                links[c] = target(t, rank)
    return values, links


def section(report: Any, stories: list, ctx: SourceContext | None) -> None:
    theaters = getattr(ctx, "_theaters", None) if ctx is not None else None
    if theaters is None:
        theaters = group(stories)  # enrich failed: rank without flags
    now = ctx.now
    sec = report.section("Theaters by activity", tone="hot")
    sec.anchor("theaters")
    if not theaters:
        sec.callout("No conflict stories this run.", label="empty", tone="info")
    else:
        values, links = map_values(theaters)
        if values:
            sec.world_map(values, links=links, label="conflict theaters by activity",
                          legend="activity = stories × sources × (1 + max severity)")
        rows = []
        for rank, t in enumerate(theaters, 1):
            rows.append([rank, Link(t.name, target(t, rank)), t.trend, t.reason, len(t.stories), len(t.sources),
                         t.max_severity, round(t.activity, 1)])
        sec.table(["#", "Theater", "Trend", "Why", "Stories", "Sources", "Max severity", "Activity"], rows)
    for rank, t in enumerate(theaters[:MAX_SECTIONS], 1):
        tone = TREND_TONE.get(t.trend, "muted")
        ts = report.section(f"{rank}. {t.name}", tone=tone)
        ts.anchor(t.anchor)
        ts.callout(t.reason or "no reason given", label=t.trend, tone=tone)
        changed = [s for s in t.stories if changed_recently(s, now)]
        new = sum(1 for s in changed if s.status == "new")
        ts.text(f"What changed in 24 h: {new} new, {len(changed) - new} updated "
                f"stor{'y' if len(changed) == 1 else 'ies'} of {len(t.stories)}.")
        changed_ids = {id(s) for s in changed}
        rows = []
        for s in t.stories:
            change = s.status.upper() if id(s) in changed_ids else "—"
            rows.append([Link(s.headline, f"#{s.anchor}"), change, len(s.sources),
                         (s.extract or {}).get("severity", "?")])
        ts.table(["Story", "24 h", "Sources", "Severity"], rows)
        if t.figures:
            ts.table(["Number (as quoted)", "Kind", "Quoted from a source (checked word for word)"],
                     [[f["figure"], f["kind"], Link(f["quote"], f["source_url"])] for f in t.figures])
    if len(theaters) > MAX_SECTIONS:
        ctx.note(f"theaters: {len(theaters) - MAX_SECTIONS} lower-ranked theater(s) are in the table only")
    source_check(report, stories, ctx)


def source_check(report: Any, stories: list, ctx: SourceContext) -> None:
    """Every configured source with what it contributed this run, so a feed
    that failed or returned nothing is visible (feeds here are unverified
    until their first run)."""
    cfg = ctx.briefing
    kept: dict[str, int] = {}
    try:
        rows = ctx.conn.execute("SELECT meta_json FROM briefing_articles WHERE briefing=? AND last_seen=?",
                                (cfg.name, ctx.now.isoformat())).fetchall()
    except Exception:  # a missing table in a test DB etc.: counts unknown, table still shown
        rows = []
    for (meta_json,) in rows:
        try:
            entry = json.loads(meta_json).get("entry", "")
        except (TypeError, ValueError, AttributeError):
            continue
        kept[entry] = kept.get(entry, 0) + 1
    in_stories: dict[str, int] = {}
    for s in stories:
        for d in s.docs:
            e = d.meta.get("entry", "")
            in_stories[e] = in_stories.get(e, 0) + 1
    out = []
    stats = getattr(ctx, "entry_stats", {}) or {}
    for i, (kind, entry) in enumerate(cfg.entries()):
        name = entry["name"]
        st = stats.get(i)
        n_kept = kept.get(name, 0)
        status = source_status(st, n_kept, f"{kind}: {name}", ctx.notes,
                               keyword_filtered=entry.get("prefilter") is not False)
        out.append([name, entry.get("group", ""), entry.get("lean", "unknown"), entry.get("reliability", "unknown"),
                    st["taken"] if st else 0, n_kept, in_stories.get(name, 0), status[:200]])
    if out:
        report.section("Sources this run", tone="muted").table(
            ["Source", "Group", "Lean", "Reliability", "Collected", "Kept", "In stories", "Status"], out)


def source_status(stat: dict | None, kept: int, label: str, notes: list[str], *,
                  keyword_filtered: bool = True) -> str:
    """One source's outcome, from the engine's per-entry collect stats
    (`ctx.entry_stats`) and its notes -- never a guess:
    not collected (max_items reached) | not available | fetch failed: ... |
    partial: N collected, then failed: ... | returned 0 items | all N items filtered by keywords | ok."""
    own = [n[len(label) + 1:].strip() for n in notes if n.startswith(label + ":")]
    if stat is None:
        return "not collected (max_items reached before this source)"
    if stat.get("error") == "not available":
        return "not available (no adapter installed)"
    taken = int(stat.get("taken") or 0)
    if stat.get("error"):
        if taken:  # yielded docs, then raised: what it gave is kept
            return f"partial: {taken} collected, then failed: {stat['error']}"
        return f"fetch failed: {stat['error']}"
    if taken == 0:
        return "returned 0 items" + (f" ({own[0]})" if own else "")
    if kept == 0 and not keyword_filtered:
        return f"none of {taken} items kept (keyword pre-filter off for this source)"
    if kept == 0:
        return f"all {taken} items filtered by keywords"
    cut = next((n for n in own if n.startswith("cut at")), None)
    return f"ok: {kept} of {taken} kept" + (f"; {cut}" if cut else "")


register_extension(NAME, enrich=enrich, render=render, section=section)
