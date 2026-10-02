"""Source lean + reliability extension (#69): the Ground News pattern --
a stacked coverage-by-lean bar and reliability badges on every story card,
plus a "Blindspot" report section for stories one side dominates.

Tags come from each source entry's own `lean` / `reliability` (set by hand
in a briefing file or in the shared `config/sources.toml` -- see
`lmsa.briefing.config.load_source_tags`); nothing here scrapes a rating
site. Registered as extension **"lean"**, `default=True` -- it runs on
every briefing with no opt-in.

- **`render`**: a `StackedBar` of the story's docs by `lean` (skipped
  entirely when every doc is "unknown" -- nothing to show), and a
  low-reliability warning (`StoryExtra`) when no doc covering the story is
  "high" or "mixed" reliability.
- **`render_sources`**: badges each source link with its `reliability`.
- **`section`**: a "Blindspot" table of stories where lean-rated coverage
  (docs with a known lean; state-media/intl/center count toward the total
  only, never toward a side; unknown is left out entirely) skews `>= BLINDSPOT_SHARE` to one side, with
  at least `MIN_RATED_SOURCES` distinct rated sources -- see
  `coverage_balance` / `is_blindspot`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ...report import Link, StackedBar, StoryExtra, StorySource
from ..sources import register_extension

NAME = "lean"

# Ground News calls this a "blindspot": most coverage from one side, with
# enough distinct sources that it isn't just one outlet's framing.
BLINDSPOT_SHARE = 0.8
MIN_RATED_SOURCES = 3

_LEFT = frozenset({"left", "center-left"})
_RIGHT = frozenset({"right", "center-right"})

# Only six report tones exist; leans map onto them for a rough left-to-right
# read (blue-ish for left, red-ish for right), not a precise scale.
LEAN_TONE = {
    "left": "info",
    "center-left": "info",
    "center": "muted",
    "center-right": "warn",
    "right": "hot",
    "state-media": "violet",
    "intl": "acc",
    "unknown": "muted",
}
RELIABILITY_TONE = {"high": "acc", "mixed": "info", "low": "warn", "unknown": "muted"}


def _lean(doc: Any) -> str:
    return str(doc.meta.get("lean") or "unknown")


def _reliability(doc: Any) -> str:
    return str(doc.meta.get("reliability") or "unknown")


def _source_name(doc: Any) -> str:
    return doc.source or str(doc.meta.get("entry") or doc.url)


def rated_sources(docs: list) -> dict[str, str]:
    """Distinct source name -> lean, for docs with a known (non-"unknown")
    lean. A source seen with more than one lean tag keeps the first one."""
    out: dict[str, str] = {}
    for d in docs:
        lean = _lean(d)
        if lean == "unknown":
            continue
        name = _source_name(d)
        out.setdefault(name, lean)
    return out


def coverage_balance(docs: list) -> dict:
    """Left-vs-right balance of a story's lean-rated coverage, by distinct
    source (not article count, so one prolific outlet can't dominate the
    share on its own). `total` counts every rated source (state-media,
    intl and center included); `leaning` is whichever side (`"left"` /
    `"right"`) has more, `None` if neither side has any. `share` is that
    side's fraction of `total` (0.0 with nothing rated)."""
    rated = rated_sources(docs)
    total = len(rated)
    left = sum(1 for lean in rated.values() if lean in _LEFT)
    right = sum(1 for lean in rated.values() if lean in _RIGHT)
    if not total or (not left and not right):
        return {"total": total, "left": left, "right": right, "leaning": None, "share": 0.0}
    leaning, count = ("left", left) if left >= right else ("right", right)
    return {"total": total, "left": left, "right": right, "leaning": leaning, "share": count / total}


def is_blindspot(docs: list) -> bool:
    """True when >= BLINDSPOT_SHARE of a story's lean-rated coverage is one
    side, with at least MIN_RATED_SOURCES distinct rated sources."""
    bal = coverage_balance(docs)
    return bal["total"] >= MIN_RATED_SOURCES and bal["leaning"] is not None and bal["share"] >= BLINDSPOT_SHARE


def render(story: Any, ctx: Any) -> list:
    docs = story.docs
    parts: list = []
    leans = [_lean(d) for d in docs]
    if leans and any(l != "unknown" for l in leans):
        counts = Counter(leans)
        parts.append(StackedBar(
            [(lean, n, LEAN_TONE.get(lean, "muted")) for lean, n in sorted(counts.items())],
            label="coverage by lean",
        ))
    if docs and not any(_reliability(d) in ("high", "mixed") for d in docs):
        parts.append(StoryExtra(
            "reliability", "No high- or mixed-reliability source covers this story yet.", tone="warn",
        ))
    return parts


def render_sources(story: Any, links: list, ctx: Any) -> list:
    by_url = {d.url: d for d in story.docs}
    out = []
    for link in links:
        d = by_url.get(link.href)
        if d is None:  # e.g. the trailing "+N more" entry (href=None)
            out.append(link)
            continue
        reliability = _reliability(d)
        out.append(StorySource(link.text, link.href, badge=reliability,
                               tone=RELIABILITY_TONE.get(reliability, "muted")))
    return out


def section(report: Any, stories: list, ctx: Any) -> None:
    rows = []
    for s in stories:
        bal = coverage_balance(s.docs)
        if bal["total"] < MIN_RATED_SOURCES or bal["leaning"] is None or bal["share"] < BLINDSPOT_SHARE:
            continue
        rows.append([Link(s.headline, f"#{s.anchor}"), bal["leaning"], f"{bal['share'] * 100:.0f}%", bal["total"]])
    sec = report.section("Blindspot", tone="warn")
    sec.text(
        f"Stories where at least {int(BLINDSPOT_SHARE * 100)}% of lean-rated coverage "
        f"(minimum {MIN_RATED_SOURCES} distinct rated sources) comes from one side. "
        "State-media, intl and center sources count toward the total but not toward either side."
    )
    if rows:
        sec.table(["Story", "Leaning", "Share", "Rated sources"], rows)
    else:
        sec.callout("No blindspots this run.", label="empty", tone="info")


register_extension(NAME, render=render, render_sources=render_sources, section=section, default=True)
