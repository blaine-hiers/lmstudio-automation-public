"""A saved `lmsa.report` page, read back as plain data for an email (#119).

The master report's `full` section mode puts a task's *whole* latest report
into the email body. A report page can't go in as-is (its `<style>`, CSS
variables, web fonts, JS and SVG don't survive Gmail / Apple Mail), so this
module parses the saved HTML back into a small, closed block vocabulary --
one block type per `lmsa.report` component -- that `lmsa.email_render` turns
into inline-styled, table-based email markup and plain text.

The page is treated as untrusted even though `lmsa.report` wrote it: only
*text* and http(s) hrefs come out of it. No attribute, style or tag is ever
passed through; every string is re-escaped by the renderer. What email can't
show (SVG maps, heatmaps, sparklines, scripts, check-off boxes, thumbnails)
is dropped, and a dropped chart leaves a one-line `Note` in its place.
Markup this parser doesn't know (e.g. a `Section.raw()` block) falls back to
its text content.

    full = load(Path("output/disaster-fusion-report-20260926-1840.html"))
    full.sections[0].title, full.sections[0].blocks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from .digest import http_url
from .report import TONES

MAX_BYTES = 20_000_000  # a saved report bigger than this isn't a report; refuse it

_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
                   "source", "track", "wbr"})
# Never text: scripts/styles, SVG (charts; its <title>s are tooltips), form controls, images.
_SKIP_TEXT = frozenset({"script", "style", "template", "svg", "img", "input", "select", "textarea", "head"})
# Inline tags that sit inside a word or sentence; any other tag boundary
# separates words.
_PHRASING = frozenset({"a", "b", "i", "em", "strong", "code", "small", "sup", "sub", "mark", "abbr", "u", "s",
                       "q", "kbd", "var", "time", "#root"})
_PCT_RE = re.compile(r"width\s*:\s*([0-9]+(?:\.[0-9]+)?)%")
_LEGEND_RE = re.compile(r"^(.*?)\s*([0-9]{1,3})%$")


# -- blocks --------------------------------------------------------------------

@dataclass
class Span:
    """A run of text, optionally a link (http(s) only; anything else is dropped
    and the text kept)."""
    text: str
    href: str | None = None


@dataclass
class TileB:
    label: str
    value: str
    delta: str = ""
    note: str = ""
    tone: str = "acc"


@dataclass
class Tiles:
    tiles: list[TileB]


@dataclass
class Para:
    text: str


@dataclass
class Note:
    """Something the email left out, e.g. "map omitted"."""
    text: str


@dataclass
class Subhead:
    text: str


@dataclass
class Cell:
    span: Span
    num: bool = False


@dataclass
class Table:
    headers: list[str]
    rows: list[list[Cell]]


@dataclass
class BarRow:
    name: str
    amount: str
    pct: float  # 0-100, the bar's fill


@dataclass
class Bars:
    rows: list[BarRow]


@dataclass
class Stacked:
    label: str
    segments: list[tuple[str, int, str]]  # (legend text e.g. "left 25%", pct, tone)


@dataclass
class Callout:
    label: str
    paras: list[str]
    tone: str = "warn"


@dataclass
class CardB:
    title: Span
    meta: str = ""
    ribbon: str = ""
    paras: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tone: str = "acc"


@dataclass
class Quote:
    text: str
    href: str | None = None
    verified: bool = True


@dataclass
class Extra:
    label: str
    span: Span
    tone: str = "muted"


@dataclass
class Source:
    span: Span
    badge: str = ""
    tone: str = "muted"


@dataclass
class StoryB:
    title: str
    badge: str = ""
    meta: str = ""
    stacked: list[Stacked] = field(default_factory=list)
    paras: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    timeline: list[tuple[str, str]] = field(default_factory=list)
    why: str = ""
    extras: list[Extra] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tone: str = "acc"


@dataclass
class SectionB:
    title: str
    tone: str = "acc"
    blocks: list = field(default_factory=list)


@dataclass
class FullReport:
    """A saved report as email data: its subtitle, the blocks above the first
    section (headline tiles), and every section in order."""
    subtitle: str = ""
    top: list = field(default_factory=list)
    sections: list[SectionB] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.top or any(s.blocks for s in self.sections))


# -- a tiny DOM ----------------------------------------------------------------

class _Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs: dict[str, str]):
        self.tag = tag
        self.attrs = attrs
        self.children: list[_Node | str] = []

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def elements(self) -> list[_Node]:
        return [c for c in self.children if isinstance(c, _Node)]

    def find(self, pred) -> _Node | None:
        for c in self.elements():
            if pred(c):
                return c
            hit = c.find(pred)
            if hit is not None:
                return hit
        return None

    def find_all(self, pred) -> list[_Node]:
        out = []
        for c in self.elements():
            if pred(c):
                out.append(c)
            else:
                out.extend(c.find_all(pred))
        return out

    def has(self, tag: str) -> bool:
        return self.find(lambda n: n.tag == tag) is not None


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, {k: (v or "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Node(tag, {k: (v or "") for k, v in attrs}))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return
        # a stray end tag: ignore it

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _dom(text: str) -> _Node:
    b = _Builder()
    b.feed(text)
    b.close()
    return b.root


def _is(cls: str, tag: str | None = None):
    return lambda n: cls in n.classes and (tag is None or n.tag == tag)


def _text(node: _Node | str | None, skip=None) -> str:
    """Whitespace-collapsed text of `node`, minus scripts/styles/SVG/controls
    and anything `skip(node)` says to leave out."""
    if node is None:
        return ""
    parts: list[str] = []

    def walk(n):
        if isinstance(n, str):
            parts.append(n)
            return
        if n.tag in _SKIP_TEXT or (skip is not None and skip(n)):
            return
        gap = "" if n.tag in _PHRASING else " "  # "<span>a</span><span>b</span>" reads "a b"
        parts.append(gap)
        for c in n.children:
            walk(c)
        parts.append(gap)

    walk(node)
    return " ".join("".join(parts).split())


def _tone(node: _Node, default: str = "acc") -> str:
    for c in node.classes:
        if c.startswith("t-") and c[2:] in TONES:
            return c[2:]
    return default


def _link(node: _Node | None) -> str | None:
    a = node if node is not None and node.tag == "a" else (node.find(lambda n: n.tag == "a") if node else None)
    return http_url(a.attrs.get("href")) if a is not None else None


def _paras(node: _Node | None) -> list[str]:
    if node is None:
        return []
    ps = [t for t in (_text(p) for p in node.find_all(lambda n: n.tag == "p")) if t]
    if ps:
        return ps
    t = _text(node)
    return [t] if t else []


def _tags(node: _Node | None) -> list[str]:
    if node is None:
        return []
    return [t for t in (_text(s) for s in node.find_all(_is("tag"))) if t]


def _child(node: _Node, pred) -> _Node | None:
    return next((c for c in node.elements() if pred(c)), None)


# -- components ----------------------------------------------------------------

def _tiles(node: _Node) -> Tiles:
    tiles = []
    for t in node.find_all(_is("tile")):
        value = t.find(_is("value"))
        tiles.append(TileB(
            label=_text(t.find(_is("label"))),
            value=_text(value, skip=_is("delta")),
            delta=_text(value.find(_is("delta")) if value else None),
            note=_text(t.find(_is("note"))),
            tone=_tone(t),
        ))
    return Tiles(tiles)


def _table(node: _Node) -> Table:
    table = node if node.tag == "table" else node.find(lambda n: n.tag == "table")
    if table is None:
        return Table([], [])
    head = table.find(lambda n: n.tag == "thead")
    headers = [_text(th) for th in head.find_all(lambda n: n.tag == "th")] if head else []
    body = table.find(lambda n: n.tag == "tbody") or table
    rows = []
    for tr in body.find_all(lambda n: n.tag == "tr"):
        if head is not None and tr.find(lambda n: n.tag == "th") and not tr.find(lambda n: n.tag == "td"):
            continue
        cells = [Cell(Span(_text(td), _link(td)), num="num" in td.classes)
                 for td in tr.elements() if td.tag in ("td", "th")]
        if cells:
            rows.append(cells)
    return Table(headers, rows)


def _bars(node: _Node) -> Bars:
    rows = []
    for bar in node.find_all(_is("bar")):
        fill = bar.find(_is("fill"))
        m = _PCT_RE.search(fill.attrs.get("style", "")) if fill else None
        pct = min(max(float(m.group(1)), 0.0), 100.0) if m else 0.0
        rows.append(BarRow(_text(bar.find(_is("name"))), _text(bar.find(_is("amt"))), pct))
    return Bars(rows)


def _stacked(node: _Node) -> Stacked:
    segs = []
    legend = node.find(_is("stacked-legend"))
    for span in (legend.elements() if legend else []):
        text = _text(span)
        m = _LEGEND_RE.match(text)
        pct = min(int(m.group(2)), 100) if m else 0
        if text:
            segs.append((text, pct, _tone(span, "muted")))
    return Stacked(_text(node.find(_is("stacked-label"))), segs)


def _callout(node: _Node) -> Callout:
    label = _text(_child(node, _is("tag")))
    return Callout(label, _paras(node) if node.find(lambda n: n.tag == "p") else
                   [t for t in [_text(node, skip=_is("tag"))] if t], _tone(node, "warn"))


def _card(node: _Node) -> CardB:
    h3 = node.find(lambda n: n.tag == "h3")
    return CardB(
        title=Span(_text(h3) or "(untitled)", _link(h3)),
        meta=_text(_child(node, _is("meta"))),
        ribbon=_text(_child(node, _is("ribbon"))),
        paras=_paras(_child(node, _is("body"))),
        tags=_tags(_child(node, _is("tags"))),
        tone=_tone(node),
    )


def _story(node: _Node) -> StoryB:
    h3 = node.find(lambda n: n.tag == "h3")
    s = StoryB(
        title=_text(h3, skip=_is("story-badge")) or "(untitled)",
        badge=_text(h3.find(_is("story-badge")) if h3 else None),
        meta=_text(_child(node, _is("meta"))),
        stacked=[_stacked(b) for b in node.find_all(_is("stacked"))],
        paras=_paras(_child(node, _is("body"))),
        tone=_tone(node),
    )
    hl = _child(node, _is("story-highlights"))
    s.highlights = [t for t in (_text(li) for li in (hl.elements() if hl else [])) if t]
    quotes = _child(node, _is("story-quotes"))
    for bq in (quotes.find_all(lambda n: n.tag == "blockquote") if quotes else []):
        s.quotes.append(Quote(
            _text(bq, skip=lambda n: bool(n.classes & {"story-cite", "story-unverified"})),
            _link(bq.find(_is("story-cite"))),
            verified=not ("unverified" in bq.classes or bq.find(_is("story-unverified"))),
        ))
    tl = _child(node, _is("story-timeline"))
    for li in (tl.elements() if tl else []):
        s.timeline.append((_text(li.find(_is("when"))), _text(li, skip=_is("when"))))
    s.why = _text(_child(node, _is("story-why")), skip=_is("tag"))
    ex = _child(node, _is("story-extras"))
    for li in (ex.elements() if ex else []):
        s.extras.append(Extra(_text(li.find(_is("tag"))), Span(_text(li, skip=_is("tag")), _link(li)),
                              _tone(li, "muted")))
    src = _child(node, _is("story-sources"))
    for item in (src.find_all(_is("story-source")) if src else []):
        badge = item.find(_is("tag"))
        s.sources.append(Source(Span(_text(item, skip=_is("tag")), _link(item)), _text(badge),
                                _tone(badge, "muted") if badge else "muted"))
    s.tags = _tags(_child(node, _is("tags")))
    return s


def _omitted(what: str) -> Note:
    return Note(f"{what} omitted: email can't show it (it's in the original report)")


def _blocks(node: _Node | str, out: list) -> None:
    """Append the blocks for one child of a report section (or of the page
    body, above the sections)."""
    if isinstance(node, str):
        t = " ".join(node.split())
        if t:
            out.append(Para(t))
        return
    c, tag = node.classes, node.tag
    if tag in ("script", "style", "template", "img", "input", "head", "meta", "link", "title") or "anchor" in c:
        return
    if tag == "svg":
        out.append(_omitted("chart"))
    elif tag == "p":
        if t := _text(node):
            out.append(Para(t))
    elif "tiles" in c:
        out.append(_tiles(node))
    elif "tile" in c:
        out.append(_tiles(_wrap(node)))
    elif "table-wrap" in c or tag == "table":
        out.append(_table(node))
    elif "bars" in c:
        out.append(_bars(node))
    elif "heatmap-wrap" in c:
        out.append(_omitted("heatmap"))
    elif "geo-map" in c:
        out.append(_omitted("map"))
    elif "callout" in c:
        out.append(_callout(node))
    elif "stacked" in c:
        out.append(_stacked(node))
    elif "cards" in c or "stories" in c:
        for child in node.children:
            _blocks(child, out)
    elif tag == "article" and "story" in c:
        out.append(_story(node))
    elif tag == "article" and "card" in c:
        out.append(_card(node))
    elif tag == "details" and "handled-hidden" in c:
        # Email has no toggle: show the checked-off cards under a subhead.
        summary = _text(node.find(lambda n: n.tag == "summary"), skip=_is("hint"))
        out.append(Subhead(summary.replace("handled hidden", "handled (hidden in the report)") or "Handled"))
        for cards in node.find_all(_is("cards")):
            _blocks(cards, out)
    else:
        # Unknown / raw() markup: its text, escaped later; a chart inside
        # still leaves its note.
        if t := _text(node):
            out.append(Para(t))
        if node.has("svg"):
            out.append(_omitted("chart"))


def _wrap(node: _Node) -> _Node:
    w = _Node("div", {"class": "tiles"})
    w.children.append(node)
    return w


def _section(node: _Node) -> SectionB:
    h2 = _child(node, lambda n: n.tag == "h2")
    sec = SectionB(_text(h2, skip=_is("idx")) or "(untitled section)", _tone(node))
    for child in node.children:
        if child is h2:
            continue
        _blocks(child, sec.blocks)
    return sec


def parse(page: str) -> FullReport:
    """A saved `lmsa.report` page as a `FullReport`. The page's masthead
    (title, kicker, sticker), ticker, hazard strip and footer are left out;
    its subtitle, top-level tiles and every section are kept."""
    root = _dom(page)
    main = root.find(lambda n: n.tag == "main") or root.find(lambda n: n.tag == "body") or root
    full = FullReport()
    for child in main.children:
        if isinstance(child, _Node):
            if child.tag == "section":
                full.sections.append(_section(child))
                continue
            if child.tag in ("header", "footer") or child.classes & {"masthead", "hazard", "ticker"}:
                if child.tag == "header" or "masthead" in child.classes:
                    full.subtitle = _text(child.find(_is("subtitle")))
                continue
        _blocks(child, full.top)
    return full


def load(path: Path) -> FullReport:
    """`parse` a saved report file. Raises OSError when it can't be read
    (missing, a directory, too big)."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise OSError(f"report file too large ({size} bytes)")
    return parse(path.read_text(encoding="utf-8", errors="replace"))
